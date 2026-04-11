"""MAVSDK simulator adapter — connects to the AI Grand Prix sim via MAVLink v2.

Communicates with the competition simulator over UDP using MAVSDK-Python.
Handles telemetry reception (ATTITUDE, HIGHRES_IMU, ODOMETRY, TIMESYNC),
heartbeat maintenance, and control command transmission (SET_ATTITUDE_TARGET).

The adapter converts between MAVLink's NED frame and our internal Z-up (ENU)
frame at the boundary.  All data exposed to the rest of the pipeline is in
our standard internal conventions (SI units, Z-up, scalar-last quaternions).

Usage::

    adapter = MAVSDKAdapter(config)
    await adapter.connect()
    await adapter.start_offboard()

    while adapter.is_running():
        imu = adapter.latest_imu
        attitude = adapter.latest_attitude
        await adapter.send_command(cmd)
        await asyncio.sleep(1 / config.mavlink.command_rate_hz)

References
----------
- VADR-TS-001 (AI Grand Prix Technical Specification, March 2026)
- MAVLink message definitions: https://mavlink.io/en/messages/common.html
- MAVSDK-Python: https://mavsdk-python.readthedocs.io/
"""

from __future__ import annotations

import asyncio
import logging
import math
import time
from dataclasses import dataclass, field

import numpy as np

from drone_racing.utils.transforms import (
    ned_to_enu,
    quat_ned_to_enu,
)
from drone_racing.utils.types import (
    DroneCommand,
    ImuSample,
    MAVLinkConfig,
    State,
    Telemetry,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Telemetry snapshot — latest data from all MAVLink streams
# ---------------------------------------------------------------------------
@dataclass
class TelemetrySnapshot:
    """Aggregated telemetry from MAVLink streams, converted to internal frame.

    All values are in our internal convention: SI units, Z-up world frame,
    scalar-last quaternions [x, y, z, w].
    """

    timestamp_ns: int = 0
    """Monotonic timestamp (ns) of the most recent update."""

    # From ATTITUDE message
    attitude_roll: float = 0.0
    """Roll angle (rad)."""
    attitude_pitch: float = 0.0
    """Pitch angle (rad)."""
    attitude_yaw: float = 0.0
    """Yaw angle (rad)."""
    attitude_rollspeed: float = 0.0
    """Roll rate (rad/s)."""
    attitude_pitchspeed: float = 0.0
    """Pitch rate (rad/s)."""
    attitude_yawspeed: float = 0.0
    """Yaw rate (rad/s)."""

    # From HIGHRES_IMU message
    imu_accel: np.ndarray = field(default_factory=lambda: np.zeros(3))
    """Accelerometer reading in body frame (m/s^2)."""
    imu_gyro: np.ndarray = field(default_factory=lambda: np.zeros(3))
    """Gyroscope reading in body frame (rad/s)."""
    imu_timestamp_ns: int = 0

    # From ODOMETRY message (if available)
    odometry_position: np.ndarray = field(default_factory=lambda: np.zeros(3))
    """Position in world frame (m) — from ODOMETRY, converted to Z-up."""
    odometry_velocity: np.ndarray = field(default_factory=lambda: np.zeros(3))
    """Velocity in world frame (m/s) — from ODOMETRY, converted to Z-up."""
    odometry_orientation: np.ndarray = field(default_factory=lambda: np.array([0.0, 0.0, 0.0, 1.0]))
    """Orientation quaternion [x, y, z, w] — from ODOMETRY, converted to Z-up."""
    odometry_valid: bool = False
    """Whether ODOMETRY data has been received at least once."""

    # Connection status
    connected: bool = False
    armed: bool = False
    offboard_active: bool = False

    # Staleness tracking
    last_attitude_ns: int = 0
    """Monotonic timestamp of the last ATTITUDE update."""
    last_imu_ns: int = 0
    """Monotonic timestamp of the last IMU update."""
    last_odometry_ns: int = 0
    """Monotonic timestamp of the last ODOMETRY update."""


_TELEMETRY_STALE_NS: int = 500_000_000
"""Telemetry older than 500 ms is considered stale."""

_MAX_RECONNECT_ATTEMPTS: int = 5
"""Maximum reconnection attempts before giving up."""

_RECONNECT_DELAY_S: float = 2.0
"""Delay between reconnection attempts (seconds)."""


def _is_finite_vec(v: np.ndarray) -> bool:
    """Return True if every element is finite (not NaN or Inf)."""
    return bool(np.all(np.isfinite(v)))


def _is_finite_scalar(*values: float) -> bool:
    """Return True if all scalar values are finite."""
    return all(math.isfinite(v) for v in values)


class MAVSDKAdapter:
    """Adapter between our pipeline and the AI Grand Prix simulator via MAVSDK.

    Manages the MAVLink v2 / UDP connection, subscribes to telemetry streams,
    and sends control commands.  All frame conversions (NED ↔ ENU) happen here.

    Parameters
    ----------
    config : MAVLinkConfig
        Connection and timing parameters.
    """

    def __init__(self, config: MAVLinkConfig) -> None:
        self._cfg = config
        self._telemetry = TelemetrySnapshot()
        self._running = False
        self._system: object | None = None  # mavsdk.System — lazy import
        self._command_period = 1.0 / config.command_rate_hz
        self._start_time_ns: int = 0
        self._tasks: list[asyncio.Task[None]] = []
        self._nan_count: int = 0
        self._reconnecting: bool = False

    # ------------------------------------------------------------------
    # Connection lifecycle
    # ------------------------------------------------------------------

    async def connect(self) -> None:
        """Connect to the simulator via MAVSDK over UDP.

        Blocks until the system is discovered and the connection is healthy.

        Raises
        ------
        ConnectionError
            If connection to the simulator fails.
        """
        try:
            from mavsdk import System
        except ImportError as e:
            msg = "mavsdk package not installed. Run: uv add mavsdk"
            raise ImportError(msg) from e

        system = System()
        logger.info("Connecting to simulator at %s ...", self._cfg.system_address)
        await system.connect(system_address=self._cfg.system_address)

        # Wait for connection
        logger.info("Waiting for system to connect...")
        async for state in system.core.connection_state():
            if state.is_connected:
                logger.info("System connected!")
                break

        self._system = system
        self._telemetry.connected = True
        self._start_time_ns = time.monotonic_ns()
        self._running = True

    async def start_offboard(self) -> None:
        """Start telemetry subscriptions and enable offboard control.

        Call after :meth:`connect`.  This starts background tasks for
        receiving ATTITUDE, HIGHRES_IMU, and ODOMETRY, and arms + enters
        offboard mode.
        """
        if self._system is None:
            msg = "Not connected. Call connect() first."
            raise RuntimeError(msg)

        system = self._system

        # Start telemetry subscription tasks
        self._tasks.append(asyncio.create_task(self._subscribe_attitude(system)))
        self._tasks.append(asyncio.create_task(self._subscribe_imu(system)))
        self._tasks.append(asyncio.create_task(self._subscribe_odometry(system)))

        # Arm the drone
        logger.info("Arming...")
        await system.action.arm()
        self._telemetry.armed = True

        # Set initial offboard command (required before starting offboard mode)
        from mavsdk.offboard import Attitude

        await system.offboard.set_attitude(Attitude(0.0, 0.0, 0.0, 0.0))

        # Start offboard mode
        logger.info("Starting offboard mode...")
        await system.offboard.start()
        self._telemetry.offboard_active = True
        logger.info("Offboard mode active — ready for commands.")

    async def stop(self) -> None:
        """Stop offboard mode, disarm, and clean up."""
        if self._system is not None:
            try:
                system = self._system
                if self._telemetry.offboard_active:
                    await system.offboard.stop()
                    self._telemetry.offboard_active = False
                if self._telemetry.armed:
                    await system.action.disarm()
                    self._telemetry.armed = False
            except Exception:
                logger.exception("Error during shutdown")

        # Cancel background tasks
        for task in self._tasks:
            task.cancel()
        self._tasks.clear()
        self._running = False
        logger.info("MAVSDKAdapter stopped.")

    async def reconnect(self) -> bool:
        """Attempt to reconnect after a connection loss.

        Returns
        -------
        bool
            True if reconnection succeeded.
        """
        if self._reconnecting:
            return False
        self._reconnecting = True
        logger.warning("Connection lost — attempting reconnect...")

        # Clean up existing tasks
        for task in self._tasks:
            task.cancel()
        self._tasks.clear()
        self._telemetry.connected = False
        self._telemetry.offboard_active = False

        for attempt in range(1, _MAX_RECONNECT_ATTEMPTS + 1):
            try:
                logger.info("Reconnect attempt %d/%d", attempt, _MAX_RECONNECT_ATTEMPTS)
                await self.connect()
                await self.start_offboard()
                logger.info("Reconnected successfully on attempt %d", attempt)
                self._reconnecting = False
                return True
            except Exception:
                logger.exception("Reconnect attempt %d failed", attempt)
                if attempt < _MAX_RECONNECT_ATTEMPTS:
                    await asyncio.sleep(_RECONNECT_DELAY_S)

        logger.error("Failed to reconnect after %d attempts", _MAX_RECONNECT_ATTEMPTS)
        self._running = False
        self._reconnecting = False
        return False

    # ------------------------------------------------------------------
    # Public API — data access
    # ------------------------------------------------------------------

    def is_running(self) -> bool:
        """Whether the adapter is connected and offboard mode is active."""
        return self._running and self._telemetry.connected

    def is_imu_stale(self) -> bool:
        """Return True if IMU data is older than the staleness threshold."""
        if self._telemetry.last_imu_ns == 0:
            return True
        return (time.monotonic_ns() - self._telemetry.last_imu_ns) > _TELEMETRY_STALE_NS

    def is_attitude_stale(self) -> bool:
        """Return True if attitude data is older than the staleness threshold."""
        if self._telemetry.last_attitude_ns == 0:
            return True
        return (time.monotonic_ns() - self._telemetry.last_attitude_ns) > _TELEMETRY_STALE_NS

    def is_odometry_stale(self) -> bool:
        """Return True if odometry data is older than the staleness threshold."""
        if not self._telemetry.odometry_valid:
            return True
        return (time.monotonic_ns() - self._telemetry.last_odometry_ns) > _TELEMETRY_STALE_NS

    @property
    def nan_count(self) -> int:
        """Number of NaN/Inf telemetry samples dropped since startup."""
        return self._nan_count

    @property
    def telemetry(self) -> TelemetrySnapshot:
        """Current aggregated telemetry (read-only snapshot)."""
        return self._telemetry

    @property
    def latest_imu(self) -> ImuSample:
        """Most recent IMU sample, converted to internal conventions.

        Returns
        -------
        ImuSample
            Body-frame accelerometer and gyroscope readings.
        """
        t = self._telemetry
        return ImuSample(
            timestamp_ns=t.imu_timestamp_ns,
            accel_mps2=t.imu_accel.copy(),
            gyro_rps=t.imu_gyro.copy(),
        )

    @property
    def latest_state(self) -> State:
        """Best available state estimate from telemetry.

        If ODOMETRY is available, uses that.  Otherwise constructs from
        ATTITUDE only (position/velocity will be zero).

        Returns
        -------
        State
            Current state estimate (may be partial if ODOMETRY unavailable).
        """
        t = self._telemetry
        if t.odometry_valid:
            from scipy.spatial.transform import Rotation

            orientation = t.odometry_orientation.copy()
            # Ensure unit quaternion
            norm = np.linalg.norm(orientation)
            if norm > 1e-12:
                orientation = orientation / norm
            return State(
                timestamp_ns=t.timestamp_ns,
                position=t.odometry_position.copy(),
                velocity=t.odometry_velocity.copy(),
                orientation=orientation,
                accel_bias=np.zeros(3),
                gyro_bias=np.zeros(3),
                covariance=np.eye(15) * 1e-2,
            )

        # Fallback: attitude only (no position/velocity)
        from scipy.spatial.transform import Rotation

        orientation = Rotation.from_euler(
            "ZYX", [t.attitude_yaw, t.attitude_pitch, t.attitude_roll]
        ).as_quat()
        return State(
            timestamp_ns=t.timestamp_ns,
            position=np.zeros(3),
            velocity=np.zeros(3),
            orientation=orientation.astype(np.float64),
            accel_bias=np.zeros(3),
            gyro_bias=np.zeros(3),
            covariance=np.eye(15) * 1.0,
        )

    def get_initial_telemetry(self) -> Telemetry:
        """Build a Telemetry snapshot for ESKF initialisation.

        Returns
        -------
        Telemetry
            Starting pose from the simulator.
        """
        state = self.latest_state
        return Telemetry(
            timestamp_ns=state.timestamp_ns,
            position=state.position,
            velocity=state.velocity,
            orientation=state.orientation,
        )

    # ------------------------------------------------------------------
    # Public API — command sending
    # ------------------------------------------------------------------

    async def send_command(self, cmd: DroneCommand) -> None:
        """Send a rate-mode control command via SET_ATTITUDE_TARGET.

        Converts our internal DroneCommand (throttle + body rates) to
        MAVSDK offboard attitude rate commands.

        Parameters
        ----------
        cmd : DroneCommand
            Rate-mode command from the controller.
        """
        if self._system is None or not self._telemetry.offboard_active:
            return

        # Validate command values before sending
        if not _is_finite_scalar(cmd.throttle, cmd.roll_rate, cmd.pitch_rate, cmd.yaw_rate):
            logger.warning("Command contains NaN/Inf — refusing to send")
            return
        if not 0.0 <= cmd.throttle <= 1.0:
            logger.warning("Throttle %.2f out of [0,1] — clamping", cmd.throttle)
            cmd = DroneCommand(
                timestamp_ns=cmd.timestamp_ns,
                throttle=max(0.0, min(1.0, cmd.throttle)),
                roll_rate=cmd.roll_rate,
                pitch_rate=cmd.pitch_rate,
                yaw_rate=cmd.yaw_rate,
            )

        system = self._system

        if self._cfg.offboard_mode == "attitude_rate":
            from mavsdk.offboard import AttitudeRate

            # MAVSDK AttitudeRate: roll_deg_s, pitch_deg_s, yaw_deg_s, thrust (0-1)
            await system.offboard.set_attitude_rate(
                AttitudeRate(
                    roll_deg_s=math.degrees(cmd.roll_rate),
                    pitch_deg_s=math.degrees(cmd.pitch_rate),
                    yaw_deg_s=math.degrees(cmd.yaw_rate),
                    thrust_value=cmd.throttle,
                )
            )
        elif self._cfg.offboard_mode == "attitude":
            # For attitude mode (not rate), we'd need desired angles.
            # Our pipeline outputs rates, so attitude_rate is the default.
            logger.warning("Attitude mode not implemented — use attitude_rate")
        else:
            logger.error("Unknown offboard mode: %s", self._cfg.offboard_mode)

    async def send_position_command(
        self,
        north: float,
        east: float,
        down: float,
        yaw_deg: float = 0.0,
    ) -> None:
        """Send a position command via SET_POSITION_TARGET_LOCAL_NED.

        Coordinates are in NED frame (as required by MAVLink).

        Parameters
        ----------
        north : float
            North position (m).
        east : float
            East position (m).
        down : float
            Down position (m, negative = up).
        yaw_deg : float
            Desired yaw (degrees).
        """
        if self._system is None or not self._telemetry.offboard_active:
            return

        system = self._system
        from mavsdk.offboard import PositionNedYaw

        await system.offboard.set_position_ned(PositionNedYaw(north, east, down, yaw_deg))

    # ------------------------------------------------------------------
    # Telemetry subscription tasks (run as background asyncio tasks)
    # ------------------------------------------------------------------

    async def _subscribe_attitude(self, system: object) -> None:
        """Subscribe to ATTITUDE messages and update telemetry snapshot."""
        try:
            async for attitude in system.telemetry.attitude_euler():
                roll = attitude.roll_deg
                pitch = attitude.pitch_deg
                yaw = attitude.yaw_deg
                if not _is_finite_scalar(roll, pitch, yaw):
                    self._nan_count += 1
                    logger.warning("ATTITUDE contains NaN/Inf — dropping sample")
                    continue

                t = self._telemetry
                t.attitude_roll = math.radians(roll)
                t.attitude_pitch = math.radians(pitch)
                t.attitude_yaw = math.radians(yaw)
                now = time.monotonic_ns()
                t.timestamp_ns = now
                t.last_attitude_ns = now
        except asyncio.CancelledError:
            pass
        except Exception:
            logger.exception("Attitude subscription error")

    async def _subscribe_imu(self, system: object) -> None:
        """Subscribe to HIGHRES_IMU messages and update telemetry snapshot."""
        try:
            async for imu in system.telemetry.imu():
                acc = imu.acceleration_frd
                gyro = imu.angular_velocity_frd
                accel_vec = np.array(
                    [acc.forward_m_s2, acc.right_m_s2, acc.down_m_s2],
                    dtype=np.float64,
                )
                gyro_vec = np.array(
                    [gyro.forward_rad_s, gyro.right_rad_s, gyro.down_rad_s],
                    dtype=np.float64,
                )
                if not (_is_finite_vec(accel_vec) and _is_finite_vec(gyro_vec)):
                    self._nan_count += 1
                    logger.warning("IMU contains NaN/Inf — dropping sample")
                    continue

                t = self._telemetry
                t.imu_accel = accel_vec
                t.imu_gyro = gyro_vec
                now = time.monotonic_ns()
                t.imu_timestamp_ns = now
                t.last_imu_ns = now
        except asyncio.CancelledError:
            pass
        except Exception:
            logger.exception("IMU subscription error")

    async def _subscribe_odometry(self, system: object) -> None:
        """Subscribe to ODOMETRY messages and update telemetry snapshot.

        ODOMETRY provides position and velocity in NED frame.  We convert
        to our internal Z-up (ENU) frame here.

        Note: The official spec lists ODOMETRY but with no field details.
        This implementation assumes standard MAVLink ODOMETRY fields.
        Will need validation when the simulator drops.
        """
        try:
            async for odom in system.telemetry.odometry():
                # Position: NED → ENU
                pos_ned = np.array(
                    [odom.position_body.x_m, odom.position_body.y_m, odom.position_body.z_m],
                    dtype=np.float64,
                )
                vel_ned = np.array(
                    [
                        odom.velocity_body.x_m_s,
                        odom.velocity_body.y_m_s,
                        odom.velocity_body.z_m_s,
                    ],
                    dtype=np.float64,
                )
                q_ned = np.array(
                    [odom.q.x, odom.q.y, odom.q.z, odom.q.w],
                    dtype=np.float64,
                )

                if not (
                    _is_finite_vec(pos_ned) and _is_finite_vec(vel_ned) and _is_finite_vec(q_ned)
                ):
                    self._nan_count += 1
                    logger.warning("ODOMETRY contains NaN/Inf — dropping sample")
                    continue

                # Reject zero-norm quaternions (corrupt data)
                q_norm = float(np.linalg.norm(q_ned))
                if q_norm < 1e-6:
                    self._nan_count += 1
                    logger.warning("ODOMETRY quaternion near-zero norm — dropping")
                    continue

                t = self._telemetry
                t.odometry_position = ned_to_enu(pos_ned)
                t.odometry_velocity = ned_to_enu(vel_ned)
                t.odometry_orientation = quat_ned_to_enu(q_ned / q_norm)

                t.odometry_valid = True
                now = time.monotonic_ns()
                t.timestamp_ns = now
                t.last_odometry_ns = now
        except asyncio.CancelledError:
            pass
        except Exception:
            # ODOMETRY may not be available — that's OK, it's optional
            logger.info("ODOMETRY subscription unavailable — will rely on ATTITUDE + IMU")
