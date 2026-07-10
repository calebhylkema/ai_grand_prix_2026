"""pymavlink-based adapter for the AI Grand Prix simulator.

Connects to the sim's MAVLink v2 UDP endpoint and the separate UDP vision
stream, exposes telemetry, vision frames, race state, and the broadcast gate
map through a small async surface that mirrors the older MAVSDKAdapter.

The choice to switch from MAVSDK to pymavlink is deliberate: the sim sends
critical race data (active gate index, full gate map, collisions) via
``ENCAPSULATED_DATA`` and ``DATA_TRANSMISSION_HANDSHAKE`` messages that
MAVSDK-Python's high-level wrappers do not expose.

References
----------
- AI-GP Simulator v1.0.3364 / PyAIPilotExample (canonical wire formats)
- VADR-TS-001 (AI Grand Prix Technical Specification, March 2026)
- MAVLink common dialect: https://mavlink.io/en/messages/common.html
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import math
import threading
import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from drone_racing.sim.encapsulated_data import (
    RaceStatus,
    RawGateEntry,
    TrackInfoAssembler,
    parse_race_status,
)
from drone_racing.sim.vision_rx import VisionReceiver
from drone_racing.utils.transforms import ned_to_enu, quat_ned_to_enu
from drone_racing.utils.types import (
    DroneCommand,
    GateWaypoint,
    ImuSample,
    MAVLinkConfig,
    State,
    Telemetry,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Telemetry snapshot — shared between receive thread and public properties.
# ---------------------------------------------------------------------------
@dataclass
class TelemetrySnapshot:
    """Aggregated, frame-converted telemetry from all subscribed streams.

    All values use the internal convention: SI units, Z-up world frame,
    scalar-last quaternions ``[x, y, z, w]``.
    """

    timestamp_ns: int = 0
    """Monotonic timestamp (ns) of the most recent update of any field."""

    # ATTITUDE
    attitude_roll: float = 0.0
    attitude_pitch: float = 0.0
    attitude_yaw: float = 0.0
    attitude_rollspeed: float = 0.0
    attitude_pitchspeed: float = 0.0
    attitude_yawspeed: float = 0.0

    # HIGHRES_IMU (body frame, FRD)
    imu_accel: np.ndarray = field(default_factory=lambda: np.zeros(3))
    imu_gyro: np.ndarray = field(default_factory=lambda: np.zeros(3))
    imu_timestamp_ns: int = 0

    # ODOMETRY / LOCAL_POSITION_NED (converted to Z-up)
    odometry_position: np.ndarray = field(default_factory=lambda: np.zeros(3))
    odometry_velocity: np.ndarray = field(default_factory=lambda: np.zeros(3))
    odometry_orientation: np.ndarray = field(default_factory=lambda: np.array([0.0, 0.0, 0.0, 1.0]))
    odometry_valid: bool = False
    odometry_reset_counter: int = 0
    """ODOMETRY.reset_counter — increments on teleport / scenario reset."""

    # Connection status
    connected: bool = False
    armed: bool = False
    offboard_active: bool = False

    # Staleness tracking
    last_attitude_ns: int = 0
    last_imu_ns: int = 0
    last_odometry_ns: int = 0
    last_heartbeat_ns: int = 0


_TELEMETRY_STALE_NS: int = 500_000_000
"""Telemetry older than 500 ms is considered stale."""


def _is_finite_vec(v: np.ndarray) -> bool:
    return bool(np.all(np.isfinite(v)))


def _is_finite_scalar(*values: float) -> bool:
    return all(math.isfinite(v) for v in values)


class MAVLinkAdapter:
    """pymavlink-based adapter between the autonomy pipeline and the AI-GP sim.

    Mirrors the public interface of the older MAVSDKAdapter so ``run_race.py``
    needs only an import swap, while additionally exposing the sim's
    broadcast gate map, race-status feed, and vision stream.

    Parameters
    ----------
    config : MAVLinkConfig
        Connection parameters.
    """

    def __init__(self, config: MAVLinkConfig) -> None:
        self._cfg = config
        self._telemetry = TelemetrySnapshot()
        self._running = False
        self._mav: Any = None  # mavutil.mavlink_connection — lazy import

        # Background workers
        self._rx_thread: threading.Thread | None = None
        self._heartbeat_thread: threading.Thread | None = None
        self._send_lock = threading.Lock()

        # Sim-specific extras
        self._track_assembler = TrackInfoAssembler()
        self._gate_map: list[RawGateEntry] = []
        self._race_status: RaceStatus | None = None
        self._collisions: list[dict[str, float]] = []

        # Vision
        self._vision: VisionReceiver | None = None

        # Diagnostics
        self._nan_count: int = 0
        self._target_system: int = 1
        self._target_component: int = 1

    # ------------------------------------------------------------------
    # Connection lifecycle
    # ------------------------------------------------------------------
    async def connect(self) -> None:
        """Open the MAVLink UDP endpoint and wait for the first heartbeat.

        Blocking pymavlink calls run on the default executor so the async
        caller is not blocked.
        """
        from pymavlink import mavutil

        loop = asyncio.get_running_loop()
        addr = self._cfg.system_address
        logger.info("Opening MAVLink endpoint %s ...", addr)
        self._mav = await loop.run_in_executor(None, lambda: mavutil.mavlink_connection(addr))
        logger.info("Waiting for sim heartbeat ...")
        hb = await loop.run_in_executor(None, self._mav.wait_heartbeat)
        self._target_system = int(self._mav.target_system)
        self._target_component = int(self._mav.target_component)
        logger.info(
            "Connected — system=%d component=%d (heartbeat type=%d)",
            self._target_system,
            self._target_component,
            int(hb.type),
        )

        self._telemetry.connected = True
        self._telemetry.last_heartbeat_ns = time.monotonic_ns()
        self._running = True

    async def start_offboard(self) -> None:
        """Start telemetry/vision rx threads and arm the drone."""
        if self._mav is None:
            raise RuntimeError("Not connected. Call connect() first.")

        # Receive thread (blocking pymavlink recv_match)
        self._rx_thread = threading.Thread(target=self._rx_loop, name="mavlink-rx", daemon=True)
        self._rx_thread.start()

        # Heartbeat thread (so the sim sees us as a healthy GCS)
        self._heartbeat_thread = threading.Thread(
            target=self._heartbeat_loop, name="mavlink-hb", daemon=True
        )
        self._heartbeat_thread.start()

        # Vision receiver (separate UDP socket)
        self._vision = VisionReceiver(
            bind_ip=self._cfg.vision_bind_ip,
            port=self._cfg.vision_port,
        )
        self._vision.start()

        # Arm via COMMAND_LONG (no MAVSDK "offboard mode" concept here —
        # arming + streaming SET_ATTITUDE_TARGET is sufficient).
        await asyncio.get_running_loop().run_in_executor(None, self._send_arm)
        self._telemetry.armed = True
        self._telemetry.offboard_active = True
        logger.info("Armed and ready for control commands.")

    async def stop(self) -> None:
        """Disarm, stop background threads, close the vision socket."""
        self._running = False

        if self._mav is not None:
            try:
                await asyncio.get_running_loop().run_in_executor(None, self._send_disarm)
            except Exception:
                logger.exception("Disarm send failed")
            self._telemetry.armed = False
            self._telemetry.offboard_active = False

        if self._rx_thread is not None:
            self._rx_thread.join(timeout=1.0)
            self._rx_thread = None
        if self._heartbeat_thread is not None:
            self._heartbeat_thread.join(timeout=1.0)
            self._heartbeat_thread = None
        if self._vision is not None:
            self._vision.stop()
            self._vision = None

        if self._mav is not None:
            with contextlib.suppress(Exception):
                self._mav.close()
            self._mav = None

        self._telemetry.connected = False
        logger.info("MAVLinkAdapter stopped.")

    async def reconnect(self) -> bool:
        """Drop the current connection and re-open from scratch.

        UDP is connectionless, so "reconnect" here means tearing down our
        state machine and re-running connect() / start_offboard().
        """
        logger.warning("Reconnect requested — tearing down and re-opening.")
        await self.stop()
        try:
            await self.connect()
            await self.start_offboard()
            return True
        except Exception:
            logger.exception("Reconnect failed")
            return False

    # ------------------------------------------------------------------
    # Public read-only accessors
    # ------------------------------------------------------------------
    def is_running(self) -> bool:
        return self._running and self._telemetry.connected

    def is_imu_stale(self) -> bool:
        if self._telemetry.last_imu_ns == 0:
            return True
        return (time.monotonic_ns() - self._telemetry.last_imu_ns) > _TELEMETRY_STALE_NS

    def is_attitude_stale(self) -> bool:
        if self._telemetry.last_attitude_ns == 0:
            return True
        return (time.monotonic_ns() - self._telemetry.last_attitude_ns) > _TELEMETRY_STALE_NS

    def is_odometry_stale(self) -> bool:
        if not self._telemetry.odometry_valid:
            return True
        return (time.monotonic_ns() - self._telemetry.last_odometry_ns) > _TELEMETRY_STALE_NS

    @property
    def nan_count(self) -> int:
        return self._nan_count

    @property
    def telemetry(self) -> TelemetrySnapshot:
        return self._telemetry

    @property
    def latest_imu(self) -> ImuSample:
        t = self._telemetry
        return ImuSample(
            timestamp_ns=t.imu_timestamp_ns,
            accel_mps2=t.imu_accel.copy(),
            gyro_rps=t.imu_gyro.copy(),
        )

    @property
    def latest_state(self) -> State:
        """Best available state estimate from telemetry."""
        from scipy.spatial.transform import Rotation

        t = self._telemetry
        if t.odometry_valid:
            orientation = t.odometry_orientation.copy()
            norm = float(np.linalg.norm(orientation))
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
        state = self.latest_state
        return Telemetry(
            timestamp_ns=state.timestamp_ns,
            position=state.position,
            velocity=state.velocity,
            orientation=state.orientation,
        )

    # ------------------------------------------------------------------
    # Sim-specific extras
    # ------------------------------------------------------------------
    @property
    def latest_frame(self) -> np.ndarray | None:
        """Most recent decoded vision frame as BGR ndarray (or None)."""
        if self._vision is None:
            return None
        return self._vision.latest_frame

    @property
    def latest_frame_timestamp_ns(self) -> int | None:
        if self._vision is None:
            return None
        ts = self._vision.latest_frame_timestamp_ns
        return ts if ts > 0 else None

    @property
    def gate_map(self) -> list[RawGateEntry]:
        """Most recently received TRACK_INFO gate list (raw NED frame).

        Empty until the sim broadcasts the map; converted to internal
        Z-up world frame via :meth:`gate_waypoints`.
        """
        return list(self._gate_map)

    @property
    def race_status(self) -> RaceStatus | None:
        return self._race_status

    @property
    def collisions(self) -> list[dict[str, float]]:
        """All collision events received since startup."""
        return list(self._collisions)

    def gate_waypoints(self) -> list[GateWaypoint]:
        """Convert the cached raw gate map to internal-frame waypoints.

        Each entry exposes its position and orientation in the internal
        Z-up world frame, plus its 4 outer corners in world coordinates
        for downstream PnP / racing-line use.
        """
        from scipy.spatial.transform import Rotation

        waypoints: list[GateWaypoint] = []
        for order, gate in enumerate(self._gate_map):
            position_w = ned_to_enu(np.array(gate.position_ned, dtype=np.float64))
            qw, qx, qy, qz = gate.orientation_ned_wxyz
            q_ned_xyzw = np.array([qx, qy, qz, qw], dtype=np.float64)
            orientation_w = quat_ned_to_enu(q_ned_xyzw)
            r_gw = Rotation.from_quat(orientation_w).as_matrix()
            half_w = 0.5 * gate.width_m
            half_h = 0.5 * gate.height_m
            local_corners = np.array(
                [
                    [-half_w, half_h, 0.0],
                    [half_w, half_h, 0.0],
                    [half_w, -half_h, 0.0],
                    [-half_w, -half_h, 0.0],
                ],
                dtype=np.float64,
            )
            corners_w = (r_gw @ local_corners.T).T + position_w
            waypoints.append(
                GateWaypoint(
                    gate_id=gate.gate_id,
                    position_w=position_w,
                    orientation_w=orientation_w,
                    corners_3d_w=corners_w,
                    order=order,
                )
            )
        return waypoints

    # ------------------------------------------------------------------
    # Command sending
    # ------------------------------------------------------------------
    async def send_command(self, cmd: DroneCommand) -> None:
        """Send a rate-mode SET_ATTITUDE_TARGET command.

        Throttle is the collective thrust ``[0, 1]``; the three rates are
        body-frame and in rad/s.
        """
        if self._mav is None or not self._telemetry.offboard_active:
            return

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

        await asyncio.get_running_loop().run_in_executor(None, self._send_attitude_target, cmd)

    async def send_position_command(
        self, north: float, east: float, down: float, yaw_deg: float = 0.0
    ) -> None:
        """Send a position command via SET_POSITION_TARGET_LOCAL_NED (NED frame)."""
        if self._mav is None or not self._telemetry.offboard_active:
            return
        await asyncio.get_running_loop().run_in_executor(
            None, self._send_position_target, north, east, down, math.radians(yaw_deg)
        )

    async def send_attitude_quaternion_command(
        self, quat_wxyz: list[float], throttle: float
    ) -> None:
        """Send an absolute attitude target via SET_ATTITUDE_TARGET (quaternion mode).

        Streams the desired body-to-NED attitude quaternion and a normalised
        thrust value, with all three body rates marked IGNORE so the sim's
        internal stabiliser handles rate control.  This sidesteps the FRD/FLU
        body-frame confusion in our PID code (discovered 2026-06-02): we tell
        the autopilot what attitude we want and let it figure out the rates.

        Parameters
        ----------
        quat_wxyz : list[float]
            Body-to-world (NED) quaternion in MAVLink ``[w, x, y, z]`` order.
        throttle : float
            Normalised collective thrust ``[0, 1]``.
        """
        if self._mav is None or not self._telemetry.offboard_active:
            return
        await asyncio.get_running_loop().run_in_executor(
            None, self._send_attitude_quaternion_target, quat_wxyz, throttle
        )

    async def send_velocity_command(self, vn: float, ve: float, vd: float) -> None:
        """Send a velocity-target command via SET_POSITION_TARGET_LOCAL_NED.

        Mirrors the AI-GP reference's ``VELOCITY_POSITION_MASK``: position,
        acceleration, yaw, and yaw_rate are all ignored, only the three NED
        velocity components are honoured.  Use this when the sim's autopilot
        ignores position-mode commands (observed 2026-06-01).

        Parameters
        ----------
        vn, ve, vd : float
            Velocity in NED frame (m/s). ``vd`` positive is downward.
        """
        if self._mav is None or not self._telemetry.offboard_active:
            return
        await asyncio.get_running_loop().run_in_executor(
            None, self._send_velocity_target, vn, ve, vd
        )

    async def send_sim_reset(self) -> None:
        """Send the sim's custom reset command (MAVLINK_CMD_SIM_RESET = 31000)."""
        if self._mav is None:
            return
        await asyncio.get_running_loop().run_in_executor(None, self._send_sim_reset)

    # ------------------------------------------------------------------
    # Blocking sends (run in executor)
    # ------------------------------------------------------------------
    def _send_attitude_target(self, cmd: DroneCommand) -> None:
        from pymavlink import mavutil

        with self._send_lock:
            type_mask = mavutil.mavlink.ATTITUDE_TARGET_TYPEMASK_ATTITUDE_IGNORE
            now_ms = int(time.monotonic() * 1000) & 0xFFFFFFFF
            self._mav.mav.set_attitude_target_send(
                now_ms,
                self._target_system,
                self._target_component,
                type_mask,
                [1.0, 0.0, 0.0, 0.0],  # ignored quaternion
                float(cmd.roll_rate),
                float(cmd.pitch_rate),
                float(cmd.yaw_rate),
                float(cmd.throttle),
            )

    def _send_position_target(self, north: float, east: float, down: float, yaw_rad: float) -> None:
        from pymavlink import mavutil

        with self._send_lock:
            # Ignore velocity/accel; use position + yaw.
            type_mask = (
                mavutil.mavlink.POSITION_TARGET_TYPEMASK_VX_IGNORE
                | mavutil.mavlink.POSITION_TARGET_TYPEMASK_VY_IGNORE
                | mavutil.mavlink.POSITION_TARGET_TYPEMASK_VZ_IGNORE
                | mavutil.mavlink.POSITION_TARGET_TYPEMASK_AX_IGNORE
                | mavutil.mavlink.POSITION_TARGET_TYPEMASK_AY_IGNORE
                | mavutil.mavlink.POSITION_TARGET_TYPEMASK_AZ_IGNORE
                | mavutil.mavlink.POSITION_TARGET_TYPEMASK_YAW_RATE_IGNORE
            )
            now_ms = int(time.monotonic() * 1000) & 0xFFFFFFFF
            self._mav.mav.set_position_target_local_ned_send(
                now_ms,
                self._target_system,
                self._target_component,
                mavutil.mavlink.MAV_FRAME_LOCAL_NED,
                type_mask,
                float(north),
                float(east),
                float(down),
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                float(yaw_rad),
                0.0,
            )

    def _send_attitude_quaternion_target(self, quat_wxyz: list[float], throttle: float) -> None:
        from pymavlink import mavutil

        with self._send_lock:
            type_mask = (
                mavutil.mavlink.ATTITUDE_TARGET_TYPEMASK_BODY_ROLL_RATE_IGNORE
                | mavutil.mavlink.ATTITUDE_TARGET_TYPEMASK_BODY_PITCH_RATE_IGNORE
                | mavutil.mavlink.ATTITUDE_TARGET_TYPEMASK_BODY_YAW_RATE_IGNORE
            )
            now_ms = int(time.monotonic() * 1000) & 0xFFFFFFFF
            self._mav.mav.set_attitude_target_send(
                now_ms,
                self._target_system,
                self._target_component,
                type_mask,
                [
                    float(quat_wxyz[0]),
                    float(quat_wxyz[1]),
                    float(quat_wxyz[2]),
                    float(quat_wxyz[3]),
                ],
                0.0,
                0.0,
                0.0,
                float(np.clip(throttle, 0.0, 1.0)),
            )

    def _send_velocity_target(self, vn: float, ve: float, vd: float) -> None:
        from pymavlink import mavutil

        with self._send_lock:
            type_mask = (
                mavutil.mavlink.POSITION_TARGET_TYPEMASK_X_IGNORE
                | mavutil.mavlink.POSITION_TARGET_TYPEMASK_Y_IGNORE
                | mavutil.mavlink.POSITION_TARGET_TYPEMASK_Z_IGNORE
                | mavutil.mavlink.POSITION_TARGET_TYPEMASK_AX_IGNORE
                | mavutil.mavlink.POSITION_TARGET_TYPEMASK_AY_IGNORE
                | mavutil.mavlink.POSITION_TARGET_TYPEMASK_AZ_IGNORE
                | mavutil.mavlink.POSITION_TARGET_TYPEMASK_YAW_IGNORE
                | mavutil.mavlink.POSITION_TARGET_TYPEMASK_YAW_RATE_IGNORE
            )
            now_ms = int(time.monotonic() * 1000) & 0xFFFFFFFF
            self._mav.mav.set_position_target_local_ned_send(
                now_ms,
                self._target_system,
                self._target_component,
                mavutil.mavlink.MAV_FRAME_LOCAL_NED,
                type_mask,
                0.0,
                0.0,
                0.0,
                float(vn),
                float(ve),
                float(vd),
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
            )

    def _send_arm(self) -> None:
        from pymavlink import mavutil

        with self._send_lock:
            self._mav.mav.command_long_send(
                self._target_system,
                self._target_component,
                mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM,
                0,
                1,
                0,
                0,
                0,
                0,
                0,
                0,
            )

    def _send_disarm(self) -> None:
        from pymavlink import mavutil

        with self._send_lock:
            self._mav.mav.command_long_send(
                self._target_system,
                self._target_component,
                mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
            )

    def _send_sim_reset(self) -> None:
        with self._send_lock:
            # MAVLINK_CMD_SIM_RESET = 31000 (custom, from PyAIPilotExample)
            self._mav.mav.command_long_send(
                self._target_system,
                self._target_component,
                31000,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
            )

    def _send_heartbeat(self) -> None:
        from pymavlink import mavutil

        with self._send_lock:
            self._mav.mav.heartbeat_send(
                mavutil.mavlink.MAV_TYPE_GCS,
                mavutil.mavlink.MAV_AUTOPILOT_INVALID,
                0,
                0,
                0,
            )

    # ------------------------------------------------------------------
    # Background loops
    # ------------------------------------------------------------------
    def _heartbeat_loop(self) -> None:
        period = 1.0 / max(0.5, self._cfg.heartbeat_rate_hz)
        while self._running and self._mav is not None:
            try:
                self._send_heartbeat()
            except Exception:
                logger.exception("Heartbeat send failed")
                return
            time.sleep(period)

    def _rx_loop(self) -> None:
        """Continuously drain incoming MAVLink messages and dispatch."""
        while self._running and self._mav is not None:
            try:
                msg = self._mav.recv_match(blocking=False)
            except ConnectionResetError:
                logger.warning("MAVLink connection reset")
                return
            except Exception:
                logger.exception("MAVLink recv error")
                return
            if msg is None:
                time.sleep(0.001)
                continue
            try:
                self._dispatch(msg)
            except Exception:
                logger.exception("Failed to handle MAVLink %s", msg.get_type())

    # ------------------------------------------------------------------
    # Message dispatch
    # ------------------------------------------------------------------
    def _dispatch(self, msg: Any) -> None:
        msg_type = msg.get_type()
        if msg_type == "BAD_DATA":
            return
        if msg_type == "HEARTBEAT":
            self._on_heartbeat(msg)
        elif msg_type == "ATTITUDE":
            self._on_attitude(msg)
        elif msg_type == "HIGHRES_IMU":
            self._on_highres_imu(msg)
        elif msg_type == "ODOMETRY":
            self._on_odometry(msg)
        elif msg_type == "LOCAL_POSITION_NED":
            self._on_local_position_ned(msg)
        elif msg_type == "ENCAPSULATED_DATA":
            self._on_encapsulated_data(msg)
        elif msg_type == "DATA_TRANSMISSION_HANDSHAKE":
            self._on_data_transmission_handshake(msg)
        elif msg_type == "COLLISION":
            self._on_collision(msg)
        # TIMESYNC, ACTUATOR_OUTPUT_STATUS are accepted but unused for now.

    def _on_heartbeat(self, msg: Any) -> None:
        self._telemetry.last_heartbeat_ns = time.monotonic_ns()

    def _on_attitude(self, msg: Any) -> None:
        roll, pitch, yaw = msg.roll, msg.pitch, msg.yaw
        roll_speed, pitch_speed, yaw_speed = msg.rollspeed, msg.pitchspeed, msg.yawspeed
        if not _is_finite_scalar(roll, pitch, yaw, roll_speed, pitch_speed, yaw_speed):
            self._nan_count += 1
            return
        t = self._telemetry
        t.attitude_roll = float(roll)
        t.attitude_pitch = float(pitch)
        t.attitude_yaw = float(yaw)
        t.attitude_rollspeed = float(roll_speed)
        t.attitude_pitchspeed = float(pitch_speed)
        t.attitude_yawspeed = float(yaw_speed)
        now = time.monotonic_ns()
        t.timestamp_ns = now
        t.last_attitude_ns = now

    def _on_highres_imu(self, msg: Any) -> None:
        accel = np.array([msg.xacc, msg.yacc, msg.zacc], dtype=np.float64)
        gyro = np.array([msg.xgyro, msg.ygyro, msg.zgyro], dtype=np.float64)
        if not (_is_finite_vec(accel) and _is_finite_vec(gyro)):
            self._nan_count += 1
            return
        t = self._telemetry
        t.imu_accel = accel
        t.imu_gyro = gyro
        now = time.monotonic_ns()
        t.imu_timestamp_ns = now
        t.last_imu_ns = now

    def _on_odometry(self, msg: Any) -> None:
        pos_ned = np.array([msg.x, msg.y, msg.z], dtype=np.float64)
        vel_ned = np.array([msg.vx, msg.vy, msg.vz], dtype=np.float64)
        # MAVLink ODOMETRY.q is scalar-first (w, x, y, z); convert to scalar-last.
        q_ned_xyzw = np.array([msg.q[1], msg.q[2], msg.q[3], msg.q[0]], dtype=np.float64)
        if not (_is_finite_vec(pos_ned) and _is_finite_vec(vel_ned) and _is_finite_vec(q_ned_xyzw)):
            self._nan_count += 1
            return
        q_norm = float(np.linalg.norm(q_ned_xyzw))
        if q_norm < 1e-6:
            self._nan_count += 1
            return

        t = self._telemetry
        t.odometry_position = ned_to_enu(pos_ned)
        t.odometry_velocity = ned_to_enu(vel_ned)
        t.odometry_orientation = quat_ned_to_enu(q_ned_xyzw / q_norm)
        t.odometry_valid = True
        t.odometry_reset_counter = int(getattr(msg, "reset_counter", 0))
        now = time.monotonic_ns()
        t.timestamp_ns = now
        t.last_odometry_ns = now

    def _on_local_position_ned(self, msg: Any) -> None:
        """Fallback when ODOMETRY is absent — position + velocity only."""
        if self._telemetry.odometry_valid:
            # ODOMETRY is richer; ignore LOCAL_POSITION_NED to avoid races.
            return
        pos_ned = np.array([msg.x, msg.y, msg.z], dtype=np.float64)
        vel_ned = np.array([msg.vx, msg.vy, msg.vz], dtype=np.float64)
        if not (_is_finite_vec(pos_ned) and _is_finite_vec(vel_ned)):
            self._nan_count += 1
            return
        t = self._telemetry
        t.odometry_position = ned_to_enu(pos_ned)
        t.odometry_velocity = ned_to_enu(vel_ned)
        # Orientation stays at whatever ATTITUDE gave us.
        t.odometry_valid = True
        now = time.monotonic_ns()
        t.timestamp_ns = now
        t.last_odometry_ns = now

    def _on_data_transmission_handshake(self, msg: Any) -> None:
        """Sim repurposes DATA_TRANSMISSION_HANDSHAKE to announce track transfers.

        ``width`` carries the transfer id; ``packets`` is the chunk count.
        """
        transfer_id = int(getattr(msg, "width", 0))
        packets = int(getattr(msg, "packets", 0))
        if packets > 0:
            self._track_assembler.handshake(transfer_id, packets)

    def _on_encapsulated_data(self, msg: Any) -> None:
        raw = bytes(msg.data)
        if not raw:
            return
        data_type = raw[0]
        if data_type == 1:  # RACE_STATUS
            rs = parse_race_status(raw)
            if rs is not None:
                self._race_status = rs
        elif data_type == 2:  # TRACK_INFO chunk
            seqnr = int(getattr(msg, "seqnr", 0))
            gates = self._track_assembler.add_chunk(seqnr, raw)
            if gates is not None and gates:
                self._gate_map = gates
                logger.info("TRACK_INFO received: %d gates", len(gates))

    def _on_collision(self, msg: Any) -> None:
        event = {
            "collision_id": float(getattr(msg, "id", 0)),
            "threat_level": float(getattr(msg, "threat_level", 0)),
            "impulse": float(getattr(msg, "horizontal_minimum_delta", 0.0)),
            "timestamp_ns": float(time.monotonic_ns()),
        }
        self._collisions.append(event)
        kind = "gate" if int(event["collision_id"]) == 1001 else "environment"
        logger.warning(
            "COLLISION (%s) threat=%d impulse=%.2f kg·m/s",
            kind,
            int(event["threat_level"]),
            event["impulse"],
        )
