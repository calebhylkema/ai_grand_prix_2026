"""Mock simulator adapter — basic physics, synthetic camera, and IMU generation.

Provides a self-contained simulation loop for integration testing.  The drone
flies through a configurable gate circuit with:

- 6-DOF rigid-body physics (Euler integration at sim rate)
- Synthetic RGB frames with coloured gate polygons for the classical detector
- Simulated IMU readings with configurable noise

Usage::

    adapter = MockSimAdapter(config, gate_waypoints)
    adapter.reset(initial_telemetry)
    while adapter.is_running():
        frame, ts = adapter.get_frame()
        imu = adapter.get_imu()
        # ... run pipeline ...
        adapter.send_command(cmd)
        adapter.step()
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING

import cv2
import numpy as np
from scipy.spatial.transform import Rotation

from drone_racing.utils.transforms import (
    GRAVITY_W,
    body_to_camera_rotation,
    quat_multiply,
    quat_normalize,
    quat_to_rotation_matrix,
    small_angle_to_quat,
)
from drone_racing.utils.types import (
    DroneCommand,
    GateWaypoint,
    ImuSample,
    State,
    Telemetry,
)

if TYPE_CHECKING:
    from drone_racing.utils.types import CameraConfig, Vec3, Vec4

logger = logging.getLogger(__name__)

# Physics constants
_G = 9.81
_HOVER_THROTTLE = 0.5
_MAX_THRUST_ACCEL = _G / _HOVER_THROTTLE  # m/s² at throttle=1.0


@dataclass(slots=True)
class MockSimConfig:
    """Configuration for the mock simulator.

    Parameters
    ----------
    sim_rate_hz : float
        Physics simulation rate (Hz).
    imu_rate_hz : float
        IMU output rate (Hz).
    camera_rate_hz : float
        Camera frame output rate (Hz).
    max_time_s : float
        Maximum simulation duration (s).
    gyro_noise_std : float
        Gyroscope noise standard deviation (rad/s).
    accel_noise_std : float
        Accelerometer noise standard deviation (m/s^2).
    gate_colour_bgr : tuple[int, int, int]
        Gate colour in BGR for rendering.
    gate_thickness : int
        Gate border thickness in pixels (-1 for filled).
    drag_coeff : float
        Linear drag coefficient (1/s).  Prevents runaway velocities.
    angular_drag : float
        Angular drag coefficient (1/s).
    """

    sim_rate_hz: float = 500.0
    imu_rate_hz: float = 200.0
    camera_rate_hz: float = 30.0
    max_time_s: float = 60.0
    gyro_noise_std: float = 0.01
    accel_noise_std: float = 0.1
    gate_colour_bgr: tuple[int, int, int] = (0, 200, 255)
    gate_thickness: int = 3
    drag_coeff: float = 0.3
    angular_drag: float = 5.0


@dataclass(slots=True)
class _PhysicsState:
    """Internal physics state for the simulated drone."""

    position: Vec3
    velocity: Vec3
    orientation: Vec4  # [x, y, z, w] scalar-last
    angular_velocity: Vec3  # body-frame rad/s
    timestamp_ns: int = 0


class MockSimAdapter:
    """Mock simulator for integration testing.

    Parameters
    ----------
    camera_config : CameraConfig
        Camera intrinsics and tilt.
    gates : list[GateWaypoint]
        Gate circuit (ordered).
    config : MockSimConfig | None
        Simulation parameters.
    """

    def __init__(
        self,
        camera_config: CameraConfig,
        gates: list[GateWaypoint],
        config: MockSimConfig | None = None,
    ) -> None:
        self._cfg = config or MockSimConfig()
        self._cam = camera_config
        self._gates = gates

        self._sim_dt = 1.0 / self._cfg.sim_rate_hz
        self._imu_period_ns = int(1e9 / self._cfg.imu_rate_hz)
        self._cam_period_ns = int(1e9 / self._cfg.camera_rate_hz)
        self._max_time_ns = int(self._cfg.max_time_s * 1e9)

        # Pre-compute camera-to-body rotation
        self._r_cb = body_to_camera_rotation(self._cam.tilt_deg)
        self._k = self._cam.intrinsic_matrix

        # State
        self._physics = _PhysicsState(
            position=np.zeros(3),
            velocity=np.zeros(3),
            orientation=np.array([0.0, 0.0, 0.0, 1.0]),
            angular_velocity=np.zeros(3),
        )
        self._last_command = DroneCommand(
            timestamp_ns=0, throttle=_HOVER_THROTTLE, roll_rate=0.0, pitch_rate=0.0, yaw_rate=0.0
        )
        self._last_imu_ns: int = 0
        self._last_cam_ns: int = 0
        self._running = False
        self._rng = np.random.default_rng(42)

        # Next gate index for completion tracking
        self._next_gate_idx = 0
        self._gates_passed = 0
        self._laps_completed = 0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def reset(self, telemetry: Telemetry) -> None:
        """Reset simulation to initial conditions.

        Parameters
        ----------
        telemetry : Telemetry
            Starting pose from the race system.
        """
        self._physics = _PhysicsState(
            position=telemetry.position.copy(),
            velocity=telemetry.velocity.copy(),
            orientation=quat_normalize(telemetry.orientation.copy()),
            angular_velocity=np.zeros(3),
            timestamp_ns=telemetry.timestamp_ns,
        )
        self._last_command = DroneCommand(
            timestamp_ns=telemetry.timestamp_ns,
            throttle=_HOVER_THROTTLE,
            roll_rate=0.0,
            pitch_rate=0.0,
            yaw_rate=0.0,
        )
        self._last_imu_ns = telemetry.timestamp_ns
        self._last_cam_ns = telemetry.timestamp_ns
        self._next_gate_idx = 0
        self._gates_passed = 0
        self._laps_completed = 0
        self._running = True
        logger.info("MockSim reset at pos=%s", telemetry.position)

    @property
    def state(self) -> State:
        """Current ground-truth state (for logging/comparison)."""
        return State(
            timestamp_ns=self._physics.timestamp_ns,
            position=self._physics.position.copy(),
            velocity=self._physics.velocity.copy(),
            orientation=self._physics.orientation.copy(),
            accel_bias=np.zeros(3),
            gyro_bias=np.zeros(3),
            covariance=np.eye(15) * 1e-6,
        )

    @property
    def gates_passed(self) -> int:
        """Number of gates passed so far."""
        return self._gates_passed

    @property
    def laps_completed(self) -> int:
        """Number of full laps completed."""
        return self._laps_completed

    def is_running(self) -> bool:
        """Whether the simulation is still active."""
        return self._running

    def send_command(self, cmd: DroneCommand) -> None:
        """Apply a control command.

        Parameters
        ----------
        cmd : DroneCommand
            Rate-mode command from the controller.
        """
        self._last_command = cmd

    def step(self) -> None:
        """Advance physics by one sim timestep."""
        if not self._running:
            return

        p = self._physics
        cmd = self._last_command
        dt = self._sim_dt

        # --- Thrust ---
        # Throttle maps linearly to acceleration along the drone's "up" axis.
        # With identity quaternion (body=world, Z-up world), thrust is world +Z.
        # The thrust direction in body frame is [0,0,1] (world-Z when level),
        # which rotates with the drone orientation.
        thrust_accel = cmd.throttle * _MAX_THRUST_ACCEL
        r_wb = quat_to_rotation_matrix(p.orientation)
        thrust_dir_world = r_wb @ np.array([0.0, 0.0, 1.0])
        accel_thrust_w = thrust_dir_world * thrust_accel

        # --- Gravity + drag ---
        accel_w = accel_thrust_w + GRAVITY_W - self._cfg.drag_coeff * p.velocity

        # --- Update linear state ---
        p.velocity = p.velocity + accel_w * dt
        p.position = p.position + p.velocity * dt

        # --- Angular dynamics ---
        # Rate commands drive angular velocity with first-order response
        desired_omega = np.array([cmd.roll_rate, cmd.pitch_rate, cmd.yaw_rate])
        omega_error = desired_omega - p.angular_velocity
        # Fast first-order convergence (bandwidth ~angular_drag)
        p.angular_velocity = p.angular_velocity + omega_error * self._cfg.angular_drag * dt
        # Clamp to prevent instability
        p.angular_velocity = np.clip(p.angular_velocity, -15.0, 15.0)

        # --- Update orientation ---
        delta_angle = p.angular_velocity * dt
        dq = small_angle_to_quat(delta_angle)
        p.orientation = quat_normalize(quat_multiply(p.orientation, dq))

        # --- Update timestamp ---
        p.timestamp_ns += int(dt * 1e9)

        # --- Gate passage check ---
        self._check_gate_passage()

        # --- Termination ---
        if p.timestamp_ns >= self._max_time_ns:
            self._running = False
            logger.info("MockSim: max time reached (%.1fs)", self._cfg.max_time_s)
        if p.position[2] < -2.0:
            self._running = False
            logger.warning("MockSim: drone crashed (z=%.2f)", p.position[2])

    def get_imu(self) -> ImuSample | None:
        """Get a simulated IMU sample if one is due.

        Returns
        -------
        ImuSample | None
            IMU reading, or None if not yet time for one.
        """
        if self._physics.timestamp_ns - self._last_imu_ns < self._imu_period_ns:
            return None

        p = self._physics
        r_wb = quat_to_rotation_matrix(p.orientation)

        # IMU measures specific force = (non-gravitational acceleration) in body frame.
        # This is thrust + drag (everything except gravity), rotated to body frame.
        cmd = self._last_command
        thrust_accel = cmd.throttle * _MAX_THRUST_ACCEL
        thrust_dir_world = r_wb @ np.array([0.0, 0.0, 1.0])
        non_grav_accel_w = thrust_dir_world * thrust_accel - self._cfg.drag_coeff * p.velocity
        accel_body = r_wb.T @ non_grav_accel_w

        # Add noise
        accel_noisy = accel_body + self._rng.normal(0, self._cfg.accel_noise_std, 3)
        gyro_noisy = p.angular_velocity + self._rng.normal(0, self._cfg.gyro_noise_std, 3)

        self._last_imu_ns = p.timestamp_ns

        return ImuSample(
            timestamp_ns=p.timestamp_ns,
            accel_mps2=accel_noisy,
            gyro_rps=gyro_noisy,
        )

    def get_frame(self) -> tuple[np.ndarray, int] | None:
        """Render a synthetic camera frame if one is due.

        Returns
        -------
        tuple[np.ndarray, int] | None
            (RGB image, timestamp_ns) or None if not yet time.
        """
        if self._physics.timestamp_ns - self._last_cam_ns < self._cam_period_ns:
            return None

        self._last_cam_ns = self._physics.timestamp_ns
        image = self._render_frame()
        return image, self._physics.timestamp_ns

    def step_until_imu(self) -> ImuSample:
        """Step physics until the next IMU sample is due.

        Returns
        -------
        ImuSample
            The next IMU reading.
        """
        while self._running:
            self.step()
            imu = self.get_imu()
            if imu is not None:
                return imu
        # Return a zero sample if sim stopped
        return ImuSample(
            timestamp_ns=self._physics.timestamp_ns,
            accel_mps2=np.array([0.0, 0.0, -_G]),
            gyro_rps=np.zeros(3),
        )

    # ------------------------------------------------------------------
    # Gate passage detection
    # ------------------------------------------------------------------

    def _check_gate_passage(self) -> None:
        """Check if the drone has passed through the next gate."""
        if len(self._gates) == 0:
            return

        gate = self._gates[self._next_gate_idx]
        r_wg = Rotation.from_quat(gate.orientation_w)
        gate_normal = r_wg.apply(np.array([0.0, 0.0, 1.0]))

        # Vector from gate centre to drone
        to_drone = self._physics.position - gate.position_w

        # Distance along gate normal (signed)
        dist_along = float(np.dot(to_drone, gate_normal))

        # Lateral distance (perpendicular to normal)
        lateral = to_drone - dist_along * gate_normal
        lateral_dist = float(np.linalg.norm(lateral))

        # Pass-through: drone is past the gate plane and within a generous radius
        gate_radius = 4.0  # generous radius for the mock sim

        if 0.0 < dist_along < 3.0 and lateral_dist < gate_radius:
            self._gates_passed += 1
            self._next_gate_idx = (self._next_gate_idx + 1) % len(self._gates)
            if self._next_gate_idx == 0:
                self._laps_completed += 1
            logger.info(
                "Gate %d passed! (total=%d, laps=%d)",
                gate.gate_id,
                self._gates_passed,
                self._laps_completed,
            )

    # ------------------------------------------------------------------
    # Rendering
    # ------------------------------------------------------------------

    def _render_frame(self) -> np.ndarray:
        """Render a synthetic RGB image with visible gates.

        Returns
        -------
        np.ndarray
            (H, W, 3) uint8 RGB image.
        """
        h, w = self._cam.height, self._cam.width
        # Grey background (desaturated — matches VQ1 description)
        image = np.full((h, w, 3), 128, dtype=np.uint8)

        r_wb = quat_to_rotation_matrix(self._physics.orientation)
        r_cw = self._r_cb @ r_wb.T  # world-to-camera rotation
        t_cw = -r_cw @ self._physics.position  # world-to-camera translation

        for gate in self._gates:
            corners_w = gate.corners_3d_w  # (N, 3) in world frame
            pixels = self._project_points(corners_w, r_cw, t_cw)
            if pixels is None:
                continue

            # Draw the gate as a coloured polygon
            pts = pixels.reshape((-1, 1, 2)).astype(np.int32)
            # Convert RGB gate colour for drawing (image is RGB)
            b, g, r = self._cfg.gate_colour_bgr
            colour_rgb = (r, g, b)
            cv2.polylines(
                image, [pts], isClosed=True, color=colour_rgb, thickness=self._cfg.gate_thickness
            )

        return image

    def _project_points(
        self,
        points_w: np.ndarray,
        r_cw: np.ndarray,
        t_cw: np.ndarray,
    ) -> np.ndarray | None:
        """Project 3D world points to 2D pixel coordinates.

        Parameters
        ----------
        points_w : np.ndarray
            (N, 3) points in world frame.
        r_cw : np.ndarray
            3x3 world-to-camera rotation.
        t_cw : np.ndarray
            (3,) world-to-camera translation.

        Returns
        -------
        np.ndarray | None
            (N, 2) pixel coordinates, or None if all points behind camera.
        """
        # Transform to camera frame
        pts_c = (r_cw @ points_w.T).T + t_cw[np.newaxis, :]

        # Check if points are in front of camera (Z > 0)
        if np.all(pts_c[:, 2] <= 0.1):
            return None

        # Only project points in front
        valid = pts_c[:, 2] > 0.1
        if not np.any(valid):
            return None

        # Project using intrinsics
        fx, fy = self._k[0, 0], self._k[1, 1]
        cx, cy = self._k[0, 2], self._k[1, 2]

        pixels = np.zeros((len(pts_c), 2), dtype=np.float64)
        pixels[:, 0] = fx * pts_c[:, 0] / np.maximum(pts_c[:, 2], 0.1) + cx
        pixels[:, 1] = fy * pts_c[:, 1] / np.maximum(pts_c[:, 2], 0.1) + cy

        # Check if any projected points are within the image bounds (with margin)
        margin = 50
        in_frame = (
            (pixels[:, 0] > -margin)
            & (pixels[:, 0] < self._cam.width + margin)
            & (pixels[:, 1] > -margin)
            & (pixels[:, 1] < self._cam.height + margin)
            & valid
        )
        if not np.any(in_frame):
            return None

        return pixels
