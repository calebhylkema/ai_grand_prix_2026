"""Core data types shared across the drone racing stack.

All values use SI units (meters, radians, seconds) unless explicitly noted.
Quaternions use scalar-last convention [x, y, z, w] to match scipy.
"""

from __future__ import annotations

import enum
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from numpy.typing import NDArray

# ---------------------------------------------------------------------------
# Aliases
# ---------------------------------------------------------------------------
Vec3 = NDArray[np.floating[Any]]  # shape (3,)
Vec4 = NDArray[np.floating[Any]]  # shape (4,)
Mat3 = NDArray[np.floating[Any]]  # shape (3, 3)
Mat4 = NDArray[np.floating[Any]]  # shape (4, 4)
CovMat = NDArray[np.floating[Any]]  # covariance matrix, shape (n, n)


# ---------------------------------------------------------------------------
# IMU
# ---------------------------------------------------------------------------
@dataclass(slots=True)
class ImuSample:
    """Single IMU measurement."""

    timestamp_ns: int
    """Capture timestamp in nanoseconds (monotonic clock)."""

    accel_mps2: Vec3
    """Linear acceleration in body frame (m/s²), includes gravity."""

    gyro_rps: Vec3
    """Angular velocity in body frame (rad/s)."""

    accel_saturated: bool = False
    """True if accelerometer hit its measurement limit (>16 g)."""


# ---------------------------------------------------------------------------
# State estimate
# ---------------------------------------------------------------------------
@dataclass(slots=True)
class State:
    """Full 16-dimensional state estimate (nominal state of ESKF).

    All quantities are in the world frame unless noted.
    """

    timestamp_ns: int
    """Time at which this state is valid (ns, monotonic clock)."""

    position: Vec3
    """Position in world frame [x, y, z] (m)."""

    velocity: Vec3
    """Velocity in world frame [vx, vy, vz] (m/s)."""

    orientation: Vec4
    """Body-to-world rotation quaternion [x, y, z, w] (scalar-last)."""

    accel_bias: Vec3
    """Accelerometer bias in body frame (m/s²)."""

    gyro_bias: Vec3
    """Gyroscope bias in body frame (rad/s)."""

    covariance: CovMat
    """15×15 error-state covariance (orientation parameterised as 3-DOF)."""

    @staticmethod
    def identity(timestamp_ns: int = 0) -> State:
        """Create a zero state with identity orientation and default covariance."""
        return State(
            timestamp_ns=timestamp_ns,
            position=np.zeros(3),
            velocity=np.zeros(3),
            orientation=np.array([0.0, 0.0, 0.0, 1.0]),  # identity quaternion
            accel_bias=np.zeros(3),
            gyro_bias=np.zeros(3),
            covariance=np.eye(15) * 1e-2,
        )


# ---------------------------------------------------------------------------
# Gate detection (from perception)
# ---------------------------------------------------------------------------
class DetectionConfidence(enum.Enum):
    """Coarse confidence level for measurement gating in the ESKF."""

    HIGH = "high"  # 6+ corners, 2–5 m range → full 6-DOF update
    MEDIUM = "medium"  # 4 corners, moderate range → position-only update
    LOW = "low"  # <2 corners or >10 m → skip update


@dataclass(slots=True)
class GateDetection:
    """A single gate observation from the perception module."""

    timestamp_ns: int
    """Camera capture timestamp (ns, monotonic clock)."""

    corners_2d: NDArray[np.floating[Any]]
    """Detected corner pixel coordinates, shape (N, 2)."""

    confidence: float
    """Scalar confidence in [0, 1]."""

    gate_id: int | None = None
    """Tracker-assigned gate identity (None if unknown)."""

    @property
    def confidence_level(self) -> DetectionConfidence:
        """Map continuous confidence + corner count to discrete level."""
        n_corners = self.corners_2d.shape[0]
        if n_corners >= 6 and self.confidence >= 0.7:
            return DetectionConfidence.HIGH
        if n_corners >= 4 and self.confidence >= 0.4:
            return DetectionConfidence.MEDIUM
        return DetectionConfidence.LOW


# ---------------------------------------------------------------------------
# PnP pose measurement (perception → estimation interface)
# ---------------------------------------------------------------------------
@dataclass(slots=True)
class PoseMeasurement:
    """6-DOF pose derived from PnP, ready for ESKF consumption."""

    timestamp_ns: int
    """Camera capture timestamp (ns)."""

    position_w: Vec3
    """Gate-derived position of the drone in world frame (m)."""

    orientation_w: Vec4
    """Gate-derived orientation quaternion [x, y, z, w] body-to-world."""

    position_noise_std: float
    """1-σ position uncertainty (m), set from config or confidence."""

    orientation_noise_std: float
    """1-σ orientation uncertainty (rad), set from config or confidence."""

    confidence: DetectionConfidence
    """Detection confidence level for gating."""

    gate_id: int | None = None
    """Which gate generated this measurement (for logging)."""


# ---------------------------------------------------------------------------
# Drone command (control output)
# ---------------------------------------------------------------------------
@dataclass(slots=True)
class DroneCommand:
    """Rate-mode command sent to the flight controller.

    All angular values are rate commands in rad/s.
    Throttle is normalised to [0, 1].
    """

    timestamp_ns: int
    """Command timestamp (ns, monotonic clock)."""

    throttle: float
    """Collective thrust command, 0.0 (off) to 1.0 (full)."""

    roll_rate: float
    """Desired roll rate (rad/s)."""

    pitch_rate: float
    """Desired pitch rate (rad/s)."""

    yaw_rate: float
    """Desired yaw rate (rad/s)."""


# ---------------------------------------------------------------------------
# Telemetry (provided at race start)
# ---------------------------------------------------------------------------
@dataclass(slots=True)
class Telemetry:
    """Initial telemetry snapshot provided by the race system."""

    timestamp_ns: int
    position: Vec3
    velocity: Vec3
    orientation: Vec4  # [x, y, z, w]


# ---------------------------------------------------------------------------
# Gate map entry
# ---------------------------------------------------------------------------
@dataclass(slots=True)
class GateWaypoint:
    """A single gate in the course map."""

    gate_id: int
    position_w: Vec3
    """Centre of the gate in world frame (m)."""

    orientation_w: Vec4
    """Gate orientation quaternion [x, y, z, w] (gate-normal = Z-axis)."""

    corners_3d_w: NDArray[np.floating[Any]]
    """Gate corner positions in world frame, shape (N, 3)."""

    order: int = 0
    """Passage order (0-indexed)."""


# ---------------------------------------------------------------------------
# Config dataclasses (loaded from TOML via utils/config.py)
# ---------------------------------------------------------------------------
@dataclass(slots=True)
class EstimationConfig:
    """Parameters for the ESKF, read from [estimation] in config TOML."""

    imu_rate_hz: float = 200.0
    gyro_noise_std: float = 0.01
    accel_noise_std: float = 0.1
    gyro_bias_std: float = 0.001
    accel_bias_std: float = 0.01
    pnp_position_noise_std: float = 0.15
    pnp_orientation_noise_std: float = 0.05
    chi2_gate_threshold: float = 9.21


@dataclass(slots=True)
class CameraConfig:
    """Camera intrinsics and extrinsic tilt."""

    fx: float = 600.0
    fy: float = 600.0
    cx: float = 640.0
    cy: float = 360.0
    width: int = 1280
    height: int = 720
    distortion: list[float] = field(default_factory=lambda: [0.0, 0.0, 0.0, 0.0, 0.0])
    tilt_deg: float = 15.0

    @property
    def intrinsic_matrix(self) -> Mat3:
        """3×3 camera intrinsic matrix K."""
        return np.array(
            [[self.fx, 0.0, self.cx], [0.0, self.fy, self.cy], [0.0, 0.0, 1.0]],
            dtype=np.float64,
        )

    @property
    def distortion_coeffs(self) -> NDArray[np.floating[Any]]:
        """Distortion coefficients as numpy array."""
        return np.array(self.distortion, dtype=np.float64)


@dataclass(slots=True)
class GateConfig:
    """Gate geometry, read from [gate] in config TOML."""

    width_m: float = 1.5
    height_m: float = 1.5
    shape: str = "square"
    num_corners: int = 8
    inner_scale: float = 0.65
    corners_3d: list[list[float]] = field(
        default_factory=lambda: [
            # Outer corners: TL, TR, BR, BL
            [-0.75, 0.75, 0.0],
            [0.75, 0.75, 0.0],
            [0.75, -0.75, 0.0],
            [-0.75, -0.75, 0.0],
            # Inner corners: TL, TR, BR, BL (65% scale)
            [-0.4875, 0.4875, 0.0],
            [0.4875, 0.4875, 0.0],
            [0.4875, -0.4875, 0.0],
            [-0.4875, -0.4875, 0.0],
        ]
    )

    @property
    def corners_3d_array(self) -> NDArray[np.floating[Any]]:
        """All gate corner coordinates as (N, 3) numpy array."""
        return np.array(self.corners_3d, dtype=np.float64)

    @property
    def outer_corners_3d_array(self) -> NDArray[np.floating[Any]]:
        """Outer 4 gate corners as (4, 3) numpy array."""
        return np.array(self.corners_3d[:4], dtype=np.float64)


@dataclass(slots=True)
class ControlConfig:
    """PID gains and physical limits, read from [control] in config TOML."""

    position_kp: list[float] = field(default_factory=lambda: [2.0, 2.0, 3.0])
    position_ki: list[float] = field(default_factory=lambda: [0.0, 0.0, 0.1])
    position_kd: list[float] = field(default_factory=lambda: [1.5, 1.5, 2.0])
    velocity_kp: list[float] = field(default_factory=lambda: [4.0, 4.0, 5.0])
    attitude_kp: list[float] = field(default_factory=lambda: [8.0, 8.0, 4.0])
    rate_kp: list[float] = field(default_factory=lambda: [12.0, 12.0, 6.0])
    max_thrust: float = 1.0
    max_roll_rate: float = 8.0
    max_pitch_rate: float = 8.0
    max_yaw_rate: float = 4.0
    max_tilt_angle: float = 1.22


@dataclass(slots=True)
class PlannerConfig:
    """Planner parameters, read from [planner] in config TOML."""

    w_progress: float = 10.0
    w_contour: float = 1.0
    w_lag: float = 1.0
    w_input: float = 0.01
    horizon_steps: int = 30
    horizon_dt: float = 0.05
    spline_smoothing: float = 0.5
    gate_approach_offset_m: float = 2.0
    max_velocity_mps: float = 25.0
