"""Fail-closed, shadow-only trajectory following for the VQ2 deployment.

This module has no command authority and deliberately does not import ``mission``,
``runtime``, ``calibration``, or ``DroneCommand``. It bridges the gate-anchored VQ2
ESKF into a reviewed course-map frame, follows a :class:`CourseReference`, and emits
hypothetical desired *actual* FRD body rates plus collective thrust for offline/shadow
evaluation. Simulator sign/scale calibration remains at the single command boundary.

Frames: map/estimator axes are NED, body is FRD, and VQ2 quaternions are scalar-first
``[w, x, y, z]``. The generic package ``State`` is intentionally not used because it
stores scalar-last quaternions.
"""

from __future__ import annotations

import math
import time
from collections import Counter
from dataclasses import dataclass
from enum import StrEnum
from typing import TYPE_CHECKING

import numpy as np
from scipy.spatial.transform import Rotation

from drone_racing.deployments.vq2.attitude import quat_to_matrix
from drone_racing.deployments.vq2.reference import (
    CourseReference,
    ReferenceSample,
    ShadowReferenceTracker,
)

if TYPE_CHECKING:
    from collections.abc import Iterable

    from drone_racing.deployments.vq2.course_map import CourseMap

G = 9.81
G_NED = np.array([0.0, 0.0, G])


def _readonly_array(value: object, shape: tuple[int, ...], name: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if array.shape != shape:
        raise ValueError(f"{name} must have shape {shape}, got {array.shape}")
    result = array.copy()
    result.setflags(write=False)
    return result


@dataclass(frozen=True, slots=True)
class VQ2EstimateSnapshot:
    """One atomic gate-anchored VQ2 ESKF snapshot.

    Non-finite values are retained here so the bridge can fail closed with a structured
    status instead of throwing from a background publisher.
    """

    arrival_time_ns: int
    anchor_gate_id: int
    position_anchor_ned_m: np.ndarray
    velocity_ned_mps: np.ndarray
    quaternion_wxyz: np.ndarray
    covariance: np.ndarray

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "position_anchor_ned_m",
            _readonly_array(self.position_anchor_ned_m, (3,), "position_anchor_ned_m"),
        )
        object.__setattr__(
            self,
            "velocity_ned_mps",
            _readonly_array(self.velocity_ned_mps, (3,), "velocity_ned_mps"),
        )
        object.__setattr__(
            self,
            "quaternion_wxyz",
            _readonly_array(self.quaternion_wxyz, (4,), "quaternion_wxyz"),
        )
        object.__setattr__(
            self,
            "covariance",
            _readonly_array(self.covariance, (15, 15), "covariance"),
        )


@dataclass(frozen=True, slots=True)
class FrameAlignment:
    """Reviewed yaw alignment from ESKF NED axes into ``gate0_local_ned``."""

    yaw_map_from_eskf_rad: float
    source_id: str
    validated: bool

    def __post_init__(self) -> None:
        if not math.isfinite(self.yaw_map_from_eskf_rad):
            raise ValueError("alignment yaw must be finite")
        if not self.source_id.strip():
            raise ValueError("alignment source_id must be non-empty")

    @property
    def rotation_map_from_eskf(self) -> np.ndarray:
        c = math.cos(self.yaw_map_from_eskf_rad)
        s = math.sin(self.yaw_map_from_eskf_rad)
        return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


@dataclass(frozen=True, slots=True)
class MapAlignedState:
    """VQ2 state expressed continuously in the reviewed course-map frame."""

    arrival_time_ns: int
    anchor_gate_id: int
    position_ned_m: np.ndarray
    velocity_ned_mps: np.ndarray
    rotation_body_to_ned: np.ndarray
    covariance: np.ndarray

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "position_ned_m", _readonly_array(self.position_ned_m, (3,), "position_ned_m")
        )
        object.__setattr__(
            self,
            "velocity_ned_mps",
            _readonly_array(self.velocity_ned_mps, (3,), "velocity_ned_mps"),
        )
        object.__setattr__(
            self,
            "rotation_body_to_ned",
            _readonly_array(self.rotation_body_to_ned, (3, 3), "rotation_body_to_ned"),
        )
        object.__setattr__(
            self, "covariance", _readonly_array(self.covariance, (15, 15), "covariance")
        )


class AlignmentStatus(StrEnum):
    """Structured result of the gate-anchor to course-map conversion."""

    OK = "ok"
    UNALIGNED = "unaligned"
    INVALID_STATE = "invalid_state"
    UNKNOWN_ANCHOR = "unknown_anchor"
    ANCHOR_MISMATCH = "anchor_mismatch"
    REANCHOR_JUMP = "reanchor_jump"


@dataclass(frozen=True, slots=True)
class AlignmentResult:
    status: AlignmentStatus
    state: MapAlignedState | None
    reanchor_jump_m: float | None = None


class MapFrameBridge:
    """Convert moving-origin ESKF snapshots into a continuous course-map state."""

    def __init__(
        self,
        course: CourseMap,
        alignment: FrameAlignment,
        *,
        max_reanchor_jump_m: float = 0.75,
    ) -> None:
        course.require_flight_ready()
        if max_reanchor_jump_m <= 0.0 or not math.isfinite(max_reanchor_jump_m):
            raise ValueError("max_reanchor_jump_m must be finite and positive")
        self.course = course
        self.alignment = alignment
        self.max_reanchor_jump_m = max_reanchor_jump_m
        self._previous: MapAlignedState | None = None

    def reset(self) -> None:
        self._previous = None

    def update(
        self,
        snapshot: VQ2EstimateSnapshot,
        *,
        active_gate_id: int | None = None,
    ) -> AlignmentResult:
        """Align one snapshot, rejecting discontinuous or inconsistent reanchors."""
        if not self.alignment.validated:
            return AlignmentResult(AlignmentStatus.UNALIGNED, None)
        if active_gate_id is not None and snapshot.anchor_gate_id != active_gate_id:
            return AlignmentResult(AlignmentStatus.ANCHOR_MISMATCH, None)
        if not _snapshot_is_valid(snapshot):
            return AlignmentResult(AlignmentStatus.INVALID_STATE, None)
        try:
            gate_center = self.course.gate(snapshot.anchor_gate_id).center
        except KeyError:
            return AlignmentResult(AlignmentStatus.UNKNOWN_ANCHOR, None)

        r_map_eskf = self.alignment.rotation_map_from_eskf
        quaternion = snapshot.quaternion_wxyz / np.linalg.norm(snapshot.quaternion_wxyz)
        r_eskf_body = quat_to_matrix(quaternion)
        position = gate_center + r_map_eskf @ snapshot.position_anchor_ned_m
        velocity = r_map_eskf @ snapshot.velocity_ned_mps
        rotation = r_map_eskf @ r_eskf_body

        # ESKF error order: world position, world velocity, LOCAL/body attitude,
        # body accel bias, body gyro bias. Only the world-vector blocks rotate.
        transform = np.eye(15)
        transform[0:3, 0:3] = r_map_eskf
        transform[3:6, 3:6] = r_map_eskf
        covariance = transform @ snapshot.covariance @ transform.T
        covariance = 0.5 * (covariance + covariance.T)
        state = MapAlignedState(
            arrival_time_ns=snapshot.arrival_time_ns,
            anchor_gate_id=snapshot.anchor_gate_id,
            position_ned_m=position,
            velocity_ned_mps=velocity,
            rotation_body_to_ned=rotation,
            covariance=covariance,
        )

        jump: float | None = None
        previous = self._previous
        self._previous = state
        if previous is not None and previous.anchor_gate_id != state.anchor_gate_id:
            jump = float(np.linalg.norm(state.position_ned_m - previous.position_ned_m))
            if jump > self.max_reanchor_jump_m:
                return AlignmentResult(AlignmentStatus.REANCHOR_JUMP, None, jump)
        return AlignmentResult(AlignmentStatus.OK, state, jump)


def _snapshot_is_valid(snapshot: VQ2EstimateSnapshot) -> bool:
    arrays = (
        snapshot.position_anchor_ned_m,
        snapshot.velocity_ned_mps,
        snapshot.quaternion_wxyz,
        snapshot.covariance,
    )
    if snapshot.arrival_time_ns < 0 or snapshot.anchor_gate_id < 0:
        return False
    if not all(np.all(np.isfinite(array)) for array in arrays):
        return False
    quaternion_norm = float(np.linalg.norm(snapshot.quaternion_wxyz))
    if not 0.99 <= quaternion_norm <= 1.01:
        return False
    if not np.allclose(snapshot.covariance, snapshot.covariance.T, atol=1e-8):
        return False
    return bool(np.min(np.linalg.eigvalsh(snapshot.covariance)) >= -1e-8)


@dataclass(frozen=True, slots=True)
class StateQualityLimits:
    """Evidence-derived limits required before a state can produce a setpoint."""

    max_age_s: float
    max_position_cov_trace_m2: float
    max_velocity_cov_trace_m2ps2: float

    def __post_init__(self) -> None:
        values = (
            self.max_age_s,
            self.max_position_cov_trace_m2,
            self.max_velocity_cov_trace_m2ps2,
        )
        if any(value <= 0.0 or not math.isfinite(value) for value in values):
            raise ValueError("state-quality limits must be finite and positive")


@dataclass(frozen=True, slots=True)
class FollowerConfig:
    """Conservative shadow follower configuration; not a flight-authority artifact."""

    quality: StateQualityLimits
    position_kp: tuple[float, float, float] = (1.2, 1.2, 1.0)
    velocity_kd: tuple[float, float, float] = (2.2, 2.2, 1.8)
    attitude_gain: float = 4.0
    lookahead_m: float = 1.0
    heading_lookahead_m: float = 2.0
    max_speed_mps: float = 1.8
    max_horizontal_accel_mps2: float = 3.5
    max_vertical_accel_mps2: float = 2.0
    max_tilt_rad: float = 0.30
    max_current_tilt_rad: float = 0.80
    hover_thrust: float = 0.30
    min_thrust: float = 0.25
    max_thrust: float = 0.45
    max_body_rate_rps: tuple[float, float, float] = (2.0, 2.0, 1.0)
    completion_progress: float = 0.995

    def __post_init__(self) -> None:
        positive = (
            *self.position_kp,
            *self.velocity_kd,
            self.attitude_gain,
            self.lookahead_m,
            self.heading_lookahead_m,
            self.max_speed_mps,
            self.max_horizontal_accel_mps2,
            self.max_vertical_accel_mps2,
            self.max_tilt_rad,
            self.max_current_tilt_rad,
            self.hover_thrust,
            self.min_thrust,
            self.max_thrust,
            *self.max_body_rate_rps,
        )
        if any(value <= 0.0 or not math.isfinite(value) for value in positive):
            raise ValueError("follower gains and limits must be finite and positive")
        if not self.min_thrust < self.hover_thrust < self.max_thrust <= 1.0:
            raise ValueError("thrust limits must contain hover and lie in (0, 1]")
        if not 0.0 < self.completion_progress <= 1.0:
            raise ValueError("completion_progress must lie in (0, 1]")


class FollowerStatus(StrEnum):
    OK = "ok"
    STALE = "stale"
    UNCERTAIN = "uncertain"
    INVALID_STATE = "invalid_state"
    INVALID_REFERENCE = "invalid_reference"
    REFERENCE_COMPLETE = "reference_complete"


@dataclass(frozen=True, slots=True)
class RateThrustSetpoint:
    """Hypothetical desired actual rates/thrust, never a simulator command."""

    collective_thrust: float
    body_rates_rps: np.ndarray
    requested_acceleration_ned_mps2: np.ndarray
    feasible_acceleration_ned_mps2: np.ndarray
    desired_rotation_body_to_ned: np.ndarray
    saturation_flags: tuple[str, ...]

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "body_rates_rps", _readonly_array(self.body_rates_rps, (3,), "body_rates_rps")
        )
        object.__setattr__(
            self,
            "requested_acceleration_ned_mps2",
            _readonly_array(
                self.requested_acceleration_ned_mps2,
                (3,),
                "requested_acceleration_ned_mps2",
            ),
        )
        object.__setattr__(
            self,
            "feasible_acceleration_ned_mps2",
            _readonly_array(
                self.feasible_acceleration_ned_mps2,
                (3,),
                "feasible_acceleration_ned_mps2",
            ),
        )
        object.__setattr__(
            self,
            "desired_rotation_body_to_ned",
            _readonly_array(
                self.desired_rotation_body_to_ned,
                (3, 3),
                "desired_rotation_body_to_ned",
            ),
        )


@dataclass(frozen=True, slots=True)
class ShadowGuidanceResult:
    status: FollowerStatus
    reference: ReferenceSample | None
    setpoint: RateThrustSetpoint | None
    position_error_ned_m: np.ndarray | None = None
    velocity_error_ned_mps: np.ndarray | None = None
    gate_bearing_rad: float | None = None
    gate_elevation_rad: float | None = None
    gate_fov_margin_px: float | None = None


@dataclass(frozen=True, slots=True)
class CameraVisibilityConfig:
    """Documented VQ2 pinhole/extrinsic model used only for shadow diagnostics."""

    fx: float = 320.0
    fy: float = 320.0
    cx: float = 320.0
    cy: float = 180.0
    width_px: int = 640
    height_px: int = 360
    tilt_up_rad: float = math.radians(20.0)

    def __post_init__(self) -> None:
        numeric = (self.fx, self.fy, self.cx, self.cy, self.tilt_up_rad)
        if not all(math.isfinite(value) for value in numeric):
            raise ValueError("camera visibility parameters must be finite")
        if self.fx <= 0.0 or self.fy <= 0.0 or self.width_px <= 0 or self.height_px <= 0:
            raise ValueError("camera focal lengths and image dimensions must be positive")


class ConservativeFollower:
    """Feed-forward geometric trajectory follower with no command authority."""

    def __init__(
        self,
        reference: CourseReference,
        course: CourseMap,
        config: FollowerConfig,
        *,
        camera: CameraVisibilityConfig | None = None,
    ) -> None:
        course.require_flight_ready()
        if course.course_id != reference.course_id:
            raise ValueError("course/reference identifiers do not match")
        self.reference = reference
        self.course = course
        self.config = config
        self.camera = CameraVisibilityConfig() if camera is None else camera
        self.tracker: ShadowReferenceTracker
        self.reset()

    def reset(self) -> None:
        """Reset per-attempt progress/heading state for deterministic replay isolation."""
        self.tracker = ShadowReferenceTracker(
            self.reference,
            lookahead_m=self.config.lookahead_m,
            heading_lookahead_m=self.config.heading_lookahead_m,
        )

    def update(self, state: MapAlignedState, *, now_ns: int) -> ShadowGuidanceResult:
        """Return a hypothetical setpoint, or no setpoint on any quality failure."""
        if not _map_state_is_valid(state) or now_ns < state.arrival_time_ns:
            return ShadowGuidanceResult(FollowerStatus.INVALID_STATE, None, None)
        age_s = (now_ns - state.arrival_time_ns) * 1e-9
        if age_s > self.config.quality.max_age_s:
            return ShadowGuidanceResult(FollowerStatus.STALE, None, None)
        position_trace = float(np.trace(state.covariance[0:3, 0:3]))
        velocity_trace = float(np.trace(state.covariance[3:6, 3:6]))
        if (
            position_trace > self.config.quality.max_position_cov_trace_m2
            or velocity_trace > self.config.quality.max_velocity_cov_trace_m2ps2
        ):
            return ShadowGuidanceResult(FollowerStatus.UNCERTAIN, None, None)
        current_tilt = math.acos(float(np.clip(state.rotation_body_to_ned[2, 2], -1.0, 1.0)))
        if current_tilt > self.config.max_current_tilt_rad:
            return ShadowGuidanceResult(FollowerStatus.INVALID_STATE, None, None)

        try:
            sample = self.tracker.update(state.position_ned_m)
        except (ValueError, FloatingPointError):
            return ShadowGuidanceResult(FollowerStatus.INVALID_REFERENCE, None, None)
        if self.tracker.progress >= self.config.completion_progress:
            return ShadowGuidanceResult(FollowerStatus.REFERENCE_COMPLETE, sample, None)

        velocity_reference, acceleration_reference = self._limited_reference(sample)
        position_error = sample.position_ned_m - state.position_ned_m
        velocity_error = velocity_reference - state.velocity_ned_mps
        requested_acceleration = (
            acceleration_reference
            + np.asarray(self.config.position_kp) * position_error
            + np.asarray(self.config.velocity_kd) * velocity_error
        )
        setpoint = acceleration_to_rate_thrust(
            requested_acceleration,
            sample.heading_rad,
            state.rotation_body_to_ned,
            self.config,
        )
        bearing, elevation, margin = _gate_visibility(self.course, state, sample, self.camera)
        return ShadowGuidanceResult(
            status=FollowerStatus.OK,
            reference=sample,
            setpoint=setpoint,
            position_error_ned_m=position_error,
            velocity_error_ned_mps=velocity_error,
            gate_bearing_rad=bearing,
            gate_elevation_rad=elevation,
            gate_fov_margin_px=margin,
        )

    def _limited_reference(self, sample: ReferenceSample) -> tuple[np.ndarray, np.ndarray]:
        speed = min(sample.speed_mps, self.config.max_speed_mps)
        velocity = sample.tangent_ned * speed
        acceleration = np.asarray(sample.acceleration_ned_mps2, dtype=np.float64).copy()
        if sample.speed_mps > self.config.max_speed_mps:
            tangent_component = float(acceleration @ sample.tangent_ned) * sample.tangent_ned
            normal_component = acceleration - tangent_component
            ratio = speed / max(sample.speed_mps, 1e-9)
            # A flat speed cap has no longitudinal feed-forward. Curvature acceleration
            # scales with v^2; velocity feedback supplies any required transition brake.
            acceleration = normal_component * ratio**2
        return velocity, acceleration


def _map_state_is_valid(state: MapAlignedState) -> bool:
    arrays = (
        state.position_ned_m,
        state.velocity_ned_mps,
        state.rotation_body_to_ned,
        state.covariance,
    )
    if state.arrival_time_ns < 0 or not all(np.all(np.isfinite(a)) for a in arrays):
        return False
    rotation = state.rotation_body_to_ned
    if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-6):
        return False
    if not math.isclose(float(np.linalg.det(rotation)), 1.0, abs_tol=1e-6):
        return False
    if not np.allclose(state.covariance, state.covariance.T, atol=1e-8):
        return False
    return bool(np.min(np.linalg.eigvalsh(state.covariance)) >= -1e-8)


def acceleration_to_rate_thrust(
    requested_acceleration: np.ndarray,
    heading_rad: float,
    current_rotation: np.ndarray,
    config: FollowerConfig,
) -> RateThrustSetpoint:
    """Convert a NED acceleration/heading target to desired actual FRD rates/thrust."""
    acceleration = np.asarray(requested_acceleration, dtype=np.float64).copy()
    flags: list[str] = []
    horizontal = float(np.linalg.norm(acceleration[:2]))
    if horizontal > config.max_horizontal_accel_mps2:
        acceleration[:2] *= config.max_horizontal_accel_mps2 / horizontal
        flags.append("horizontal_acceleration")
    vertical = float(
        np.clip(
            acceleration[2],
            -config.max_vertical_accel_mps2,
            config.max_vertical_accel_mps2,
        )
    )
    if not math.isclose(vertical, float(acceleration[2]), abs_tol=1e-12):
        flags.append("vertical_acceleration")
    acceleration[2] = vertical

    required_force = acceleration - G_NED
    force_norm = float(np.linalg.norm(required_force))
    if force_norm <= 1e-9:
        desired_down = np.array([0.0, 0.0, 1.0])
    else:
        desired_down = -required_force / force_norm
    cos_tilt = float(np.clip(desired_down[2], -1.0, 1.0))
    if cos_tilt < math.cos(config.max_tilt_rad):
        horizontal_down = desired_down[:2]
        horizontal_norm = float(np.linalg.norm(horizontal_down))
        if horizontal_norm > 1e-12:
            desired_down = np.array(
                [
                    *(math.sin(config.max_tilt_rad) * horizontal_down / horizontal_norm),
                    math.cos(config.max_tilt_rad),
                ]
            )
        else:
            desired_down = np.array([0.0, 0.0, 1.0])
        flags.append("tilt")

    # Preserve the requested vertical acceleration after applying the tilt cone. This
    # gives vertical thrust priority; horizontal acceleration becomes whatever the
    # feasible thrust direction can produce.
    specific_thrust = max(0.0, (G - acceleration[2]) / max(desired_down[2], 1e-6))
    feasible_force = -specific_thrust * desired_down
    feasible_acceleration = G_NED + feasible_force

    heading = np.array([math.cos(heading_rad), math.sin(heading_rad), 0.0])
    desired_right = np.cross(desired_down, heading)
    right_norm = float(np.linalg.norm(desired_right))
    if right_norm <= 1e-9:
        desired_right = np.array([-math.sin(heading_rad), math.cos(heading_rad), 0.0])
    else:
        desired_right /= right_norm
    desired_forward = np.cross(desired_right, desired_down)
    desired_forward /= np.linalg.norm(desired_forward)
    desired_rotation = np.column_stack((desired_forward, desired_right, desired_down))

    rotation_error = current_rotation.T @ desired_rotation
    body_rates = config.attitude_gain * Rotation.from_matrix(rotation_error).as_rotvec()
    rate_limit = np.asarray(config.max_body_rate_rps)
    limited_rates = np.clip(body_rates, -rate_limit, rate_limit)
    if not np.allclose(limited_rates, body_rates, atol=1e-12):
        flags.append("body_rate")

    current_body_up = -current_rotation[:, 2]
    thrust_acceleration = float(feasible_force @ current_body_up)
    raw_collective = config.hover_thrust * max(0.0, thrust_acceleration) / G
    collective = float(np.clip(raw_collective, config.min_thrust, config.max_thrust))
    if not math.isclose(collective, raw_collective, abs_tol=1e-12):
        flags.append("collective_thrust")

    return RateThrustSetpoint(
        collective_thrust=collective,
        body_rates_rps=limited_rates,
        requested_acceleration_ned_mps2=acceleration,
        feasible_acceleration_ned_mps2=feasible_acceleration,
        desired_rotation_body_to_ned=desired_rotation,
        saturation_flags=tuple(dict.fromkeys(flags)),
    )


def _gate_visibility(
    course: CourseMap,
    state: MapAlignedState,
    sample: ReferenceSample,
    camera: CameraVisibilityConfig,
) -> tuple[float | None, float | None, float | None]:
    if sample.active_gate is None:
        return None, None, None
    gate_body = state.rotation_body_to_ned.T @ (
        course.gate(sample.active_gate).center - state.position_ned_m
    )
    t = camera.tilt_up_rad
    axis = np.array([[0.0, 0.0, 1.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    tilt = np.array(
        [[math.cos(t), 0.0, math.sin(t)], [0.0, 1.0, 0.0], [-math.sin(t), 0.0, math.cos(t)]]
    )
    camera_to_body = tilt @ axis
    gate_camera = camera_to_body.T @ gate_body
    bearing = float(math.atan2(gate_camera[0], gate_camera[2]))
    elevation = float(math.atan2(-gate_camera[1], gate_camera[2]))
    if gate_camera[2] <= 1e-9:
        return bearing, elevation, -1.0e9
    u = camera.fx * gate_camera[0] / gate_camera[2] + camera.cx
    v = camera.fy * gate_camera[1] / gate_camera[2] + camera.cy
    margin = float(min(u, camera.width_px - u, v, camera.height_px - v))
    return bearing, elevation, margin


@dataclass(frozen=True, slots=True)
class GuidanceTraceMetrics:
    sample_count: int
    valid_setpoints: int
    status_counts: dict[str, int]
    progress_start: float
    progress_end: float
    backward_progress_jumps: int
    tracking_rmse_m: float
    tracking_max_m: float
    velocity_rmse_mps: float
    cross_track_rmse_m: float
    along_track_rmse_m: float
    vertical_rmse_m: float
    thrust_min: float
    thrust_max: float
    max_tilt_rad: float
    max_body_rate_rps: float
    saturation_counts: dict[str, int]
    longest_saturation_run: int
    max_reanchor_jump_m: float
    minimum_gate_clearance_m: float
    minimum_gate_fov_margin_px: float
    latency_mean_ms: float
    latency_p95_ms: float
    latency_max_ms: float


def evaluate_guidance_trace(
    bridge: MapFrameBridge,
    follower: ConservativeFollower,
    snapshots: Iterable[VQ2EstimateSnapshot],
    *,
    active_gate_ids: Iterable[int] | None = None,
) -> GuidanceTraceMetrics:
    """Replay anchored estimates through alignment and guidance without commands."""
    snapshot_list = list(snapshots)
    gate_ids: list[int | None] = (
        [None] * len(snapshot_list)
        if active_gate_ids is None
        else [int(gate_id) for gate_id in active_gate_ids]
    )
    if len(gate_ids) != len(snapshot_list):
        raise ValueError("active_gate_ids length must match snapshots")
    if not snapshot_list:
        raise ValueError("guidance trace must contain at least one snapshot")

    statuses: Counter[str] = Counter()
    saturation_counts: Counter[str] = Counter()
    progress: list[float] = []
    position_errors: list[float] = []
    velocity_errors: list[float] = []
    cross_errors: list[float] = []
    along_errors: list[float] = []
    vertical_errors: list[float] = []
    thrust: list[float] = []
    tilt: list[float] = []
    rates: list[float] = []
    fov_margins: list[float] = []
    jumps: list[float] = []
    latencies: list[float] = []
    saturation_run = 0
    longest_saturation = 0

    bridge.reset()
    follower.reset()
    for snapshot, active_gate in zip(snapshot_list, gate_ids, strict=True):
        started = time.perf_counter_ns()
        aligned = bridge.update(snapshot, active_gate_id=active_gate)
        if aligned.reanchor_jump_m is not None:
            jumps.append(aligned.reanchor_jump_m)
        if aligned.state is None:
            statuses[aligned.status.value] += 1
            latencies.append((time.perf_counter_ns() - started) * 1e-6)
            saturation_run = 0
            continue
        result = follower.update(aligned.state, now_ns=snapshot.arrival_time_ns)
        statuses[result.status.value] += 1
        latencies.append((time.perf_counter_ns() - started) * 1e-6)
        if result.reference is None or result.setpoint is None:
            saturation_run = 0
            continue

        sample = result.reference
        setpoint = result.setpoint
        progress.append(sample.progress)
        position_error = np.asarray(result.position_error_ned_m)
        velocity_error = np.asarray(result.velocity_error_ned_mps)
        along = float(position_error @ sample.tangent_ned)
        cross = position_error - along * sample.tangent_ned
        position_errors.append(float(np.linalg.norm(position_error)))
        velocity_errors.append(float(np.linalg.norm(velocity_error)))
        cross_errors.append(float(np.linalg.norm(cross)))
        along_errors.append(abs(along))
        vertical_errors.append(abs(float(position_error[2])))
        thrust.append(setpoint.collective_thrust)
        desired_down = setpoint.desired_rotation_body_to_ned[:, 2]
        tilt.append(math.acos(float(np.clip(desired_down[2], -1.0, 1.0))))
        rates.append(float(np.max(np.abs(setpoint.body_rates_rps))))
        if result.gate_fov_margin_px is not None:
            fov_margins.append(result.gate_fov_margin_px)
        saturation_counts.update(setpoint.saturation_flags)
        if setpoint.saturation_flags:
            saturation_run += 1
            longest_saturation = max(longest_saturation, saturation_run)
        else:
            saturation_run = 0

    latency = np.asarray(latencies)
    valid = len(position_errors)
    return GuidanceTraceMetrics(
        sample_count=len(snapshot_list),
        valid_setpoints=valid,
        status_counts=dict(statuses),
        progress_start=progress[0] if progress else math.nan,
        progress_end=progress[-1] if progress else math.nan,
        backward_progress_jumps=int(np.count_nonzero(np.diff(progress) < -1e-12)),
        tracking_rmse_m=_rms(position_errors),
        tracking_max_m=max(position_errors, default=math.nan),
        velocity_rmse_mps=_rms(velocity_errors),
        cross_track_rmse_m=_rms(cross_errors),
        along_track_rmse_m=_rms(along_errors),
        vertical_rmse_m=_rms(vertical_errors),
        thrust_min=min(thrust, default=math.nan),
        thrust_max=max(thrust, default=math.nan),
        max_tilt_rad=max(tilt, default=math.nan),
        max_body_rate_rps=max(rates, default=math.nan),
        saturation_counts=dict(saturation_counts),
        longest_saturation_run=longest_saturation,
        max_reanchor_jump_m=max(jumps, default=0.0),
        minimum_gate_clearance_m=min(
            report.clearance_m for report in follower.reference.gate_passages
        ),
        minimum_gate_fov_margin_px=min(fov_margins, default=math.nan),
        latency_mean_ms=float(np.mean(latency)),
        latency_p95_ms=float(np.quantile(latency, 0.95)),
        latency_max_ms=float(np.max(latency)),
    )


def _rms(values: list[float]) -> float:
    if not values:
        return math.nan
    array = np.asarray(values, dtype=np.float64)
    return float(np.sqrt(np.mean(array**2)))
