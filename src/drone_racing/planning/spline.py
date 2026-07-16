"""Arc-length-parameterized racing lines and gate-passage validation.

The reference generator is deliberately separate from MPCC. It builds a conservative
open course by default, passes through approach/center/exit waypoints for every gate,
resamples by physical arc length, and applies lateral plus longitudinal acceleration
limits to the speed profile. MPCC and fallback controllers consume the same immutable
``RacingLine`` contract.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np
from scipy.interpolate import CubicSpline

if TYPE_CHECKING:
    from drone_racing.utils.types import PlannerConfig, Vec3

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class RacingLine:
    """Precomputed reference path indexed by normalized physical arc length."""

    positions: np.ndarray
    """(N, 3) positions in one documented world frame (m)."""

    tangents: np.ndarray
    """(N, 3) unit path tangents."""

    curvatures: np.ndarray
    """(N,) curvature magnitudes (1/m)."""

    speeds: np.ndarray
    """(N,) acceleration-feasible speed limits (m/s)."""

    arc_lengths: np.ndarray
    """(N,) cumulative physical path length (m)."""

    total_length: float
    """Total physical path length (m)."""

    closed: bool = False
    """Whether the final sample closes the path back to the first."""

    accelerations: np.ndarray = field(default_factory=lambda: np.empty((0, 3)))
    """(N, 3) trajectory feed-forward acceleration in the line frame (m/s^2)."""

    def __post_init__(self) -> None:
        """Populate trajectory acceleration for legacy/manual ``RacingLine`` fixtures."""
        if self.accelerations.size == 0:
            self.accelerations = _reference_accelerations(
                self.tangents,
                self.speeds,
                self.arc_lengths,
            )
        acceleration = np.asarray(self.accelerations, dtype=np.float64)
        if acceleration.shape != self.positions.shape or not np.all(np.isfinite(acceleration)):
            raise ValueError("accelerations must be finite and match racing-line positions")
        self.accelerations = acceleration

    def at_progress(self, progress: float) -> tuple[Vec3, Vec3, float]:
        """Interpolate position, tangent, and speed at arc-length progress.

        Parameters
        ----------
        progress : float
            Normalized physical path distance in ``[0, 1]``.
        """
        s = float(np.clip(progress, 0.0, 1.0))
        if self.total_length <= 1e-12:
            return self.positions[0].copy(), self.tangents[0].copy(), float(self.speeds[0])
        distance = s * self.total_length
        idx_hi = int(np.searchsorted(self.arc_lengths, distance, side="right"))
        idx_hi = min(max(idx_hi, 1), len(self.positions) - 1)
        idx_lo = idx_hi - 1
        segment = self.arc_lengths[idx_hi] - self.arc_lengths[idx_lo]
        alpha = 0.0 if segment <= 1e-12 else (distance - self.arc_lengths[idx_lo]) / segment

        position = (1.0 - alpha) * self.positions[idx_lo] + alpha * self.positions[idx_hi]
        tangent = (1.0 - alpha) * self.tangents[idx_lo] + alpha * self.tangents[idx_hi]
        norm = float(np.linalg.norm(tangent))
        if norm > 1e-12:
            tangent = tangent / norm
        speed = (1.0 - alpha) * self.speeds[idx_lo] + alpha * self.speeds[idx_hi]
        return position, tangent, float(speed)

    def kinematics_at_progress(self, progress: float) -> tuple[Vec3, Vec3, float, Vec3, float]:
        """Interpolate position, tangent, speed, acceleration, and curvature.

        The acceleration is the derivative of the reference velocity with respect to
        time, so it contains both longitudinal speed-profile and curvature feed-forward.
        """
        s = float(np.clip(progress, 0.0, 1.0))
        position, tangent, speed = self.at_progress(s)
        if self.total_length <= 1e-12:
            return position, tangent, speed, self.accelerations[0].copy(), float(self.curvatures[0])
        distance = s * self.total_length
        idx_hi = int(np.searchsorted(self.arc_lengths, distance, side="right"))
        idx_hi = min(max(idx_hi, 1), len(self.positions) - 1)
        idx_lo = idx_hi - 1
        segment = self.arc_lengths[idx_hi] - self.arc_lengths[idx_lo]
        alpha = 0.0 if segment <= 1e-12 else (distance - self.arc_lengths[idx_lo]) / segment
        acceleration = (1.0 - alpha) * self.accelerations[idx_lo] + alpha * self.accelerations[
            idx_hi
        ]
        curvature = (1.0 - alpha) * self.curvatures[idx_lo] + alpha * self.curvatures[idx_hi]
        return position, tangent, speed, acceleration, float(curvature)

    def closest_progress(
        self,
        position: Vec3,
        previous_progress: float | None = None,
        search_window: float = 0.15,
        max_backtrack: float = 0.0,
    ) -> float:
        """Project a position onto the path without unsafe branch jumping.

        With ``previous_progress=None`` the whole path is searched. Supplying previous
        progress restricts projection to a local forward window and permits only the
        explicitly requested backtrack. This is the mode live progress estimators must
        use near crossings or adjacent legs.
        """
        if search_window <= 0.0 or max_backtrack < 0.0:
            raise ValueError("search_window must be positive and max_backtrack non-negative")
        query = np.asarray(position, dtype=np.float64)
        if query.shape != (3,) or not np.all(np.isfinite(query)):
            raise ValueError("position must be a finite 3-vector")

        segment_indices = np.arange(len(self.positions) - 1)
        previous: float | None = None
        if previous_progress is not None:
            previous = float(np.clip(previous_progress, 0.0, 1.0))
            lo = max(0.0, previous - max_backtrack)
            hi = min(1.0, previous + search_window)
            normalized = self.arc_lengths / max(self.total_length, 1e-12)
            mask = (normalized[1:] >= lo) & (normalized[:-1] <= hi)
            segment_indices = segment_indices[mask]
            if not len(segment_indices):
                return previous

        starts = self.positions[segment_indices]
        vectors = self.positions[segment_indices + 1] - starts
        lengths_sq = np.sum(vectors * vectors, axis=1)
        numerators = np.sum((query - starts) * vectors, axis=1)
        alpha = np.divide(
            numerators,
            lengths_sq,
            out=np.zeros_like(numerators),
            where=lengths_sq > 1e-12,
        )
        alpha = np.clip(alpha, 0.0, 1.0)
        projections = starts + alpha[:, np.newaxis] * vectors
        local_index = int(np.argmin(np.sum((projections - query) ** 2, axis=1)))
        segment_index = int(segment_indices[local_index])
        distance = self.arc_lengths[segment_index] + alpha[local_index] * (
            self.arc_lengths[segment_index + 1] - self.arc_lengths[segment_index]
        )
        progress = float(distance / max(self.total_length, 1e-12))
        if previous is not None:
            progress = max(progress, previous - max_backtrack)
        return float(np.clip(progress, 0.0, 1.0))


@dataclass(frozen=True, slots=True)
class GatePassageReport:
    """Geometric feasibility of one reference-path gate crossing."""

    gate_id: int
    progress: float
    offset_m: tuple[float, float]
    clearance_m: float
    direction_dot: float
    passed: bool


def build_racing_line(
    gate_positions: np.ndarray,
    gate_orientations: np.ndarray,
    config: PlannerConfig,
    num_points: int = 200,
    closed: bool = False,
    gate_normals: np.ndarray | None = None,
) -> RacingLine:
    """Build an arc-length reference through ordered gate constraints.

    ``gate_normals`` should be supplied when a deployment has explicit traversal
    normals. The quaternion local-Z convention remains for pre-sim callers.
    """
    if len(gate_positions) < 2:
        raise ValueError(f"Need ≥2 gates to build a racing line, got {len(gate_positions)}")
    _validate_builder_inputs(gate_positions, gate_orientations, gate_normals, num_points)
    waypoints = _build_waypoints(
        gate_positions,
        gate_orientations,
        config,
        closed,
        gate_normals=gate_normals,
    )
    positions, derivatives, curvatures = _fit_spline(waypoints, num_points, closed)
    derivative_norms = np.maximum(np.linalg.norm(derivatives, axis=1, keepdims=True), 1e-12)
    tangents = derivatives / derivative_norms

    arc_lengths = np.zeros(num_points)
    arc_lengths[1:] = np.cumsum(np.linalg.norm(np.diff(positions, axis=0), axis=1))
    total_length = float(arc_lengths[-1])
    speeds = _curvature_speed_profile(curvatures, arc_lengths, config, closed=closed)
    logger.info(
        "Racing line: %d gates, %.1f m, %d arc-length samples, closed=%s",
        len(gate_positions),
        total_length,
        num_points,
        closed,
    )
    return RacingLine(
        positions=positions,
        tangents=tangents,
        curvatures=curvatures,
        speeds=speeds,
        arc_lengths=arc_lengths,
        total_length=total_length,
        closed=closed,
    )


def evaluate_gate_passages(
    racing_line: RacingLine,
    gate_positions: np.ndarray,
    gate_normals: np.ndarray,
    inner_sizes_m: np.ndarray,
    *,
    vehicle_radius_m: float,
    tracking_margin_m: float,
    position_std_m: np.ndarray | None = None,
    uncertainty_sigma: float = 3.0,
) -> tuple[GatePassageReport, ...]:
    """Check ordered plane crossing, aperture offset, and conservative clearance.

    The aperture is shrunk by vehicle radius, tracking margin, and the requested map
    uncertainty multiple. A path must cross in the positive-normal direction.
    """
    positions = np.asarray(gate_positions, dtype=np.float64)
    normals = np.asarray(gate_normals, dtype=np.float64)
    sizes = np.asarray(inner_sizes_m, dtype=np.float64)
    if positions.ndim != 2 or positions.shape[1] != 3:
        raise ValueError("gate_positions must have shape (G, 3)")
    if normals.shape != positions.shape or sizes.shape != (len(positions), 2):
        raise ValueError("gate_normals/inner_sizes_m shapes do not match gate_positions")
    if vehicle_radius_m <= 0.0 or tracking_margin_m < 0.0 or uncertainty_sigma < 0.0:
        raise ValueError("invalid clearance or uncertainty parameter")
    normal_norms = np.linalg.norm(normals, axis=1)
    if not np.allclose(normal_norms, 1.0, atol=1e-6):
        raise ValueError("gate_normals must be unit vectors")
    std = np.zeros_like(positions) if position_std_m is None else np.asarray(position_std_m)
    if std.shape != positions.shape or np.any(std < 0.0) or not np.all(np.isfinite(std)):
        raise ValueError("position_std_m must be finite, non-negative, and shape (G, 3)")

    reports: list[GatePassageReport] = []
    minimum_progress = 0.0
    for gate_id, (center, normal, size, gate_std) in enumerate(
        zip(positions, normals, sizes, std, strict=True)
    ):
        signed = (racing_line.positions - center) @ normal
        candidates = np.flatnonzero(signed[:-1] * signed[1:] <= 0.0)
        best: tuple[float, np.ndarray, np.ndarray] | None = None
        for idx in candidates:
            denominator = signed[idx] - signed[idx + 1]
            alpha = 0.0 if abs(denominator) <= 1e-12 else signed[idx] / denominator
            crossing = (1.0 - alpha) * racing_line.positions[idx] + alpha * racing_line.positions[
                idx + 1
            ]
            distance = racing_line.arc_lengths[idx] + alpha * (
                racing_line.arc_lengths[idx + 1] - racing_line.arc_lengths[idx]
            )
            progress = float(distance / max(racing_line.total_length, 1e-12))
            direction = racing_line.positions[idx + 1] - racing_line.positions[idx]
            if progress + 1e-9 < minimum_progress:
                continue
            if best is None or np.linalg.norm(crossing - center) < np.linalg.norm(best[1] - center):
                best = (progress, crossing, direction)
        if best is None:
            reports.append(
                GatePassageReport(gate_id, math.nan, (math.nan, math.nan), -math.inf, 0.0, False)
            )
            continue

        progress, crossing, direction = best
        minimum_progress = progress
        basis_a, basis_b = _plane_basis(normal)
        error = crossing - center
        offset = (float(error @ basis_a), float(error @ basis_b))
        uncertainty = uncertainty_sigma * float(np.max(gate_std))
        margins = np.asarray(size) / 2.0 - vehicle_radius_m - tracking_margin_m - uncertainty
        clearance = float(np.min(margins - np.abs(offset)))
        direction_norm = float(np.linalg.norm(direction))
        direction_dot = (
            0.0 if direction_norm <= 1e-12 else float(direction @ normal / direction_norm)
        )
        reports.append(
            GatePassageReport(
                gate_id=gate_id,
                progress=progress,
                offset_m=offset,
                clearance_m=clearance,
                direction_dot=direction_dot,
                passed=clearance > 0.0 and direction_dot > 0.0,
            )
        )
    return tuple(reports)


def _validate_builder_inputs(
    gate_positions: np.ndarray,
    gate_orientations: np.ndarray,
    gate_normals: np.ndarray | None,
    num_points: int,
) -> None:
    positions = np.asarray(gate_positions)
    orientations = np.asarray(gate_orientations)
    if positions.ndim != 2 or positions.shape[1] != 3:
        raise ValueError("gate_positions must have shape (G, 3)")
    if orientations.shape != (len(positions), 4):
        raise ValueError("gate_orientations must have shape (G, 4)")
    if not np.all(np.isfinite(positions)) or not np.all(np.isfinite(orientations)):
        raise ValueError("gate positions/orientations must be finite")
    if num_points < 4:
        raise ValueError("num_points must be at least 4")
    if gate_normals is not None:
        normals = np.asarray(gate_normals, dtype=np.float64)
        if normals.shape != (len(positions), 3):
            raise ValueError("gate_normals must have shape (G, 3)")
        if not np.all(np.isfinite(normals)) or not np.allclose(
            np.linalg.norm(normals, axis=1), 1.0, atol=1e-6
        ):
            raise ValueError("gate_normals must contain finite unit vectors")


def _build_waypoints(
    gate_positions: np.ndarray,
    gate_orientations: np.ndarray,
    config: PlannerConfig,
    closed: bool,
    *,
    gate_normals: np.ndarray | None,
) -> np.ndarray:
    from scipy.spatial.transform import Rotation

    normals = (
        np.asarray(gate_normals, dtype=np.float64)
        if gate_normals is not None
        else Rotation.from_quat(gate_orientations).apply(
            np.tile(np.array([0.0, 0.0, 1.0]), (len(gate_positions), 1))
        )
    )
    offset = config.gate_approach_offset_m
    waypoints: list[np.ndarray] = []
    for center, normal in zip(gate_positions, normals, strict=True):
        waypoints.extend((center - offset * normal, center, center + offset * normal))
    if closed:
        waypoints.append(gate_positions[0] - offset * normals[0])
    return np.asarray(waypoints, dtype=np.float64)


def _fit_spline(
    waypoints: np.ndarray,
    num_points: int,
    closed: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    chord_lengths = np.linalg.norm(np.diff(waypoints, axis=0), axis=1)
    if np.any(chord_lengths <= 1e-9):
        raise ValueError("approach/center/exit waypoints contain duplicates")
    parameter = np.zeros(len(waypoints))
    parameter[1:] = np.cumsum(chord_lengths)
    parameter /= parameter[-1]
    boundary = "periodic" if closed and len(waypoints) >= 4 else "not-a-knot"
    splines = [CubicSpline(parameter, waypoints[:, axis], bc_type=boundary) for axis in range(3)]

    dense_count = max(1000, num_points * 10)
    dense_parameter = np.linspace(0.0, 1.0, dense_count)
    dense_positions = np.column_stack([spline(dense_parameter) for spline in splines])
    dense_arc = np.zeros(dense_count)
    dense_arc[1:] = np.cumsum(np.linalg.norm(np.diff(dense_positions, axis=0), axis=1))
    if dense_arc[-1] <= 1e-12:
        raise ValueError("racing-line spline has zero length")
    target_arc = np.linspace(0.0, dense_arc[-1], num_points)
    sample_parameter = np.interp(target_arc, dense_arc, dense_parameter)

    positions = np.column_stack([spline(sample_parameter) for spline in splines])
    first = np.column_stack([spline(sample_parameter, 1) for spline in splines])
    second = np.column_stack([spline(sample_parameter, 2) for spline in splines])
    cross_magnitude = np.linalg.norm(np.cross(first, second), axis=1)
    first_magnitude = np.linalg.norm(first, axis=1)
    curvature = np.divide(
        cross_magnitude,
        first_magnitude**3,
        out=np.zeros_like(cross_magnitude),
        where=first_magnitude > 1e-12,
    )
    return positions, first, curvature


def _curvature_speed_profile(
    curvatures: np.ndarray,
    arc_lengths: np.ndarray,
    config: PlannerConfig,
    *,
    closed: bool,
) -> np.ndarray:
    min_velocity = float(getattr(config, "min_velocity_mps", 1.0))
    max_lateral = float(getattr(config, "max_lateral_accel_mps2", 8.0))
    max_acceleration = float(getattr(config, "max_longitudinal_accel_mps2", 4.0))
    max_braking = float(getattr(config, "max_braking_accel_mps2", 6.0))
    if (
        min(
            min_velocity,
            max_lateral,
            max_acceleration,
            max_braking,
        )
        <= 0.0
    ):
        raise ValueError("planner speed and acceleration limits must be positive")
    if min_velocity > config.max_velocity_mps:
        raise ValueError("minimum velocity exceeds maximum velocity")

    safe_curvature = np.maximum(curvatures, 1e-9)
    speeds = np.minimum(
        np.sqrt(max_lateral / safe_curvature),
        config.max_velocity_mps,
    )
    speeds = np.clip(speeds, min_velocity, config.max_velocity_mps)
    distances = np.diff(arc_lengths)
    for index, distance in enumerate(distances):
        reachable = math.sqrt(speeds[index] ** 2 + 2.0 * max_acceleration * distance)
        speeds[index + 1] = min(speeds[index + 1], reachable)
    for index in range(len(speeds) - 2, -1, -1):
        reachable = math.sqrt(speeds[index + 1] ** 2 + 2.0 * max_braking * distances[index])
        speeds[index] = min(speeds[index], reachable)
    if closed:
        seam_speed = min(speeds[0], speeds[-1])
        speeds[0] = seam_speed
        speeds[-1] = seam_speed
    return np.asarray(speeds, dtype=np.float64)


def _reference_accelerations(
    tangents: np.ndarray,
    speeds: np.ndarray,
    arc_lengths: np.ndarray,
) -> np.ndarray:
    """Differentiate ``velocity = tangent * speed`` along physical arc length.

    For a path parameterized by distance ``s``, ``ds/dt = speed`` and therefore
    ``acceleration = speed * d(velocity)/ds``. This produces the signed curvature
    acceleration vector as well as acceleration/braking along the tangent.
    """
    tangent = np.asarray(tangents, dtype=np.float64)
    speed = np.asarray(speeds, dtype=np.float64)
    distance = np.asarray(arc_lengths, dtype=np.float64)
    if tangent.ndim != 2 or tangent.shape[1] != 3 or speed.shape != (len(tangent),):
        raise ValueError("tangents and speeds have inconsistent shapes")
    if distance.shape != speed.shape or np.any(np.diff(distance) < 0.0):
        raise ValueError("arc lengths must be non-decreasing")
    velocity = tangent * speed[:, np.newaxis]
    unique_distance, inverse, counts = np.unique(distance, return_inverse=True, return_counts=True)
    if len(unique_distance) < 2:
        return np.zeros_like(velocity)
    # Legacy/manual RacingLine fixtures may repeat a corner sample. Collapse those
    # equal-distance velocities before differentiating, then interpolate the finite
    # derivative back to every original row. Builder output is strictly increasing.
    unique_velocity = np.zeros((len(unique_distance), 3), dtype=np.float64)
    np.add.at(unique_velocity, inverse, velocity)
    unique_velocity /= counts[:, np.newaxis]
    edge_order = 2 if len(unique_distance) >= 3 else 1
    velocity_gradient = np.gradient(  # type: ignore[call-overload]
        unique_velocity, unique_distance, axis=0, edge_order=edge_order
    )
    gradient = np.column_stack(
        [np.interp(distance, unique_distance, velocity_gradient[:, axis]) for axis in range(3)]
    )
    return np.asarray(speed[:, np.newaxis] * gradient, dtype=np.float64)


def _plane_basis(normal: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    reference = np.array([0.0, 0.0, 1.0])
    if abs(float(normal @ reference)) > 0.9:
        reference = np.array([0.0, 1.0, 0.0])
    axis_a = np.cross(normal, reference)
    axis_a /= np.linalg.norm(axis_a)
    axis_b = np.cross(normal, axis_a)
    return axis_a, axis_b
