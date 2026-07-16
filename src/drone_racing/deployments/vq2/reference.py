"""Offline/shadow-only VQ2 course-reference construction and evaluation.

Nothing in this module sends commands. A reference can only be constructed from a
flight-ready :class:`CourseMap`; historical and candidate maps fail closed. The live
mission may integrate this module only after estimator and controller promotion gates
are met.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from drone_racing.planning.spline import (
    GatePassageReport,
    RacingLine,
    build_racing_line,
    evaluate_gate_passages,
)

if TYPE_CHECKING:
    from drone_racing.deployments.vq2.course_map import CourseMap
    from drone_racing.utils.types import PlannerConfig


@dataclass(frozen=True, slots=True)
class CourseReference:
    """Validated immutable map-to-trajectory product."""

    course_id: str
    racing_line: RacingLine
    gate_passages: tuple[GatePassageReport, ...]


@dataclass(frozen=True, slots=True)
class ReferenceSample:
    """One shadow planner output with no command authority."""

    progress: float
    target_progress: float
    position_ned_m: np.ndarray
    tangent_ned: np.ndarray
    speed_mps: float
    velocity_ned_mps: np.ndarray
    acceleration_ned_mps2: np.ndarray
    curvature_1pm: float
    heading_rad: float
    active_gate: int | None


@dataclass(frozen=True, slots=True)
class ReferenceTraceMetrics:
    """Offline comparison between recorded position and a course reference."""

    sample_count: int
    progress_start: float
    progress_end: float
    backward_jumps: int
    tracking_rmse_m: float
    tracking_max_m: float
    minimum_gate_clearance_m: float


def build_course_reference(
    course: CourseMap,
    config: PlannerConfig,
    *,
    num_points: int = 600,
    uncertainty_sigma: float = 3.0,
) -> CourseReference:
    """Construct and geometrically validate an open VQ2 racing reference."""
    course.require_flight_ready()
    centers = np.asarray([gate.center_ned_m for gate in course.gates], dtype=np.float64)
    normals = np.asarray([gate.normal_ned for gate in course.gates], dtype=np.float64)
    sizes = np.asarray([gate.inner_size_m for gate in course.gates], dtype=np.float64)
    std = np.asarray([gate.position_std_m for gate in course.gates], dtype=np.float64)
    identity_quaternions = np.tile(np.array([0.0, 0.0, 0.0, 1.0]), (len(centers), 1))
    line = build_racing_line(
        centers,
        identity_quaternions,
        config,
        num_points=num_points,
        closed=False,
        gate_normals=normals,
    )
    passages = evaluate_gate_passages(
        line,
        centers,
        normals,
        sizes,
        vehicle_radius_m=course.drone_radius_m,
        tracking_margin_m=course.tracking_margin_m,
        position_std_m=std,
        uncertainty_sigma=uncertainty_sigma,
    )
    failed = [report.gate_id for report in passages if not report.passed]
    if failed:
        raise ValueError(f"reference is not aperture/direction feasible at gates {failed}")
    if any(
        later.progress <= earlier.progress
        for earlier, later in zip(passages, passages[1:], strict=False)
    ):
        raise ValueError("reference does not cross gates in strictly increasing order")
    return CourseReference(course.course_id, line, passages)


class ShadowReferenceTracker:
    """Monotonic local progress tracker for offline/shadow evaluation."""

    def __init__(
        self,
        reference: CourseReference,
        *,
        lookahead_m: float = 1.0,
        heading_lookahead_m: float = 2.0,
        search_window: float = 0.15,
    ) -> None:
        if lookahead_m < 0.0 or heading_lookahead_m < 0.0 or search_window <= 0.0:
            raise ValueError("lookaheads must be non-negative and search window positive")
        self.reference = reference
        self.lookahead_m = lookahead_m
        self.heading_lookahead_m = heading_lookahead_m
        self.search_window = search_window
        self.progress = 0.0
        self._initialized = False
        self._heading_rad = 0.0

    def update(self, position_ned_m: np.ndarray) -> ReferenceSample:
        """Update monotonic progress and return a lookahead reference sample."""
        line = self.reference.racing_line
        query_position = np.asarray(position_ned_m, dtype=np.float64)
        if self._initialized:
            self.progress = line.closest_progress(
                query_position,
                previous_progress=self.progress,
                search_window=self.search_window,
                max_backtrack=0.0,
            )
        else:
            self.progress = line.closest_progress(query_position)
            self._initialized = True
        lookahead_progress = self.lookahead_m / max(line.total_length, 1e-12)
        target_progress = min(1.0, self.progress + lookahead_progress)
        target_position, tangent, speed, acceleration, curvature = line.kinematics_at_progress(
            target_progress
        )
        velocity = tangent * speed
        heading_progress = min(
            1.0,
            self.progress + self.heading_lookahead_m / max(line.total_length, 1e-12),
        )
        _, heading_tangent, _ = line.at_progress(heading_progress)
        if float(np.linalg.norm(heading_tangent[:2])) > 1e-9:
            self._heading_rad = float(np.arctan2(heading_tangent[1], heading_tangent[0]))
        active_gate = next(
            (
                passage.gate_id
                for passage in self.reference.gate_passages
                if passage.progress + 1e-9 >= self.progress
            ),
            None,
        )
        return ReferenceSample(
            progress=self.progress,
            target_progress=target_progress,
            position_ned_m=target_position,
            tangent_ned=tangent,
            speed_mps=speed,
            velocity_ned_mps=velocity,
            acceleration_ned_mps2=acceleration,
            curvature_1pm=curvature,
            heading_rad=self._heading_rad,
            active_gate=active_gate,
        )


def evaluate_reference_trace(
    reference: CourseReference,
    positions_ned_m: np.ndarray,
) -> ReferenceTraceMetrics:
    """Replay world-NED positions through the monotonic shadow tracker."""
    positions = np.asarray(positions_ned_m, dtype=np.float64)
    if positions.ndim != 2 or positions.shape[1] != 3 or not len(positions):
        raise ValueError("positions_ned_m must be a non-empty (N, 3) array")
    if not np.all(np.isfinite(positions)):
        raise ValueError("positions_ned_m contains non-finite values")
    tracker = ShadowReferenceTracker(reference, lookahead_m=0.0)
    progress: list[float] = []
    errors: list[float] = []
    for position in positions:
        sample = tracker.update(position)
        nearest, _, _ = reference.racing_line.at_progress(sample.progress)
        progress.append(sample.progress)
        errors.append(float(np.linalg.norm(position - nearest)))
    progress_array = np.asarray(progress)
    error_array = np.asarray(errors)
    return ReferenceTraceMetrics(
        sample_count=len(positions),
        progress_start=float(progress_array[0]),
        progress_end=float(progress_array[-1]),
        backward_jumps=int(np.count_nonzero(np.diff(progress_array) < -1e-12)),
        tracking_rmse_m=float(np.sqrt(np.mean(error_array**2))),
        tracking_max_m=float(np.max(error_array)),
        minimum_gate_clearance_m=min(report.clearance_m for report in reference.gate_passages),
    )
