"""Tests for flight-gated VQ2 reference generation and shadow tracking."""

from __future__ import annotations

import numpy as np
import pytest

from drone_racing.deployments.vq2.course_map import (
    CourseMap,
    GateLeg,
    GateSpec,
    MapEvidence,
    MapFrame,
    MapStatus,
    load_course_map,
)
from drone_racing.deployments.vq2.reference import (
    ShadowReferenceTracker,
    build_course_reference,
    evaluate_reference_trace,
)
from drone_racing.utils.types import PlannerConfig


def _validated_course() -> CourseMap:
    evidence = MapEvidence("structured-v2", ("a", "b", "c"), 3, 30)
    gates = (
        GateSpec(0, (0.0, 0.0, 0.0), (0.01, 0.01, 0.01), (1.5, 1.5), (1.0, 0.0, 0.0), evidence),
        GateSpec(1, (9.2, 6.4, -0.6), (0.01, 0.01, 0.01), (1.5, 1.5), (1.0, 0.0, 0.0), evidence),
    )
    return CourseMap(
        schema_version=1,
        course_id="test-course",
        status=MapStatus.VALIDATED,
        frame=MapFrame.GATE0_LOCAL_NED,
        units="si",
        drone_radius_m=0.14,
        tracking_margin_m=0.20,
        gates=gates,
        legs=(GateLeg(0, 1, 1, (3.0, 16.0), (0.4, 1.9)),),
    )


def test_historical_map_cannot_build_reference() -> None:
    with pytest.raises(ValueError, match="not flight-ready"):
        build_course_reference(load_course_map("vq2_historical_v0"), PlannerConfig())


def test_validated_course_builds_open_feasible_reference() -> None:
    reference = build_course_reference(_validated_course(), PlannerConfig())
    assert reference.racing_line.closed is False
    assert all(report.passed for report in reference.gate_passages)
    assert reference.gate_passages[0].progress < reference.gate_passages[1].progress


def test_shadow_tracker_is_monotonic_and_reports_next_gate() -> None:
    reference = build_course_reference(_validated_course(), PlannerConfig())
    tracker = ShadowReferenceTracker(reference)
    samples = [tracker.update(position) for position in reference.racing_line.positions[::50]]
    progress = [sample.progress for sample in samples]
    assert np.all(np.diff(progress) >= 0.0)
    assert samples[0].active_gate == 0
    assert samples[-1].active_gate in {1, None}
    assert all(
        np.allclose(sample.velocity_ned_mps, sample.tangent_ned * sample.speed_mps)
        for sample in samples
    )
    assert all(np.all(np.isfinite(sample.acceleration_ned_mps2)) for sample in samples)
    assert all(np.isfinite(sample.curvature_1pm) for sample in samples)
    assert all(np.isfinite(sample.heading_rad) for sample in samples)


def test_trace_evaluator_reports_zero_error_on_reference() -> None:
    reference = build_course_reference(_validated_course(), PlannerConfig())
    metrics = evaluate_reference_trace(reference, reference.racing_line.positions)
    assert metrics.backward_jumps == 0
    assert metrics.progress_end > 0.99
    assert metrics.tracking_rmse_m < 1e-9
