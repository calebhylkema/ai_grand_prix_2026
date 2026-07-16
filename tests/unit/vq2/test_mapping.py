"""Tests for course-map evidence estimation."""

from __future__ import annotations

import numpy as np
import pytest

from drone_racing.deployments.vq2.mapping import estimate_pair_delta


def _pair(run: str, fwd: float, right: float, up: float) -> dict[str, object]:
    return {"run": run, "d_fwd": fwd, "d_right": right, "d_up": up}


def test_repeated_frames_do_not_inflate_independent_runs() -> None:
    pairs = [_pair("run-a", 9.0, 6.0, 0.5) for _ in range(100)]
    estimate = estimate_pair_delta(pairs, perception_revision="structured-v2")
    assert estimate.independent_runs == 1
    assert estimate.observation_count == 100
    assert estimate.flight_evidence_ready is False


def test_estimate_uses_ned_down_sign_and_run_medians() -> None:
    pairs = [
        _pair("run-a", 9.0, 6.0, 0.5),
        _pair("run-a", 99.0, 99.0, 99.0),
        _pair("run-a", 9.0, 6.0, 0.5),
        _pair("run-b", 9.2, 6.2, 0.7),
        _pair("run-c", 9.1, 6.1, 0.6),
    ]
    estimate = estimate_pair_delta(pairs, perception_revision="structured-v2")
    assert estimate.delta_ned_m == pytest.approx((9.1, 6.1, -0.6))
    assert np.all(np.asarray(estimate.position_std_m) > 0.0)
    assert estimate.flight_evidence_ready is True


def test_legacy_revision_can_never_be_flight_ready() -> None:
    pairs = [
        _pair("run-a", 9.0, 6.0, 0.5),
        _pair("run-b", 9.1, 6.1, 0.6),
        _pair("run-c", 9.2, 6.2, 0.7),
    ]
    estimate = estimate_pair_delta(
        pairs,
        perception_revision="legacy-pre-structured-unknown-tilt",
    )
    assert estimate.flight_evidence_ready is False
    assert "obsolete/unknown" in estimate.readiness_reason


def test_zero_observations_are_rejected() -> None:
    with pytest.raises(ValueError, match="zero observations"):
        estimate_pair_delta([], perception_revision="structured-v2")
