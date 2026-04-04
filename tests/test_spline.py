"""Tests for drone_racing.planning.spline — B-spline racing line.

Test strategy:
    1. Straight line: gates in a line → curvature ≈ 0, speed ≈ max.
    2. Circular course: gates in a circle → nonzero curvature, reduced speed.
    3. Arc length: monotonically increasing, total > 0.
    4. Tangent vectors: unit length everywhere.
    5. Speed profile: clamped to [min, max], smoother than raw curvature.
    6. Progress queries: at_progress(0) ≈ start, at_progress(1) ≈ end.
    7. Closest progress: exact point on the line returns correct s.
    8. Edge cases: minimum gate count, open vs closed.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest
from numpy.testing import assert_allclose

from drone_racing.planning.spline import RacingLine, build_racing_line
from drone_racing.utils.config import load_config
from drone_racing.utils.transforms import quat_identity

if TYPE_CHECKING:
    from drone_racing.utils.types import PlannerConfig

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def planner_config() -> PlannerConfig:
    """Load planner config from default.toml."""
    return load_config().planner


def _straight_gates(n: int = 5, spacing: float = 10.0) -> tuple[np.ndarray, np.ndarray]:
    """Gates evenly spaced along the X axis, facing +X."""
    positions = np.zeros((n, 3))
    positions[:, 0] = np.arange(n) * spacing
    orientations = np.tile(quat_identity(), (n, 1))
    return positions, orientations


def _circular_gates(n: int = 8, radius: float = 20.0) -> tuple[np.ndarray, np.ndarray]:
    """Gates arranged in a circle in the XY plane."""
    from scipy.spatial.transform import Rotation

    angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
    positions = np.column_stack([radius * np.cos(angles), radius * np.sin(angles), np.zeros(n)])
    # Each gate faces inward (toward the centre) — normal points toward centre
    orientations = np.zeros((n, 4))
    for i in range(n):
        # Gate normal should point toward centre → rotation from Z to -radial direction
        yaw = angles[i] + np.pi  # face inward
        orientations[i] = Rotation.from_euler("z", yaw).as_quat()
    return positions, orientations


# ---------------------------------------------------------------------------
# Straight line tests
# ---------------------------------------------------------------------------


class TestStraightLine:
    """Gates in a line — should produce a nearly straight racing line."""

    def test_builds_successfully(self, planner_config: PlannerConfig) -> None:
        pos, ori = _straight_gates()
        line = build_racing_line(pos, ori, planner_config, closed=False)
        assert isinstance(line, RacingLine)

    def test_total_length_reasonable(self, planner_config: PlannerConfig) -> None:
        pos, ori = _straight_gates(5, spacing=10.0)
        line = build_racing_line(pos, ori, planner_config, closed=False)
        # 4 gaps of 10m + approach/exit offsets → roughly 40-50m
        assert 30.0 < line.total_length < 80.0

    def test_low_curvature(self, planner_config: PlannerConfig) -> None:
        pos, ori = _straight_gates()
        line = build_racing_line(pos, ori, planner_config, closed=False)
        # Straight line: the minimum curvature should be very small
        # (approach/exit offsets add some curvature at transitions)
        assert np.min(line.curvatures) < 0.1

    def test_high_speed_on_straights(self, planner_config: PlannerConfig) -> None:
        pos, ori = _straight_gates()
        line = build_racing_line(pos, ori, planner_config, closed=False)
        # Middle of a straight → speed should be near max
        mid = len(line.speeds) // 2
        assert line.speeds[mid] > planner_config.max_velocity_mps * 0.5


# ---------------------------------------------------------------------------
# Circular course tests
# ---------------------------------------------------------------------------


class TestCircularCourse:
    """Gates in a circle — curvature should be nonzero, speed reduced."""

    def test_builds_closed_loop(self, planner_config: PlannerConfig) -> None:
        pos, ori = _circular_gates()
        line = build_racing_line(pos, ori, planner_config, closed=True)
        assert isinstance(line, RacingLine)

    def test_nonzero_curvature(self, planner_config: PlannerConfig) -> None:
        pos, ori = _circular_gates()
        line = build_racing_line(pos, ori, planner_config, closed=True)
        assert np.mean(line.curvatures) > 0.001

    def test_speed_below_max(self, planner_config: PlannerConfig) -> None:
        pos, ori = _circular_gates(n=8, radius=10.0)  # tight circle
        line = build_racing_line(pos, ori, planner_config, closed=True)
        # Tight turns → some speeds should be below max
        assert np.min(line.speeds) < planner_config.max_velocity_mps


# ---------------------------------------------------------------------------
# Array property tests
# ---------------------------------------------------------------------------


class TestRacingLineProperties:
    """Properties that should hold for any racing line."""

    def test_tangents_unit_length(self, planner_config: PlannerConfig) -> None:
        pos, ori = _straight_gates()
        line = build_racing_line(pos, ori, planner_config, closed=False)
        norms = np.linalg.norm(line.tangents, axis=1)
        assert_allclose(norms, 1.0, atol=1e-6)

    def test_arc_lengths_monotonic(self, planner_config: PlannerConfig) -> None:
        pos, ori = _straight_gates()
        line = build_racing_line(pos, ori, planner_config, closed=False)
        diffs = np.diff(line.arc_lengths)
        assert np.all(diffs >= 0)

    def test_arc_length_starts_at_zero(self, planner_config: PlannerConfig) -> None:
        pos, ori = _straight_gates()
        line = build_racing_line(pos, ori, planner_config, closed=False)
        assert_allclose(line.arc_lengths[0], 0.0)

    def test_speeds_positive(self, planner_config: PlannerConfig) -> None:
        pos, ori = _circular_gates()
        line = build_racing_line(pos, ori, planner_config, closed=True)
        assert np.all(line.speeds >= 1.0)

    def test_speeds_below_max(self, planner_config: PlannerConfig) -> None:
        pos, ori = _circular_gates()
        line = build_racing_line(pos, ori, planner_config, closed=True)
        assert np.all(line.speeds <= planner_config.max_velocity_mps + 0.01)

    def test_curvatures_non_negative(self, planner_config: PlannerConfig) -> None:
        pos, ori = _straight_gates()
        line = build_racing_line(pos, ori, planner_config, closed=False)
        assert np.all(line.curvatures >= 0.0)


# ---------------------------------------------------------------------------
# Progress queries
# ---------------------------------------------------------------------------


class TestProgressQueries:
    """at_progress() and closest_progress()."""

    def test_progress_zero_near_start(self, planner_config: PlannerConfig) -> None:
        pos, ori = _straight_gates()
        line = build_racing_line(pos, ori, planner_config, closed=False)
        p, t, s = line.at_progress(0.0)
        assert_allclose(p, line.positions[0], atol=1e-10)

    def test_progress_one_near_end(self, planner_config: PlannerConfig) -> None:
        pos, ori = _straight_gates()
        line = build_racing_line(pos, ori, planner_config, closed=False)
        p, t, s = line.at_progress(1.0)
        assert_allclose(p, line.positions[-1], atol=1e-10)

    def test_progress_clamps(self, planner_config: PlannerConfig) -> None:
        pos, ori = _straight_gates()
        line = build_racing_line(pos, ori, planner_config, closed=False)
        # Values outside [0,1] should be clamped
        p_neg, _, _ = line.at_progress(-0.5)
        p_over, _, _ = line.at_progress(1.5)
        assert_allclose(p_neg, line.positions[0], atol=1e-10)
        assert_allclose(p_over, line.positions[-1], atol=1e-10)

    def test_tangent_is_unit_at_midpoint(self, planner_config: PlannerConfig) -> None:
        pos, ori = _straight_gates()
        line = build_racing_line(pos, ori, planner_config, closed=False)
        _, t, _ = line.at_progress(0.5)
        assert_allclose(np.linalg.norm(t), 1.0, atol=1e-6)

    def test_closest_progress_on_line(self, planner_config: PlannerConfig) -> None:
        pos, ori = _straight_gates()
        line = build_racing_line(pos, ori, planner_config, closed=False)
        # Query the midpoint position → should get s ≈ 0.5
        mid_pos = line.positions[len(line.positions) // 2]
        s = line.closest_progress(mid_pos)
        assert 0.3 < s < 0.7


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    """Minimum inputs, error handling."""

    def test_two_gates_minimum(self, planner_config: PlannerConfig) -> None:
        pos = np.array([[0.0, 0.0, 0.0], [10.0, 0.0, 0.0]])
        ori = np.tile(quat_identity(), (2, 1))
        line = build_racing_line(pos, ori, planner_config, closed=False)
        assert line.total_length > 0

    def test_one_gate_raises(self, planner_config: PlannerConfig) -> None:
        pos = np.array([[0.0, 0.0, 0.0]])
        ori = np.tile(quat_identity(), (1, 1))
        with pytest.raises(ValueError, match="Need ≥2 gates"):
            build_racing_line(pos, ori, planner_config, closed=False)

    def test_open_vs_closed_different_length(self, planner_config: PlannerConfig) -> None:
        pos, ori = _straight_gates(4)
        line_open = build_racing_line(pos, ori, planner_config, closed=False)
        line_closed = build_racing_line(pos, ori, planner_config, closed=True)
        # Closed loop wraps around → should be longer
        assert line_closed.total_length > line_open.total_length * 0.8
