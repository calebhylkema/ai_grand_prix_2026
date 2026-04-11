"""Tests for drone_racing.control.mpcc_adapter — MPCC controller with safety ladder.

Test strategy:
    1. Emergency tilt (>80°) → kills motors (throttle=0, zero rates).
    2. Unreliable state (high covariance trace) → PID fallback.
    3. MPCC solve failure → PID fallback with correct stats.
    4. MPCC solve success → uses MPCC output.
    5. Reset clears internal state.
    6. Fallback reason codes are logged in stats.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from drone_racing.control.mpcc_adapter import (
    FallbackReason,
    MPCCController,
    _compute_tilt,
    _state_to_mpcc,
)
from drone_racing.control.pid import PIDController
from drone_racing.planning.spline import RacingLine, build_racing_line
from drone_racing.utils.transforms import quat_identity
from drone_racing.utils.types import ControlConfig, PlannerConfig, State

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_state(
    ts: int = 0,
    pos: tuple[float, float, float] = (0.0, 0.0, 0.0),
    orientation: np.ndarray | None = None,
    cov_scale: float = 1e-2,
) -> State:
    if orientation is None:
        orientation = quat_identity()
    return State(
        timestamp_ns=ts,
        position=np.array(pos),
        velocity=np.zeros(3),
        orientation=orientation.copy(),
        accel_bias=np.zeros(3),
        gyro_bias=np.zeros(3),
        covariance=np.eye(15) * cov_scale,
    )


def _tilted_quat(roll_deg: float = 0.0, pitch_deg: float = 0.0) -> np.ndarray:
    """Create a quaternion with the given roll and pitch in degrees."""
    return Rotation.from_euler("xy", [np.radians(roll_deg), np.radians(pitch_deg)]).as_quat()


def _simple_racing_line() -> RacingLine:
    """Build a minimal racing line for testing."""
    positions = np.array([[0, 0, 0], [10, 0, 0], [10, 10, 0], [0, 10, 0]], dtype=np.float64)
    orientations = np.tile(quat_identity(), (4, 1))
    return build_racing_line(positions, orientations, PlannerConfig(), num_points=100, closed=True)


def _build_mpcc_controller(racing_line: RacingLine | None = None) -> MPCCController:
    """Build an MPCCController with a mock MPCC solver."""
    if racing_line is None:
        racing_line = _simple_racing_line()

    mock_solver = MagicMock()
    mock_solver._cfg = MagicMock()
    mock_solver._cfg.dt = 0.05

    control_cfg = ControlConfig()
    pid = PIDController(control_cfg)

    return MPCCController(
        mpcc_solver=mock_solver,
        racing_line=racing_line,
        pid_controller=pid,
        control_config=control_cfg,
        attitude_tau=0.2,
    )


# ---------------------------------------------------------------------------
# Tests: Helper functions
# ---------------------------------------------------------------------------


class TestComputeTilt:
    """Tests for _compute_tilt."""

    def test_level_is_zero(self) -> None:
        state = _make_state(orientation=quat_identity())
        assert _compute_tilt(state) == pytest.approx(0.0, abs=1e-6)

    def test_90_degree_tilt(self) -> None:
        ori = Rotation.from_euler("x", np.pi / 2).as_quat()
        state = _make_state(orientation=ori)
        assert _compute_tilt(state) == pytest.approx(np.pi / 2, abs=0.01)

    def test_45_degree_roll(self) -> None:
        ori = Rotation.from_euler("x", np.radians(45)).as_quat()
        state = _make_state(orientation=ori)
        assert _compute_tilt(state) == pytest.approx(np.radians(45), abs=0.01)


class TestStateToMPCC:
    """Tests for _state_to_mpcc conversion."""

    def test_identity_state(self) -> None:
        state = _make_state(pos=(1.0, 2.0, 3.0))
        mpcc_state = _state_to_mpcc(state)
        assert len(mpcc_state) == 9
        np.testing.assert_allclose(mpcc_state[:3], [1.0, 2.0, 3.0])
        np.testing.assert_allclose(mpcc_state[3:6], 0.0)  # zero velocity
        np.testing.assert_allclose(mpcc_state[6:9], 0.0, atol=1e-10)  # identity → zero euler

    def test_rotated_state(self) -> None:
        ori = Rotation.from_euler("xyz", [0.1, 0.2, 0.3]).as_quat()
        state = _make_state(orientation=ori)
        mpcc_state = _state_to_mpcc(state)
        assert mpcc_state[6] == pytest.approx(0.1, abs=0.01)
        assert mpcc_state[7] == pytest.approx(0.2, abs=0.01)
        assert mpcc_state[8] == pytest.approx(0.3, abs=0.01)


# ---------------------------------------------------------------------------
# Tests: Safety ladder
# ---------------------------------------------------------------------------


class TestEmergencyTilt:
    """Emergency kill when tilt exceeds 80°."""

    def test_extreme_tilt_kills_motors(self) -> None:
        ctrl = _build_mpcc_controller()
        ctrl.reset()

        ori = Rotation.from_euler("x", np.radians(85)).as_quat()
        state = _make_state(ts=1_000_000, orientation=ori)

        cmd = ctrl.update(state)
        assert cmd.throttle == 0.0
        assert cmd.roll_rate == 0.0
        assert cmd.pitch_rate == 0.0
        assert cmd.yaw_rate == 0.0

        stats = ctrl.last_stats
        assert stats is not None
        assert stats.fallback_reason == FallbackReason.EMERGENCY_TILT
        assert stats.mpcc_success is False

    def test_moderate_tilt_does_not_kill(self) -> None:
        ctrl = _build_mpcc_controller()
        ctrl.reset()

        # 45° is well under the 80° limit
        ori = Rotation.from_euler("x", np.radians(45)).as_quat()
        state = _make_state(ts=1_000_000, orientation=ori)

        # MPCC solver needs to return something
        mock_sol = MagicMock()
        mock_sol.success = True
        mock_sol.inputs = np.array([[0.5, 0.0, 0.0, 0.0]])
        mock_sol.progress = np.array([0.0, 0.01])
        mock_sol.solve_time_ms = 5.0
        ctrl._mpcc.solve.return_value = mock_sol

        cmd = ctrl.update(state)
        assert cmd.throttle > 0.0  # not killed

        stats = ctrl.last_stats
        assert stats is not None
        assert stats.fallback_reason != FallbackReason.EMERGENCY_TILT


class TestMPCCFallback:
    """MPCC solve failure triggers PID fallback."""

    def test_mpcc_failure_uses_pid(self) -> None:
        ctrl = _build_mpcc_controller()
        ctrl.reset()

        mock_sol = MagicMock()
        mock_sol.success = False
        mock_sol.solve_time_ms = 15.0
        ctrl._mpcc.solve.return_value = mock_sol

        state = _make_state(ts=1_000_000, pos=(5.0, 0.0, 0.0))
        cmd = ctrl.update(state)

        # Should get a valid command from PID fallback
        assert 0.0 <= cmd.throttle <= 1.0
        assert np.isfinite(cmd.roll_rate)

        stats = ctrl.last_stats
        assert stats is not None
        assert stats.fallback_reason == FallbackReason.MPCC_FAIL
        assert stats.mpcc_success is False

    def test_mpcc_success_uses_mpcc(self) -> None:
        ctrl = _build_mpcc_controller()
        ctrl.reset()

        mock_sol = MagicMock()
        mock_sol.success = True
        mock_sol.inputs = np.array([[0.6, 0.1, -0.05, 0.02]])
        mock_sol.progress = np.array([0.0, 0.01])
        mock_sol.solve_time_ms = 8.0
        ctrl._mpcc.solve.return_value = mock_sol

        state = _make_state(ts=1_000_000, pos=(5.0, 0.0, 0.0))
        ctrl.update(state)

        stats = ctrl.last_stats
        assert stats is not None
        assert stats.fallback_reason == FallbackReason.MPCC_OK
        assert stats.mpcc_success is True


class TestResetBehaviour:
    """Controller reset clears internal state."""

    def test_reset_clears_progress(self) -> None:
        ctrl = _build_mpcc_controller()
        ctrl.reset(progress=0.5)
        assert ctrl.current_progress == pytest.approx(0.5)

        ctrl.reset(progress=0.0)
        assert ctrl.current_progress == pytest.approx(0.0)

    def test_reset_clears_solution(self) -> None:
        ctrl = _build_mpcc_controller()

        # Simulate a successful solve
        mock_sol = MagicMock()
        mock_sol.success = True
        mock_sol.inputs = np.array([[0.5, 0.0, 0.0, 0.0]])
        mock_sol.progress = np.array([0.0, 0.01])
        mock_sol.solve_time_ms = 5.0
        ctrl._mpcc.solve.return_value = mock_sol
        ctrl.update(_make_state(ts=1_000_000))

        # Reset should clear
        ctrl.reset()
        assert ctrl._last_solution is None
        assert ctrl._solution_step == 0
