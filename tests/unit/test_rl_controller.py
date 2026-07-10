"""Tests for drone_racing.control.rl_controller — RL policy inference controller.

Test strategy:
    1. Observation vector matches rl_env layout (14-D, correct order).
    2. Action mapping [-1, 1] -> DroneCommand matches rl_env.step().
    3. Progress tracking updates correctly.
    4. Reset clears state.
    5. Stats are populated after update.
    6. from_checkpoint raises FileNotFoundError for missing path.
    7. Observation normalization applies correctly when stats are provided.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from drone_racing.control.rl_controller import ObsNormStats, RLController, RLControllerStats
from drone_racing.planning.spline import build_racing_line
from drone_racing.utils.transforms import quat_identity
from drone_racing.utils.types import ControlConfig, PlannerConfig, State

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_state(
    ts: int = 0,
    pos: tuple[float, float, float] = (0.0, 0.0, 0.0),
    vel: tuple[float, float, float] = (0.0, 0.0, 0.0),
    orientation: np.ndarray | None = None,
) -> State:
    if orientation is None:
        orientation = quat_identity()
    return State(
        timestamp_ns=ts,
        position=np.array(pos, dtype=np.float64),
        velocity=np.array(vel, dtype=np.float64),
        orientation=orientation.copy(),
        accel_bias=np.zeros(3),
        gyro_bias=np.zeros(3),
        covariance=np.eye(15) * 1e-2,
    )


def _simple_racing_line():  # noqa: ANN202
    """Build a minimal racing line for testing."""
    positions = np.array([[0, 0, 0], [10, 0, 0], [10, 10, 0], [0, 10, 0]], dtype=np.float64)
    orientations = np.tile(quat_identity(), (4, 1))
    return build_racing_line(positions, orientations, PlannerConfig(), num_points=100, closed=True)


def _mock_model(action: np.ndarray | None = None) -> MagicMock:
    """Create a mock SB3 model that returns a fixed action."""
    if action is None:
        action = np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float32)
    model = MagicMock()
    model.predict.return_value = (action, None)
    return model


def _build_controller(
    action: np.ndarray | None = None,
    control_config: ControlConfig | None = None,
    obs_norm: ObsNormStats | None = None,
) -> RLController:
    """Build an RLController with a mock model."""
    return RLController(
        model=_mock_model(action),
        racing_line=_simple_racing_line(),
        control_config=control_config or ControlConfig(),
        obs_norm=obs_norm,
    )


# ---------------------------------------------------------------------------
# Tests: Observation
# ---------------------------------------------------------------------------


class TestBuildObs:
    """Tests for observation vector construction."""

    def test_obs_shape(self) -> None:
        ctrl = _build_controller()
        state = _make_state()
        obs = ctrl._build_obs(state)
        assert obs.shape == (14,)
        assert obs.dtype == np.float32

    def test_obs_layout_matches_env(self) -> None:
        """Verify obs layout: [pos(3), vel(3), euler(3), rel_gate(3), progress, speed]."""
        pos = (1.0, 2.0, 3.0)
        vel = (0.5, -0.3, 0.1)
        ctrl = _build_controller()
        state = _make_state(pos=pos, vel=vel)
        obs = ctrl._build_obs(state)

        # Position
        np.testing.assert_allclose(obs[0:3], pos, atol=1e-5)
        # Velocity
        np.testing.assert_allclose(obs[3:6], vel, atol=1e-5)
        # Euler angles for identity quaternion should be ~0
        np.testing.assert_allclose(obs[6:9], 0.0, atol=1e-5)
        # Speed = |vel|
        expected_speed = float(np.linalg.norm(vel))
        assert obs[13] == pytest.approx(expected_speed, abs=1e-5)

    def test_obs_progress_in_range(self) -> None:
        ctrl = _build_controller()
        state = _make_state(pos=(5.0, 0.0, 0.0))
        obs = ctrl._build_obs(state)
        assert 0.0 <= obs[12] <= 1.0


# ---------------------------------------------------------------------------
# Tests: Action mapping
# ---------------------------------------------------------------------------


class TestActionMapping:
    """Tests for action -> DroneCommand mapping."""

    def test_zero_action_gives_half_throttle(self) -> None:
        """Action [0,0,0,0] -> throttle=0.5, zero rates."""
        ctrl = _build_controller(action=np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float32))
        state = _make_state()
        cmd = ctrl.update(state)
        assert cmd.throttle == pytest.approx(0.5, abs=1e-6)
        assert cmd.roll_rate == pytest.approx(0.0, abs=1e-6)
        assert cmd.pitch_rate == pytest.approx(0.0, abs=1e-6)
        assert cmd.yaw_rate == pytest.approx(0.0, abs=1e-6)

    def test_full_throttle(self) -> None:
        """Action [1,...] -> throttle=1.0."""
        ctrl = _build_controller(action=np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32))
        cmd = ctrl.update(_make_state())
        assert cmd.throttle == pytest.approx(1.0, abs=1e-6)

    def test_min_throttle(self) -> None:
        """Action [-1,...] -> throttle=0.0."""
        ctrl = _build_controller(action=np.array([-1.0, 0.0, 0.0, 0.0], dtype=np.float32))
        cmd = ctrl.update(_make_state())
        assert cmd.throttle == pytest.approx(0.0, abs=1e-6)

    def test_rate_scaling(self) -> None:
        """Action [0, 1, -1, 0.5] -> max_roll, -max_pitch, 0.5*max_yaw."""
        cfg = ControlConfig()
        ctrl = _build_controller(
            action=np.array([0.0, 1.0, -1.0, 0.5], dtype=np.float32),
            control_config=cfg,
        )
        cmd = ctrl.update(_make_state())
        assert cmd.roll_rate == pytest.approx(cfg.max_roll_rate, abs=1e-5)
        assert cmd.pitch_rate == pytest.approx(-cfg.max_pitch_rate, abs=1e-5)
        assert cmd.yaw_rate == pytest.approx(0.5 * cfg.max_yaw_rate, abs=1e-5)

    def test_rates_clamped(self) -> None:
        """Actions beyond [-1, 1] are still clamped to physical limits."""
        cfg = ControlConfig()
        ctrl = _build_controller(
            action=np.array([0.0, 2.0, -2.0, 2.0], dtype=np.float32),
            control_config=cfg,
        )
        cmd = ctrl.update(_make_state())
        assert abs(cmd.roll_rate) <= cfg.max_roll_rate + 1e-6
        assert abs(cmd.pitch_rate) <= cfg.max_pitch_rate + 1e-6
        assert abs(cmd.yaw_rate) <= cfg.max_yaw_rate + 1e-6

    def test_timestamp_propagated(self) -> None:
        ctrl = _build_controller()
        state = _make_state(ts=123456789)
        cmd = ctrl.update(state)
        assert cmd.timestamp_ns == 123456789


# ---------------------------------------------------------------------------
# Tests: State management
# ---------------------------------------------------------------------------


class TestStateManagement:
    """Tests for reset and progress tracking."""

    def test_reset_clears_progress(self) -> None:
        ctrl = _build_controller()
        ctrl._current_progress = 0.5
        ctrl.reset(progress=0.0)
        assert ctrl.current_progress == 0.0

    def test_reset_sets_progress(self) -> None:
        ctrl = _build_controller()
        ctrl.reset(progress=0.75)
        assert ctrl.current_progress == pytest.approx(0.75)

    def test_stats_populated_after_update(self) -> None:
        ctrl = _build_controller()
        assert ctrl.last_stats is None
        ctrl.update(_make_state())
        stats = ctrl.last_stats
        assert stats is not None
        assert isinstance(stats, RLControllerStats)
        assert stats.action_raw.shape == (4,)
        assert stats.speed >= 0.0
        assert stats.contour_error >= 0.0

    def test_model_called_with_deterministic(self) -> None:
        """Verify the model is called with deterministic=True by default."""
        model = _mock_model()
        ctrl = RLController(
            model=model,
            racing_line=_simple_racing_line(),
        )
        ctrl.update(_make_state())
        model.predict.assert_called_once()
        _, kwargs = model.predict.call_args
        assert kwargs.get("deterministic") is True


# ---------------------------------------------------------------------------
# Tests: Observation normalization
# ---------------------------------------------------------------------------


class TestObsNormalization:
    """Tests for observation normalization."""

    def test_normalize_zero_mean_unit_var(self) -> None:
        """Normalization with known mean/var produces expected output."""
        mean = np.array([2.0, 0.0], dtype=np.float64)
        var = np.array([4.0, 1.0], dtype=np.float64)
        stats = ObsNormStats(mean=mean, var=var, clip=10.0)

        raw = np.array([4.0, 1.0], dtype=np.float32)
        normed = stats.normalize(raw)

        # (4 - 2) / sqrt(4) = 1.0,  (1 - 0) / sqrt(1) = 1.0
        np.testing.assert_allclose(normed, [1.0, 1.0], atol=1e-5)

    def test_normalize_clips(self) -> None:
        """Extreme values are clipped to [-clip, clip]."""
        stats = ObsNormStats(
            mean=np.zeros(2, dtype=np.float64),
            var=np.ones(2, dtype=np.float64) * 1e-10,  # tiny variance → huge normalised
            clip=5.0,
        )
        raw = np.array([100.0, -100.0], dtype=np.float32)
        normed = stats.normalize(raw)
        np.testing.assert_allclose(normed, [5.0, -5.0], atol=1e-5)

    def test_controller_applies_normalization(self) -> None:
        """When obs_norm is set, the model receives normalised observations."""
        # Create normalization stats for 14-D obs
        mean = np.full(14, 5.0, dtype=np.float64)
        var = np.full(14, 25.0, dtype=np.float64)
        norm = ObsNormStats(mean=mean, var=var, clip=10.0)

        model = _mock_model()
        ctrl = RLController(
            model=model,
            racing_line=_simple_racing_line(),
            obs_norm=norm,
        )
        ctrl.update(_make_state())

        # Check that the obs passed to model.predict was normalised
        call_args = model.predict.call_args
        obs_passed = call_args[0][0]
        # Raw obs at origin is mostly zeros; (0 - 5) / sqrt(25) = -1.0 for most dims
        # (not all — progress and relative_gate vary), but the key check is
        # that normalisation was applied (values shifted from raw)
        assert obs_passed.dtype == np.float32
        assert not np.allclose(obs_passed, 0.0)  # would be ~0 without normalization

    def test_controller_works_without_normalization(self) -> None:
        """When obs_norm is None, raw observations are passed through."""
        model = _mock_model()
        ctrl = RLController(
            model=model,
            racing_line=_simple_racing_line(),
            obs_norm=None,
        )
        ctrl.update(_make_state(pos=(1.0, 2.0, 3.0)))

        call_args = model.predict.call_args
        obs_passed = call_args[0][0]
        # First 3 values should be the raw position
        np.testing.assert_allclose(obs_passed[0:3], [1.0, 2.0, 3.0], atol=1e-5)


# ---------------------------------------------------------------------------
# Tests: from_checkpoint
# ---------------------------------------------------------------------------


class TestFromCheckpoint:
    """Tests for checkpoint loading."""

    def test_missing_checkpoint_raises(self) -> None:
        with pytest.raises(FileNotFoundError, match="RL checkpoint not found"):
            RLController.from_checkpoint(
                "/nonexistent/model.zip",
                _simple_racing_line(),
            )

    def test_missing_sb3_raises_import_error(self, tmp_path: object) -> None:
        """If stable-baselines3 is not installed, raise ImportError."""
        import tempfile

        with tempfile.NamedTemporaryFile(suffix=".zip", delete=False) as f:
            f.write(b"fake")
            fake_path = f.name

        with (
            patch.dict("sys.modules", {"stable_baselines3": None}),
            pytest.raises(ImportError, match="RL dependencies"),
        ):
            RLController.from_checkpoint(fake_path, _simple_racing_line())
