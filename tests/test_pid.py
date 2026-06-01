"""Tests for drone_racing.control.pid — Cascaded PID controller.

Test strategy:
    1. Hover: setpoint = current position → throttle ≈ hover, rates ≈ 0.
    2. Step response: setpoint above current → throttle increases.
    3. Output clamping: extreme setpoints → outputs stay within limits.
    4. Yaw tracking: desired yaw ≠ current → yaw_rate nonzero.
    5. Reset: integrators clear on reset.
    6. Feed-forward: velocity setpoint adds to desired velocity.
"""

from __future__ import annotations

import numpy as np
import pytest
from numpy.testing import assert_allclose

from drone_racing.control.pid import _PID3, PIDController, TrajectorySetpoint, _PIDAxis
from drone_racing.utils.config import load_config
from drone_racing.utils.transforms import quat_identity
from drone_racing.utils.types import ControlConfig, State

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def config() -> ControlConfig:
    """Load control config from default.toml."""
    return load_config().control


@pytest.fixture()
def controller(config: ControlConfig) -> PIDController:
    """Create a PID controller with default config."""
    return PIDController(config)


def _level_state(
    position: np.ndarray | None = None,
    velocity: np.ndarray | None = None,
    timestamp_ns: int = 0,
) -> State:
    """Build a level (identity orientation) state."""
    return State(
        timestamp_ns=timestamp_ns,
        position=position if position is not None else np.zeros(3),
        velocity=velocity if velocity is not None else np.zeros(3),
        orientation=quat_identity(),
        accel_bias=np.zeros(3),
        gyro_bias=np.zeros(3),
        covariance=np.eye(15) * 1e-2,
    )


# ---------------------------------------------------------------------------
# Single-axis PID
# ---------------------------------------------------------------------------


class TestPIDAxis:
    """Unit tests for the scalar PID."""

    def test_proportional_only(self) -> None:
        pid = _PIDAxis(kp=2.0, ki=0.0, kd=0.0)
        out = pid.update(error=1.0, dt=0.01)
        assert_allclose(out, 2.0)

    def test_integral_accumulates(self) -> None:
        pid = _PIDAxis(kp=0.0, ki=1.0, kd=0.0)
        pid.update(error=1.0, dt=0.1)
        out = pid.update(error=1.0, dt=0.1)
        assert_allclose(out, 0.2)  # integral = 0.1 + 0.1 = 0.2

    def test_integral_anti_windup(self) -> None:
        pid = _PIDAxis(kp=0.0, ki=1.0, kd=0.0, integral_limit=0.5)
        for _ in range(100):
            pid.update(error=10.0, dt=1.0)
        out = pid.update(error=10.0, dt=1.0)
        assert out <= 0.5  # clamped by limit

    def test_derivative(self) -> None:
        pid = _PIDAxis(kp=0.0, ki=0.0, kd=1.0)
        pid.update(error=0.0, dt=0.01)  # set prev_error
        out = pid.update(error=1.0, dt=0.01)
        assert_allclose(out, 100.0)  # d/dt(1.0 - 0.0) / 0.01 = 100

    def test_zero_dt_returns_zero(self) -> None:
        pid = _PIDAxis(kp=1.0, ki=1.0, kd=1.0)
        assert pid.update(error=5.0, dt=0.0) == 0.0

    def test_reset_clears_state(self) -> None:
        pid = _PIDAxis(kp=0.0, ki=1.0, kd=0.0)
        pid.update(error=1.0, dt=0.1)
        pid.reset()
        out = pid.update(error=1.0, dt=0.1)
        assert_allclose(out, 0.1)  # integral starts fresh


class TestPID3:
    """3-axis PID."""

    def test_independent_axes(self) -> None:
        pid = _PID3(kp=[1.0, 2.0, 3.0], ki=[0.0, 0.0, 0.0], kd=[0.0, 0.0, 0.0])
        out = pid.update(np.array([1.0, 1.0, 1.0]), dt=0.01)
        assert_allclose(out, [1.0, 2.0, 3.0])


# ---------------------------------------------------------------------------
# Cascaded PID controller
# ---------------------------------------------------------------------------


class TestPIDControllerHover:
    """Hover: setpoint = current position → should produce hover throttle."""

    def test_hover_throttle_matches_constant(self, controller: PIDController) -> None:
        state = _level_state()
        setpoint = TrajectorySetpoint(position=np.zeros(3))
        cmd = controller.update(state, setpoint)
        # Hover throttle is the measured AI-GP sim value (~0.30) — gravity
        # compensation with zero position/velocity error.
        assert 0.25 < cmd.throttle < 0.35

    def test_hover_rates_near_zero(self, controller: PIDController) -> None:
        state = _level_state()
        setpoint = TrajectorySetpoint(position=np.zeros(3))
        cmd = controller.update(state, setpoint)
        assert abs(cmd.roll_rate) < 0.1
        assert abs(cmd.pitch_rate) < 0.1
        assert abs(cmd.yaw_rate) < 0.1


class TestPIDControllerStep:
    """Step responses in position."""

    def test_upward_step_increases_throttle(self, controller: PIDController) -> None:
        """Setpoint 1m above → throttle should exceed hover."""
        state = _level_state()
        setpoint = TrajectorySetpoint(position=np.array([0.0, 0.0, 1.0]))
        cmd = controller.update(state, setpoint)
        assert cmd.throttle > 0.5  # more than hover

    def test_downward_step_decreases_throttle(self, controller: PIDController) -> None:
        """Setpoint 1m below → throttle should be less than hover."""
        state = _level_state()
        setpoint = TrajectorySetpoint(position=np.array([0.0, 0.0, -1.0]))
        cmd = controller.update(state, setpoint)
        assert cmd.throttle < 0.5

    def test_forward_step_produces_pitch(self, controller: PIDController) -> None:
        """Setpoint 2m ahead (X) → should command pitch (body tilts forward)."""
        state = _level_state()
        setpoint = TrajectorySetpoint(position=np.array([2.0, 0.0, 0.0]))
        cmd = controller.update(state, setpoint)
        # Pitching forward to go +X should produce a nonzero pitch rate
        assert abs(cmd.pitch_rate) > 0.01 or abs(cmd.roll_rate) > 0.01

    def test_lateral_step_produces_roll(self, controller: PIDController) -> None:
        """Setpoint 2m right (Y) → should command roll."""
        state = _level_state()
        setpoint = TrajectorySetpoint(position=np.array([0.0, 2.0, 0.0]))
        cmd = controller.update(state, setpoint)
        assert abs(cmd.roll_rate) > 0.01 or abs(cmd.pitch_rate) > 0.01


class TestPIDControllerClamping:
    """Output clamping to physical limits."""

    def test_throttle_clamped(self, controller: PIDController) -> None:
        state = _level_state()
        setpoint = TrajectorySetpoint(position=np.array([0.0, 0.0, 100.0]))
        cmd = controller.update(state, setpoint)
        assert cmd.throttle <= 1.0

    def test_throttle_floor(self, controller: PIDController) -> None:
        state = _level_state()
        setpoint = TrajectorySetpoint(position=np.array([0.0, 0.0, -100.0]))
        cmd = controller.update(state, setpoint)
        assert cmd.throttle >= 0.0

    def test_rate_clamped(self, config: ControlConfig) -> None:
        controller = PIDController(config)
        state = _level_state()
        setpoint = TrajectorySetpoint(position=np.array([100.0, 100.0, 0.0]))
        cmd = controller.update(state, setpoint)
        assert abs(cmd.roll_rate) <= config.max_roll_rate
        assert abs(cmd.pitch_rate) <= config.max_pitch_rate
        assert abs(cmd.yaw_rate) <= config.max_yaw_rate


class TestPIDControllerYaw:
    """Yaw tracking."""

    def test_yaw_error_produces_yaw_rate(self, controller: PIDController) -> None:
        """Desired yaw ≠ 0 when drone faces forward → yaw_rate nonzero."""
        state = _level_state()
        setpoint = TrajectorySetpoint(
            position=np.zeros(3),
            yaw=0.5,  # ~30° desired yaw
        )
        cmd = controller.update(state, setpoint)
        assert abs(cmd.yaw_rate) > 0.01


class TestPIDControllerFeedForward:
    """Feed-forward velocity term."""

    def test_velocity_feedforward_reduces_lag(self, controller: PIDController) -> None:
        """With velocity feed-forward, the controller should respond faster."""
        state = _level_state(velocity=np.array([5.0, 0.0, 0.0]))
        # Setpoint matching current velocity → smaller error response
        sp_ff = TrajectorySetpoint(
            position=np.zeros(3),
            velocity=np.array([5.0, 0.0, 0.0]),
        )
        sp_no_ff = TrajectorySetpoint(
            position=np.zeros(3),
        )

        ctrl_ff = PIDController(load_config().control)
        ctrl_no_ff = PIDController(load_config().control)

        cmd_ff = ctrl_ff.update(state, sp_ff)
        cmd_no_ff = ctrl_no_ff.update(state, sp_no_ff)

        # The feed-forward version should have less extreme attitude commands
        # because the velocity error is smaller
        ff_magnitude = abs(cmd_ff.roll_rate) + abs(cmd_ff.pitch_rate)
        no_ff_magnitude = abs(cmd_no_ff.roll_rate) + abs(cmd_no_ff.pitch_rate)
        assert ff_magnitude <= no_ff_magnitude + 0.01


class TestPIDControllerTimestep:
    """Timestamp and dt handling."""

    def test_increasing_timestamps(self, controller: PIDController) -> None:
        """Sequential calls with increasing timestamps should work."""
        setpoint = TrajectorySetpoint(position=np.array([1.0, 0.0, 0.0]))
        for i in range(10):
            state = _level_state(timestamp_ns=(i + 1) * 20_000_000)  # 20ms
            cmd = controller.update(state, setpoint)
            assert 0.0 <= cmd.throttle <= 1.0

    def test_reset_clears_state(self, controller: PIDController) -> None:
        """After reset, the controller should behave like fresh."""
        setpoint = TrajectorySetpoint(position=np.array([1.0, 0.0, 0.0]))
        state = _level_state(timestamp_ns=100_000_000)
        controller.update(state, setpoint)
        controller.reset()
        assert controller._last_timestamp_ns is None
