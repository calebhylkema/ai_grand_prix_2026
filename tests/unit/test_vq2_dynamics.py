"""Tests for the NED, rate/thrust VQ2 planning model."""

from __future__ import annotations

import casadi as ca
import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from drone_racing.planning.dynamics import (
    NU,
    NX,
    VQ2DynamicsConfig,
    continuous_vq2_dynamics,
    fit_vq2_acceleration_model,
    fit_vq2_rate_response,
    rk4_vq2_step,
    vq2_input_bounds,
)


def _evaluate_dynamics(state: np.ndarray, command: np.ndarray) -> np.ndarray:
    x = ca.SX.sym("x", NX)
    u = ca.SX.sym("u", NU)
    fn = ca.Function("vq2_dyn", [x, u], [continuous_vq2_dynamics(x, u, VQ2DynamicsConfig())])
    return np.asarray(fn(state, command)).reshape(-1)


def test_level_hover_is_equilibrium_in_ned() -> None:
    derivative = _evaluate_dynamics(np.zeros(NX), np.array([0.30, 0.0, 0.0, 0.0]))
    assert derivative == pytest.approx(np.zeros(NX), abs=1e-10)


def test_negative_pitch_accelerates_forward() -> None:
    state = np.zeros(NX)
    state[7] = -0.2
    derivative = _evaluate_dynamics(state, np.array([0.30, 0.0, 0.0, 0.0]))
    assert derivative[3] > 0.0  # +north / local forward


def test_positive_roll_accelerates_right() -> None:
    state = np.zeros(NX)
    state[6] = 0.2
    derivative = _evaluate_dynamics(state, np.array([0.30, 0.0, 0.0, 0.0]))
    assert derivative[4] > 0.0  # +east / local right


def test_body_rate_input_integrates_attitude() -> None:
    x = ca.SX.sym("x", NX)
    u = ca.SX.sym("u", NU)
    fn = ca.Function("vq2_step", [x, u], [rk4_vq2_step(x, u, 0.1, VQ2DynamicsConfig())])
    next_state = np.asarray(fn(np.zeros(NX), np.array([0.30, 1.0, 0.0, 0.0]))).reshape(-1)
    assert next_state[6] == pytest.approx(0.1, abs=1e-6)


def test_rate_thrust_bounds_match_contract() -> None:
    lower, upper = vq2_input_bounds(VQ2DynamicsConfig())
    assert lower == pytest.approx([0.10, -3.0, -3.0, -3.0])
    assert upper == pytest.approx([0.55, 3.0, 3.0, 3.0])


def test_acceleration_fit_recovers_hover_and_drag() -> None:
    rng = np.random.default_rng(7)
    count = 300
    rpy = rng.normal(0.0, 0.15, (count, 3))
    velocity = rng.normal(0.0, 2.0, (count, 3))
    thrust = rng.uniform(0.2, 0.45, count)
    true_hover = 0.30
    true_drag = np.array([0.25, 0.35, 0.20])
    body_down = Rotation.from_euler("xyz", rpy).as_matrix()[:, :, 2]
    acceleration = (
        np.array([0.0, 0.0, 9.81])
        - (9.81 / true_hover) * thrust[:, None] * body_down
        - true_drag * velocity
    )
    acceleration += rng.normal(0.0, 0.02, acceleration.shape)
    fit = fit_vq2_acceleration_model(
        rpy,
        velocity,
        thrust,
        acceleration,
        source_id="synthetic-held-out-contract-test",
    )
    assert fit.config.hover_thrust == pytest.approx(true_hover, abs=0.002)
    assert fit.config.drag_coeff == pytest.approx(true_drag, abs=0.01)
    assert fit.acceleration_rmse_mps2 < 0.05


def test_acceleration_fit_rejects_raw_short_capture() -> None:
    with pytest.raises(ValueError, match="at least 20"):
        fit_vq2_acceleration_model(
            np.zeros((5, 3)),
            np.zeros((5, 3)),
            np.full(5, 0.3),
            np.zeros((5, 3)),
            source_id="too-short",
        )


def test_rate_response_fit_recovers_gain_delay_and_bias() -> None:
    rng = np.random.default_rng(4)
    time_s = np.arange(0.0, 8.0, 0.01)
    commands = rng.normal(0.0, 1.0, (len(time_s), 3))
    gains = np.array([0.98, 1.03, 0.95])
    delays = np.array([0.03, 0.05, 0.02])
    biases = np.array([0.01, -0.02, 0.005])
    measured = np.empty_like(commands)
    for axis in range(3):
        delayed = np.interp(time_s - delays[axis], time_s, commands[:, axis])
        measured[:, axis] = gains[axis] * delayed + biases[axis]
    measured += rng.normal(0.0, 0.005, measured.shape)
    fit = fit_vq2_rate_response(
        time_s,
        commands,
        measured,
        source_id="synthetic-rate-contract-test",
    )
    assert fit.gain == pytest.approx(gains, abs=0.01)
    assert fit.delay_s == pytest.approx(delays, abs=0.006)
    assert fit.bias_rps == pytest.approx(biases, abs=0.01)
    assert max(fit.rmse_rps) < 0.02
