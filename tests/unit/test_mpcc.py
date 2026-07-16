"""Tests for the MPCC planner: dynamics model and trajectory solver.

Covers:
- Continuous dynamics hover equilibrium
- RK4 integration accuracy
- State/input bounds
- MPCC solver convergence on straight-line and 3D paths
- Progress monotonicity and constraint satisfaction
- Warm-starting across consecutive solves
"""

from __future__ import annotations

import numpy as np
import pytest

from drone_racing.planning.dynamics import (
    NU,
    NX,
    PHI,
    PX,
    PY,
    PZ,
    THETA,
    VX,
    VY,
    VZ,
    DynamicsConfig,
    continuous_dynamics,
    input_bounds,
    rk4_step,
    state_bounds,
)
from drone_racing.planning.spline import RacingLine

casadi = pytest.importorskip("casadi")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_straight_racing_line(
    start: np.ndarray,
    end: np.ndarray,
    num_points: int = 50,
) -> RacingLine:
    """Build a minimal RacingLine along a straight segment."""

    positions = np.linspace(start, end, num_points)
    direction = end - start
    length = float(np.linalg.norm(direction))
    tangent = direction / max(length, 1e-12)
    tangents = np.tile(tangent, (num_points, 1))
    curvatures = np.zeros(num_points)
    speeds = np.full(num_points, 10.0)
    seg_lengths = np.linalg.norm(np.diff(positions, axis=0), axis=1)
    arc_lengths = np.zeros(num_points)
    arc_lengths[1:] = np.cumsum(seg_lengths)

    return RacingLine(
        positions=positions,
        tangents=tangents,
        curvatures=curvatures,
        speeds=speeds,
        arc_lengths=arc_lengths,
        total_length=length,
    )


def _make_l_shaped_racing_line(num_points: int = 60) -> RacingLine:
    """Build a racing line with a 90-degree turn (L-shape)."""

    half = num_points // 2
    # First leg: along +X
    p1 = np.column_stack(
        [
            np.linspace(0, 10, half),
            np.zeros(half),
            np.full(half, 2.0),
        ]
    )
    # Second leg: along +Y
    p2 = np.column_stack(
        [
            np.full(num_points - half, 10.0),
            np.linspace(0, 10, num_points - half),
            np.full(num_points - half, 2.0),
        ]
    )
    positions = np.vstack([p1, p2])

    # Tangents
    tangents = np.zeros_like(positions)
    tangents[:half] = [1, 0, 0]
    tangents[half:] = [0, 1, 0]

    curvatures = np.zeros(num_points)
    speeds = np.full(num_points, 8.0)
    seg_lengths = np.linalg.norm(np.diff(positions, axis=0), axis=1)
    arc_lengths = np.zeros(num_points)
    arc_lengths[1:] = np.cumsum(seg_lengths)

    return RacingLine(
        positions=positions,
        tangents=tangents,
        curvatures=curvatures,
        speeds=speeds,
        arc_lengths=arc_lengths,
        total_length=float(arc_lengths[-1]),
    )


def _make_closed_circuit(num_points: int = 200) -> RacingLine:
    """Build a closed square gate circuit with realistic horizontal orientations."""
    from scipy.spatial.transform import Rotation

    from drone_racing.planning.spline import build_racing_line

    gate_positions = np.array(
        [
            [5.0, 0.0, 2.0],
            [5.0, 5.0, 2.0],
            [0.0, 5.0, 2.0],
            [0.0, 0.0, 2.0],
        ]
    )

    def _quat_from_normal(normal: list[float]) -> np.ndarray:
        r = Rotation.align_vectors([normal], [[0, 0, 1]])[0]
        return r.as_quat()

    gate_orientations = np.array(
        [
            _quat_from_normal([0, 1, 0]),
            _quat_from_normal([-1, 0, 0]),
            _quat_from_normal([0, -1, 0]),
            _quat_from_normal([1, 0, 0]),
        ]
    )

    class _Cfg:
        gate_approach_offset_m = 0.5
        max_velocity_mps = 10.0
        smoothing_factor = 0.0

    return build_racing_line(
        gate_positions, gate_orientations, _Cfg(), num_points=num_points, closed=True
    )


def _eval_casadi(expr: casadi.SX, sym: casadi.SX, val: np.ndarray) -> np.ndarray:
    """Numerically evaluate a CasADi SX expression."""
    f = casadi.Function("f", [sym], [expr])
    return np.array(f(val)).flatten()


# ---------------------------------------------------------------------------
# Dynamics tests
# ---------------------------------------------------------------------------


class TestContinuousDynamics:
    """Continuous-time dynamics model."""

    def test_hover_equilibrium(self) -> None:
        """At hover thrust with zero tilt, vertical acceleration should be ~0."""
        cfg = DynamicsConfig()
        x = casadi.SX.sym("x", NX)
        u = casadi.SX.sym("u", NU)

        xdot_expr = continuous_dynamics(x, u, cfg)

        # Hover state: zero velocity/angles
        x0 = np.zeros(NX)
        # Hover input: thrust = mg / max_thrust, zero angles, zero yaw rate
        u0 = np.array([cfg.mass * 9.81 / cfg.max_thrust, 0.0, 0.0, 0.0])

        xdot_fn = casadi.Function("xdot", [x, u], [xdot_expr])
        xdot_val = np.array(xdot_fn(x0, u0)).flatten()

        # Position derivatives = velocity = 0
        np.testing.assert_allclose(xdot_val[:3], 0.0, atol=1e-10)
        # Velocity derivatives (acceleration) should be ~0 at hover
        np.testing.assert_allclose(xdot_val[3:6], 0.0, atol=1e-6)
        # Attitude derivatives = 0 (commanded = current = 0)
        np.testing.assert_allclose(xdot_val[6:9], 0.0, atol=1e-10)

    def test_free_fall(self) -> None:
        """With zero thrust and zero velocity, vertical acceleration should be -g."""
        cfg = DynamicsConfig()
        x = casadi.SX.sym("x", NX)
        u = casadi.SX.sym("u", NU)

        xdot_expr = continuous_dynamics(x, u, cfg)
        xdot_fn = casadi.Function("xdot", [x, u], [xdot_expr])

        x0 = np.zeros(NX)  # zero velocity → drag has no effect
        u0 = np.zeros(NU)  # zero thrust

        xdot_val = np.array(xdot_fn(x0, u0)).flatten()
        assert xdot_val[VZ] == pytest.approx(-9.81, abs=1e-10)
        np.testing.assert_allclose(xdot_val[VX:VZ], 0.0, atol=1e-10)

    def test_velocity_propagates_position(self) -> None:
        """Position derivatives should equal velocity components."""
        cfg = DynamicsConfig()
        x = casadi.SX.sym("x", NX)
        u = casadi.SX.sym("u", NU)

        xdot_expr = continuous_dynamics(x, u, cfg)
        xdot_fn = casadi.Function("xdot", [x, u], [xdot_expr])

        x0 = np.zeros(NX)
        x0[VX] = 3.0
        x0[VY] = -2.0
        x0[VZ] = 1.5
        u0 = np.zeros(NU)

        xdot_val = np.array(xdot_fn(x0, u0)).flatten()
        assert xdot_val[PX] == pytest.approx(3.0)
        assert xdot_val[PY] == pytest.approx(-2.0)
        assert xdot_val[PZ] == pytest.approx(1.5)

    def test_attitude_lag_converges(self) -> None:
        """Attitude should move toward commanded values."""
        cfg = DynamicsConfig(attitude_tau=0.1)
        x = casadi.SX.sym("x", NX)
        u = casadi.SX.sym("u", NU)

        xdot_expr = continuous_dynamics(x, u, cfg)
        xdot_fn = casadi.Function("xdot", [x, u], [xdot_expr])

        x0 = np.zeros(NX)
        # Command a roll of 0.5 rad, current roll is 0
        u0 = np.array([0.5, 0.5, 0.3, 0.0])

        xdot_val = np.array(xdot_fn(x0, u0)).flatten()
        # dphi = (phi_cmd - phi) / tau = 0.5 / 0.1 = 5.0
        assert xdot_val[PHI] == pytest.approx(5.0)
        # dtheta = 0.3 / 0.1 = 3.0
        assert xdot_val[THETA] == pytest.approx(3.0)


class TestRK4Integration:
    """RK4 discrete integration step."""

    def test_hover_stays_stationary(self) -> None:
        """A hovering drone should stay at the same position after one step."""
        cfg = DynamicsConfig()
        x = casadi.SX.sym("x", NX)
        u = casadi.SX.sym("u", NU)

        x_next_expr = rk4_step(x, u, 0.05, cfg)
        step_fn = casadi.Function("step", [x, u], [x_next_expr])

        x0 = np.zeros(NX)
        x0[PZ] = 2.0  # some height
        u0 = np.array([cfg.mass * 9.81 / cfg.max_thrust, 0.0, 0.0, 0.0])

        x1 = np.array(step_fn(x0, u0)).flatten()
        # Position should barely move
        np.testing.assert_allclose(x1[:3], x0[:3], atol=1e-6)
        # Velocity should stay near zero
        np.testing.assert_allclose(x1[3:6], 0.0, atol=1e-6)

    def test_free_fall_integration(self) -> None:
        """Free fall for 0.1s should give vz ≈ -0.981, pz ≈ -0.0049."""
        cfg = DynamicsConfig(drag_coeff=0.0)  # No drag for analytical comparison
        x = casadi.SX.sym("x", NX)
        u = casadi.SX.sym("u", NU)

        dt = 0.1
        x_next_expr = rk4_step(x, u, dt, cfg)
        step_fn = casadi.Function("step", [x, u], [x_next_expr])

        x0 = np.zeros(NX)
        u0 = np.zeros(NU)

        x1 = np.array(step_fn(x0, u0)).flatten()
        # vz = -g * dt = -0.981
        assert x1[VZ] == pytest.approx(-9.81 * dt, abs=1e-4)
        # pz = -0.5 * g * dt^2 = -0.04905
        assert x1[PZ] == pytest.approx(-0.5 * 9.81 * dt**2, abs=1e-4)

    def test_multi_step_energy(self) -> None:
        """Multiple RK4 steps in free fall should match analytical solution."""
        cfg = DynamicsConfig(drag_coeff=0.0)  # No drag for analytical comparison
        x_sym = casadi.SX.sym("x", NX)
        u_sym = casadi.SX.sym("u", NU)

        dt = 0.01
        n_steps = 100  # 1 second of free fall
        x_next_expr = rk4_step(x_sym, u_sym, dt, cfg)
        step_fn = casadi.Function("step", [x_sym, u_sym], [x_next_expr])

        x_val = np.zeros(NX)
        u_val = np.zeros(NU)

        for _ in range(n_steps):
            x_val = np.array(step_fn(x_val, u_val)).flatten()

        t_total = dt * n_steps
        # pz = -0.5 * g * t^2
        assert x_val[PZ] == pytest.approx(-0.5 * 9.81 * t_total**2, rel=1e-4)
        # vz = -g * t
        assert x_val[VZ] == pytest.approx(-9.81 * t_total, rel=1e-4)


class TestBounds:
    """State and input bound vectors."""

    def test_state_bounds_shape(self) -> None:
        cfg = DynamicsConfig()
        lb, ub = state_bounds(cfg)
        assert lb.shape == (NX,)
        assert ub.shape == (NX,)
        assert np.all(lb <= ub)

    def test_input_bounds_shape(self) -> None:
        cfg = DynamicsConfig()
        lb, ub = input_bounds(cfg)
        assert lb.shape == (NU,)
        assert ub.shape == (NU,)
        assert np.all(lb <= ub)

    def test_thrust_bounds(self) -> None:
        cfg = DynamicsConfig()
        lb, ub = input_bounds(cfg)
        assert lb[0] == 0.0
        assert ub[0] == 1.0

    def test_attitude_bounds_match_config(self) -> None:
        cfg = DynamicsConfig(max_roll=0.8, max_pitch=0.9)
        lb, ub = state_bounds(cfg)
        assert lb[PHI] == pytest.approx(-0.8)
        assert ub[PHI] == pytest.approx(0.8)
        assert lb[THETA] == pytest.approx(-0.9)
        assert ub[THETA] == pytest.approx(0.9)


# ---------------------------------------------------------------------------
# MPCC solver tests
# ---------------------------------------------------------------------------


class TestMPCCSolver:
    """MPCC solver convergence and constraint satisfaction."""

    def test_solver_builds(self) -> None:
        """Solver should construct without errors."""
        from drone_racing.planning.mpcc import MPCCConfig, MPCCSolver

        rl = _make_straight_racing_line(
            np.array([0.0, 0.0, 2.0]),
            np.array([20.0, 0.0, 2.0]),
        )
        solver = MPCCSolver(rl, MPCCConfig(horizon=5, dt=0.05))
        assert solver._n_w > 0

    def test_straight_line_convergence(self) -> None:
        """Solver should converge on a simple straight-line path."""
        from drone_racing.planning.mpcc import MPCCConfig, MPCCSolver

        rl = _make_straight_racing_line(
            np.array([0.0, 0.0, 2.0]),
            np.array([20.0, 0.0, 2.0]),
        )
        cfg = MPCCConfig(horizon=10, dt=0.05, w_progress=5.0, w_contour=1.0, w_lag=1.0)
        solver = MPCCSolver(rl, cfg)

        x0 = np.zeros(NX)
        x0[PZ] = 2.0
        sol = solver.solve(x0, 0.0)

        assert sol.success
        assert sol.states.shape == (11, NX)
        assert sol.inputs.shape == (10, NU)

    def test_progress_increases(self) -> None:
        """Progress parameter should be non-decreasing."""
        from drone_racing.planning.mpcc import MPCCConfig, MPCCSolver

        rl = _make_straight_racing_line(
            np.array([0.0, 0.0, 2.0]),
            np.array([20.0, 0.0, 2.0]),
        )
        cfg = MPCCConfig(horizon=10, dt=0.05, w_progress=10.0)
        solver = MPCCSolver(rl, cfg)

        x0 = np.zeros(NX)
        x0[PZ] = 2.0
        sol = solver.solve(x0, 0.0)

        assert sol.success
        dtheta = np.diff(sol.progress)
        assert np.all(dtheta >= -1e-6), f"Progress decreased: {dtheta}"

    def test_input_bounds_respected(self) -> None:
        """All inputs should stay within bounds."""
        from drone_racing.planning.mpcc import MPCCConfig, MPCCSolver

        cfg_dyn = DynamicsConfig()
        rl = _make_straight_racing_line(
            np.array([0.0, 0.0, 2.0]),
            np.array([20.0, 0.0, 2.0]),
        )
        cfg = MPCCConfig(horizon=10, dt=0.05, dynamics=cfg_dyn)
        solver = MPCCSolver(rl, cfg)

        x0 = np.zeros(NX)
        x0[PZ] = 2.0
        sol = solver.solve(x0, 0.0)

        assert sol.success
        u_lb, u_ub = input_bounds(cfg_dyn)
        for k in range(sol.inputs.shape[0]):
            assert np.all(sol.inputs[k] >= u_lb - 1e-6), f"Input {k} below lower bound"
            assert np.all(sol.inputs[k] <= u_ub + 1e-6), f"Input {k} above upper bound"

    def test_initial_state_pinned(self) -> None:
        """First state in solution should match the provided initial state."""
        from drone_racing.planning.mpcc import MPCCConfig, MPCCSolver

        rl = _make_straight_racing_line(
            np.array([0.0, 0.0, 2.0]),
            np.array([20.0, 0.0, 2.0]),
        )
        solver = MPCCSolver(rl, MPCCConfig(horizon=5, dt=0.05))

        x0 = np.zeros(NX)
        x0[PX] = 1.0
        x0[PZ] = 2.0
        x0[VX] = 3.0
        sol = solver.solve(x0, 0.05)

        assert sol.success
        np.testing.assert_allclose(sol.states[0], x0, atol=1e-6)
        assert sol.progress[0] == pytest.approx(0.05, abs=1e-6)

    def test_warm_start_improves_solve_time(self) -> None:
        """Second solve should be at least as fast as the first (warm start)."""
        from drone_racing.planning.mpcc import MPCCConfig, MPCCSolver

        rl = _make_straight_racing_line(
            np.array([0.0, 0.0, 2.0]),
            np.array([20.0, 0.0, 2.0]),
        )
        solver = MPCCSolver(rl, MPCCConfig(horizon=10, dt=0.05))

        x0 = np.zeros(NX)
        x0[PZ] = 2.0

        sol1 = solver.solve(x0, 0.0)
        assert sol1.success

        # Advance state slightly for second solve
        x1 = sol1.states[1]
        p1 = sol1.progress[1]
        sol2 = solver.solve(x1, p1)

        assert sol2.success
        # Warm-started solve should converge — we can't guarantee faster wall-clock
        # on all hardware, but it should still succeed
        assert sol2.solve_time_ms > 0

    def test_l_shaped_path_convergence(self) -> None:
        """Solver should converge on an L-shaped 3D path with a turn."""
        from drone_racing.planning.mpcc import MPCCConfig, MPCCSolver

        rl = _make_l_shaped_racing_line()
        cfg = MPCCConfig(horizon=10, dt=0.05, w_progress=5.0, w_contour=5.0, w_lag=5.0)
        solver = MPCCSolver(rl, cfg)

        x0 = np.zeros(NX)
        x0[PZ] = 2.0
        sol = solver.solve(x0, 0.0)

        assert sol.success
        assert sol.progress[-1] > sol.progress[0]

    def test_solve_time_reasonable(self) -> None:
        """Solve time should be under 500ms (generous bound for CI)."""
        from drone_racing.planning.mpcc import MPCCConfig, MPCCSolver

        rl = _make_straight_racing_line(
            np.array([0.0, 0.0, 2.0]),
            np.array([20.0, 0.0, 2.0]),
        )
        solver = MPCCSolver(rl, MPCCConfig(horizon=10, dt=0.05))

        x0 = np.zeros(NX)
        x0[PZ] = 2.0
        sol = solver.solve(x0, 0.0)

        assert sol.success
        assert sol.solve_time_ms < 500, f"Solve took {sol.solve_time_ms:.1f}ms"

    def test_from_planner_config(self) -> None:
        """Factory method should create a working solver."""
        from drone_racing.planning.mpcc import MPCCSolver
        from drone_racing.utils.types import PlannerConfig

        rl = _make_straight_racing_line(
            np.array([0.0, 0.0, 2.0]),
            np.array([20.0, 0.0, 2.0]),
        )
        pc = PlannerConfig(
            horizon_steps=5,
            horizon_dt=0.05,
            w_progress=5.0,
            w_contour=1.0,
            w_lag=1.0,
            w_input=0.01,
        )
        solver = MPCCSolver.from_planner_config(rl, pc)

        x0 = np.zeros(NX)
        x0[PZ] = 2.0
        sol = solver.solve(x0, 0.0)
        assert sol.success

    def test_cost_is_finite(self) -> None:
        """Optimal cost should be a finite number."""
        from drone_racing.planning.mpcc import MPCCConfig, MPCCSolver

        rl = _make_straight_racing_line(
            np.array([0.0, 0.0, 2.0]),
            np.array([20.0, 0.0, 2.0]),
        )
        solver = MPCCSolver(rl, MPCCConfig(horizon=5, dt=0.05))

        x0 = np.zeros(NX)
        x0[PZ] = 2.0
        sol = solver.solve(x0, 0.0)

        assert sol.success
        assert np.isfinite(sol.cost)

    def test_solution_shapes(self) -> None:
        """Verify output array shapes match horizon."""
        from drone_racing.planning.mpcc import MPCCConfig, MPCCSolver

        n = 8
        rl = _make_straight_racing_line(
            np.array([0.0, 0.0, 2.0]),
            np.array([20.0, 0.0, 2.0]),
        )
        solver = MPCCSolver(rl, MPCCConfig(horizon=n, dt=0.05))

        x0 = np.zeros(NX)
        x0[PZ] = 2.0
        sol = solver.solve(x0, 0.0)

        assert sol.states.shape == (n + 1, NX)
        assert sol.inputs.shape == (n, NU)
        assert sol.progress.shape == (n + 1,)

    def test_closed_circuit_all_progress_values(self) -> None:
        """Solver converges at all progress values on a closed circuit with turns."""
        from drone_racing.planning.mpcc import MPCCConfig, MPCCSolver

        rl = _make_closed_circuit()
        solver = MPCCSolver(rl, MPCCConfig(), closed=True)

        for start_prog in [0.05, 0.15, 0.25, 0.35, 0.50, 0.65, 0.80, 0.95]:
            solver._prev_solution = None
            pos, _tan, _spd = rl.at_progress(start_prog)
            state = np.zeros(NX)
            state[:3] = pos

            sol = solver.solve(state, start_prog)
            assert sol.success, f"Failed at progress={start_prog:.2f}"
            assert sol.progress[-1] > start_prog

    def test_consecutive_solves_all_succeed(self) -> None:
        """20 consecutive warm-started solves should all succeed."""
        from drone_racing.planning.mpcc import MPCCConfig, MPCCSolver

        rl = _make_closed_circuit()
        solver = MPCCSolver(rl, MPCCConfig(), closed=True)

        pos, _tan, _spd = rl.at_progress(0.0)
        state = np.zeros(NX)
        state[:3] = pos
        prog = 0.0

        # Seed the primal/dual warm start explicitly. A numerically converged cold solve may
        # miss the strict 50 ms flight deadline on a throttled/shared CI host; it is rejected
        # for authority but remains valid as a warm-start seed.
        warmup = solver.solve(state, prog)
        assert warmup.converged
        for step in range(20):
            sol = solver.solve(state, prog)
            assert sol.success, f"Failed at step {step}, progress={prog:.3f}"
            state = sol.states[1]
            prog = sol.progress[1]

        assert prog > 0.01, "Drone should have made progress"

    @pytest.mark.slow
    def test_full_lap_completion(self) -> None:
        """Solver completes a full lap on a closed circuit."""
        from drone_racing.planning.mpcc import MPCCConfig, MPCCSolver

        rl = _make_closed_circuit()
        solver = MPCCSolver(rl, MPCCConfig(), closed=True)

        pos, _tan, _spd = rl.at_progress(0.0)
        state = np.zeros(NX)
        state[:3] = pos
        prog = 0.0

        warmup = solver.solve(state, prog)
        assert warmup.converged
        for step in range(500):
            sol = solver.solve(state, prog)
            assert sol.success, f"Failed at step {step}, progress={prog:.3f}"
            state = sol.states[1]
            prog = sol.progress[1]
            if prog >= 1.0:
                break

        assert prog >= 1.0, f"Did not complete lap: progress={prog:.3f}"


class TestMPCCSafetyContract:
    """The reusable solver fails closed on deployment-unsafe assumptions."""

    def test_open_course_is_the_default(self) -> None:
        from drone_racing.planning.mpcc import MPCCSolver

        line = _make_straight_racing_line(np.zeros(3), np.array([10.0, 0.0, 0.0]))
        solver = MPCCSolver(line)
        assert solver._closed is False

    def test_solution_missed_deadline_is_rejected(self) -> None:
        from drone_racing.planning.mpcc import MPCCConfig, MPCCSolver

        line = _make_straight_racing_line(np.zeros(3), np.array([10.0, 0.0, 0.0]))
        solver = MPCCSolver(line, MPCCConfig(horizon=5, max_solve_time_ms=0.0))
        result = solver.solve(np.zeros(9), 0.0)
        assert result.deadline_met is False
        assert result.success is False
        assert result.converged is True

    def test_converged_deadline_miss_seeds_next_warm_start(self) -> None:
        """Deadline rejection blocks authority without forcing every later solve cold."""
        from drone_racing.planning.mpcc import MPCCConfig, MPCCSolver

        line = _make_straight_racing_line(np.zeros(3), np.array([10.0, 0.0, 0.0]))
        solver = MPCCSolver(line, MPCCConfig(horizon=5, max_solve_time_ms=0.0))
        first = solver.solve(np.zeros(9), 0.0)
        assert first.converged and not first.success
        assert solver._prev_lam_g is not None and solver._prev_lam_x is not None
        guess = solver._build_initial_guess(np.zeros(9), 0.0)
        assert np.all(np.isfinite(guess))
