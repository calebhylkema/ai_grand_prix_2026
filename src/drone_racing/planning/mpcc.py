"""Model Predictive Contouring Control (MPCC) for time-optimal drone racing.

Maximises progress along a reference racing line while minimising contouring
error (lateral deviation) and lag error (behind/ahead of schedule).  Based on
the formulation from Romero, Foehn & Scaramuzza (IEEE T-RO 2022).

Cost function::

    J = Σ_{k=0}^{N-1} [ -w_p * Δθ + w_c * e_c² + w_l * e_l² + w_u * ||u||² ]

where θ is the progress parameter along the racing line, e_c is the
contouring error (perpendicular distance from path), and e_l is the lag
error (distance along path from expected position).

The solver uses CasADi with IPOPT for nonlinear optimisation.  Path
interpolation uses CasADi's compiled ``interpolant`` for efficiency —
this avoids the O(N) symbolic loop that would otherwise make the NLP
graph enormous and slow to evaluate.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import casadi as ca
import numpy as np

from drone_racing.planning.dynamics import (
    NU,
    NX,
    PX,
    PY,
    PZ,
    DynamicsConfig,
    input_bounds,
    rk4_step,
    state_bounds,
)

if TYPE_CHECKING:
    from drone_racing.planning.spline import RacingLine
    from drone_racing.utils.types import PlannerConfig

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class MPCCConfig:
    """MPCC solver configuration.

    Parameters
    ----------
    horizon : int
        Number of prediction steps.
    dt : float
        Time step per prediction step (s).
    w_progress : float
        Weight on progress maximisation (negative cost → maximise).
    w_contour : float
        Weight on contouring error (perpendicular to path).
    w_lag : float
        Weight on lag error (along path).
    w_input : float
        Weight on input magnitude (regularisation).
    w_input_rate : float
        Weight on input rate of change (smoothness).
    max_progress_per_step : float
        Maximum progress increment per step (prevents wrap-around issues).
    dynamics : DynamicsConfig
        Quadrotor dynamics parameters.
    solver_opts : dict
        Options passed to IPOPT solver.
    """

    horizon: int = 20
    dt: float = 0.05
    w_progress: float = 20.0
    w_contour: float = 5.0
    w_lag: float = 5.0
    w_input: float = 0.05
    w_input_rate: float = 0.1
    max_progress_per_step: float = 0.03
    max_solve_time_ms: float = 50.0
    dynamics: DynamicsConfig = field(default_factory=DynamicsConfig)
    solver_opts: dict[str, object] = field(default_factory=dict)


@dataclass(slots=True)
class MPCCSolution:
    """Result of an MPCC solve.

    Parameters
    ----------
    success : bool
        Whether the solver converged.
    states : np.ndarray
        (N+1, 9) predicted state trajectory.
    inputs : np.ndarray
        (N, 4) optimal input sequence.
    progress : np.ndarray
        (N+1,) progress parameter along racing line.
    cost : float
        Optimal cost value.
    solve_time_ms : float
        Wall-clock solve time in milliseconds.
    """

    success: bool
    converged: bool
    states: np.ndarray
    inputs: np.ndarray
    progress: np.ndarray
    cost: float
    solve_time_ms: float
    deadline_met: bool
    solver_status: str


class MPCCSolver:
    """MPCC trajectory optimiser using CasADi + IPOPT.

    Parameters
    ----------
    racing_line : RacingLine
        Reference path from the spline planner.
    config : MPCCConfig | None
        Solver parameters.  Uses defaults if None.
    """

    def __init__(
        self,
        racing_line: RacingLine,
        config: MPCCConfig | None = None,
        closed: bool = False,
    ) -> None:
        self._cfg = config or MPCCConfig()
        self._racing_line = racing_line
        self._closed = closed
        self._prev_solution: MPCCSolution | None = None
        self._prev_lam_g: np.ndarray | None = None
        self._prev_lam_x: np.ndarray | None = None

        # Build CasADi interpolants for the racing line (compiled, fast)
        self._build_path_interpolants()

        # Build numpy propagation function for initial guess generation
        self._build_numpy_propagator()

        # Pre-build the NLP
        self._build_nlp()

    def _build_numpy_propagator(self) -> None:
        """Build a CasADi Function for fast numpy-level state propagation."""
        x_sym = ca.SX.sym("x", NX)
        u_sym = ca.SX.sym("u", NU)
        x_next = rk4_step(x_sym, u_sym, self._cfg.dt, self._cfg.dynamics)
        self._propagate_fn = ca.Function("prop", [x_sym, u_sym], [x_next])

    def _propagate_numpy(self, x: np.ndarray, u: np.ndarray, dt: float) -> np.ndarray:
        """Propagate state one step using the pre-built CasADi function.

        Parameters
        ----------
        x : np.ndarray
            Current state (9,).
        u : np.ndarray
            Input (4,).
        dt : float
            Time step (unused — baked into the function).

        Returns
        -------
        np.ndarray
            Next state (9,).
        """
        return np.array(self._propagate_fn(x, u)).flatten()

    def _build_path_interpolants(self) -> None:
        """Build CasADi interpolant functions for path position and tangent.

        Uses ``ca.interpolant`` with linear interpolation — this compiles to
        efficient C code internally, avoiding the O(N) symbolic graph that
        would result from looping over all racing line points.
        """
        positions = self._racing_line.positions  # (M, 3)
        tangents = self._racing_line.tangents  # (M, 3)
        n_pts = len(positions)

        # Progress grid: uniform in [0, 1]
        grid = np.linspace(0.0, 1.0, n_pts).tolist()

        # CasADi interpolant expects 1D values, so build one per component.
        # Use B-spline interpolation for C² smoothness — linear interpolants
        # have discontinuous gradients at grid points, which causes IPOPT to
        # cycle between two near-optimal points without converging.
        self._pos_interp: list[ca.Function] = []
        self._tan_interp: list[ca.Function] = []

        for dim in range(3):
            self._pos_interp.append(
                ca.interpolant(f"pos_{dim}", "bspline", [grid], positions[:, dim].tolist())
            )
            self._tan_interp.append(
                ca.interpolant(f"tan_{dim}", "bspline", [grid], tangents[:, dim].tolist())
            )

    def _path_at_progress(self, theta: ca.SX) -> tuple[ca.SX, ca.SX]:
        """Interpolate the racing line at a symbolic progress value.

        Uses pre-built CasADi interpolants for efficiency.

        Parameters
        ----------
        theta : ca.SX
            Progress parameter in [0, 1].

        Returns
        -------
        tuple[ca.SX, ca.SX]
            (position (3,), unit_tangent (3,)) at progress theta.
        """
        # For closed tracks, wrap theta to [0, 1) using fmod.
        # For open tracks, clamp to [0, 1].
        if self._closed:
            theta_c = ca.fmod(theta, 1.0)
            # fmod can return negative for negative inputs
            theta_c = ca.fmin(ca.fmax(theta_c, 0.0), 1.0 - 1e-6)
        else:
            theta_c = ca.fmin(ca.fmax(theta, 0.0), 1.0)

        pos = ca.vertcat(
            self._pos_interp[0](theta_c),
            self._pos_interp[1](theta_c),
            self._pos_interp[2](theta_c),
        )
        tan = ca.vertcat(
            self._tan_interp[0](theta_c),
            self._tan_interp[1](theta_c),
            self._tan_interp[2](theta_c),
        )

        # Normalise tangent
        tan_norm = ca.norm_2(tan) + 1e-12
        tan = tan / tan_norm

        return pos, tan

    def _build_nlp(self) -> None:
        """Construct the CasADi NLP structure (called once at init)."""
        n = self._cfg.horizon
        dt = self._cfg.dt
        dyn_cfg = self._cfg.dynamics

        # --- Decision variables ---
        # States: x_0 .. x_N  (N+1 states, each 9D)
        # Inputs: u_0 .. u_{N-1}  (N inputs, each 4D)
        # Progress: theta_0 .. theta_N  (N+1 scalars)
        self._x_vars: list[ca.SX] = []
        self._u_vars: list[ca.SX] = []
        self._theta_vars: list[ca.SX] = []

        for k in range(n + 1):
            self._x_vars.append(ca.SX.sym(f"x_{k}", NX))
            self._theta_vars.append(ca.SX.sym(f"theta_{k}", 1))
        for k in range(n):
            self._u_vars.append(ca.SX.sym(f"u_{k}", NU))

        # Pack into a single decision vector
        # Layout per step k: [x_k (9), theta_k (1), u_k (4)]
        # Last step: [x_N (9), theta_N (1)] — no input
        w: list[ca.SX] = []
        for k in range(n + 1):
            w.append(self._x_vars[k])
            w.append(self._theta_vars[k])
            if k < n:
                w.append(self._u_vars[k])
        self._w = ca.vertcat(*w)

        # Compute per-step size for indexing
        self._step_size = NX + 1 + NU  # state + progress + input
        self._last_step_size = NX + 1  # final step has no input

        # --- Bounds ---
        x_lb, x_ub = state_bounds(dyn_cfg)
        u_lb, u_ub = input_bounds(dyn_cfg)

        lbw: list[float] = []
        ubw: list[float] = []

        # For closed tracks, progress can grow beyond 1.0 (wraps via fmod in
        # the path lookup).  Upper bound allows up to 2 full laps of progress
        # in the horizon.  For open tracks, clamp to [0, 1].
        theta_ub = 2.0 if self._closed else 1.0

        for k in range(n + 1):
            lbw.extend(x_lb.tolist())
            ubw.extend(x_ub.tolist())
            lbw.append(0.0)
            ubw.append(theta_ub)
            if k < n:
                lbw.extend(u_lb.tolist())
                ubw.extend(u_ub.tolist())

        # --- Constraints and cost ---
        g: list[ca.SX] = []
        lbg: list[float] = []
        ubg: list[float] = []
        total_cost = ca.SX(0)

        max_dp = self._cfg.max_progress_per_step

        for k in range(n):
            xk = self._x_vars[k]
            uk = self._u_vars[k]
            xk1 = self._x_vars[k + 1]
            thetak = self._theta_vars[k]
            thetak1 = self._theta_vars[k + 1]

            # Dynamics: x_{k+1} = f(x_k, u_k)
            xk1_pred = rk4_step(xk, uk, dt, dyn_cfg)
            g.append(xk1 - xk1_pred)
            lbg.extend([0.0] * NX)
            ubg.extend([0.0] * NX)

            # Progress monotonicity: 0 <= theta_{k+1} - theta_k <= max_dp
            d_theta = thetak1 - thetak
            g.append(d_theta)
            lbg.append(0.0)
            ubg.append(max_dp)

            # --- Contouring and lag errors ---
            pos_k = ca.vertcat(xk[PX], xk[PY], xk[PZ])
            ref_pos, ref_tangent = self._path_at_progress(thetak)

            error_vec = pos_k - ref_pos

            # Lag error: component along tangent
            e_lag = ca.dot(error_vec, ref_tangent)

            # Contouring error²: total error² - lag² (Pythagorean decomposition).
            # With B-spline path interpolation this stays non-negative; any tiny
            # numerical negativity from cancellation is harmless.
            e_contour_sq = ca.dot(error_vec, error_vec) - e_lag**2

            # Stage cost
            total_cost += (
                -self._cfg.w_progress * d_theta
                + self._cfg.w_contour * e_contour_sq
                + self._cfg.w_lag * e_lag**2
                + self._cfg.w_input * ca.dot(uk, uk)
            )

            # Input rate cost (smoothness)
            if k > 0:
                du = uk - self._u_vars[k - 1]
                total_cost += self._cfg.w_input_rate * ca.dot(du, du)

        self._g = ca.vertcat(*g)
        self._lbg = np.array(lbg)
        self._ubg = np.array(ubg)
        self._lbw = np.array(lbw)
        self._ubw = np.array(ubw)

        # --- Build solver ---
        nlp = {"x": self._w, "f": total_cost, "g": self._g}

        opts: dict[str, object] = {
            "ipopt.print_level": 0,
            "ipopt.max_iter": 50,
            "ipopt.tol": 1e-3,
            "ipopt.acceptable_tol": 1e-2,
            "ipopt.acceptable_iter": 3,
            "ipopt.warm_start_init_point": "yes",
            "ipopt.warm_start_bound_push": 1e-6,
            "ipopt.warm_start_mult_bound_push": 1e-6,
            "ipopt.mu_strategy": "adaptive",
            "ipopt.mu_init": 1e-3,
            "print_time": 0,
        }
        opts.update(self._cfg.solver_opts)

        self._solver = ca.nlpsol("mpcc", "ipopt", nlp, opts)

        self._n_w = self._w.shape[0]
        self._w0 = np.zeros(self._n_w)

        logger.info(
            "MPCC NLP built: %d vars, %d constraints, horizon=%d, dt=%.3fs",
            self._n_w,
            self._g.shape[0],
            n,
            dt,
        )

    def _build_initial_guess(
        self,
        current_state: np.ndarray,
        current_progress: float,
    ) -> np.ndarray:
        """Build an initial guess for the NLP decision variables.

        If a previous solution exists, shifts it forward by one step
        (warm-start). Otherwise, uses a straight-line hover guess.

        Parameters
        ----------
        current_state : np.ndarray
            Current drone state (9,).
        current_progress : float
            Current progress along the racing line [0, 1].

        Returns
        -------
        np.ndarray
            Initial guess vector.
        """
        n = self._cfg.horizon
        hover_thrust = self._cfg.dynamics.mass * 9.81 / self._cfg.dynamics.max_thrust

        if self._prev_solution is not None and self._prev_solution.converged:
            # Shift previous solution forward by one step.
            # Use RK4 to propagate the last state (keeps dynamics consistent).
            prev = self._prev_solution
            w0 = np.zeros(self._n_w)
            offset = 0

            for k in range(n + 1):
                src_k = min(k + 1, n)
                w0[offset : offset + NX] = prev.states[src_k]
                offset += NX
                w0[offset] = prev.progress[src_k]
                offset += 1
                if k < n:
                    src_u = min(k + 1, n - 1)
                    w0[offset : offset + NU] = prev.inputs[src_u]
                    offset += NU

            # Override initial state and progress
            w0[:NX] = current_state
            w0[NX] = current_progress
            return w0

        # Cold start: forward-simulate from current state using hover inputs,
        # with progress estimated from the resulting positions.
        w0 = np.zeros(self._n_w)
        offset = 0

        # Build a CasADi function for propagation

        x_s = np.copy(current_state)
        u_hover = np.array([hover_thrust, 0.0, 0.0, 0.0])

        for k in range(n + 1):
            w0[offset : offset + NX] = x_s
            offset += NX

            # Progress from position
            prog_k = self._racing_line.closest_progress(x_s[:3])
            # Ensure monotonicity
            if k > 0:
                prog_k = max(prog_k, w0[offset - NX - 1])
            w0[offset] = prog_k
            offset += 1

            if k < n:
                w0[offset : offset + NU] = u_hover
                offset += NU
                # Propagate state numerically (not symbolic)
                x_s = self._propagate_numpy(x_s, u_hover, self._cfg.dt)

        return w0

    def solve(
        self,
        current_state: np.ndarray,
        current_progress: float,
    ) -> MPCCSolution:
        """Solve the MPCC for one control cycle.

        Parameters
        ----------
        current_state : np.ndarray
            Current drone state (9,): [px, py, pz, vx, vy, vz, phi, theta, psi].
        current_progress : float
            Current progress along the racing line [0, 1].

        Returns
        -------
        MPCCSolution
            Optimal trajectory and inputs.
        """
        n = self._cfg.horizon
        t_start = time.perf_counter()

        # --- Set initial state constraint ---
        lbw = self._lbw.copy()
        ubw = self._ubw.copy()

        # x_0 is fixed
        lbw[:NX] = current_state
        ubw[:NX] = current_state

        # theta_0 is fixed
        lbw[NX] = current_progress
        ubw[NX] = current_progress

        # --- Initial guess ---
        w0 = self._build_initial_guess(current_state, current_progress)

        # --- Solve with primal + dual warm-start ---
        solver_args: dict[str, object] = {
            "x0": w0,
            "lbx": lbw,
            "ubx": ubw,
            "lbg": self._lbg,
            "ubg": self._ubg,
        }
        if self._prev_lam_g is not None:
            solver_args["lam_g0"] = self._prev_lam_g
        if self._prev_lam_x is not None:
            solver_args["lam_x0"] = self._prev_lam_x
        sol = self._solver(**solver_args)

        t_end = time.perf_counter()
        solve_time_ms = (t_end - t_start) * 1000.0

        # --- Extract solution ---
        w_opt = np.array(sol["x"]).flatten()
        solver_stats = self._solver.stats()
        solver_success = bool(solver_stats["success"])
        cost = float(sol["f"])

        states = np.zeros((n + 1, NX))
        progress = np.zeros(n + 1)
        inputs = np.zeros((n, NU))

        offset = 0
        for k in range(n + 1):
            states[k, :] = w_opt[offset : offset + NX]
            offset += NX
            progress[k] = w_opt[offset]
            offset += 1
            if k < n:
                inputs[k, :] = w_opt[offset : offset + NU]
                offset += NU

        finite = bool(
            np.all(np.isfinite(states))
            and np.all(np.isfinite(inputs))
            and np.all(np.isfinite(progress))
            and np.isfinite(cost)
        )
        deadline_met = solve_time_ms <= self._cfg.max_solve_time_ms
        converged = solver_success and finite
        success = converged and deadline_met

        # A converged solution that missed the flight deadline must never receive command
        # authority (``success`` remains false), but it is still numerically safe and useful
        # as the next solver's warm start. Discarding it forced every subsequent solve to
        # cold-start and turned one startup miss into a permanent deadline failure.
        self._w0 = w_opt
        self._prev_lam_g = np.array(sol["lam_g"]).flatten() if converged else None
        self._prev_lam_x = np.array(sol["lam_x"]).flatten() if converged else None

        result = MPCCSolution(
            success=success,
            converged=converged,
            states=states,
            inputs=inputs,
            progress=progress,
            cost=cost,
            solve_time_ms=solve_time_ms,
            deadline_met=deadline_met,
            solver_status=str(solver_stats.get("return_status", "unknown")),
        )
        self._prev_solution = result

        logger.debug(
            "MPCC solved: success=%s, cost=%.3f, time=%.1fms, progress=%.3f→%.3f",
            success,
            cost,
            solve_time_ms,
            progress[0],
            progress[-1],
        )

        return result

    @staticmethod
    def from_planner_config(
        racing_line: RacingLine,
        planner_config: PlannerConfig,
        closed: bool = False,
    ) -> MPCCSolver:
        """Create an MPCCSolver from the standard PlannerConfig.

        Parameters
        ----------
        racing_line : RacingLine
            Reference path.
        planner_config : PlannerConfig
            Planner parameters from TOML config.
        closed : bool
            Whether the racing line is a closed loop.

        Returns
        -------
        MPCCSolver
            Configured solver.
        """
        mpcc_cfg = MPCCConfig(
            horizon=planner_config.horizon_steps,
            dt=planner_config.horizon_dt,
            w_progress=planner_config.w_progress,
            w_contour=planner_config.w_contour,
            w_lag=planner_config.w_lag,
            w_input=planner_config.w_input,
        )
        return MPCCSolver(racing_line, mpcc_cfg, closed=closed)
