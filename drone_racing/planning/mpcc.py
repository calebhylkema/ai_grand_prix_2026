"""Model Predictive Contouring Control (MPCC) for time-optimal drone racing.

Maximises progress along a reference racing line while minimising contouring
error (lateral deviation) and lag error (behind/ahead of schedule).  Based on
the formulation from Romero, Foehn & Scaramuzza (IEEE T-RO 2022).

Cost function::

    J = Σ_{k=0}^{N-1} [ -w_p * Δθ + w_c * e_c² + w_l * e_l² + w_u * ||u||² ]

where θ is the progress parameter along the racing line, e_c is the
contouring error (perpendicular distance from path), and e_l is the lag
error (distance along path from expected position).

The solver uses CasADi with IPOPT for nonlinear optimisation.
"""

from __future__ import annotations

import logging
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
    dynamics : DynamicsConfig
        Quadrotor dynamics parameters.
    solver_opts : dict
        Options passed to IPOPT solver.
    """

    horizon: int = 20
    dt: float = 0.05
    w_progress: float = 10.0
    w_contour: float = 5.0
    w_lag: float = 5.0
    w_input: float = 0.01
    w_input_rate: float = 0.1
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
    states: np.ndarray
    inputs: np.ndarray
    progress: np.ndarray
    cost: float
    solve_time_ms: float


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
    ) -> None:
        self._cfg = config or MPCCConfig()
        self._racing_line = racing_line

        # Pre-build the NLP
        self._build_nlp()

    def _build_nlp(self) -> None:
        """Construct the CasADi NLP structure (called once at init)."""
        n_horizon = self._cfg.horizon
        dt = self._cfg.dt
        dyn_cfg = self._cfg.dynamics

        # --- Decision variables ---
        # States: x_0, x_1, ..., x_N  (N+1 states)
        # Inputs: u_0, u_1, ..., u_{N-1}  (N inputs)
        # Progress: theta_0, theta_1, ..., theta_N  (N+1 progress values)

        self._x_vars: list[ca.SX] = []
        self._u_vars: list[ca.SX] = []
        self._theta_vars: list[ca.SX] = []

        for k in range(n_horizon + 1):
            self._x_vars.append(ca.SX.sym(f"x_{k}", NX))
            self._theta_vars.append(ca.SX.sym(f"theta_{k}", 1))
        for k in range(n_horizon):
            self._u_vars.append(ca.SX.sym(f"u_{k}", NU))

        # Pack into a single decision vector
        w: list[ca.SX] = []
        for k in range(n_horizon + 1):
            w.append(self._x_vars[k])
            w.append(self._theta_vars[k])
            if k < n_horizon:
                w.append(self._u_vars[k])
        self._w = ca.vertcat(*w)

        # --- Constraints ---
        g: list[ca.SX] = []  # constraint expressions
        lbg: list[float] = []  # lower bounds
        ubg: list[float] = []  # upper bounds

        # --- Cost ---
        total_cost = ca.SX(0)

        # Bounds on decision variables
        lbw: list[float] = []
        ubw: list[float] = []

        x_lb, x_ub = state_bounds(dyn_cfg)
        u_lb, u_ub = input_bounds(dyn_cfg)

        for k in range(n_horizon + 1):
            # State bounds
            lbw.extend(x_lb.tolist())
            ubw.extend(x_ub.tolist())
            # Progress bounds: [0, 1]
            lbw.append(0.0)
            ubw.append(1.0)

            if k < n_horizon:
                # Input bounds
                lbw.extend(u_lb.tolist())
                ubw.extend(u_ub.tolist())

        # --- Dynamics constraints + cost ---
        for k in range(n_horizon):
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

            # Progress monotonicity: theta_{k+1} >= theta_k
            g.append(thetak1 - thetak)
            lbg.append(0.0)
            ubg.append(1.0)  # Don't allow more than full lap per step

            # --- Contouring and lag errors ---
            # Get reference position and tangent at progress theta_k
            pos_k = ca.vertcat(xk[PX], xk[PY], xk[PZ])

            # Reference path: interpolate from discretised racing line
            ref_pos, ref_tangent = self._path_at_progress(thetak)

            # Error vector from reference to drone
            error_vec = pos_k - ref_pos

            # Lag error: component along tangent
            e_lag = ca.dot(error_vec, ref_tangent)

            # Contouring error: perpendicular component
            e_contour_sq = ca.dot(error_vec, error_vec) - e_lag**2

            # Progress increment
            d_theta = thetak1 - thetak

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
        self._total_cost = total_cost

        # --- Build solver ---
        nlp = {"x": self._w, "f": total_cost, "g": self._g}

        opts: dict[str, object] = {
            "ipopt.print_level": 0,
            "ipopt.max_iter": 200,
            "ipopt.tol": 1e-4,
            "ipopt.warm_start_init_point": "yes",
            "print_time": 0,
        }
        opts.update(self._cfg.solver_opts)

        self._solver = ca.nlpsol("mpcc", "ipopt", nlp, opts)

        # Store dimensions for warm-starting
        self._n_w = self._w.shape[0]
        self._w0 = np.zeros(self._n_w)

        logger.info(
            "MPCC NLP built: %d vars, %d constraints, horizon=%d, dt=%.3fs",
            self._n_w,
            self._g.shape[0],
            n_horizon,
            dt,
        )

    def _path_at_progress(self, theta: ca.SX) -> tuple[ca.SX, ca.SX]:
        """Interpolate the racing line at a symbolic progress value.

        Uses piecewise-linear interpolation over the discretised racing line
        points. This creates a CasADi expression that can be differentiated.

        Parameters
        ----------
        theta : ca.SX
            Progress parameter in [0, 1].

        Returns
        -------
        tuple[ca.SX, ca.SX]
            (position (3,), unit_tangent (3,)) at progress theta.
        """
        positions = self._racing_line.positions  # (M, 3)
        tangents = self._racing_line.tangents  # (M, 3)
        n_pts = len(positions)

        # Clamp theta to [0, 1]
        theta_c = ca.fmin(ca.fmax(theta, 0.0), 1.0)

        # Continuous index
        idx_f = theta_c * (n_pts - 1)
        idx_lo = ca.floor(idx_f)
        alpha = idx_f - idx_lo

        # We use a lookup approach: sum over all segments with an indicator
        # For efficiency with CasADi, use a direct linear interpolation
        # by constructing the interpolation with CasADi's interpolant
        # However, for simplicity and robustness, we pre-build a CasADi
        # interpolant using the discretised points.

        # Simple approach: piecewise constant lookup with interpolation
        # Use CasADi's conditional to handle this
        pos = ca.SX.zeros(3)
        tan = ca.SX.zeros(3)

        for i in range(n_pts - 1):
            # Weight for this segment: 1 when idx_lo == i, 0 otherwise
            # Smooth approximation using soft indicator
            w_i = ca.fmax(0, 1 - ca.fabs(idx_lo - i))

            p0 = positions[i]
            p1 = positions[i + 1]
            t0 = tangents[i]
            t1 = tangents[i + 1]

            pos += w_i * ((1 - alpha) * p0 + alpha * p1)
            tan += w_i * ((1 - alpha) * t0 + alpha * t1)

        # Normalise tangent
        tan_norm = ca.norm_2(tan) + 1e-12
        tan = tan / tan_norm

        return pos, tan

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
        import time

        n_horizon = self._cfg.horizon

        t_start = time.perf_counter()

        # --- Set initial state constraint ---
        # The first state is fixed to the current state
        lbw = self._lbw.copy()
        ubw = self._ubw.copy()

        # State x_0 is fixed
        # Find the offset of x_0 in the decision vector
        # Layout per step k: x_k (NX), theta_k (1), [u_k (NU) if k < N]
        # x_0 starts at index 0
        lbw[:NX] = current_state
        ubw[:NX] = current_state

        # theta_0 is fixed to current progress
        lbw[NX] = current_progress
        ubw[NX] = current_progress

        # --- Warm start ---
        w0 = self._w0.copy()
        # Set initial guess for x_0
        w0[:NX] = current_state
        w0[NX] = current_progress

        # Propagate initial guess: constant input, linear progress
        offset = 0
        for k in range(n_horizon + 1):
            # x_k
            w0[offset : offset + NX] = current_state
            offset += NX
            # theta_k
            w0[offset] = min(current_progress + k * 0.01, 1.0)
            offset += 1
            # u_k
            if k < n_horizon:
                # Hover input as initial guess
                hover_thrust = self._cfg.dynamics.mass * 9.81 / self._cfg.dynamics.max_thrust
                w0[offset + 0] = hover_thrust  # thrust
                w0[offset + 1] = 0.0  # roll
                w0[offset + 2] = 0.0  # pitch
                w0[offset + 3] = 0.0  # yaw rate
                offset += NU

        # --- Solve ---
        sol = self._solver(
            x0=w0,
            lbx=lbw,
            ubx=ubw,
            lbg=self._lbg,
            ubg=self._ubg,
        )

        t_end = time.perf_counter()
        solve_time_ms = (t_end - t_start) * 1000.0

        # --- Extract solution ---
        w_opt = np.array(sol["x"]).flatten()
        success = self._solver.stats()["success"]
        cost = float(sol["f"])

        # Unpack states, inputs, progress
        states = np.zeros((n_horizon + 1, NX))
        progress = np.zeros(n_horizon + 1)
        inputs = np.zeros((n_horizon, NU))

        offset = 0
        for k in range(n_horizon + 1):
            states[k, :] = w_opt[offset : offset + NX]
            offset += NX
            progress[k] = w_opt[offset]
            offset += 1
            if k < n_horizon:
                inputs[k, :] = w_opt[offset : offset + NU]
                offset += NU

        # Save for warm-starting next solve
        self._w0 = w_opt

        logger.debug(
            "MPCC solved: success=%s, cost=%.3f, time=%.1fms, progress=%.3f→%.3f",
            success,
            cost,
            solve_time_ms,
            progress[0],
            progress[-1],
        )

        return MPCCSolution(
            success=success,
            states=states,
            inputs=inputs,
            progress=progress,
            cost=cost,
            solve_time_ms=solve_time_ms,
        )

    @staticmethod
    def from_planner_config(
        racing_line: RacingLine,
        planner_config: PlannerConfig,
    ) -> MPCCSolver:
        """Create an MPCCSolver from the standard PlannerConfig.

        Parameters
        ----------
        racing_line : RacingLine
            Reference path.
        planner_config : PlannerConfig
            Planner parameters from TOML config.

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
        return MPCCSolver(racing_line, mpcc_cfg)
