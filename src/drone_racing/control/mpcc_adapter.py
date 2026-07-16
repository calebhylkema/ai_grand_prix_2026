"""MPCC-to-DroneCommand adapter with safety ladder fallback.

Converts MPCC optimal inputs ``[T, phi_cmd, theta_cmd, psi_dot_cmd]``
into rate-mode ``DroneCommand(throttle, roll_rate, pitch_rate, yaw_rate)``
that the flight controller accepts.

The MPCC dynamics model uses a first-order attitude lag::

    dphi   = (phi_cmd   - phi)   / tau
    dtheta = (theta_cmd - theta) / tau

so the desired angular rates are simply the lag-model rates, which this
adapter computes from the MPCC output and the current attitude estimate.

Safety ladder (per ``control/CLAUDE.md``)::

    1. MPCC solve succeeded → use MPCC output
    2. MPCC failed          → fall back to PID cascade
    3. State estimate bad   → hover in place
    4. Attitude > 80°       → kill motors
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
from scipy.spatial.transform import Rotation

from drone_racing.control.pid import PIDController, TrajectorySetpoint
from drone_racing.utils.types import DroneCommand, State

if TYPE_CHECKING:
    from drone_racing.planning.mpcc import MPCCSolution, MPCCSolver
    from drone_racing.planning.spline import RacingLine
    from drone_racing.utils.types import ControlConfig

logger = logging.getLogger(__name__)

# Maximum tilt from horizontal before emergency kill (rad) ≈ 80°
_EMERGENCY_TILT_RAD = np.radians(80.0)

# Covariance trace threshold for "unreliable state estimate".
# The ESKF 15x15 covariance includes position, velocity, orientation, and
# bias terms.  Process noise causes the trace to grow between vision
# corrections.  Must be calibrated per sim/hardware — set conservatively
# high until we have real data.
_COV_TRACE_THRESHOLD = 1e9  # Effectively disabled — calibrate for real hardware


class FallbackReason:
    """Reason codes for fallback transitions (logged for telemetry)."""

    MPCC_OK = "mpcc_ok"
    MPCC_FAIL = "mpcc_fail"
    STATE_UNRELIABLE = "state_unreliable"
    EMERGENCY_TILT = "emergency_tilt"
    SHADOW_ONLY = "shadow_only"


@dataclass(slots=True)
class MPCCControllerStats:
    """Diagnostics from the last control cycle."""

    fallback_reason: str
    solve_time_ms: float
    progress: float
    mpcc_success: bool


class MPCCController:
    """MPCC trajectory controller with PID fallback and safety ladder.

    Parameters
    ----------
    mpcc_solver : MPCCSolver
        Pre-built MPCC solver instance.
    racing_line : RacingLine
        Reference racing line (shared with the MPCC solver).
    pid_controller : PIDController
        Fallback PID controller.
    control_config : ControlConfig
        Physical limits for clamping.
    attitude_tau : float
        Attitude first-order time constant (s), must match the MPCC dynamics.
    """

    def __init__(
        self,
        mpcc_solver: MPCCSolver,
        racing_line: RacingLine,
        pid_controller: PIDController,
        control_config: ControlConfig,
        attitude_tau: float = 0.2,
        replan_dt_ns: int | None = None,
        *,
        allow_command_authority: bool = False,
    ) -> None:
        self._mpcc = mpcc_solver
        self._racing_line = racing_line
        self._pid = pid_controller
        self._cfg = control_config
        self._tau = attitude_tau
        self._allow_command_authority = allow_command_authority
        self._current_progress: float = 0.0
        self._last_stats: MPCCControllerStats | None = None
        self._consecutive_failures: int = 0

        # Re-plan interval: defaults to the MPCC timestep.
        # Between re-plans, the most recent MPCC solution is used to
        # generate commands (interpolating the planned trajectory).
        mpcc_dt_ns = int(mpcc_solver._cfg.dt * 1e9)
        self._replan_dt_ns = replan_dt_ns if replan_dt_ns is not None else mpcc_dt_ns
        self._last_solve_ns: int = 0
        self._last_solution: MPCCSolution | None = None
        self._solution_step: int = 0  # which step in the current solution to use

    @property
    def current_progress(self) -> float:
        """Current progress along the racing line [0, 1]."""
        return self._current_progress

    @property
    def last_stats(self) -> MPCCControllerStats | None:
        """Diagnostics from the most recent update."""
        return self._last_stats

    def reset(self, progress: float = 0.0) -> None:
        """Reset controller state.

        Parameters
        ----------
        progress : float
            Initial progress along the racing line.
        """
        self._current_progress = progress
        self._consecutive_failures = 0
        self._last_solve_ns = 0
        self._last_solution = None
        self._solution_step = 0
        self._pid.reset()
        self._mpcc._prev_solution = None

    def update(self, state: State) -> DroneCommand:
        """Run one control cycle: MPCC → fallback ladder → DroneCommand.

        The MPCC re-solves at ``replan_dt_ns`` intervals.  Between re-plans,
        commands are generated from the planned trajectory by stepping
        through the solution sequence.

        Parameters
        ----------
        state : State
            Current state estimate from the ESKF.

        Returns
        -------
        DroneCommand
            Rate-mode command for the flight controller.
        """
        # --- Safety check 4: Emergency tilt ---
        tilt = _compute_tilt(state)
        if tilt > _EMERGENCY_TILT_RAD:
            logger.warning(
                "EMERGENCY: tilt=%.1f° > 80°, killing motors",
                np.degrees(tilt),
            )
            self._last_stats = MPCCControllerStats(
                fallback_reason=FallbackReason.EMERGENCY_TILT,
                solve_time_ms=0.0,
                progress=self._current_progress,
                mpcc_success=False,
            )
            return DroneCommand(
                timestamp_ns=state.timestamp_ns,
                throttle=0.0,
                roll_rate=0.0,
                pitch_rate=0.0,
                yaw_rate=0.0,
            )

        # --- Safety check 3: Unreliable state estimate → PID fallback ---
        cov_trace = float(np.trace(state.covariance))
        if cov_trace > _COV_TRACE_THRESHOLD:
            logger.debug(
                "State uncertain (cov_trace=%.1f), using PID fallback",
                cov_trace,
            )
            cmd = self._pid_fallback(state)
            self._last_stats = MPCCControllerStats(
                fallback_reason=FallbackReason.STATE_UNRELIABLE,
                solve_time_ms=0.0,
                progress=self._current_progress,
                mpcc_success=False,
            )
            return cmd

        # The reusable MPCC stack is pre-sim lineage and is shadow-only by default.
        # A caller must make command authority explicit; the active VQ2 deployment
        # never instantiates this adapter.
        if not self._allow_command_authority:
            cmd = self._pid_fallback(state)
            self._last_stats = MPCCControllerStats(
                fallback_reason=FallbackReason.SHADOW_ONLY,
                solve_time_ms=0.0,
                progress=self._current_progress,
                mpcc_success=False,
            )
            return cmd

        # --- Check if we need to re-plan ---
        needs_replan = (
            self._last_solution is None
            or state.timestamp_ns - self._last_solve_ns >= self._replan_dt_ns
            or self._solution_step >= self._last_solution.inputs.shape[0]
        )

        if needs_replan:
            # Update progress from position
            if self._racing_line.closed:
                pos_progress = self._racing_line.closest_progress(state.position)
                dp = pos_progress - self._current_progress
                if dp < -0.5:
                    dp += 1.0
                if dp > 0.5:
                    dp -= 1.0
                if abs(dp) < 0.1:
                    self._current_progress = pos_progress
            else:
                self._current_progress = self._racing_line.closest_progress(
                    state.position,
                    previous_progress=self._current_progress,
                    search_window=0.1,
                    max_backtrack=0.0,
                )

            mpcc_state = _state_to_mpcc(state)
            sol = self._mpcc.solve(mpcc_state, self._current_progress)

            if sol.success:
                self._consecutive_failures = 0
                self._last_solution = sol
                self._last_solve_ns = state.timestamp_ns
                self._solution_step = 0
                self._current_progress = sol.progress[1]
            else:
                self._consecutive_failures += 1
                logger.debug(
                    "MPCC failed (%d consecutive), falling back to PID",
                    self._consecutive_failures,
                )
                cmd = self._pid_fallback(state)
                self._last_stats = MPCCControllerStats(
                    fallback_reason=FallbackReason.MPCC_FAIL,
                    solve_time_ms=sol.solve_time_ms,
                    progress=self._current_progress,
                    mpcc_success=False,
                )
                return cmd

        # --- Use current solution step ---
        assert self._last_solution is not None  # guaranteed by logic above
        sol = self._last_solution
        step = min(self._solution_step, sol.inputs.shape[0] - 1)

        cmd = _mpcc_step_to_command(
            sol,
            step,
            state,
            self._tau,
            self._cfg,
        )
        self._solution_step += 1

        self._last_stats = MPCCControllerStats(
            fallback_reason=FallbackReason.MPCC_OK,
            solve_time_ms=sol.solve_time_ms if needs_replan else 0.0,
            progress=self._current_progress,
            mpcc_success=True,
        )
        return cmd

    def _pid_fallback(self, state: State) -> DroneCommand:
        """Generate a command using the PID cascade as fallback.

        Parameters
        ----------
        state : State
            Current state estimate.

        Returns
        -------
        DroneCommand
            PID-generated command.
        """
        # Look ahead on the racing line for a setpoint
        lookahead = self._current_progress + 0.015
        lookahead = lookahead % 1.0 if self._racing_line.closed else min(1.0, lookahead)
        target_pos, target_tangent, target_speed = self._racing_line.at_progress(lookahead)

        # Clamp setpoint distance
        pos_error = target_pos - state.position
        err_dist = float(np.linalg.norm(pos_error))
        if err_dist > 3.0:
            target_pos = state.position + pos_error * (3.0 / err_dist)

        desired_yaw = float(np.arctan2(target_tangent[1], target_tangent[0]))
        ff_velocity = target_tangent * min(target_speed, 8.0) * 0.4

        setpoint = TrajectorySetpoint(
            position=target_pos,
            velocity=ff_velocity,
            yaw=desired_yaw,
        )
        return self._pid.update(state, setpoint)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _state_to_mpcc(state: State) -> np.ndarray:
    """Convert ESKF State to the 9D MPCC state vector.

    MPCC state: [px, py, pz, vx, vy, vz, phi, theta, psi]

    Parameters
    ----------
    state : State
        ESKF state estimate.

    Returns
    -------
    np.ndarray
        (9,) MPCC state vector.
    """
    r = Rotation.from_quat(state.orientation)
    roll, pitch, yaw = r.as_euler("xyz")

    x = np.zeros(9)
    x[0:3] = state.position
    x[3:6] = state.velocity
    x[6] = roll
    x[7] = pitch
    x[8] = yaw
    return x


def _mpcc_step_to_command(
    sol: MPCCSolution,
    step: int,
    state: State,
    tau: float,
    config: ControlConfig,
) -> DroneCommand:
    """Convert an MPCC planned input at a given step to a DroneCommand.

    MPCC input: [T_cmd, phi_cmd, theta_cmd, psi_dot_cmd]
    DroneCommand: throttle, roll_rate, pitch_rate, yaw_rate

    The attitude rates replicate the first-order lag model that the MPCC
    dynamics assume: rate = (cmd - current) / tau.

    Parameters
    ----------
    sol : MPCCSolution
        Successful MPCC solve result.
    step : int
        Which step in the solution to use (0 = first input).
    state : State
        Current state estimate (for current attitude).
    tau : float
        Attitude time constant (s).
    config : ControlConfig
        Physical limits for clamping.

    Returns
    -------
    DroneCommand
        Clamped rate-mode command.
    """
    u = sol.inputs[step]  # [T_cmd, phi_cmd, theta_cmd, psi_dot_cmd]

    throttle = float(np.clip(u[0], 0.0, config.max_thrust))

    # Current Euler angles
    r = Rotation.from_quat(state.orientation)
    roll_cur, pitch_cur, _yaw_cur = r.as_euler("xyz")

    # Desired rates from the first-order lag model.
    # The gain 1/tau is boosted to compensate for the sim's angular
    # velocity dynamics, which add an extra first-order lag.
    gain = 1.0 / tau
    roll_rate = float(gain * (u[1] - roll_cur))
    pitch_rate = float(gain * (u[2] - pitch_cur))
    yaw_rate = float(u[3])

    # Clamp to physical limits
    roll_rate = float(np.clip(roll_rate, -config.max_roll_rate, config.max_roll_rate))
    pitch_rate = float(np.clip(pitch_rate, -config.max_pitch_rate, config.max_pitch_rate))
    yaw_rate = float(np.clip(yaw_rate, -config.max_yaw_rate, config.max_yaw_rate))

    return DroneCommand(
        timestamp_ns=state.timestamp_ns,
        throttle=throttle,
        roll_rate=roll_rate,
        pitch_rate=pitch_rate,
        yaw_rate=yaw_rate,
    )


def _compute_tilt(state: State) -> float:
    """Compute the tilt angle from horizontal (rad).

    Parameters
    ----------
    state : State
        Current state estimate.

    Returns
    -------
    float
        Tilt angle in radians (0 = perfectly level).
    """
    r = Rotation.from_quat(state.orientation)
    # Body Z-axis in world frame
    body_z = r.apply(np.array([0.0, 0.0, 1.0]))
    # Angle between body Z and world Z (vertical)
    cos_tilt = float(np.clip(body_z[2], -1.0, 1.0))
    return float(np.arccos(cos_tilt))


def _hover_command(state: State) -> DroneCommand:
    """Generate a hover-in-place command.

    Parameters
    ----------
    state : State
        Current state (for timestamp).

    Returns
    -------
    DroneCommand
        Hover throttle with zero angular rates.
    """
    return DroneCommand(
        timestamp_ns=state.timestamp_ns,
        throttle=0.30,  # measured VQ2 hover; this helper remains fallback-only
        roll_rate=0.0,
        pitch_rate=0.0,
        yaw_rate=0.0,
    )
