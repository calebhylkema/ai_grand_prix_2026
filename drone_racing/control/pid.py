"""Cascaded PID controller for quadrotor trajectory tracking.

Architecture (inner loops run faster than outer loops)::

    Position PID → Velocity PID → Attitude → Rate PID → DroneCommand

The controller converts a desired position + yaw into rate-mode commands
(throttle, roll_rate, pitch_rate, yaw_rate) that the flight controller accepts.

All computations use SI units (metres, radians, seconds).  Gains are read
from :class:`~drone_racing.utils.types.ControlConfig`.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np

from drone_racing.utils.transforms import (
    GRAVITY_W,
    quat_to_rotation_matrix,
)
from drone_racing.utils.types import DroneCommand, State

if TYPE_CHECKING:
    from drone_racing.utils.types import ControlConfig, Vec3

logger = logging.getLogger(__name__)

# Gravity magnitude (m/s²)
_G = 9.81

# Hover throttle approximation (normalised) — the throttle needed to
# counteract gravity when the drone is level.  Will need tuning once we
# have the real mass / thrust curve.
_HOVER_THROTTLE = 0.5


# ---------------------------------------------------------------------------
# Trajectory setpoint
# ---------------------------------------------------------------------------
@dataclass(slots=True)
class TrajectorySetpoint:
    """Desired state for the controller to track.

    At minimum, ``position`` must be set.  Optional fields allow
    feed-forward terms for tighter tracking.
    """

    position: Vec3
    """Desired position in world frame (m)."""

    velocity: Vec3 = field(default_factory=lambda: np.zeros(3))
    """Desired velocity in world frame (m/s) — feed-forward."""

    yaw: float = 0.0
    """Desired yaw angle (rad) in world frame."""

    acceleration: Vec3 = field(default_factory=lambda: np.zeros(3))
    """Desired acceleration in world frame (m/s²) — feed-forward."""


# ---------------------------------------------------------------------------
# Single-axis PID with anti-windup
# ---------------------------------------------------------------------------
class _PIDAxis:
    """Scalar PID controller with integrator anti-windup (clamping).

    Parameters
    ----------
    kp : float
        Proportional gain.
    ki : float
        Integral gain.
    kd : float
        Derivative gain.
    integral_limit : float
        Symmetric clamp on the integrator state.
    """

    __slots__ = ("_kp", "_ki", "_kd", "_integral", "_prev_error", "_integral_limit")

    def __init__(
        self,
        kp: float,
        ki: float,
        kd: float,
        integral_limit: float = 5.0,
    ) -> None:
        self._kp = kp
        self._ki = ki
        self._kd = kd
        self._integral = 0.0
        self._prev_error: float | None = None
        self._integral_limit = integral_limit

    def reset(self) -> None:
        """Reset integrator and derivative memory."""
        self._integral = 0.0
        self._prev_error = None

    def update(self, error: float, dt: float) -> float:
        """Compute PID output for one time step.

        Parameters
        ----------
        error : float
            Setpoint − measurement.
        dt : float
            Time step (s).  Must be > 0.

        Returns
        -------
        float
            Control output.
        """
        if dt <= 0.0:
            return 0.0

        # Proportional
        p_term = self._kp * error

        # Integral with anti-windup
        self._integral += error * dt
        self._integral = np.clip(self._integral, -self._integral_limit, self._integral_limit)
        i_term = self._ki * self._integral

        # Derivative (on error; first call uses zero)
        d_term = 0.0 if self._prev_error is None else self._kd * (error - self._prev_error) / dt
        self._prev_error = error

        return float(p_term + i_term + d_term)


# ---------------------------------------------------------------------------
# 3-axis PID helper
# ---------------------------------------------------------------------------
class _PID3:
    """Three independent PID axes (x, y, z).

    Parameters
    ----------
    kp : list[float]
        Proportional gains [x, y, z].
    ki : list[float]
        Integral gains [x, y, z].
    kd : list[float]
        Derivative gains [x, y, z].
    integral_limit : float
        Symmetric integrator clamp per axis.
    """

    __slots__ = ("_axes",)

    def __init__(
        self,
        kp: list[float],
        ki: list[float],
        kd: list[float],
        integral_limit: float = 5.0,
    ) -> None:
        self._axes = tuple(_PIDAxis(kp[i], ki[i], kd[i], integral_limit) for i in range(3))

    def reset(self) -> None:
        """Reset all axes."""
        for ax in self._axes:
            ax.reset()

    def update(self, error: Vec3, dt: float) -> Vec3:
        """Compute 3-axis PID output.

        Parameters
        ----------
        error : Vec3
            Error vector [x, y, z].
        dt : float
            Time step (s).

        Returns
        -------
        Vec3
            Output vector [x, y, z].
        """
        return np.array(
            [self._axes[i].update(float(error[i]), dt) for i in range(3)],
            dtype=np.float64,
        )


# ---------------------------------------------------------------------------
# Cascaded PID controller
# ---------------------------------------------------------------------------
class PIDController:
    """Cascaded PID controller for quadrotor trajectory tracking.

    Cascade: Position PID → Velocity PID → desired thrust + attitude →
    Rate PID → DroneCommand.

    Parameters
    ----------
    config : ControlConfig
        PID gains and physical limits from configuration.
    """

    def __init__(self, config: ControlConfig) -> None:
        self._cfg = config

        # Position loop (outermost)
        self._pos_pid = _PID3(
            config.position_kp,
            config.position_ki,
            config.position_kd,
        )

        # Velocity loop
        self._vel_pid = _PID3(
            config.velocity_kp,
            [0.0, 0.0, 0.0],  # No integral on velocity (position loop handles it)
            [0.0, 0.0, 0.0],
        )

        # Rate loop (innermost — P-only is standard for rate)
        self._rate_pid = _PID3(
            config.rate_kp,
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
        )

        self._last_timestamp_ns: int | None = None

    def reset(self) -> None:
        """Reset all PID integrators and derivative memory."""
        self._pos_pid.reset()
        self._vel_pid.reset()
        self._rate_pid.reset()
        self._last_timestamp_ns = None

    def update(self, state: State, setpoint: TrajectorySetpoint) -> DroneCommand:
        """Run one full cascade iteration and produce a command.

        Parameters
        ----------
        state : State
            Current state estimate from the ESKF.
        setpoint : TrajectorySetpoint
            Desired position/velocity/yaw.

        Returns
        -------
        DroneCommand
            Clamped rate-mode command for the flight controller.
        """
        # Compute dt
        if self._last_timestamp_ns is None:
            dt = 0.02  # Default 50 Hz on first call
        else:
            dt_ns = state.timestamp_ns - self._last_timestamp_ns
            dt = max(dt_ns * 1e-9, 1e-6)
        self._last_timestamp_ns = state.timestamp_ns

        # --- Position loop ---
        pos_error = setpoint.position - state.position
        desired_velocity = self._pos_pid.update(pos_error, dt) + setpoint.velocity

        # --- Velocity loop ---
        vel_error = desired_velocity - state.velocity
        desired_accel_w = self._vel_pid.update(vel_error, dt) + setpoint.acceleration

        # --- Acceleration → thrust + attitude ---
        # Total desired acceleration including gravity compensation
        # In world frame: a_desired = a_command - g  →  a_command = a_desired - g_vec
        # (GRAVITY_W is [0,0,-9.81], so subtracting it adds +9.81 on Z)
        thrust_vec_w = desired_accel_w - GRAVITY_W  # world-frame thrust direction

        # Thrust magnitude (normalised to [0, 1])
        thrust_magnitude = float(np.linalg.norm(thrust_vec_w))
        # Map physical acceleration to normalised throttle
        # At hover: thrust = g, throttle = _HOVER_THROTTLE
        throttle = _HOVER_THROTTLE * thrust_magnitude / _G
        throttle = float(np.clip(throttle, 0.0, self._cfg.max_thrust))

        # Desired body Z-axis (thrust direction in world frame, normalised)
        if thrust_magnitude > 1e-6:
            z_des = thrust_vec_w / thrust_magnitude
        else:
            z_des = np.array([0.0, 0.0, 1.0])

        # Build desired rotation matrix from thrust direction + desired yaw
        # 1. Desired yaw gives us the x-axis direction projected onto horizontal
        yaw_des = setpoint.yaw
        x_c = np.array([np.cos(yaw_des), np.sin(yaw_des), 0.0])

        # 2. y_des = z_des × x_c (right-hand rule), then normalise
        y_des = np.cross(z_des, x_c)
        y_norm = float(np.linalg.norm(y_des))
        y_des = y_des / y_norm if y_norm > 1e-6 else np.array([0.0, 1.0, 0.0])

        # 3. x_des = y_des × z_des
        x_des = np.cross(y_des, z_des)

        r_des = np.column_stack([x_des, y_des, z_des])  # desired rotation matrix

        # --- Attitude error → desired angular rates ---
        r_cur = quat_to_rotation_matrix(state.orientation)
        # Error rotation: R_err = R_des^T @ R_cur  →  extract to angle-axis
        r_err = r_des.T @ r_cur
        # Extract error angles (small-angle: off-diagonal elements)
        # This is the standard "vee map" for SO(3) error
        angle_error = (
            np.array(
                [r_err[2, 1] - r_err[1, 2], r_err[0, 2] - r_err[2, 0], r_err[1, 0] - r_err[0, 1]],
                dtype=np.float64,
            )
            * 0.5
        )

        # Attitude P-controller → desired body rates
        att_kp = np.array(self._cfg.attitude_kp, dtype=np.float64)
        desired_rates_body = -att_kp * angle_error

        # --- Rate loop ---
        # Current body angular rates from gyro (approximation: use zero if not available)
        # In a full system, we'd read gyro directly; here we use the state estimate.
        # For now, assume we're tracking from zero angular rate.
        # The rate PID takes (desired_rate - current_rate) as error.
        # We don't have current angular rate in State, so we skip the rate PID
        # and directly output the attitude controller's desired rates.
        # This is equivalent to a pure P attitude + P rate controller.
        roll_rate = float(desired_rates_body[0])
        pitch_rate = float(desired_rates_body[1])
        yaw_rate = float(desired_rates_body[2])

        # --- Clamp to physical limits ---
        roll_rate = float(np.clip(roll_rate, -self._cfg.max_roll_rate, self._cfg.max_roll_rate))
        pitch_rate = float(np.clip(pitch_rate, -self._cfg.max_pitch_rate, self._cfg.max_pitch_rate))
        yaw_rate = float(np.clip(yaw_rate, -self._cfg.max_yaw_rate, self._cfg.max_yaw_rate))

        return DroneCommand(
            timestamp_ns=state.timestamp_ns,
            throttle=throttle,
            roll_rate=roll_rate,
            pitch_rate=pitch_rate,
            yaw_rate=yaw_rate,
        )
