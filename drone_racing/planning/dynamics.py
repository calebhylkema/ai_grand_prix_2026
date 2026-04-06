"""Point-mass quadrotor dynamics model for MPCC trajectory optimisation.

Simplified model suitable for receding-horizon planning at 50–100 Hz.
The full 6-DOF attitude dynamics are left to the inner-loop controller
(PID cascade or rate controller); the planner treats the drone as a
3D point mass with bounded thrust and angular rate inputs.

State vector (9-D)::

    x = [px, py, pz, vx, vy, vz, phi, theta, psi]

where (phi, theta, psi) are roll, pitch, yaw Euler angles.  This is
acceptable for the planner because we never approach gimbal-lock
attitudes during racing (tilt < 70°).

Input vector (4-D)::

    u = [T, phi_cmd, theta_cmd, psi_dot_cmd]

where T is normalised collective thrust [0, 1], phi_cmd and theta_cmd
are commanded roll/pitch angles (rad), and psi_dot_cmd is yaw rate (rad/s).

The attitude is modelled as a first-order lag (time constant tau) to
approximate the inner-loop tracking bandwidth.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import casadi as ca
import numpy as np

logger = logging.getLogger(__name__)

# Gravity
_G = 9.81


@dataclass(slots=True)
class DynamicsConfig:
    """Parameters for the point-mass quadrotor dynamics.

    Parameters
    ----------
    mass : float
        Drone mass (kg).
    max_thrust : float
        Maximum collective thrust (N).  T_normalised * max_thrust = force.
    attitude_tau : float
        First-order attitude tracking time constant (s).  Smaller = snappier.
    max_roll : float
        Maximum roll angle (rad).
    max_pitch : float
        Maximum pitch angle (rad).
    max_yaw_rate : float
        Maximum yaw rate (rad/s).
    """

    mass: float = 1.5
    max_thrust: float = 29.43
    attitude_tau: float = 0.2
    max_roll: float = 0.8
    max_pitch: float = 0.8
    max_yaw_rate: float = 3.0
    drag_coeff: float = 0.3


# State indices
PX, PY, PZ = 0, 1, 2
VX, VY, VZ = 3, 4, 5
PHI, THETA, PSI = 6, 7, 8
NX = 9

# Input indices
T_CMD, PHI_CMD, THETA_CMD, PSI_DOT_CMD = 0, 1, 2, 3
NU = 4


def continuous_dynamics(
    x: ca.SX,
    u: ca.SX,
    config: DynamicsConfig,
) -> ca.SX:
    """Continuous-time point-mass quadrotor dynamics.

    Parameters
    ----------
    x : ca.SX
        State vector (9,).
    u : ca.SX
        Input vector (4,).
    config : DynamicsConfig
        Physical parameters.

    Returns
    -------
    ca.SX
        State derivative dx/dt (9,).
    """
    # Unpack state
    vx = x[VX]
    vy = x[VY]
    vz = x[VZ]
    phi = x[PHI]
    theta = x[THETA]
    psi = x[PSI]

    # Unpack inputs
    t_cmd = u[T_CMD]
    phi_cmd = u[PHI_CMD]
    theta_cmd = u[THETA_CMD]
    psi_dot_cmd = u[PSI_DOT_CMD]

    # Thrust force (normalised → Newtons)
    thrust = t_cmd * config.max_thrust

    # Acceleration in world frame (simplified rotation: small-angle body Z → world)
    # For a point-mass model, thrust acts along the body Z-axis which is
    # tilted by roll (phi) and pitch (theta) from vertical.
    ax = (thrust / config.mass) * (
        ca.cos(psi) * ca.sin(theta) + ca.sin(psi) * ca.sin(phi)
    ) - config.drag_coeff * vx
    ay = (thrust / config.mass) * (
        ca.sin(psi) * ca.sin(theta) - ca.cos(psi) * ca.sin(phi)
    ) - config.drag_coeff * vy
    az = (thrust / config.mass) * ca.cos(phi) * ca.cos(theta) - _G - config.drag_coeff * vz

    # First-order attitude lag
    tau = config.attitude_tau
    dphi = (phi_cmd - phi) / tau
    dtheta = (theta_cmd - theta) / tau
    dpsi = psi_dot_cmd

    # Assemble dx/dt
    xdot = ca.vertcat(
        vx,  # dpx
        vy,  # dpy
        vz,  # dpz
        ax,  # dvx
        ay,  # dvy
        az,  # dvz
        dphi,  # dphi
        dtheta,  # dtheta
        dpsi,  # dpsi
    )

    return xdot


def rk4_step(
    x: ca.SX,
    u: ca.SX,
    dt: float,
    config: DynamicsConfig,
) -> ca.SX:
    """Single RK4 integration step.

    Parameters
    ----------
    x : ca.SX
        Current state (9,).
    u : ca.SX
        Input (4,), held constant over the step.
    dt : float
        Time step (s).
    config : DynamicsConfig
        Physical parameters.

    Returns
    -------
    ca.SX
        Next state (9,).
    """
    k1 = continuous_dynamics(x, u, config)
    k2 = continuous_dynamics(x + dt / 2 * k1, u, config)
    k3 = continuous_dynamics(x + dt / 2 * k2, u, config)
    k4 = continuous_dynamics(x + dt * k3, u, config)
    return x + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)


def state_bounds(config: DynamicsConfig) -> tuple[np.ndarray, np.ndarray]:
    """Return lower and upper bounds on the state vector.

    Parameters
    ----------
    config : DynamicsConfig
        Physical parameters.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        (lower_bounds, upper_bounds) each shape (9,).
    """
    inf = 1e6
    lb = np.array([-inf, -inf, -inf, -inf, -inf, -inf, -config.max_roll, -config.max_pitch, -inf])
    ub = np.array([inf, inf, inf, inf, inf, inf, config.max_roll, config.max_pitch, inf])
    return lb, ub


def input_bounds(config: DynamicsConfig) -> tuple[np.ndarray, np.ndarray]:
    """Return lower and upper bounds on the input vector.

    Parameters
    ----------
    config : DynamicsConfig
        Physical parameters.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        (lower_bounds, upper_bounds) each shape (4,).
    """
    lb = np.array([0.0, -config.max_roll, -config.max_pitch, -config.max_yaw_rate])
    ub = np.array([1.0, config.max_roll, config.max_pitch, config.max_yaw_rate])
    return lb, ub
