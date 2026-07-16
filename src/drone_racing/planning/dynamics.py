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
import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

import casadi as ca
import numpy as np

if TYPE_CHECKING:
    from numpy.typing import NDArray

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


@dataclass(frozen=True, slots=True)
class VQ2DynamicsConfig:
    """Identified VQ2 point-mass model using NED and rate/thrust inputs.

    Unlike :class:`DynamicsConfig`, this contract matches the only simulator-honored
    interface: normalized collective thrust plus FRD body rates. Values are safe
    initialization bounds, not claims of identification; flight code must load a
    reviewed identified artifact before using the model for commands.
    """

    hover_thrust: float = 0.30
    min_thrust: float = 0.10
    max_thrust: float = 0.55
    drag_coeff: tuple[float, float, float] = (0.3, 0.3, 0.3)
    max_body_rate: tuple[float, float, float] = (3.0, 3.0, 3.0)
    rate_gain: tuple[float, float, float] = (1.0, 1.0, 1.0)
    rate_delay_s: tuple[float, float, float] = (0.0, 0.0, 0.0)
    max_roll: float = 0.8
    max_pitch: float = 0.8

    def __post_init__(self) -> None:
        if not 0.0 < self.min_thrust < self.hover_thrust < self.max_thrust <= 1.0:
            raise ValueError("VQ2 thrust bounds must contain hover and lie in (0, 1]")
        if any(value < 0.0 or not np.isfinite(value) for value in self.drag_coeff):
            raise ValueError("VQ2 drag coefficients must be finite and non-negative")
        if any(value <= 0.0 or not np.isfinite(value) for value in self.max_body_rate):
            raise ValueError("VQ2 body-rate bounds must be finite and positive")
        if any(value <= 0.0 or not np.isfinite(value) for value in self.rate_gain):
            raise ValueError("VQ2 rate gains must be finite and positive")
        if any(value < 0.0 or not np.isfinite(value) for value in self.rate_delay_s):
            raise ValueError("VQ2 rate delays must be finite and non-negative")


@dataclass(frozen=True, slots=True)
class DynamicsFit:
    """Held-out-ready result of fitting the VQ2 acceleration model."""

    config: VQ2DynamicsConfig
    sample_count: int
    acceleration_rmse_mps2: float
    source_id: str


@dataclass(frozen=True, slots=True)
class RateResponseFit:
    """Per-axis command-to-measured-body-rate identification result."""

    gain: tuple[float, float, float]
    delay_s: tuple[float, float, float]
    bias_rps: tuple[float, float, float]
    rmse_rps: tuple[float, float, float]
    sample_count: int
    source_id: str


def continuous_vq2_dynamics(
    x: ca.SX,
    u: ca.SX,
    config: VQ2DynamicsConfig,
) -> ca.SX:
    """VQ2 NED point-mass dynamics with ``[thrust, p, q, r]`` inputs.

    State remains ``[north,east,down,vn,ve,vd,roll,pitch,yaw]``. Body rates use
    FRD signs. Positive pitch is nose-up (braking); negative pitch accelerates
    forward, matching the calibrated VQ2 command convention before the one command
    boundary applies simulator sign/gain correction.
    """
    vx, vy, vz = x[VX], x[VY], x[VZ]
    phi, theta, psi = x[PHI], x[THETA], x[PSI]
    thrust_command = u[0]
    p_rate = u[1] * config.rate_gain[0]
    q_rate = u[2] * config.rate_gain[1]
    r_rate = u[3] * config.rate_gain[2]

    cphi, sphi = ca.cos(phi), ca.sin(phi)
    ctheta, stheta = ca.cos(theta), ca.sin(theta)
    cpsi, spsi = ca.cos(psi), ca.sin(psi)
    # Third column of R_body_to_NED for ZYX Euler angles.
    body_down_ned = ca.vertcat(
        cpsi * stheta * cphi + spsi * sphi,
        spsi * stheta * cphi - cpsi * sphi,
        ctheta * cphi,
    )
    specific_thrust = _G * thrust_command / config.hover_thrust
    drag = ca.vertcat(*config.drag_coeff) * ca.vertcat(vx, vy, vz)
    acceleration = ca.vertcat(0.0, 0.0, _G) - specific_thrust * body_down_ned - drag

    # Exact body-rate to ZYX Euler-rate map (safe inside configured pitch bounds).
    tan_theta = ca.tan(theta)
    inv_cos_theta = 1.0 / ca.cos(theta)
    phi_dot = p_rate + sphi * tan_theta * q_rate + cphi * tan_theta * r_rate
    theta_dot = cphi * q_rate - sphi * r_rate
    psi_dot = sphi * inv_cos_theta * q_rate + cphi * inv_cos_theta * r_rate
    return ca.vertcat(vx, vy, vz, acceleration, phi_dot, theta_dot, psi_dot)


def rk4_vq2_step(x: ca.SX, u: ca.SX, dt: float, config: VQ2DynamicsConfig) -> ca.SX:
    """Integrate the VQ2 rate/thrust model by one fixed RK4 step."""
    k1 = continuous_vq2_dynamics(x, u, config)
    k2 = continuous_vq2_dynamics(x + dt / 2.0 * k1, u, config)
    k3 = continuous_vq2_dynamics(x + dt / 2.0 * k2, u, config)
    k4 = continuous_vq2_dynamics(x + dt * k3, u, config)
    return x + dt / 6.0 * (k1 + 2.0 * k2 + 2.0 * k3 + k4)


def vq2_input_bounds(config: VQ2DynamicsConfig) -> tuple[np.ndarray, np.ndarray]:
    """Return ``[thrust, p, q, r]`` bounds for the VQ2 model."""
    rates = np.asarray(config.max_body_rate, dtype=np.float64)
    return (
        np.concatenate(([config.min_thrust], -rates)),
        np.concatenate(([config.max_thrust], rates)),
    )


def fit_vq2_acceleration_model(
    attitudes_rpy: NDArray[np.floating],
    velocities_ned_mps: NDArray[np.floating],
    thrust_commands: NDArray[np.floating],
    accelerations_ned_mps2: NDArray[np.floating],
    *,
    source_id: str,
    min_samples: int = 20,
) -> DynamicsFit:
    """Fit thrust scale and axis drag from synchronized flight-state samples.

    This is an offline identification boundary. It requires world-NED acceleration,
    attitude, velocity, and the command actually sent at the calibrated boundary.
    Raw accelerometer specific force is not a substitute for world acceleration.
    """
    from scipy.optimize import lsq_linear
    from scipy.spatial.transform import Rotation

    rpy = np.asarray(attitudes_rpy, dtype=np.float64)
    velocity = np.asarray(velocities_ned_mps, dtype=np.float64)
    thrust = np.asarray(thrust_commands, dtype=np.float64)
    acceleration = np.asarray(accelerations_ned_mps2, dtype=np.float64)
    count = len(thrust)
    if count < min_samples:
        raise ValueError(f"need at least {min_samples} synchronized samples, got {count}")
    if rpy.shape != (count, 3) or velocity.shape != (count, 3) or acceleration.shape != (count, 3):
        raise ValueError("attitude, velocity, and acceleration must all have shape (N, 3)")
    if not all(np.all(np.isfinite(array)) for array in (rpy, velocity, thrust, acceleration)):
        raise ValueError("dynamics-fit inputs contain non-finite values")
    if not source_id.strip():
        raise ValueError("source_id must identify the fit dataset")

    rotations = Rotation.from_euler("xyz", rpy).as_matrix()
    body_down = rotations[:, :, 2]
    # a - g = -scale*command*b3 - diag(drag)*velocity.
    design = np.zeros((count * 3, 4), dtype=np.float64)
    target = (acceleration - np.array([0.0, 0.0, _G])).reshape(-1)
    for axis in range(3):
        rows = np.arange(count) * 3 + axis
        design[rows, 0] = -thrust * body_down[:, axis]
        design[rows, axis + 1] = -velocity[:, axis]
    fit = lsq_linear(design, target, bounds=(0.0, np.inf))
    thrust_scale = float(fit.x[0])
    if thrust_scale <= _G or not math.isfinite(thrust_scale):
        raise ValueError("identified thrust scale cannot support a valid normalized hover")
    hover = _G / thrust_scale
    if not 0.05 < hover < 0.95:
        raise ValueError(f"identified hover command {hover:.3f} is implausible")
    residual = design @ fit.x - target
    rmse = float(np.sqrt(np.mean(residual**2)))
    config = VQ2DynamicsConfig(
        hover_thrust=hover,
        min_thrust=max(0.01, hover * 0.33),
        max_thrust=min(1.0, max(hover * 1.83, hover + 0.1)),
        drag_coeff=tuple(float(x) for x in fit.x[1:]),  # type: ignore[arg-type]
    )
    return DynamicsFit(
        config=config, sample_count=count, acceleration_rmse_mps2=rmse, source_id=source_id
    )


def fit_vq2_rate_response(
    time_s: NDArray[np.floating],
    commanded_rates_rps: NDArray[np.floating],
    measured_rates_rps: NDArray[np.floating],
    *,
    source_id: str,
    max_delay_s: float = 0.2,
    min_samples: int = 40,
) -> RateResponseFit:
    """Fit per-axis rate gain, bias, and transport/response delay offline.

    Candidate delays are evaluated at the recording's median sample interval. The fit
    expects *desired actual* rates before the single calibrated simulator boundary,
    not raw inverted/scaled MAVLink values.
    """
    timestamps = np.asarray(time_s, dtype=np.float64)
    commanded = np.asarray(commanded_rates_rps, dtype=np.float64)
    measured = np.asarray(measured_rates_rps, dtype=np.float64)
    count = len(timestamps)
    if count < min_samples:
        raise ValueError(f"need at least {min_samples} rate samples, got {count}")
    if commanded.shape != (count, 3) or measured.shape != (count, 3):
        raise ValueError("commanded and measured rates must have shape (N, 3)")
    if not all(np.all(np.isfinite(array)) for array in (timestamps, commanded, measured)):
        raise ValueError("rate-fit inputs contain non-finite values")
    if np.any(np.diff(timestamps) <= 0.0):
        raise ValueError("rate-fit timestamps must be strictly increasing")
    if max_delay_s < 0.0 or not source_id.strip():
        raise ValueError("max_delay_s must be non-negative and source_id non-empty")
    dt = float(np.median(np.diff(timestamps)))
    delays = np.arange(0.0, max_delay_s + 0.5 * dt, dt)
    gains: list[float] = []
    best_delays: list[float] = []
    biases: list[float] = []
    errors: list[float] = []
    for axis in range(3):
        best: tuple[float, float, float, float] | None = None
        for delay in delays:
            shifted_time = timestamps - delay
            valid = shifted_time >= timestamps[0]
            if np.count_nonzero(valid) < min_samples:
                continue
            delayed_command = np.interp(shifted_time[valid], timestamps, commanded[:, axis])
            design = np.column_stack([delayed_command, np.ones(len(delayed_command))])
            coefficients, *_ = np.linalg.lstsq(design, measured[valid, axis], rcond=None)
            prediction = design @ coefficients
            rmse = float(np.sqrt(np.mean((prediction - measured[valid, axis]) ** 2)))
            candidate = (rmse, float(delay), float(coefficients[0]), float(coefficients[1]))
            if best is None or candidate[0] < best[0]:
                best = candidate
        if best is None or best[2] <= 0.0:
            raise ValueError(f"could not identify a positive response on rate axis {axis}")
        errors.append(best[0])
        best_delays.append(best[1])
        gains.append(best[2])
        biases.append(best[3])
    return RateResponseFit(
        gain=tuple(gains),  # type: ignore[arg-type]
        delay_s=tuple(best_delays),  # type: ignore[arg-type]
        bias_rps=tuple(biases),  # type: ignore[arg-type]
        rmse_rps=tuple(errors),  # type: ignore[arg-type]
        sample_count=count,
        source_id=source_id,
    )


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
