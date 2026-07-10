"""Geometric position/trajectory controller on the ESKF state (VQ2).

Reactive point-and-servo can't thread the gate-1 turn (it can't translate onto the
moving-turn centerline in time). This controller uses the ESKF's metric relative
position + velocity + attitude to fly a smooth intercept that arrives CENTERED in
position AND velocity: a desired acceleration (PD on relative position, velocity
damped) is turned into a desired thrust vector -> desired attitude -> body rates.
Standard acceleration->attitude quadrotor control (Lee et al., simplified).

Conventions: world NED, body FRD (thrust along body -z = "up"), gravity = +Z down.
"""

from __future__ import annotations

import numpy as np

G = 9.81
G_WORLD = np.array([0.0, 0.0, G])  # gravity, NED (down +)


def _rotvec_from_matrix(r: np.ndarray) -> np.ndarray:
    """Rotation vector (axis*angle) of a rotation matrix, robust for 0..pi."""
    cos_a = (np.trace(r) - 1.0) / 2.0
    cos_a = max(-1.0, min(1.0, cos_a))
    ang = float(np.arccos(cos_a))
    if ang < 1e-6:
        return np.array([r[2, 1] - r[1, 2], r[0, 2] - r[2, 0], r[1, 0] - r[0, 1]]) * 0.5
    axis = np.array([r[2, 1] - r[1, 2], r[0, 2] - r[2, 0], r[1, 0] - r[0, 1]])
    axis = axis / (2.0 * np.sin(ang))
    return axis * ang


def geometric_control(
    p_rel: np.ndarray,  # gate position relative to drone, world NED (gate - drone)
    v: np.ndarray,  # drone velocity, world NED
    r: np.ndarray,  # attitude body->world (3x3)
    *,
    hover_thrust: float = 0.30,
    kp: float = 1.2,
    kd: float = 2.2,
    k_rate: float = 6.0,
    max_tilt: float = 0.6,  # rad, limit commanded lean (~34 deg)
    max_acc: float = 8.0,  # m/s^2, limit desired horizontal accel
    max_rate: float = 3.0,
    thr_min: float = 0.1,
    thr_max: float = 0.55,
) -> tuple[float, np.ndarray, np.ndarray]:
    """Return (collective_thrust[0..1], body_rates[3] rad/s, a_des[3] world) for a
    smooth intercept of the gate. `p_rel` = gate minus drone (world)."""
    # Desired acceleration: PD to the gate, damped by velocity (arrive with low speed).
    a_des = kp * np.asarray(p_rel, float) - kd * np.asarray(v, float)
    n = float(np.linalg.norm(a_des))
    if n > max_acc:
        a_des = a_des * (max_acc / n)

    # Required specific-force (world): f = a_des - g_world. Hover => (0,0,-G) = up.
    f_world = a_des - G_WORLD
    tmag = float(np.linalg.norm(f_world))
    if tmag < 1e-3:
        f_world = -G_WORLD
        tmag = G

    # Desired body-DOWN axis (world) = -f_world/|f| (thrust is along body-up).
    b3 = -f_world / tmag
    # Limit tilt: keep b3 within max_tilt of straight-down (0,0,1).
    down = np.array([0.0, 0.0, 1.0])
    cos_tilt = float(np.dot(b3, down))
    if cos_tilt < np.cos(max_tilt):
        # project b3 toward `down` to the tilt cone
        horiz = b3 - cos_tilt * down
        hn = float(np.linalg.norm(horiz))
        if hn > 1e-6:
            b3 = np.cos(max_tilt) * down + np.sin(max_tilt) * (horiz / hn)
            b3 = b3 / np.linalg.norm(b3)

    # Desired heading (world horizontal): point body-forward at the gate.
    yaw = np.arctan2(p_rel[1], p_rel[0])  # NED: atan2(east, north)
    h = np.array([np.cos(yaw), np.sin(yaw), 0.0])
    b2 = np.cross(b3, h)  # body-right = down x forward
    b2n = float(np.linalg.norm(b2))
    b2 = np.array([0.0, 1.0, 0.0]) if b2n < 1e-6 else b2 / b2n
    b1 = np.cross(b2, b3)  # body-forward
    r_des = np.column_stack([b1, b2, b3])

    # Attitude error -> body rates.
    r_err = r.T @ r_des
    omega = k_rate * _rotvec_from_matrix(r_err)
    omega = np.clip(omega, -max_rate, max_rate)

    # Collective: scale hover by required specific force along the CURRENT body-up.
    body_up_world = -r[:, 2]
    thrust_acc = float(np.dot(f_world, body_up_world))  # project onto actual thrust axis
    collective = hover_thrust * max(0.0, thrust_acc) / G
    collective = float(np.clip(collective, thr_min, thr_max))
    return collective, omega, a_des
