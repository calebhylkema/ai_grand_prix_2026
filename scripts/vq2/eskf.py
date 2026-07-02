"""15-state error-state Kalman filter (visual-inertial) for VQ2.

State (nominal): position p, velocity v (world NED), attitude q (body->world),
accel bias ba, gyro bias bg. Error state (15): dp, dv, dtheta, dba, dbg, with the
LOCAL (body) attitude error convention  q_true = q (x) dq,  dq ~ [1, dtheta/2].

Measurement: the gate is a STATIC world landmark. We anchor the world origin AT the
current target gate, so the drone's position p equals its position relative to that
gate. PnP gives the gate in the camera frame; converted to body it is
  z = CAM_TO_BODY @ p_cam
and the predicted measurement is  h(x) = -R^T @ p  (gate at origin, drone at p).
On passing a gate (active_gate_index changes) we RE-ANCHOR: reset p from the new
gate's PnP, keeping v/q/biases (continuous).

Conventions match estimator.py (validated): world NED (g=+Z down), body FRD, quat
scalar-first [w,x,y,z] body->world.
"""

from __future__ import annotations

import numpy as np
from estimator import G, quat_mul, quat_to_euler, quat_to_matrix


def skew(v: np.ndarray) -> np.ndarray:
    x, y, z = v
    return np.array([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]])


def quat_from_rotvec(r: np.ndarray) -> np.ndarray:
    ang = float(np.linalg.norm(r))
    if ang < 1e-9:
        q = np.array([1.0, r[0] / 2, r[1] / 2, r[2] / 2])
    else:
        axis = r / ang
        q = np.array([np.cos(ang / 2), *(np.sin(ang / 2) * axis)])
    return q / np.linalg.norm(q)


class ESKF:
    def __init__(
        self,
        sig_a: float = 0.5,  # accel noise (m/s^2/sqrt(Hz))
        sig_g: float = 0.02,  # gyro noise (rad/s/sqrt(Hz))
        sig_ba: float = 0.02,  # accel bias random walk
        sig_bg: float = 0.002,  # gyro bias random walk
        pnp_pos_sig: float = 0.4,  # gate position measurement noise (m)
    ) -> None:
        self.p = np.zeros(3)
        self.v = np.zeros(3)
        self.q = np.array([1.0, 0.0, 0.0, 0.0])
        self.ba = np.zeros(3)
        self.bg = np.zeros(3)
        self.P = np.eye(15) * 0.01
        self.P[0:3, 0:3] *= 100.0  # position (relative to gate) initially unknown
        self.P[3:6, 3:6] *= 1.0
        self.sig_a, self.sig_g = sig_a, sig_g
        self.sig_ba, self.sig_bg = sig_ba, sig_bg
        self.Rm = np.eye(3) * pnp_pos_sig**2
        self._init_att = False

    # ------------------------------------------------------------------
    def initialise_attitude(self, accel: np.ndarray) -> None:
        ax, ay, az = accel
        roll = np.arctan2(-ay, -az)
        pitch = np.arctan2(ax, np.hypot(ay, az))
        cr, sr = np.cos(roll / 2), np.sin(roll / 2)
        cp, sp = np.cos(pitch / 2), np.sin(pitch / 2)
        self.q = np.array([cr * cp, sr * cp, cr * sp, -sr * sp])
        self.q /= np.linalg.norm(self.q)
        self._init_att = True

    @property
    def R(self) -> np.ndarray:  # noqa: N802
        return quat_to_matrix(self.q)

    @property
    def euler(self) -> tuple[float, float, float]:
        return quat_to_euler(self.q)

    @property
    def initialised(self) -> bool:
        return self._init_att

    # ------------------------------------------------------------------
    def predict(self, accel: np.ndarray, gyro: np.ndarray, dt: float) -> None:
        if not self._init_att:
            self.initialise_attitude(accel)
            return
        a_m = np.asarray(accel, float)
        w = np.asarray(gyro, float) - self.bg
        r = self.R
        # Reject collision/saturation spikes for the velocity/position integration
        # (attitude still propagates via gyro); prevents blow-up from impacts.
        if abs(float(np.linalg.norm(a_m)) - G) > 25.0:
            self.q = quat_mul(self.q, quat_from_rotvec(w * dt))
            self.q /= np.linalg.norm(self.q)
            return
        a_world = r @ (a_m - self.ba) + np.array([0.0, 0.0, G])
        # nominal propagation
        self.p = self.p + self.v * dt + 0.5 * a_world * dt * dt
        self.v = self.v + a_world * dt
        self.q = quat_mul(self.q, quat_from_rotvec(w * dt))
        self.q /= np.linalg.norm(self.q)
        # error-state transition (local attitude error)
        f = np.eye(15)
        f[0:3, 3:6] = np.eye(3) * dt
        f[3:6, 6:9] = -r @ skew(a_m - self.ba) * dt
        f[3:6, 9:12] = -r * dt
        f[6:9, 6:9] = np.eye(3) - skew(w) * dt
        f[6:9, 12:15] = -np.eye(3) * dt
        # process noise (diagonal)
        q = np.zeros(15)
        q[3:6] = (self.sig_a * dt) ** 2
        q[6:9] = (self.sig_g * dt) ** 2
        q[9:12] = self.sig_ba**2 * dt
        q[12:15] = self.sig_bg**2 * dt
        self.P = f @ self.P @ f.T + np.diag(q)

    # ------------------------------------------------------------------
    def update_gravity(self, accel: np.ndarray) -> None:
        """Constrain roll/pitch from the gravity direction when |accel|~g. Without this
        the attitude is only weakly observable from the gate and diverges under motion."""
        amag = float(np.linalg.norm(accel))
        if abs(amag - G) > 1.5:  # accel corrupted by thrust/motion → skip
            return
        z = np.asarray(accel, float) / amag  # measured up-dir in body (~ R^T (0,0,-1))
        r = self.R
        u = np.array([0.0, 0.0, -1.0])
        h = r.T @ u
        y = z - h
        h_jac = np.zeros((3, 15))
        # d(R^T u)/d(dtheta) = +[R^T u]x  (note: gate's h=-R^T p gives -[R^T p]x)
        h_jac[:, 6:9] = skew(r.T @ u)
        rn = np.eye(3) * (0.3**2)
        s = h_jac @ self.P @ h_jac.T + rn
        k = self.P @ h_jac.T @ np.linalg.inv(s)
        self._inject(k @ y)
        self.P = (np.eye(15) - k @ h_jac) @ self.P

    def update_gate(self, gate_body: np.ndarray) -> float:
        """Fuse a gate-in-body position measurement. Returns innovation norm (m)."""
        r = self.R
        z = np.asarray(gate_body, float)
        h = -r.T @ self.p
        y = z - h
        h_jac = np.zeros((3, 15))
        h_jac[:, 0:3] = -r.T
        h_jac[:, 6:9] = -skew(r.T @ self.p)
        s = h_jac @ self.P @ h_jac.T + self.Rm
        # Chi-square innovation gate: reject outliers (wrong-gate detections / jumps)
        # instead of fusing them, which would corrupt the filter.
        nis = float(y @ np.linalg.solve(s, y))
        if nis > 16.0:  # ~chi2(3) 99.9%
            return float(np.linalg.norm(y))
        k = self.P @ h_jac.T @ np.linalg.inv(s)
        self._inject(k @ y)
        self.P = (np.eye(15) - k @ h_jac) @ self.P
        return float(np.linalg.norm(y))

    def _inject(self, dx: np.ndarray) -> None:
        self.p = self.p + dx[0:3]
        self.v = self.v + dx[3:6]
        self.q = quat_mul(self.q, quat_from_rotvec(dx[6:9]))
        self.q /= np.linalg.norm(self.q)
        self.ba = self.ba + dx[9:12]
        self.bg = self.bg + dx[12:15]

    def reanchor(self, gate_body: np.ndarray) -> None:
        """Move the world origin to a newly-targeted gate: p = -R @ z. Keep v/q/bias."""
        z = np.asarray(gate_body, float)
        self.p = -self.R @ z
        self.P[0:3, 0:3] = np.eye(3) * 4.0  # position uncertain right after re-anchor
