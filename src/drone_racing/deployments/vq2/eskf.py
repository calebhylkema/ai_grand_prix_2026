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

from dataclasses import dataclass

import numpy as np

from drone_racing.deployments.vq2.attitude import G, quat_mul, quat_to_euler, quat_to_matrix

_SPIKE_G = 25.0  # |‖a‖-g| beyond this (m/s^2) = impact/saturation; discard specific force
_SPIKE_VEL_STD = 3.0  # m/s — one-time velocity uncertainty added per collision (an impact
#                       can change velocity abruptly); bounded so it recovers via updates
NIS_GATE = 16.0  # chi²(3) ~99.9% innovation gate for gate measurements


@dataclass
class GateUpdateInfo:
    """Structured diagnostics from a gate measurement update."""

    accepted: bool
    nis: float
    residual: float  # innovation norm (m)
    threshold: float


def _sym(m: np.ndarray) -> np.ndarray:
    """Symmetrize a covariance matrix (guards drift from finite-precision updates)."""
    return 0.5 * (m + m.T)


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
        self._spike_active = False  # debounce: impulse covariance added once per collision

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
        """Propagate the nominal state and covariance one IMU step.

        On a collision/saturation spike the specific-force reading is discarded
        (velocity is held rather than integrated), but the covariance is STILL
        propagated with inflated accel process noise so the filter widens instead of
        freezing on a corrupted sample.
        """
        if not self._init_att:
            self.initialise_attitude(accel)
            return
        if dt <= 0.0:
            return
        a_m = np.asarray(accel, float)
        w = np.asarray(gyro, float) - self.bg
        r = self.R
        spike = abs(float(np.linalg.norm(a_m)) - G) > _SPIKE_G
        a_lin = np.zeros(3) if spike else a_m - self.ba  # specific force (0 = untrusted)

        # nominal propagation
        if spike:
            self.p = self.p + self.v * dt  # hold velocity; the accel is untrustworthy
        else:
            a_world = r @ a_lin + np.array([0.0, 0.0, G])
            self.p = self.p + self.v * dt + 0.5 * a_world * dt * dt
            self.v = self.v + a_world * dt
        self.q = quat_mul(self.q, quat_from_rotvec(w * dt))
        self.q /= np.linalg.norm(self.q)

        # error-state transition (local attitude error)
        f = np.eye(15)
        f[0:3, 3:6] = np.eye(3) * dt
        f[6:9, 6:9] = np.eye(3) - skew(w) * dt
        f[6:9, 12:15] = -np.eye(3) * dt
        if not spike:
            # accel couples into velocity via attitude error and accel-bias error; on a
            # spike we ignore the specific force entirely, so BOTH couplings drop out
            # (consistent with a_lin=0 — no accel information flows to velocity).
            f[3:6, 6:9] = -r @ skew(a_lin) * dt
            f[3:6, 9:12] = -r * dt

        # Discrete process noise. Gyro (attitude) and bias random walks always apply.
        # Accel white noise (density sig_a) drives velocity and, by integration, position
        # — but ONLY when the specific force is trusted. First-order G·Qc·Gᵀ; the previous
        # (sig_a·dt)² was ~dt too small (≈100× at dt=0.01).
        i3 = np.eye(3)
        q = np.zeros((15, 15))
        q[6:9, 6:9] = self.sig_g**2 * dt * i3
        q[9:12, 9:12] = self.sig_ba**2 * dt * i3
        q[12:15, 12:15] = self.sig_bg**2 * dt * i3
        if spike:
            # accel discarded → no accel-driven position/velocity process noise. A
            # collision can change velocity abruptly by an unknown impulse: add a bounded
            # velocity uncertainty ONCE per event (rising edge), not every spike sample,
            # so a sustained saturation does not repeatedly re-inflate the filter.
            if not self._spike_active:
                q[3:6, 3:6] = _SPIKE_VEL_STD**2 * i3
                self._spike_active = True
        else:
            self._spike_active = False
            sa2 = self.sig_a**2
            q[0:3, 0:3] = sa2 * dt**3 / 3.0 * i3
            q[0:3, 3:6] = sa2 * dt**2 / 2.0 * i3
            q[3:6, 0:3] = sa2 * dt**2 / 2.0 * i3
            q[3:6, 3:6] = sa2 * dt * i3
        self.P = _sym(f @ self.P @ f.T + q)

    # ------------------------------------------------------------------
    def _joseph(self, h_jac: np.ndarray, y: np.ndarray, rn: np.ndarray) -> None:
        """Joseph-form covariance update + state injection for innovation ``y``.

        Uses a linear solve (no explicit inverse) and the numerically stable
        ``(I-KH)P(I-KH)ᵀ + KRKᵀ`` form, which stays symmetric PSD.
        """
        pht = self.P @ h_jac.T
        s = h_jac @ pht + rn
        k = np.linalg.solve(s, pht.T).T  # K = P Hᵀ S⁻¹
        ikh = np.eye(15) - k @ h_jac
        self.P = _sym(ikh @ self.P @ ikh.T + k @ rn @ k.T)
        self._inject(k @ y)

    def update_gravity(self, accel: np.ndarray) -> None:
        """Constrain roll/pitch from the gravity direction when |accel|~g. Without this
        the attitude is only weakly observable from the gate and diverges under motion.

        NOTE: gravity is only valid at rest — the caller must gate this to ground.
        """
        amag = float(np.linalg.norm(accel))
        if abs(amag - G) > 1.5:  # accel corrupted by thrust/motion → skip
            return
        z = np.asarray(accel, float) / amag  # measured up-dir in body (~ R^T (0,0,-1))
        r = self.R
        u = np.array([0.0, 0.0, -1.0])
        y = z - r.T @ u
        h_jac = np.zeros((3, 15))
        # d(R^T u)/d(dtheta) = +[R^T u]x  (note: gate's h=-R^T p gives -[R^T p]x)
        h_jac[:, 6:9] = skew(r.T @ u)
        self._joseph(h_jac, y, np.eye(3) * (0.3**2))

    def update_gate(
        self, gate_body: np.ndarray, meas_cov: np.ndarray | None = None
    ) -> GateUpdateInfo:
        """Fuse a gate-in-body position measurement (chi²-gated).

        Parameters
        ----------
        gate_body : np.ndarray
            Gate position in the body frame (from PnP → camera → body).
        meas_cov : np.ndarray | None
            (3,3) measurement covariance. Defaults to the isotropic ``Rm`` when None;
            perception supplies a distance/geometry-aware covariance in shadow fusion.

        Returns
        -------
        GateUpdateInfo
            Whether the measurement was accepted, its NIS, residual, and the threshold.
        """
        r = self.R
        z = np.asarray(gate_body, float)
        h = -r.T @ self.p
        y = z - h
        h_jac = np.zeros((3, 15))
        h_jac[:, 0:3] = -r.T
        h_jac[:, 6:9] = -skew(r.T @ self.p)
        rn = self.Rm if meas_cov is None else np.asarray(meas_cov, float)
        s = h_jac @ self.P @ h_jac.T + rn
        nis = float(y @ np.linalg.solve(s, y))
        resid = float(np.linalg.norm(y))
        # Chi-square innovation gate: reject outliers (wrong-gate detections / jumps)
        # rather than fusing them, which would corrupt the filter.
        if nis > NIS_GATE:
            return GateUpdateInfo(False, nis, resid, NIS_GATE)
        self._joseph(h_jac, y, rn)
        return GateUpdateInfo(True, nis, resid, NIS_GATE)

    def _inject(self, dx: np.ndarray) -> None:
        self.p = self.p + dx[0:3]
        self.v = self.v + dx[3:6]
        dtheta = dx[6:9]
        self.q = quat_mul(self.q, quat_from_rotvec(dtheta))
        self.q /= np.linalg.norm(self.q)
        self.ba = self.ba + dx[9:12]
        self.bg = self.bg + dx[12:15]
        # error-state reset: fold the applied attitude error into the covariance
        # (local/right perturbation), so P stays consistent with the new quaternion.
        g = np.eye(15)
        g[6:9, 6:9] = np.eye(3) - skew(0.5 * dtheta)
        self.P = _sym(g @ self.P @ g.T)

    def reanchor(self, gate_body: np.ndarray, meas_cov: np.ndarray | None = None) -> None:
        """Move the world origin to a newly-targeted gate: p = -R @ z. Keep v/q/bias.

        The first anchor bypasses the NIS gate (there is no prior position to test
        against), so the CALLER must only reanchor from a confirmed, metric-quality
        target (tracker-confirmed FULL/POSITION_ONLY) — never a raw nearest blob.

        Position is redefined relative to a fresh anchor. Its uncertainty comes from the
        measurement covariance (body frame, rotated to world) plus the attitude-induced
        term  Jθ·Pθ·Jθᵀ  with  Jθ = R·[z]ₓ = ∂p/∂δθ  (a tilt error rotates the lever arm
        ``z``). Position is then DECORRELATED from the rest of the state — including
        attitude — as a deliberate conservative approximation. Retaining the full
        position↔attitude cross-covariance ``Ppθ = Jθ·Pθ`` is more correct in principle
        but numerically UNSTABLE at the large gate lever arm (~8 m, so ‖Jθ‖≈8): a Monte
        Carlo over the synthetic trajectory diverged on some seeds (RMS → hundreds of m)
        with the cross-term but stayed at ~0.26 m without it (see
        ``test_reanchor_decorrelation_is_stable``). The variance term is kept; the
        correlation is dropped.

        Parameters
        ----------
        gate_body : np.ndarray
            Gate position in the body frame.
        meas_cov : np.ndarray | None
            (3,3) body-frame measurement covariance; defaults to the isotropic ``Rm``.
        """
        z = np.asarray(gate_body, float)
        r = self.R
        self.p = -r @ z
        mcov = self.Rm if meas_cov is None else np.asarray(meas_cov, float)
        j_theta = r @ skew(z)  # d p / d(dtheta)  for  p = -R z, local attitude error
        p_pos = r @ mcov @ r.T + j_theta @ self.P[6:9, 6:9] @ j_theta.T
        self.P[0:3, :] = 0.0
        self.P[:, 0:3] = 0.0
        self.P[0:3, 0:3] = _sym(p_pos)
