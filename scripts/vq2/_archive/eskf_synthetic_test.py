"""Validate the ESKF math on a synthetic trajectory with KNOWN ground truth.

Forward-simulate a smooth flight (chosen world accel + body rates), synthesize the
exact IMU (specific force + gyro, with bias + noise) and gate-in-body measurements it
would produce, run the ESKF on them, and check it recovers the true state. If any
Jacobian/sign is wrong this blows up — catching it offline before the sim.
"""

from __future__ import annotations

import numpy as np
from eskf import ESKF, quat_from_rotvec
from estimator import G, quat_mul, quat_to_matrix

rng = np.random.default_rng(0)


def true_inputs(t: float) -> tuple[np.ndarray, np.ndarray]:
    """(world accel, body angular rate) at time t. Rest for 0.5s, then maneuver."""
    if t < 0.5:
        return np.zeros(3), np.zeros(3)
    tt = t - 0.5
    a_world = np.array(
        [
            1.2 * np.sin(0.8 * tt),  # forward accel/decel (North)
            0.5 * np.sin(1.3 * tt),  # lateral (East)
            0.4 * np.sin(1.1 * tt),  # vertical (Down)
        ]
    )
    omega = np.array(
        [
            0.4 * np.sin(1.5 * tt),  # roll rate
            0.3 * np.sin(0.9 * tt),  # pitch rate
            0.25 * np.sin(0.6 * tt),  # yaw rate
        ]
    )
    return a_world, omega


def main() -> None:
    dt = 0.01
    steps = 500
    ba_true = np.array([0.10, -0.05, 0.08])
    bg_true = np.array([0.010, -0.008, 0.005])
    sig_a, sig_g, sig_pnp = 0.05, 0.002, 0.05

    # ground truth, started at a known position relative to the gate (origin)
    p = np.array([-8.0, 1.0, -0.5])
    v = np.zeros(3)
    q = np.array([1.0, 0.0, 0.0, 0.0])

    kf = ESKF(pnp_pos_sig=0.08)

    perr, verr, terr = [], [], []
    for k in range(steps):
        t = k * dt
        a_world, omega = true_inputs(t)
        # advance ground truth
        Rm = quat_to_matrix(q)  # noqa: N806
        v = v + a_world * dt
        p = p + v * dt + 0.5 * a_world * dt * dt
        q = quat_mul(q, quat_from_rotvec(omega * dt))
        q /= np.linalg.norm(q)
        Rm = quat_to_matrix(q)  # noqa: N806

        # synthesize IMU: a_m = R^T (a_world - g) + ba + noise ; w_m = omega + bg + noise
        a_m = Rm.T @ (a_world - np.array([0, 0, G])) + ba_true + rng.normal(0, sig_a, 3)
        w_m = omega + bg_true + rng.normal(0, sig_g, 3)

        kf.predict(a_m, w_m, dt)
        kf.update_gravity(a_m)

        # gate-in-body measurement every 3rd step (~33 Hz), gate at world origin
        if k % 3 == 0 and t > 0.05:
            gate_body = -Rm.T @ p + rng.normal(0, sig_pnp, 3)
            if k == 6:  # first obs: anchor (large initial innovation), then fuse
                kf.reanchor(gate_body)
            else:
                kf.update_gate(gate_body)

        if t > 1.0:  # let it converge before scoring
            perr.append(np.linalg.norm(kf.p - p))
            verr.append(np.linalg.norm(kf.v - v))
            # attitude error angle (deg)
            dq = quat_mul(np.array([q[0], -q[1], -q[2], -q[3]]), kf.q)
            terr.append(np.degrees(2 * np.arcsin(min(1.0, np.linalg.norm(dq[1:])))))

    print("--- ESKF synthetic validation (RMS after convergence) ---")
    print(f"  position error: {np.sqrt(np.mean(np.square(perr))):.3f} m")
    print(f"  velocity error: {np.sqrt(np.mean(np.square(verr))):.3f} m/s")
    print(f"  attitude error: {np.sqrt(np.mean(np.square(terr))):.3f} deg")
    print(f"  accel-bias est {np.round(kf.ba, 3)}  (true {ba_true})")
    print(f"  gyro-bias  est {np.round(kf.bg, 4)}  (true {bg_true})")
    ok = (
        np.sqrt(np.mean(np.square(perr))) < 0.3
        and np.sqrt(np.mean(np.square(verr))) < 0.3
        and np.sqrt(np.mean(np.square(terr))) < 5.0
    )
    print("  RESULT:", "PASS OK" if ok else "FAIL X (math/sign error)")


if __name__ == "__main__":
    main()
