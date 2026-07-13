"""Unit tests locking in the VQ2 estimation math (attitude filter + ESKF).

Run: uv run pytest tests/unit/vq2/test_estimation.py -q

These encode the ground-truth-free / synthetic validations we relied on:
- attitude estimator reads the known rest pose with zero gravity error,
- the ESKF recovers a synthetic trajectory (position/velocity/attitude) from
  synthesized IMU + gate measurements — guarding the Jacobians/sign conventions.
"""

from __future__ import annotations

import math

import numpy as np

from drone_racing.deployments.vq2.attitude import AttitudeEstimator, G, quat_mul, quat_to_matrix
from drone_racing.deployments.vq2.eskf import ESKF, quat_from_rotvec


def test_quat_from_rotvec_small_angle() -> None:
    q = quat_from_rotvec(np.array([0.0, 0.0, 0.0]))
    assert np.allclose(q, [1.0, 0.0, 0.0, 0.0])
    # 90 deg about z
    q = quat_from_rotvec(np.array([0.0, 0.0, math.pi / 2]))
    assert abs(q[0] - math.cos(math.pi / 4)) < 1e-6
    assert abs(q[3] - math.sin(math.pi / 4)) < 1e-6


def test_attitude_estimator_rest_pose() -> None:
    """At the VQ2 start pose (accel (-3,0,-9.34)) the filter reads roll~0, pitch~-17.8."""
    est = AttitudeEstimator()
    accel = np.array([-3.0, 0.0, -9.34])
    gyro = np.zeros(3)
    for _ in range(300):
        est.update(gyro, accel, 0.01)
    roll, pitch, _ = est.euler
    assert abs(math.degrees(roll)) < 1.0
    assert abs(math.degrees(pitch) - (-17.8)) < 1.0
    assert est.mean_grav_err_deg < 0.5


def test_in_flight_accel_is_ignored() -> None:
    """Run-E (2026-07-04): in steady cruise the gyro is quiet and |a|=g·sec(18°)≈1.05g
    — inside every gate we had — so the accel correction slowly erased real nose-down
    pitch (+12° bias → vz read 'climb' while the drone skimmed the floor into an
    obstacle). In flight the accel must be ignored entirely: gyro-only + vision roll."""
    est = AttitudeEstimator()
    est.initialise(np.array([0.0, 0.0, -G]))
    est.in_flight = True
    # pitch nose-down 15° via gyro, then cruise steadily for 6 s while the accel
    # reads pure body-z thrust at 1.05g ("level" — the cruise lie).
    for _ in range(50):
        est.update(np.array([0.0, math.radians(-15.0 / 0.5), 0.0]), np.zeros(3), 0.01)
    lying_accel = np.array([0.0, 0.0, -1.05 * G])
    for _ in range(600):
        est.update(np.zeros(3), lying_accel, 0.01)
    _, pitch, _ = est.euler
    assert abs(math.degrees(pitch) - (-15.0)) < 0.5, "cruise accel corrupted the pitch"


def test_correct_roll_fixes_hidden_bank() -> None:
    """THE run-3 crash (2026-07-03): the accel correction erases real bank (a quad's
    accel reads body-z thrust = 'level' at any bank). correct_roll must be able to
    drag the estimate to the vision-measured bank and hold it against that pull."""
    est = AttitudeEstimator()
    est.initialise(np.array([0.0, 0.0, -G]))
    # filter believes level; vision (world-upright gate) says banked 40 deg left.
    # ~1.5 s of sightings at 50 Hz: the slew limit walks the estimate in smoothly.
    for _ in range(75):
        est.correct_roll(math.radians(-40.0), gain=0.5)
        # accel keeps insisting "level" between sightings (thrust along body -z)
        est.update(np.zeros(3), np.array([0.0, 0.0, -G]), 0.02)
    roll, _, _ = est.euler
    assert abs(math.degrees(roll) - (-40.0)) < 5.0


def test_correct_roll_slew_limited() -> None:
    """Run-4 lesson: a single call must NOT slam a big error into the estimate —
    step-driving the attitude P-loop against vision latency limit-cycled at ±38°."""
    est = AttitudeEstimator()
    est.initialise(np.array([0.0, 0.0, -G]))
    est.correct_roll(math.radians(-40.0), gain=0.5, max_step=0.02)
    roll, _, _ = est.euler
    assert abs(roll - (-0.02)) < 1e-6  # exactly one slew step toward the measurement


def test_correct_roll_wraps_angle() -> None:
    """A near-pi disagreement corrects the short way round (no 2*pi spin-up)."""
    est = AttitudeEstimator()
    est.initialise(np.array([0.0, 0.0, -G]))
    est.correct_roll(math.radians(170.0), gain=1.0)
    roll, _, _ = est.euler
    assert 0.0 < roll < math.radians(171.0)  # moved toward +170 the short way


def test_correct_roll_zero_error_is_noop() -> None:
    est = AttitudeEstimator()
    est.initialise(np.array([0.0, 0.0, -G]))
    q0 = est.q.copy()
    est.correct_roll(0.0, gain=0.5)
    assert np.allclose(est.q, q0, atol=1e-9)


def _run_synthetic(seed: int = 0) -> tuple[float, float, float]:
    rng = np.random.default_rng(seed)
    dt = 0.01
    steps = 500
    ba_true = np.array([0.10, -0.05, 0.08])
    bg_true = np.array([0.010, -0.008, 0.005])
    sig_a, sig_g, sig_pnp = 0.05, 0.002, 0.05
    p = np.array([-8.0, 1.0, -0.5])
    v = np.zeros(3)
    q = np.array([1.0, 0.0, 0.0, 0.0])
    kf = ESKF(pnp_pos_sig=0.08)
    perr, verr, terr = [], [], []
    for k in range(steps):
        t = k * dt
        if t < 0.5:
            a_world, omega = np.zeros(3), np.zeros(3)
        else:
            tt = t - 0.5
            a_world = np.array(
                [1.2 * math.sin(0.8 * tt), 0.5 * math.sin(1.3 * tt), 0.4 * math.sin(1.1 * tt)]
            )
            omega = np.array(
                [0.4 * math.sin(1.5 * tt), 0.3 * math.sin(0.9 * tt), 0.25 * math.sin(0.6 * tt)]
            )
        v = v + a_world * dt
        p = p + v * dt + 0.5 * a_world * dt * dt
        q = quat_mul(q, quat_from_rotvec(omega * dt))
        q /= np.linalg.norm(q)
        rmat = quat_to_matrix(q)
        a_m = rmat.T @ (a_world - np.array([0, 0, G])) + ba_true + rng.normal(0, sig_a, 3)
        w_m = omega + bg_true + rng.normal(0, sig_g, 3)
        kf.predict(a_m, w_m, dt)
        # Gravity aiding is only valid near-static (specific force ≈ g); feeding it
        # mid-maneuver injects motion accel into attitude. Real flight gates this to the
        # ground; the synthetic mirrors that discipline.
        if abs(float(np.linalg.norm(a_m)) - G) < 0.1:
            kf.update_gravity(a_m)
        if k % 3 == 0 and t > 0.05:
            z = -rmat.T @ p + rng.normal(0, sig_pnp, 3)
            if k == 6:  # first gate obs: anchor (large initial innovation), then fuse
                kf.reanchor(z)
            else:
                kf.update_gate(z)
        if t > 1.0:
            perr.append(np.linalg.norm(kf.p - p))
            verr.append(np.linalg.norm(kf.v - v))
            dq = quat_mul(np.array([q[0], -q[1], -q[2], -q[3]]), kf.q)
            terr.append(math.degrees(2 * math.asin(min(1.0, np.linalg.norm(dq[1:])))))
    return (
        float(np.sqrt(np.mean(np.square(perr)))),
        float(np.sqrt(np.mean(np.square(verr)))),
        float(np.sqrt(np.mean(np.square(terr)))),
    )


def test_eskf_recovers_synthetic_trajectory() -> None:
    # Tightened after the covariance/Joseph repair (was 0.4/0.4/5.0 with the ~100×
    # under-scaled process noise); the corrected filter tracks measurably better.
    perr, verr, terr = _run_synthetic()
    assert perr < 0.35, f"position error too high: {perr}"
    assert verr < 0.30, f"velocity error too high: {verr}"
    # yaw is only weakly observable (gate updates get chi2-gated during the maneuver);
    # roll/pitch stay good via gravity. 5deg is fine — we steer by gate bearing anyway.
    assert terr < 5.0, f"attitude error too high: {terr}"


def test_eskf_rest_no_drift() -> None:
    """On the ground (rest) the ESKF velocity must not drift."""
    kf = ESKF()
    accel = np.array([-3.0, 0.0, -9.34])
    gyro = np.zeros(3)
    for _ in range(500):
        kf.predict(accel, gyro, 0.01)
        kf.update_gravity(accel)
    assert float(np.linalg.norm(kf.v)) < 0.05, f"rest velocity drifted: {kf.v}"
