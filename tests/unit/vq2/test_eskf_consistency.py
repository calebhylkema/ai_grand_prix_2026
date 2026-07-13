"""Consistency tests for the repaired VQ2 ESKF (timing, covariance, Joseph update).

Run: uv run pytest tests/unit/vq2/test_eskf_consistency.py -q

These guard the 2026-07-10 filter repair: proper process-noise discretization, the
collision-spike-safe covariance propagation, and the Joseph/reset covariance update.
"""

from __future__ import annotations

import numpy as np

from drone_racing.deployments.vq2.attitude import G
from drone_racing.deployments.vq2.eskf import ESKF


def _ready(sig_a: float = 0.5) -> ESKF:
    """An attitude-initialised filter with negligible initial covariance."""
    kf = ESKF(sig_a=sig_a)
    kf.initialise_attitude(np.array([0.0, 0.0, -G]))
    kf.P = np.eye(15) * 1e-12
    return kf


def test_predict_zero_dt_is_noop() -> None:
    kf = _ready()
    kf.v = np.array([1.0, 2.0, 3.0])
    p0, v0, pcov = kf.p.copy(), kf.v.copy(), kf.P.copy()
    kf.predict(np.array([0.0, 0.0, -G]), np.zeros(3), 0.0)
    assert np.array_equal(kf.p, p0) and np.array_equal(kf.v, v0)
    assert np.array_equal(kf.P, pcov)


def test_predict_is_rate_invariant_for_constant_accel() -> None:
    """One 0.02 s step must equal two 0.01 s steps — the point of per-sample dt."""
    a = np.array([0.4, -0.3, -G + 0.2])
    coarse = _ready()
    coarse.v = np.array([0.5, -0.2, 0.1])
    coarse.predict(a, np.zeros(3), 0.02)
    fine = _ready()
    fine.v = np.array([0.5, -0.2, 0.1])
    fine.predict(a, np.zeros(3), 0.01)
    fine.predict(a, np.zeros(3), 0.01)
    assert np.allclose(coarse.p, fine.p, atol=1e-9)
    assert np.allclose(coarse.v, fine.v, atol=1e-9)


def test_process_covariance_matches_monte_carlo() -> None:
    """Predicted velocity variance ≈ empirical variance of integrated accel noise."""
    sig_a, dt, n, trials = 0.5, 0.01, 50, 6000
    # filter growth (x-axis has no gravity term, so it isolates process noise)
    kf = _ready(sig_a=sig_a)
    for _ in range(n):
        kf.predict(np.zeros(3), np.zeros(3), dt)
    filt_var = kf.P[3, 3]
    # Monte-Carlo: discrete accel noise std = sig_a/sqrt(dt) integrated to velocity
    rng = np.random.default_rng(0)
    vx = np.zeros(trials)
    for _ in range(n):
        vx += rng.normal(0.0, sig_a / np.sqrt(dt), trials) * dt
    mc_var = float(np.var(vx))
    # ≈ analytic n·σ²·dt (a hair more: accel-bias uncertainty also feeds velocity)
    assert abs(filt_var - n * sig_a**2 * dt) / (n * sig_a**2 * dt) < 0.02
    assert abs(filt_var - mc_var) / mc_var < 0.15  # matches Monte-Carlo


def _psd_symmetric(p: np.ndarray) -> bool:
    return bool(np.allclose(p, p.T, atol=1e-9) and np.min(np.linalg.eigvalsh(p)) > -1e-9)


def test_joseph_update_keeps_covariance_psd_symmetric() -> None:
    kf = ESKF()
    kf.initialise_attitude(np.array([-3.0, 0.0, -9.34]))
    kf.reanchor(np.array([0.0, 0.0, 6.0]))
    for _ in range(20):
        kf.predict(np.array([-3.0, 0.0, -9.34]), np.array([0.02, -0.01, 0.03]), 0.01)
        kf.update_gate(np.array([0.05, -0.02, 5.8]))
        assert _psd_symmetric(kf.P)


def test_collision_spike_holds_velocity_and_inflates_covariance() -> None:
    kf = _ready()
    kf.v = np.array([2.0, 0.0, 0.0])
    normal = kf.P[3, 3]
    kf.predict(np.array([0.0, 0.0, -200.0]), np.zeros(3), 0.01)  # impact spike
    assert np.allclose(kf.v, [2.0, 0.0, 0.0]), "velocity held through the spike"
    assert kf.P[3, 3] > normal + 1.0, "velocity uncertainty inflated by the impulse term"
    assert kf.P[6, 6] > 0.0, "attitude covariance still propagates (gyro)"
    assert _psd_symmetric(kf.P)


def test_collision_spike_inflation_recovers() -> None:
    """One collision must not permanently overwhelm the filter — updates bring P back."""
    kf = ESKF()
    kf.initialise_attitude(np.array([-3.0, 0.0, -9.34]))
    kf.reanchor(np.array([0.0, 0.0, 6.0]))
    for _ in range(10):
        kf.predict(np.array([-3.0, 0.0, -9.34]), np.zeros(3), 0.01)
        kf.update_gate(np.array([0.0, 0.0, 6.0]))
    kf.predict(np.array([0.0, 0.0, -200.0]), np.zeros(3), 0.01)  # collision
    spiked = kf.P[3, 3]
    assert spiked > 1.0, "spike should transiently inflate velocity covariance"
    for _ in range(40):  # normal operation resumes
        kf.predict(np.array([-3.0, 0.0, -9.34]), np.zeros(3), 0.01)
        kf.update_gate(np.array([0.0, 0.0, 6.0]))
    assert kf.P[3, 3] < 0.1 * spiked, "velocity covariance recovers after the collision"
    assert _psd_symmetric(kf.P)


def test_wrong_gate_measurement_rejected_by_nis() -> None:
    kf = ESKF()
    kf.initialise_attitude(np.array([-3.0, 0.0, -9.34]))
    kf.reanchor(np.array([0.0, 0.0, 6.0]))
    for _ in range(5):
        kf.predict(np.array([-3.0, 0.0, -9.34]), np.zeros(3), 0.01)
        kf.update_gate(np.array([0.0, 0.0, 6.0]))
    p_before, v_before = kf.p.copy(), kf.v.copy()
    info = kf.update_gate(np.array([50.0, -40.0, 3.0]))  # absurd jump
    assert not info.accepted and info.nis > info.threshold
    assert np.allclose(kf.p, p_before) and np.allclose(kf.v, v_before)


def test_reanchor_uses_measurement_and_attitude_covariance() -> None:
    from drone_racing.deployments.vq2.eskf import skew

    kf = ESKF()
    kf.initialise_attitude(np.array([0.0, 0.0, -G]))  # R = I
    kf.P = np.eye(15) * 1e-9
    p_theta = np.diag([0.01, 0.02, 0.03])
    kf.P[6:9, 6:9] = p_theta
    z = np.array([1.0, 0.5, 6.0])  # large lever arm → attitude term dominates
    mcov = np.diag([0.05, 0.05, 0.20])
    kf.reanchor(z, meas_cov=mcov)
    # position cov = measurement + attitude-induced Jθ·Pθ·Jθᵀ
    j_theta = skew(z)  # R = I
    assert np.allclose(kf.P[0:3, 0:3], mcov + j_theta @ p_theta @ j_theta.T)
    # position deliberately decorrelated from the rest (conservative; see docstring)
    assert np.allclose(kf.P[0:3, 3:], 0.0) and np.allclose(kf.P[3:, 0:3], 0.0)
    assert np.min(np.linalg.eigvalsh(kf.P[0:3, 0:3])) > 0.0


def test_nis_gate_does_not_auto_recover_a_wrong_gate_sequence() -> None:
    """A persistently wrong (but plausible-range) gate measurement must keep failing the
    NIS gate — the filter must not silently relax/adapt to accept a wrong-gate stream."""
    kf = ESKF()
    kf.initialise_attitude(np.array([-3.0, 0.0, -9.34]))
    kf.reanchor(np.array([0.0, 0.0, 6.0]))
    accepted = 0
    for _ in range(40):
        kf.predict(np.array([-3.0, 0.0, -9.34]), np.zeros(3), 0.01)
        info = kf.update_gate(np.array([0.0, 20.0, 6.0]))  # 20 m lateral jump = wrong gate
        accepted += int(info.accepted)
    assert accepted == 0  # never auto-accepts the inconsistent measurement


def test_reanchor_decorrelation_is_stable() -> None:
    """Justifies dropping the position↔attitude cross-covariance: a short filter run
    across several noise seeds stays bounded after a reanchor (retaining the lever-arm
    cross-term diverges on some seeds — see reanchor docstring)."""
    for seed in range(6):
        r = np.random.default_rng(seed)
        kf = ESKF()
        kf.initialise_attitude(np.array([-3.0, 0.0, -9.34]))
        z0 = np.array([2.0, -1.0, 8.0])
        kf.reanchor(z0)
        for _ in range(200):
            kf.predict(np.array([-3.0, 0.0, -9.34]), r.normal(0, 0.01, 3), 0.01)
            kf.update_gate(z0 + r.normal(0, 0.05, 3))
        assert float(np.linalg.norm(kf.p - (-kf.R @ z0))) < 1.0
        assert np.all(np.isfinite(kf.P))


def test_collision_impulse_debounced_to_once_per_event() -> None:
    """A sustained saturation must add the impulse covariance once, not every sample."""
    from drone_racing.deployments.vq2.eskf import _SPIKE_VEL_STD

    kf = _ready()
    kf.predict(np.array([0.0, 0.0, -200.0]), np.zeros(3), 0.01)  # spike sample 1
    after_one = kf.P[3, 3]
    for _ in range(5):  # sustained saturation
        kf.predict(np.array([0.0, 0.0, -200.0]), np.zeros(3), 0.01)
    # 5 more spike samples must not each add another impulse of variance
    assert kf.P[3, 3] < after_one + _SPIKE_VEL_STD**2, "impulse re-added every sample"
