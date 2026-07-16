"""Deterministic tests for the pre-pass PnP closure + ESKF-speed validation (Phase 2)."""

from __future__ import annotations

import math

from drone_racing.deployments.vq2.speed import (
    ClosureConfig,
    ClosureEstimate,
    ClosureSample,
    pnp_closure_estimate,
    validate_gate0_speed,
)

MS = 1_000_000
NOW = 5_000_000_000


def _samples(
    ranges: list[float], *, dt_ms: int = 33, bearing_deg: float = 1.0, track: int = 1, race: int = 0
) -> list[ClosureSample]:
    out = []
    t = NOW - (len(ranges) - 1) * dt_ms * MS
    for r in ranges:
        out.append(ClosureSample(t, track, race, r, math.radians(bearing_deg)))
        t += dt_ms * MS
    return out


# ------------------------------------------------------------------ closure estimate
def test_clean_decreasing_range_gives_closure() -> None:
    # 6.0 → 5.0 m over 3 × 33 ms ≈ 0.099 s → ~10 m/s? 1.0 m / 0.099 s ≈ 10.1
    est = pnp_closure_estimate(_samples([6.0, 5.67, 5.33, 5.0]), now_ns=NOW)
    assert est.valid and est.reason == "ok"
    assert 9.0 < est.speed_mps < 11.0
    assert est.n_samples == 4


def test_duplicate_timestamps_deduplicated() -> None:
    s = _samples([6.0, 5.6, 5.2])
    dup = [s[0], s[0], s[1], s[1], s[2]]  # each frame republished
    est = pnp_closure_estimate(dup, now_ns=NOW)
    assert est.valid and est.n_samples == 3  # de-duplicated to 3 unique frames


def test_increasing_range_step_rejected() -> None:
    # a spurious range increase (5.2 → 6.0) must not count as a closing step
    est = pnp_closure_estimate(_samples([5.6, 5.2, 6.0, 5.0]), now_ns=NOW)
    # the one bad step is dropped; the remaining clean steps still yield a positive closure
    assert est.valid
    assert est.speed_mps > 0.0


def test_far_off_axis_samples_excluded() -> None:
    # the downstream +38° gate must never enter the gate-0 closure window
    good = _samples([5.6, 5.2, 4.8], bearing_deg=1.0)
    bad = _samples([22.0, 22.0, 22.0], bearing_deg=38.0)
    est = pnp_closure_estimate(good + bad, now_ns=NOW)
    assert est.valid and est.n_samples == 3  # only the 3 near-centred samples used


def test_mixed_track_rejected() -> None:
    s = [
        ClosureSample(NOW - 66 * MS, 1, 0, 5.6, math.radians(1.0)),
        ClosureSample(NOW - 33 * MS, 1, 0, 5.2, math.radians(1.0)),
        ClosureSample(NOW, 9, 0, 5.0, math.radians(1.0)),  # a different physical track id
    ]
    est = pnp_closure_estimate(s, now_ns=NOW)
    assert not est.valid and est.reason == "mixed_track"


def test_too_few_samples() -> None:
    est = pnp_closure_estimate(_samples([5.6]), now_ns=NOW)
    assert not est.valid and est.reason in ("few_samples", "few_steps")


def test_stale_window_excluded() -> None:
    old = _samples([6.0, 5.6, 5.2])
    old = [
        ClosureSample(s.arrival_ns - 2_000_000_000, s.track_id, 0, s.slant_range_m, s.bearing_rad)
        for s in old
    ]
    est = pnp_closure_estimate(old, now_ns=NOW)
    assert not est.valid and est.reason == "no_samples"


def test_impossible_closure_step_rejected() -> None:
    cfg = ClosureConfig()
    # 5.6 → 0.1 over one 33 ms step ≈ 166 m/s → rejected; only clean steps kept
    est = pnp_closure_estimate(_samples([5.6, 0.1, 0.0]), now_ns=NOW, config=cfg)
    assert not est.valid  # no clean steps survive


# ------------------------------------------------------------------ speed validation
def _closure(v: float = 8.0) -> ClosureEstimate:
    return ClosureEstimate(True, v, 3, 2, 0.2, "ok")


def test_speed_validation_accepts_agreement() -> None:
    val = validate_gate0_speed(
        anchor_gate=0,
        initialised=True,
        eskf_speed_mps=8.2,
        vel_cov_trace=0.5,
        fix_age_s=0.02,
        agreement_age_s=float("inf"),
        identity_jumped=False,
        closure=_closure(8.5),
    )
    assert val.eligible and val.reason == "ok"
    assert val.speed_mps == 8.2 and val.residual_mps < 0.5


def test_speed_validation_rejects_disagreement() -> None:
    val = validate_gate0_speed(
        anchor_gate=0,
        initialised=True,
        eskf_speed_mps=13.0,
        vel_cov_trace=0.5,
        fix_age_s=0.02,
        agreement_age_s=float("inf"),
        identity_jumped=False,
        closure=_closure(8.0),
    )
    assert not val.eligible and val.reason == "disagreement"
    assert math.isnan(val.speed_mps)


def test_speed_validation_rejects_stale_fix() -> None:
    val = validate_gate0_speed(
        anchor_gate=0,
        initialised=True,
        eskf_speed_mps=8.0,
        vel_cov_trace=0.5,
        fix_age_s=0.4,
        agreement_age_s=float("inf"),
        identity_jumped=False,
        closure=_closure(),
    )
    assert not val.eligible and val.reason == "stale"


def test_speed_validation_stale_fix_not_coasted() -> None:
    # The ESKF over-reads once gate 0 clips, so a stale fix is refused even with a recent prior
    # agreement — coasting on the inflated ESKF velocity is deliberately NOT trusted.
    val = validate_gate0_speed(
        anchor_gate=0,
        initialised=True,
        eskf_speed_mps=8.0,
        vel_cov_trace=0.5,
        fix_age_s=0.4,
        agreement_age_s=0.10,
        identity_jumped=False,
        closure=_closure(),
    )
    assert not val.eligible and val.reason == "stale"


def test_speed_validation_rejects_high_cov() -> None:
    val = validate_gate0_speed(
        anchor_gate=0,
        initialised=True,
        eskf_speed_mps=8.0,
        vel_cov_trace=9.0,
        fix_age_s=0.02,
        agreement_age_s=float("inf"),
        identity_jumped=False,
        closure=_closure(),
    )
    assert not val.eligible and val.reason == "high_cov"


def test_speed_validation_rejects_wrong_anchor() -> None:
    val = validate_gate0_speed(
        anchor_gate=1,
        initialised=True,
        eskf_speed_mps=8.0,
        vel_cov_trace=0.5,
        fix_age_s=0.02,
        agreement_age_s=float("inf"),
        identity_jumped=False,
        closure=_closure(),
    )
    assert not val.eligible and val.reason == "wrong_anchor"


def test_speed_validation_rejects_identity_jump() -> None:
    val = validate_gate0_speed(
        anchor_gate=0,
        initialised=True,
        eskf_speed_mps=8.0,
        vel_cov_trace=0.5,
        fix_age_s=0.02,
        agreement_age_s=float("inf"),
        identity_jumped=True,
        closure=_closure(),
    )
    assert not val.eligible and val.reason == "identity_jump"


def test_speed_validation_rejects_nonfinite_and_out_of_range() -> None:
    for bad in (float("nan"), -1.0, 25.0):
        val = validate_gate0_speed(
            anchor_gate=0,
            initialised=True,
            eskf_speed_mps=bad,
            vel_cov_trace=0.5,
            fix_age_s=0.02,
            agreement_age_s=float("inf"),
            identity_jumped=False,
            closure=_closure(),
        )
        assert not val.eligible and val.reason == "speed_out_of_range"


def test_speed_validation_rejects_no_closure() -> None:
    bad = ClosureEstimate(False, float("nan"), 1, 0, float("nan"), "few_samples")
    val = validate_gate0_speed(
        anchor_gate=0,
        initialised=True,
        eskf_speed_mps=8.0,
        vel_cov_trace=0.5,
        fix_age_s=0.02,
        agreement_age_s=float("inf"),
        identity_jumped=False,
        closure=bad,
    )
    assert not val.eligible and val.reason == "no_closure"
