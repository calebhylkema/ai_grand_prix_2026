"""Deterministic tests for the isolated short-horizon bank observer (Phase C).

The observer holds its own scalar-first quaternion, seeds from the control attitude, propagates
the raw body gyro (validated right-bank-positive sign), never uses accelerometer levelling,
optionally accepts a slew-limited high-quality PnP roll correction, and resets between attempts.
"""

from __future__ import annotations

import math

from drone_racing.deployments.vq2.visual_handoff import (
    BankObserverConfig,
    _BankObserver,
    _quat_from_euler_zyx,
    _quat_roll,
)

MS = 1_000_000
NOW = 5_000_000_000


def _run(obs: _BankObserver, gyro, *, seconds: float, dt_ms: int = 8, start: int = NOW) -> int:
    t = start
    steps = int(seconds * 1000 / dt_ms)
    for _ in range(steps):
        t += dt_ms * MS
        obs.propagate(gyro, t)
    return t


# ------------------------------------------------------------------ quaternion integration
def test_positive_roll_rate_integrates_to_positive_bank() -> None:
    # +1.0 rad/s body roll (right-bank-positive) for 0.5 s → ~0.5 rad right bank.
    obs = _BankObserver()
    obs.seed(0.0, 0.0, 0.0, NOW)
    _run(obs, (1.0, 0.0, 0.0), seconds=0.5)
    assert math.isclose(obs.bank_rad(), 0.5, abs_tol=0.02)


def test_negative_roll_rate_integrates_to_negative_bank() -> None:
    obs = _BankObserver()
    obs.seed(0.0, 0.0, 0.0, NOW)
    _run(obs, (-0.8, 0.0, 0.0), seconds=0.5)
    assert math.isclose(obs.bank_rad(), -0.4, abs_tol=0.02)


def test_seed_bank_is_the_starting_estimate() -> None:
    obs = _BankObserver()
    obs.seed(math.radians(20.0), math.radians(-10.0), math.radians(30.0), NOW)
    assert math.isclose(obs.bank_rad(), math.radians(20.0), abs_tol=1e-6)


def test_pure_yaw_rate_level_does_not_change_bank() -> None:
    # At ZERO pitch the ZYX roll is yaw-independent: a pure body yaw rate leaves the bank at 0,
    # even seeded at a large yaw. (At nonzero pitch a body-yaw rate sweeps the pitched body-z /
    # lift vector sideways — a REAL bank change — so this invariance only holds level.)
    obs = _BankObserver()
    obs.seed(0.0, 0.0, math.radians(80.0), NOW)
    _run(obs, (0.0, 0.0, 1.5), seconds=0.6)  # spin in yaw only
    assert math.isclose(obs.bank_rad(), 0.0, abs_tol=math.radians(1.0))


def test_roll_rate_at_large_yaw_still_banks() -> None:
    obs = _BankObserver()
    obs.seed(0.0, math.radians(-18.0), math.radians(80.0), NOW)
    _run(obs, (1.0, 0.0, 0.0), seconds=0.4)
    assert obs.bank_rad() > math.radians(15.0)  # a real right bank is tracked


# ------------------------------------------------------------------ dt / gap / dup handling
def test_gap_beyond_max_invalidates() -> None:
    obs = _BankObserver(BankObserverConfig(max_gyro_gap_s=0.08))
    obs.seed(0.0, 0.0, 0.0, NOW)
    obs.propagate((1.0, 0.0, 0.0), NOW + 200 * MS)  # 200 ms > 80 ms gap
    assert not obs.established
    assert obs.source.startswith("stale")


def test_nonfinite_gyro_invalidates() -> None:
    obs = _BankObserver()
    obs.seed(0.0, 0.0, 0.0, NOW)
    obs.propagate((float("nan"), 0.0, 0.0), NOW + 8 * MS)
    assert not obs.established


def test_duplicate_and_backward_timestamps_hold() -> None:
    obs = _BankObserver()
    obs.seed(0.0, 0.0, 0.0, NOW)
    obs.propagate((1.0, 0.0, 0.0), NOW + 8 * MS)
    b = obs.bank_rad()
    obs.propagate((1.0, 0.0, 0.0), NOW + 8 * MS)  # duplicate stamp → dt=0 → hold
    obs.propagate((1.0, 0.0, 0.0), NOW + 4 * MS)  # backward stamp → dt<0 → hold
    assert obs.bank_rad() == b
    assert obs.established


# ------------------------------------------------------------------ PnP correction gates
def test_pnp_correction_accepted_moves_roll_toward_vision() -> None:
    obs = _BankObserver()
    obs.seed(math.radians(20.0), 0.0, 0.0, NOW)
    before = obs.bank_rad()
    obs.correct_pnp(math.radians(30.0), True)  # good-quality vision roll 10° away
    assert obs.pnp_accepted == 1
    assert before < obs.bank_rad() <= math.radians(30.0)  # moved toward, slew-limited
    assert obs.source == "gyro+pnp"


def test_pnp_correction_rejected_when_quality_low() -> None:
    obs = _BankObserver()
    obs.seed(math.radians(20.0), 0.0, 0.0, NOW)
    b = obs.bank_rad()
    obs.correct_pnp(math.radians(30.0), False)  # quality gate not met
    assert obs.pnp_accepted == 0 and obs.bank_rad() == b


def test_pnp_correction_rejected_on_large_innovation() -> None:
    obs = _BankObserver()
    obs.seed(math.radians(0.0), 0.0, 0.0, NOW)
    obs.correct_pnp(math.radians(60.0), True)  # 60° innovation > 25° gate
    assert obs.pnp_rejected == 1 and obs.pnp_accepted == 0


def test_pnp_step_is_slew_limited() -> None:
    obs = _BankObserver(BankObserverConfig(pnp_max_step_rad=math.radians(3.0)))
    obs.seed(0.0, 0.0, 0.0, NOW)
    obs.correct_pnp(math.radians(20.0), True)  # within innovation gate but big
    assert obs.bank_rad() <= math.radians(3.0) + 1e-9  # one call moves ≤ the slew cap


# ------------------------------------------------------------------ reset / isolation
def test_reset_clears_all_state() -> None:
    obs = _BankObserver()
    obs.seed(math.radians(20.0), 0.0, 0.0, NOW)
    _run(obs, (1.0, 0.0, 0.0), seconds=0.2)
    obs.correct_pnp(math.radians(25.0), True)
    obs.reset()
    assert not obs.established
    assert obs.pnp_accepted == 0 and obs.pnp_rejected == 0
    assert math.isnan(obs.bank_rad())
    assert obs.source == "idle"


def test_quat_helpers_roundtrip() -> None:
    q = _quat_from_euler_zyx(math.radians(15.0), math.radians(-20.0), math.radians(40.0))
    assert math.isclose(_quat_roll(q), math.radians(15.0), abs_tol=1e-9)
