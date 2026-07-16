"""Deterministic pre-pass speed estimation + validation for the VQ2 gate-0 approach.

Pure, log-only helpers used by the Training-only gate-0 exit-speed controller. They let a
scalar shadow-ESKF *horizontal* speed be trusted ONLY during the gate-0 anchored/visually-
observed window, and only when an independent PnP range-closure estimate agrees:

- :func:`pnp_closure_estimate` — robust closing speed from a short history of genuine,
  accepted, near-centred gate-0 metric observations (median pairwise ``-Δrange/Δt``).
- :func:`validate_gate0_speed` — gate the scalar ESKF horizontal speed on anchor==0,
  freshness, low velocity covariance, a sane range, and agreement with the PnP closure.

No estimator/filter state crosses threads here — these operate on immutable value inputs,
so the controller that calls them stays a deterministic function of its recorded inputs.
"""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class ClosureSample:
    """One accepted, near-centred gate-0 metric observation for closure estimation."""

    arrival_ns: int
    track_id: int
    race_gate_id: int
    slant_range_m: float
    bearing_rad: float


@dataclass(frozen=True, slots=True)
class ClosureConfig:
    """Bounds for the PnP range-closure estimate."""

    max_bearing_rad: float = math.radians(10.0)
    # Floor of 2 unique frames (one clean closing step); ≥3 are used when available. The
    # short genuine gate-0 metric window (~0.6 s, sometimes only 2 unique frames) means 3 is
    # not always available, and the independent ESKF-agreement check is the real safety net.
    min_samples: int = 2
    max_window_s: float = 0.6  # only use samples within this recent window
    max_closure_mps: float = 25.0  # a step faster than this is impossible → reject the step
    range_increase_tol_m: float = 0.3  # tolerate this much apparent range growth (PnP noise)
    min_dt_s: float = 0.005  # ignore duplicate-timestamp steps

    def __post_init__(self) -> None:
        if self.min_samples < 2:
            raise ValueError("closure needs at least two samples")
        for v in (self.max_bearing_rad, self.max_window_s, self.max_closure_mps, self.min_dt_s):
            if not math.isfinite(v) or v <= 0.0:
                raise ValueError("closure config values must be finite and positive")


@dataclass(frozen=True, slots=True)
class ClosureEstimate:
    """Robust gate-0 closing-speed estimate (log-only)."""

    valid: bool
    speed_mps: float
    n_samples: int
    n_steps: int
    spread_mps: float  # max−min of the accepted pairwise steps (0 for a single step)
    reason: str


@dataclass(frozen=True, slots=True)
class SpeedValidationConfig:
    """Eligibility bounds for trusting the scalar ESKF horizontal speed pre-pass."""

    max_fix_age_s: float = 0.15
    # A recent PnP agreement lets the ESKF speed be trusted (coasting) this long afterwards.
    # The drone accelerates hardest in the final ~0.6 s after gate 0 clips to bearing-only, so
    # this coast window must span that acceleration; 0.5 s is short enough that ESKF velocity
    # drift on the straight approach stays small.
    max_agreement_age_s: float = 0.5
    max_vel_cov_trace: float = 4.0
    max_speed_mps: float = 20.0
    agree_abs_mps: float = 3.0
    agree_frac: float = 0.35  # allow the larger of agree_abs_mps or this fraction of closure

    def __post_init__(self) -> None:
        for v in (
            self.max_fix_age_s,
            self.max_agreement_age_s,
            self.max_vel_cov_trace,
            self.max_speed_mps,
            self.agree_abs_mps,
            self.agree_frac,
        ):
            if not math.isfinite(v) or v <= 0.0:
                raise ValueError("speed-validation config values must be finite and positive")


@dataclass(frozen=True, slots=True)
class SpeedValidation:
    """Whether the scalar ESKF horizontal speed may be used for gate-0 exit control."""

    eligible: bool
    speed_mps: float  # the validated ESKF horizontal speed (NaN when not eligible)
    reason: str
    eskf_speed_mps: float
    closure_mps: float
    residual_mps: float


def pnp_closure_estimate(
    samples: list[ClosureSample], *, now_ns: int, config: ClosureConfig | None = None
) -> ClosureEstimate:
    """Robust gate-0 closing speed = median of accepted pairwise ``-Δrange/Δt`` steps.

    Filters to a single physical track (race gate 0), near-centred (|bearing| ≤ limit),
    finite decreasing range, unique timestamps within the recent window. Individual steps
    with nonpositive/near-zero dt, range increase beyond tolerance, or impossible closure are
    rejected. Returns ``valid=False`` with a reason if fewer than ``min_samples`` remain.
    """
    cfg = ClosureConfig() if config is None else config
    # keep only recent, near-centred, race-0, finite-range samples
    window = [
        s
        for s in samples
        if s.race_gate_id == 0
        and math.isfinite(s.slant_range_m)
        and s.slant_range_m > 0.0
        and math.isfinite(s.bearing_rad)
        and abs(s.bearing_rad) <= cfg.max_bearing_rad
        and 0.0 <= (now_ns - s.arrival_ns) * 1e-9 <= cfg.max_window_s
    ]
    if not window:
        return ClosureEstimate(False, float("nan"), 0, 0, float("nan"), "no_samples")
    # de-duplicate identical timestamps (perception republishes a frame until a new one),
    # keeping one sample per unique arrival, and require a single track id
    unique: dict[int, ClosureSample] = {}
    for s in window:
        unique.setdefault(s.arrival_ns, s)
    ordered = sorted(unique.values(), key=lambda s: s.arrival_ns)
    track_ids = {s.track_id for s in ordered}
    if len(track_ids) != 1:
        return ClosureEstimate(False, float("nan"), len(ordered), 0, float("nan"), "mixed_track")
    if len(ordered) < cfg.min_samples:
        return ClosureEstimate(False, float("nan"), len(ordered), 0, float("nan"), "few_samples")
    steps: list[float] = []
    for a, b in zip(ordered, ordered[1:], strict=False):
        dt = (b.arrival_ns - a.arrival_ns) * 1e-9
        if dt < cfg.min_dt_s:
            continue
        d_range = b.slant_range_m - a.slant_range_m  # negative when closing
        if d_range > cfg.range_increase_tol_m:
            continue  # range grew beyond noise → not a clean closing step
        closure = -d_range / dt
        if not math.isfinite(closure) or abs(closure) > cfg.max_closure_mps:
            continue
        steps.append(closure)
    if len(steps) < cfg.min_samples - 1:
        return ClosureEstimate(
            False, float("nan"), len(ordered), len(steps), float("nan"), "few_steps"
        )
    steps.sort()
    n = len(steps)
    median = steps[n // 2] if n % 2 else 0.5 * (steps[n // 2 - 1] + steps[n // 2])
    spread = steps[-1] - steps[0]
    return ClosureEstimate(True, float(median), len(ordered), n, float(spread), "ok")


def validate_gate0_speed(
    *,
    anchor_gate: int,
    initialised: bool,
    eskf_speed_mps: float,
    vel_cov_trace: float,
    fix_age_s: float,
    agreement_age_s: float,
    identity_jumped: bool,
    closure: ClosureEstimate,
    config: SpeedValidationConfig | None = None,
) -> SpeedValidation:
    """Return whether the scalar ESKF horizontal speed is trustworthy for gate-0 control.

    Fails CLOSED with an explicit reason on wrong anchor, uninitialised filter, stale fix
    (unless a recent PnP agreement is fresh), high velocity covariance, out-of-range speed,
    invalid PnP closure, disagreement beyond the tolerance, or a detected identity jump.
    ``agreement_age_s`` = seconds since the last accepted ESKF/PnP agreement (``inf`` if none).
    """
    cfg = SpeedValidationConfig() if config is None else config
    fail = lambda reason: SpeedValidation(  # noqa: E731 - local shorthand
        False,
        float("nan"),
        reason,
        eskf_speed_mps,
        closure.speed_mps if closure.valid else float("nan"),
        float("nan"),
    )
    if identity_jumped:
        return fail("identity_jump")
    if anchor_gate != 0 or not initialised:
        return fail("wrong_anchor")
    if not (math.isfinite(eskf_speed_mps) and 0.0 <= eskf_speed_mps <= cfg.max_speed_mps):
        return fail("speed_out_of_range")
    if not math.isfinite(vel_cov_trace) or vel_cov_trace > cfg.max_vel_cov_trace:
        return fail("high_cov")
    # Trust the scalar ESKF velocity ONLY with a FRESH gate-0 fix AND an agreeing PnP closure.
    # Measured (run_20260715_194949): during the metric window the ESKF agrees with PnP
    # (~4–6 m/s), but once gate 0 clips to bearing-only the ESKF velocity diverges (velcov
    # grows) and OVER-READS ~2× (position-derived ~4 m/s vs ESKF ~10). A stale-fix "coast" on
    # the ESKF is therefore NOT trustworthy and is refused — the drone is actually slow enough
    # for the corner, and trusting the inflated coast would spuriously refuse a feasible one.
    _ = agreement_age_s  # retained in the signature for diagnostics; coasting is not trusted
    fresh_fix = math.isfinite(fix_age_s) and 0.0 <= fix_age_s <= cfg.max_fix_age_s
    if not fresh_fix:
        return fail("stale")
    if not closure.valid:
        return fail("no_closure")
    residual = abs(eskf_speed_mps - closure.speed_mps)
    tol = max(cfg.agree_abs_mps, cfg.agree_frac * abs(closure.speed_mps))
    if residual > tol:
        return SpeedValidation(
            False, float("nan"), "disagreement", eskf_speed_mps, closure.speed_mps, residual
        )
    return SpeedValidation(
        True, float(eskf_speed_mps), "ok", eskf_speed_mps, closure.speed_mps, residual
    )
