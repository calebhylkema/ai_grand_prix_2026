"""Training-only visual gate-0→1 handoff controller for the VQ2 deployment.

This is a SEPARATE visual-servo subsystem, not another ``BRAKE``/``SCAN`` conditional.
It exists to fix one measured failure: after gate 0 the legacy mission enters a blind
1.6 s ``BRAKE`` flare that levels the bank which should bend the velocity vector right,
then a ``SCAN`` that yaws (which does not steer) — discarding the gate-1 sighting it
already has and colliding. The handoff replaces that blind sequence with an
image-relative right turn onto the already-tracked provisional gate-1 target.

Scope / boundaries (enforced, not just documented):

- **Training only.** Both the Training TOML permission and an explicit CLI flag are
  required; Qualification rejects the flag (checked by the mission). Default-off: with
  the controller absent or ``IDLE`` the mission command is byte-for-byte legacy.
- **Command interface unchanged.** The controller emits desired *actual* FRD body rates
  + collective thrust — the exact tuple ``Flight.command`` already accepts. It never
  touches ``RATE_GAIN``, ``send_att``, the calibrated boundary, the rate/thrust clamps,
  arming, or any abort.
- **ESKF roll ONLY.** Short-horizon bank feedback uses the shadow ESKF roll channel and
  its covariance and nothing else — never ESKF velocity, position, pitch, yaw, or global
  heading (regression-locked in the tests).
- **Perception is authoritative on identity.** The handoff consumes the worker's
  confirmed :class:`StructuredTarget` snapshots (active + provisional next gate); it never
  loosens association / NIS / range / right-turn / slant-band guards and never fuses a
  far PnP pose.

Deterministic: every output is a pure function of the input sequence (all timing comes
from ``VisualHandoffInput.now_ns``; no wall-clock reads). :meth:`VisualHandoffController.reset`
gives complete per-attempt isolation.

Frames: body FRD (x forward, y right, z down). A right turn is a POSITIVE roll target
(right-wing-down) and a positive yaw rate. Nose-up (speed-bleeding) is a positive pitch
rate. These sign conventions match ``runtime`` (``accel_guidance``/``level_rates``).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import StrEnum
from typing import TYPE_CHECKING

from drone_racing.deployments.vq2.speed import (
    ClosureConfig,
    ClosureSample,
    SpeedValidationConfig,
    pnp_closure_estimate,
    validate_gate0_speed,
)

if TYPE_CHECKING:
    from drone_racing.deployments.vq2.gate_relative import StructuredTarget

Command = tuple[float, float, float, float]  # (thrust, roll_rate, pitch_rate, yaw_rate) rps

# The reviewed image-row height target and thrust envelope (shared with the legacy servo
# constants but held locally so this module never imports runtime). Do not widen.
CY_TARGET = 130.0
_DEG = math.radians
_G = 9.80665  # standard gravity, for the coordinated-turn bank geometry

Quat = tuple[float, float, float, float]  # scalar-first (w, x, y, z), body->world


def _quat_norm(q: Quat) -> Quat:
    w, x, y, z = q
    n = math.sqrt(w * w + x * x + y * y + z * z)
    if n < 1e-12:
        return (1.0, 0.0, 0.0, 0.0)
    return (w / n, x / n, y / n, z / n)


def _quat_from_euler_zyx(roll: float, pitch: float, yaw: float) -> Quat:
    """Scalar-first quaternion from ZYX (yaw·pitch·roll) Euler angles."""
    cr, sr = math.cos(roll / 2), math.sin(roll / 2)
    cp, sp = math.cos(pitch / 2), math.sin(pitch / 2)
    cy, sy = math.cos(yaw / 2), math.sin(yaw / 2)
    return _quat_norm(
        (
            cy * cp * cr + sy * sp * sr,
            cy * cp * sr - sy * sp * cr,
            cy * sp * cr + sy * cp * sr,
            sy * cp * cr - cy * sp * sr,
        )
    )


def _quat_roll(q: Quat) -> float:
    """ZYX Euler roll (== gravity-tilt bank) from a scalar-first body->world quaternion."""
    w, x, y, z = q
    return math.atan2(2.0 * (w * x + y * z), 1.0 - 2.0 * (x * x + y * y))


def _quat_integrate_body_rate(q: Quat, gyro: tuple[float, float, float], dt: float) -> Quat:
    """First-order body-rate quaternion propagation: q += 0.5·q⊗[0,ω]·dt (mirrors attitude.py)."""
    w, x, y, z = q
    gx, gy, gz = gyro
    # quat_mul(q, [0, gx, gy, gz]) with scalar-first convention
    dw = -(x * gx + y * gy + z * gz)
    dx = w * gx + y * gz - z * gy
    dy = w * gy - x * gz + z * gx
    dz = w * gz + x * gy - y * gx
    return _quat_norm((w + 0.5 * dw * dt, x + 0.5 * dx * dt, y + 0.5 * dy * dt, z + 0.5 * dz * dt))


@dataclass(frozen=True)
class BankObserverConfig:
    """Short-horizon bank observer + its PnP-roll correction gates (Training-only)."""

    max_gyro_gap_s: float = 0.08  # a body-rate gap beyond this invalidates the observer
    # The seed is the already-validated control attitude, so an age-0 seed is itself a usable
    # bank estimate; propagation only refines it. 0.0 = trust immediately (a gyro gap still
    # invalidates it, and the command-integral envelope still cross-checks it).
    warmup_s: float = 0.0
    pnp_correct_gain: float = 0.4  # slew-limited pull toward a high-quality vision roll
    pnp_max_step_rad: float = _DEG(3.0)  # per-correction cap (bounds a bad PnP roll)
    pnp_max_innovation_rad: float = _DEG(25.0)  # reject a vision roll this far from the observer


class _BankObserver:
    """Isolated, gyro-propagated short-horizon roll/bank estimate for the visual handoff.

    Lives only inside the Training visual-handoff subsystem: it holds its OWN quaternion,
    seeds from the best validated pre-turn attitude, propagates the raw body gyro (validated
    right-bank-positive sign — see the attempt-2 audit), never uses accelerometer levelling,
    optionally accepts a slew-limited high-quality PnP roll correction, resets between
    attempts, and is fully disconnected from the general ESKF and the normal controller.
    """

    def __init__(self, config: BankObserverConfig | None = None) -> None:
        self.config = config or BankObserverConfig()
        self.reset()

    def reset(self) -> None:
        self._q: Quat | None = None
        self._last_ns: int | None = None
        self._seed_ns: int | None = None
        self._source = "idle"  # idle | gyro | gyro+pnp | stale
        self._pnp_accepted = 0
        self._pnp_rejected = 0
        self._last_innovation_rad = float("nan")

    @property
    def established(self) -> bool:
        return self._q is not None

    @property
    def source(self) -> str:
        return self._source

    @property
    def pnp_accepted(self) -> int:
        return self._pnp_accepted

    @property
    def pnp_rejected(self) -> int:
        return self._pnp_rejected

    @property
    def last_innovation_rad(self) -> float:
        return self._last_innovation_rad

    def bank_rad(self) -> float:
        return _quat_roll(self._q) if self._q is not None else float("nan")

    def age_s(self, now_ns: int) -> float:
        if self._seed_ns is None:
            return float("inf")
        return max(0.0, (now_ns - self._seed_ns) * 1e-9)

    def warmed_up(self, now_ns: int) -> bool:
        return self.established and self.age_s(now_ns) >= self.config.warmup_s

    def seed(self, roll: float, pitch: float, yaw: float, now_ns: int) -> None:
        """Seed from a validated pre-turn attitude (idempotent while already established)."""
        if not (math.isfinite(roll) and math.isfinite(pitch) and math.isfinite(yaw)):
            return
        self._q = _quat_from_euler_zyx(roll, pitch, yaw)
        self._last_ns = now_ns
        self._seed_ns = now_ns
        self._source = "gyro"
        self._pnp_accepted = 0
        self._pnp_rejected = 0
        self._last_innovation_rad = float("nan")

    def propagate(self, gyro: tuple[float, float, float], now_ns: int) -> None:
        """Propagate the observer quaternion with the raw body gyro; fail closed on gaps."""
        if self._q is None or self._last_ns is None:
            return
        if not all(math.isfinite(g) for g in gyro):
            self._invalidate("nonfinite")
            return
        dt = (now_ns - self._last_ns) * 1e-9
        if dt <= 0.0:
            return  # duplicate / out-of-order tick: hold
        if dt > self.config.max_gyro_gap_s:
            self._invalidate("gap")
            return
        self._q = _quat_integrate_body_rate(self._q, gyro, dt)
        self._last_ns = now_ns

    def correct_pnp(self, vision_roll: float, quality_ok: bool) -> None:
        """Optionally nudge the observer roll toward a HIGH-QUALITY vision roll (slew-limited).

        ``quality_ok`` is the caller's high-quality gate (a full-pose, metric, confident upright
        gate sighting). The innovation gate + per-call slew bound the effect of a bad roll.
        """
        cfg = self.config
        if self._q is None or not quality_ok or not math.isfinite(vision_roll):
            return
        cur = _quat_roll(self._q)
        innovation = math.atan2(math.sin(vision_roll - cur), math.cos(vision_roll - cur))
        self._last_innovation_rad = innovation
        if abs(innovation) > cfg.pnp_max_innovation_rad:
            self._pnp_rejected += 1
            return
        step = max(
            -cfg.pnp_max_step_rad, min(cfg.pnp_max_step_rad, cfg.pnp_correct_gain * innovation)
        )
        # body-x rotation by `step` (right-multiply), matching attitude.correct_roll
        w, x, y, z = self._q
        c, s = math.cos(step / 2), math.sin(step / 2)
        self._q = _quat_norm((w * c - x * s, w * s + x * c, y * c + z * s, z * c - y * s))
        self._pnp_accepted += 1
        self._source = "gyro+pnp"

    def _invalidate(self, why: str) -> None:
        self._q = None
        self._last_ns = None
        self._source = f"stale:{why}"


class VisualHandoffState(StrEnum):
    """The handoff state machine. Only IDLE hands the command straight to legacy."""

    IDLE = "idle"  # legacy owns commands; observe the provisional gate-1 target
    PRETURN = "preturn"  # late gate-0 traversal: overlay right roll/yaw only
    TURN = "turn"  # race counter advanced 0→1: visual right-turn replaces BRAKE/SCAN
    CAPTURE = "capture"  # bearing converging / in-band metric: taper bank, keep steering
    COMMIT = "commit"  # gate close+centred: taper roll/yaw, carry momentum through
    RECOVERY = "recovery"  # (legacy) short structured dropout: bounded extrapolation + decay
    REACQUIRE_ARC = "reacquire_arc"  # bounded (≤1.2 s) right-arc continuation to re-find gate 1
    SAFE_EXIT = "safe_exit"  # persistent loss / wrong id / unsafe: bounded level, then terminate


@dataclass(frozen=True, slots=True)
class VisualHandoffConfig:
    """Bounded, reviewed parameters for the conservative first campaign controller."""

    # -- PRETURN entry gates --------------------------------------------------------
    preturn_bearing_min_rad: float = _DEG(8.0)
    preturn_bearing_max_rad: float = _DEG(50.0)
    provisional_max_age_s: float = 0.080
    provisional_min_hits: int = 5
    gate0_close_area_px2: float = 30000.0
    gate0_close_visible_fraction: float = 0.60
    gate0_close_slant_m: float = 6.5
    gate0_bearing_limit_rad: float = _DEG(15.0)
    # Gate 0 de-selects from the structured pipeline right as the provisional gate-1 track
    # confirms (they barely miss overlapping in the live capture). Latch a recent close +
    # aligned gate-0 sighting so PRETURN can still trigger on the confirmed provisional as
    # gate 0 passes, and tolerate a brief provisional gap through the 0→1 promotion.
    gate0_proximity_memory_s: float = 0.25
    preturn_provisional_grace_s: float = 0.12
    # -- LOGICAL provisional gate-1 identity (above raw tracker ids). The raw provisional churns
    # (e.g. track 0 → 0.23 s gap → track 2, the SAME physical gate; track 4 is the downstream
    # imposter). Continuity is judged on the ESKF-independent world azimuth = control_yaw +
    # bearing (a fixed gate holds it ~constant as the drone yaws). A guarded raw-id handover
    # keeps the logical identity through the gap while rejecting the imposter's large azimuth
    # jump. PRETURN stays engaged through a logical gap up to logical_max_gap_s.
    logical_max_gap_s: float = 0.35
    logical_azimuth_residual_rad: float = _DEG(15.0)
    logical_min_hits: int = 3
    logical_azimuth_rate_max_rps: float = _DEG(300.0)  # implausibly fast world-azimuth slew
    logical_fov_limit_rad: float = _DEG(50.0)
    # -- roll feedback validity + observer arbitration ------------------------------
    roll_state_max_age_s: float = 0.080
    roll_cov_trace_max_rad2: float = 0.08
    roll_abs_limit_rad: float = _DEG(45.0)
    roll_feedback_gain: float = 3.0
    roll_rate_clip_rps: float = 1.0
    # Attempt-2 audit: the shadow ESKF roll is sign-DEFECTIVE through the turn (raw gyro +
    # command-integral + no-flip quaternion integration all agree with the control attitude that
    # the drone banked RIGHT to a real ~48° peak, while the shadow ESKF read the wrong sign then
    # reversed to +45° and spuriously tripped roll_invalid). So roll validity + feedback now
    # arbitrate a dedicated gyro observer (primary) against the command-integral bank (envelope),
    # with the control attitude + shadow ESKF as cross-checks only.
    bank_observer: BankObserverConfig = BankObserverConfig()
    roll_disagreement_rad: float = _DEG(15.0)  # observer-vs-command-integral tolerance
    roll_disagreement_persist: int = 4  # consecutive ticks over tolerance → fail closed
    shadow_degraded_rad: float = _DEG(20.0)  # |corrected-shadow − observer| beyond this = degraded
    # The shadow ESKF roll is sign-INVERTED vs the command/control roll convention (as the
    # drone banks right, control-att roll rises + while ESKF roll goes −) AND under-reads the
    # fast turn (reads ~−10° at a true ~60° bank). So: (1) correct the sign for command-frame
    # feedback, and (2) hard-cap the bank with a FEEDFORWARD integral of the commanded roll
    # rate (the sim tracks commanded actual rate ~1.5×), so a mis-tracking ESKF roll can never
    # run the bank away to the 60° tilt abort. Both are plumbing/model fixes, not threshold
    # relaxations. Measured from run_20260713_203831.
    eskf_roll_feedback_sign: float = -1.0
    # Commanded→actual roll-rate ratio. MEASURED at ~0.89–0.95 (median 0.92–0.95, IQR
    # 0.59–0.99) across the sustained-roll segments of run_20260713_203831 and _204817 —
    # i.e. the sim tracks the calibrated desired-actual roll rate ~1:1, NOT 1.5×. 1.0 is a
    # mild safety margin (feedforward stops the bank at ≈0.9×target, ≤ target), with the 35°
    # hard cap + 45° roll-validity + 60° tilt abort as backstops.
    roll_command_actual_gain: float = 1.0
    # Feedforward integral hard cap. DERATED 38°→35° after attempt 2 (run_20260716_001807):
    # the audit (raw gyro + command-integral + no-flip quaternion integration all agree with the
    # control attitude) showed the drone REALLY over-banked to ~48° — a genuine roll-rate-inertia
    # overshoot of the 38° geometry target, past the 45° validity. Lowering the command target
    # (geometry cap 38→34° below) + this feedforward cap keeps the true peak under 45° with
    # margin. Roll-validity 45° and tilt-abort 60° UNCHANGED.
    bank_feedforward_cap_rad: float = _DEG(35.0)  # absolute clamp on the feedforward integral
    # -- GEOMETRY-DERIVED CORNER BANK (Phase 3) -------------------------------------
    # The gate-0→1 corner is a coordinated turn of radius R = (x²+y²)/(2y) from the mined map
    # offsets (gate 1 ≈ +9.64 m forward / +6.52 m right of gate 0 → R ≈ 10.4 m). The steady
    # bank to hold that radius at ground speed v is atan(v²/(g·R)). We freeze a CONSERVATIVE
    # upper-CI pre-pass speed (Phase 0 reconciled the physical pass speed to ~9.0 ± 1.5 m/s;
    # `ctrl_vh_spd_exit`≈6.9 is a STALE PnP closure, NOT the pass speed) and clamp the bank to
    # the reviewed 38° max. Used as the TURN/REACQUIRE bank ceiling; CAPTURE tapers off it;
    # PRETURN is unaffected (stays ≤20°).
    corner_offset_fwd_m: float = 9.64
    corner_offset_right_m: float = 6.52
    corner_bank_speed_mps: float = 9.5  # frozen conservative upper-CI pre-pass ground speed
    # The geometry-derived request (recorded as bank_geometry_raw) is clamped to this AUTHORITY
    # cap. DERATED 38°→34° after attempt 2's real ~48° over-bank — a safety derating (leaving
    # ~11° of dynamic-overshoot + validity margin below 45°), NOT a claim that 34° is optimal.
    corner_bank_max_rad: float = _DEG(34.0)  # authority cap on the commanded geometry bank
    # -- PRETURN command ------------------------------------------------------------
    preturn_bank_gain: float = 0.6  # bank target = gain * provisional bearing, then clip
    preturn_bank_min_rad: float = _DEG(10.0)
    preturn_bank_max_rad: float = _DEG(20.0)
    preturn_roll_rate_cap_rps: float = 0.8
    preturn_yaw_cap_rps: float = 0.35
    # -- TURN bank schedule (right only) --------------------------------------------
    turn_bank_hi_bearing_rad: float = _DEG(15.0)
    turn_bank_hi_rad: float = _DEG(30.0)  # attempt 6: under-turn fix, +2° (capped at 30°)
    turn_bank_lo_bearing_rad: float = _DEG(5.0)
    turn_bank_lo_rad: float = _DEG(12.0)
    turn_bank_min_rad: float = _DEG(8.0)
    turn_bank_cap_rad: float = _DEG(30.0)
    turn_yaw_gain: float = 1.6
    turn_yaw_cap_rps: float = 0.8
    turn_pitch_up_rps: float = 0.35
    turn_pitch_up_s: float = 0.25
    # CORNER BRAKE (scoped extension). The measured blocker is that the drone flies PAST
    # gate 1: after the pass its velocity carries it forward and the gate's world azimuth
    # sweeps behind (perception-confirmed), never entering the 3–16 m band. Bleeding forward
    # speed with a bounded nose-up brake while off-axis was TESTED in attempts 7 (0.30/1.0 s)
    # and 8 (0.45/1.2 s): it reduced the nose-down attitude (−27°→+1°) but did NOT decelerate
    # the drone (ESKF horizontal speed stayed ~12→16 m/s) and did NOT stop the fly-past. Kept
    # at a conservative bound (open-loop; no ESKF pitch/velocity used; 60° tilt abort
    # backstops it). The residual blocker is velocity-vector redirection, which needs the
    # off-limits ESKF velocity guidance — see docs/project/status.md.
    corner_brake_pitch_rps: float = 0.30
    corner_brake_max_s: float = 1.0
    corner_brake_bearing_rad: float = _DEG(12.0)  # only brake while off-axis by more than this
    # -- gate-0 EXIT-SPEED control (Phase 3). The corner (R≈10.4 m) needs ~7.7 m/s at 30°
    # bank, but the drone accelerates from ~8 to ~11–12 m/s in the final 0.6–0.7 s. Bleed
    # that with a bounded nose-up brake owning PITCH ONLY during the anchored gate-0 window,
    # using a PnP-validated scalar ESKF speed. PRETURN keeps roll/yaw; thrust stays legacy.
    # Target/engage reverted to the PRETURN-preserving values (attempt 3's lower target braked
    # so hard it pitched the camera up and broke the gate-1 provisional confirmation → PRETURN
    # never engaged and legacy tumbled). Attempt 4 instead RAISES the impulse cap so the brake
    # can overcome the ~−18° nose-down TRIM and reach an absolute nose-up (real deceleration)
    # — a ≤15° impulse only reached ~−6° absolute (still nose-down → still accelerating).
    entry_speed_target_mps: float = 7.0
    entry_speed_deadband_mps: float = 0.5
    entry_speed_engage_mps: float = 7.5  # engage the brake only above this
    entry_speed_kp: float = 0.30
    entry_speed_pitch_cap_rps: float = 0.60  # never exceed +0.60 rad/s nose-up
    # Attempt 4: 15°→22° so the cumulative nose-up can overcome the ~−18° nose-down trim and
    # reach a slightly nose-up ABSOLUTE attitude (the point where the drone actually
    # decelerates). Still bounded; the 60° tilt abort remains the hard safety backstop.
    entry_speed_impulse_cap_rad: float = _DEG(22.0)
    entry_speed_bearing_limit_rad: float = _DEG(10.0)
    entry_speed_cy_band_px: float = 120.0  # gate-0 vertical error safety band
    # The measured real corner-entry speed is ~8.5 m/s (PnP closure + ESKF position agree; the
    # ESKF velocity over-reads to ~11). That is above the 7.7 m/s a 30° bank needs for the
    # R≈10.4 m corner (and a 30° bank + 1.2 s arc only turns ~52° of the ~90° corner at 8.5
    # m/s), so it is REFUSED: attempting it tumbles into the corner clutter, whereas refusing
    # is a clean bounded SAFE_EXIT with no collision. Threading needs ≤~7 m/s, which the
    # launch-limited bounded brake cannot reliably reach.
    entry_speed_refuse_corner_mps: float = 8.5
    entry_speed_material_impulse_rad: float = _DEG(3.0)  # skip TURN corner-brake beyond this
    closure: ClosureConfig = ClosureConfig()
    speed_validation: SpeedValidationConfig = SpeedValidationConfig()
    # -- CAPTURE --------------------------------------------------------------------
    capture_bearing_rad: float = _DEG(10.0)
    capture_frames: int = 3
    capture_metric_bearing_rad: float = _DEG(15.0)
    capture_bank_cap_rad: float = _DEG(22.0)
    capture_bank_gain: float = 1.4  # bank = gain * bearing, clipped to capture_bank_cap
    capture_yaw_cap_rps: float = 0.6
    # -- COMMIT ---------------------------------------------------------------------
    commit_bearing_rad: float = _DEG(8.0)
    commit_cy_err_px: float = 50.0
    commit_area_px2: float = 15000.0
    commit_slant_m: float = 3.5
    commit_taper_s: float = 0.20
    commit_max_s: float = 1.0
    # -- reviewed slant band (never weakened) ---------------------------------------
    slant_band_m: tuple[float, float] = (3.0, 16.0)
    # -- vertical (image-row) thrust ------------------------------------------------
    thrust_min: float = 0.29
    # Local nose-level thrust ceiling for the TURN/REACQUIRE phases. Raised 0.36→0.38 so the
    # steeper geometry bank (up to 38°) keeps enough vertical component through the corner
    # (cos 38° ≈ 0.79). This is a LOCAL cap; the mission's global calibrated thrust clamp is
    # UNCHANGED and still bounds the final command downstream.
    thrust_max: float = 0.38
    thrust_hover: float = 0.30
    thrust_kp_px: float = 0.20
    thrust_kd_px: float = 0.05
    thrust_descend_area_px2: float = 2500.0
    thrust_descend_max_bearing_rad: float = _DEG(20.0)
    thrust_tilt_cos_floor: float = 0.85
    # -- dropout / recovery ---------------------------------------------------------
    dropout_hold_s: float = 0.150
    dropout_recovery_s: float = 0.350
    dropout_decay_tau_s: float = 0.20
    dropout_bearing_rate_cap_rps: float = 2.0  # bounded extrapolation slope
    recovery_bank_cap_rad: float = _DEG(15.0)
    recovery_yaw_cap_rps: float = 0.35
    recovery_thrust: float = 0.33
    # -- REACQUIRE_ARC: bounded right-arc continuation when the confirmed gate-1 target is
    # temporarily lost. NOT the old unbounded SCAN — a hard 1.2 s cap, then SAFE_EXIT. At
    # +0.8 rad/s a 35° camera rotation is ~0.76 s, so 1.2 s covers the last +29–35° LOS plus
    # dynamics/handover margin. Uses ONLY sign-corrected roll + the feedforward hard cap.
    reacquire_max_s: float = 1.2
    reacquire_bank_rad: float = _DEG(25.0)  # attempt 6: under-turn fix, +5° (was 20°, ≤30°)
    reacquire_yaw_rps: float = 0.8
    reacquire_thrust: float = 0.33  # within the 0.32–0.34 band (also clamped downstream)
    reacquire_min_hits: int = 3  # guarded NEW-id reacquisition confirmation
    reacquire_capture_bearing_rad: float = _DEG(15.0)  # ≤ this on reacquire → CAPTURE not TURN
    reacquire_continuity_margin_rad: float = _DEG(20.0)  # bearing-continuity imposter guard
    reacquire_fov_limit_rad: float = _DEG(50.0)
    reacquire_target_max_age_s: float = 0.15  # a candidate older than this is stale
    # -- SAFE_EXIT / limits ---------------------------------------------------------
    max_time_to_capture_s: float = 2.0
    bank_abort_rad: float = _DEG(45.0)
    wrong_side_bearing_margin_rad: float = _DEG(12.0)
    safe_exit_hold_s: float = 0.40
    # -- authority handover (slew) --------------------------------------------------
    handover_s: float = 0.12
    slew_spans: Command = (0.20, 2.0, 4.0, 1.6)  # per-channel full slew span (thr, r, p, y)

    def __post_init__(self) -> None:
        positives = (
            self.provisional_max_age_s,
            self.roll_state_max_age_s,
            self.roll_cov_trace_max_rad2,
            self.roll_abs_limit_rad,
            self.roll_feedback_gain,
            self.roll_rate_clip_rps,
            self.turn_yaw_cap_rps,
            self.commit_max_s,
            self.dropout_hold_s,
            self.dropout_recovery_s,
            self.dropout_decay_tau_s,
            self.max_time_to_capture_s,
            self.handover_s,
        )
        if any(v <= 0.0 or not math.isfinite(v) for v in positives):
            raise ValueError("visual-handoff config values must be finite and positive")
        if self.preturn_bearing_min_rad >= self.preturn_bearing_max_rad:
            raise ValueError("preturn bearing band must satisfy min < max")
        if self.provisional_min_hits < 1 or self.capture_frames < 1:
            raise ValueError("hit/frame counts must be >= 1")
        lo, hi = self.slant_band_m
        if not 0.0 < lo < hi:
            raise ValueError("slant_band_m must satisfy 0 < lo < hi")
        if not 0.0 < self.thrust_min < self.thrust_max <= 1.0:
            raise ValueError("thrust envelope must satisfy 0 < min < max <= 1")
        if self.dropout_recovery_s <= self.dropout_hold_s:
            raise ValueError("dropout_recovery_s must exceed dropout_hold_s")


@dataclass(frozen=True, slots=True)
class VisualHandoffInput:
    """One immutable control-loop input for the handoff (built by the mission).

    ``active_target`` is the worker's SELECTED structured target (race gate == active);
    ``provisional_target`` is the worker's provisional NEXT-gate target (race gate ==
    active + 1). ESKF roll/covariance/age come from the atomic :class:`ShadowState`
    snapshot — no other estimator channel is provided (roll-only by construction).
    """

    now_ns: int
    training: bool
    flag_enabled: bool
    active_gate: int
    anchor_gate: int
    active_target: StructuredTarget
    provisional_target: StructuredTarget
    eskf_roll_rad: float
    eskf_roll_cov_rad2: float
    eskf_att_cov_trace_rad2: float
    state_age_s: float
    legacy_command: Command
    abort_reason: str = ""
    # -- gate-0 exit-speed control (Phase 3): a scalar, validated shadow-ESKF HORIZONTAL
    # speed usable ONLY in the anchored/visually-observed gate-0 window (validated against a
    # PnP range-closure estimate the controller computes from the active gate-0 metric
    # stream). Defaults keep existing constructors + replay valid; NaN fails closed.
    eskf_speed_mps: float = float("nan")
    eskf_vel_cov_trace: float = float("nan")
    eskf_fix_age_s: float = float("nan")
    eskf_initialised: bool = False
    # -- control-attitude YAW (gyro-integrated, FRD, from the accel-levelled AttitudeEstimator)
    # for the logical provisional-gate continuity cue only. Its roll/pitch are acceleration-
    # contaminated but its yaw is a clean short-horizon gyro integral, and it is NOT the shadow
    # ESKF yaw. Used to form target_world_azimuth = control_yaw + bearing (frame: world NED, sign
    # +right; a fixed gate holds a ~constant world azimuth as the drone yaws). NaN → no cue.
    control_yaw_rad: float = float("nan")
    # -- dedicated bank-observer inputs (Phase C). ``body_gyro_rps`` is the raw body angular
    # rate the control loop already reads (validated right-bank-positive roll sign, NOT the
    # GYRO_SIGN-flipped shadow-ESKF gyro). ``control_roll_rad``/``control_pitch_rad`` seed the
    # observer from the control attitude at PRETURN. All default NaN → observer never establishes
    # (a recording predating the observer replays exactly on the pre-observer path).
    body_gyro_rps: tuple[float, float, float] = (float("nan"),) * 3
    control_roll_rad: float = float("nan")
    control_pitch_rad: float = float("nan")

    @property
    def enabled(self) -> bool:
        return self.training and self.flag_enabled


@dataclass(frozen=True, slots=True)
class VisualHandoffStatus:
    """Structured, auditable per-loop diagnostics (recorded; never fed back to control)."""

    state: VisualHandoffState
    reason: str
    source: str  # "legacy" | "visual"
    active: bool  # visual authority engaged this tick
    locked_track_id: int
    input_track_id: int
    input_sequence: int
    bearing_rad: float
    bearing_rate_rps: float
    predicted_bearing_rad: float
    cy_err_px: float
    cy_rate_pxps: float
    dropout_age_s: float
    roll_target_rad: float
    roll_estimate_rad: float
    att_cov_trace_rad2: float
    roll_valid: bool
    time_in_state_s: float
    transition_age_s: float
    safe_exit_reason: str
    authoritative_channels: tuple[bool, bool, bool, bool]
    entered: bool
    exited: bool
    slew_limited: bool
    succeeded: bool
    # -- REACQUIRE_ARC + termination diagnostics (schema-v9) -----------------------------
    terminate: bool = False  # attempt should end now (SAFE_EXIT hold elapsed) → mission stops
    reacquire_elapsed_s: float = 0.0
    reacquire_yaw_since_seen_rad: float = 0.0
    reacquire_prev_track_id: int = -1
    reacquire_candidate_track_id: int = -1
    reacquire_candidate_hits: int = 0
    reacquire_continuity_residual_rad: float = float("nan")
    reacquire_result: str = ""  # "" | original | new | rejected:<reason> | timeout
    # -- gate-0 exit-speed control (Phase 3) diagnostics -------------------------------
    speed_active: bool = False  # pitch authority engaged for exit-speed control this tick
    speed_reason: str = ""  # eligibility / revoke reason
    eskf_speed_mps: float = float("nan")
    vel_cov_trace: float = float("nan")
    gate0_fix_age_s: float = float("nan")
    pnp_closure_mps: float = float("nan")
    pnp_closure_samples: int = 0
    speed_residual_mps: float = float("nan")
    speed_error_mps: float = float("nan")
    speed_pitch_rate_rps: float = 0.0
    speed_pitch_impulse_rad: float = 0.0
    exit_speed_mps: float = float("nan")  # captured at the 0→1 advance
    corner_permitted: bool = True  # False → exit speed too high, corner refused
    # -- logical provisional identity (Phase 1) diagnostics -----------------------------
    logical_prov_id: int = -1  # the logical gate-1 identity (first raw id locked)
    logical_raw_id: int = -1  # the raw tracker id currently backing it
    logical_prev_raw_id: int = -1
    logical_streak: int = 0
    logical_azimuth_rad: float = float("nan")  # world azimuth = control_yaw + bearing
    logical_azimuth_residual_rad: float = float("nan")  # residual on the last handover check
    logical_gap_s: float = 0.0  # time since the logical target was last seen
    logical_reacquired: int = 0  # count of guarded raw-id handovers
    logical_reason: str = ""  # continue | established | reacquired | rejected:<why> | none
    control_yaw_rad: float = float("nan")  # the control-attitude yaw the tick used (for replay)
    body_gyro_rps: tuple[float, float, float] = (
        float("nan"),
        float("nan"),
        float("nan"),
    )  # the raw body rates the observer integrated this tick (for exact replay)
    # -- geometry-derived corner bank (Phase 3) -----------------------------------------
    corner_bank_speed_mps: float = float("nan")  # frozen conservative speed used for the bank
    corner_bank_raw_rad: float = float("nan")  # raw (unclamped) geometry bank == bank_geometry_raw
    corner_bank_rad: float = float("nan")  # geometry-derived, clamped turn bank target
    bank_authority_cap_rad: float = float("nan")  # the command authority cap the clamp applied
    # -- bank observer + roll arbitration (Phase C/D) diagnostics -----------------------
    observer_bank_rad: float = float("nan")  # dedicated gyro-observer bank (primary roll signal)
    observer_source: str = "idle"  # idle | gyro | gyro+pnp | stale:<why>
    observer_age_s: float = float("inf")
    observer_pnp_accepted: int = 0
    observer_pnp_rejected: int = 0
    observer_pnp_innovation_rad: float = float("nan")
    cmd_integral_bank_rad: float = float("nan")  # independent envelope (commanded-rate integral)
    roll_feedback_source: str = "none"  # observer | none (which signal fed the roll law)
    roll_arbitration: str = ""  # ok | fail:<why>
    shadow_roll_degraded: bool = False  # shadow ESKF disagrees while observer+integral agree
    active_roll_degraded: bool = False  # control attitude spikes while observer+integral agree
    preturn_reject: str = ""  # first failing PRETURN eligibility gate (empty when engaged)


@dataclass(frozen=True, slots=True)
class VisualHandoffDecision:
    """The command selection + full status for one loop (log-only carrier).

    ``selected`` is what the mission sends. ``requested`` is the handoff law's desired
    command for its authoritative channels (``None`` in IDLE); non-authoritative channels
    always carry the legacy value.
    """

    legacy: Command
    requested: Command | None
    selected: Command
    status: VisualHandoffStatus

    @property
    def state(self) -> VisualHandoffState:
        return self.status.state

    @property
    def active(self) -> bool:
        return self.status.active


_ACTIVE_STATES = frozenset(
    {
        VisualHandoffState.PRETURN,
        VisualHandoffState.TURN,
        VisualHandoffState.CAPTURE,
        VisualHandoffState.COMMIT,
        VisualHandoffState.RECOVERY,
        VisualHandoffState.REACQUIRE_ARC,
        VisualHandoffState.SAFE_EXIT,
    }
)
_FULL_AUTHORITY = (True, True, True, True)
_ROLL_YAW_ONLY = (False, True, False, True)  # PRETURN: override roll + yaw only
_NO_AUTHORITY = (False, False, False, False)


def _finite(*values: float) -> bool:
    return all(math.isfinite(v) for v in values)


def _wrap_angle(a: float) -> float:
    """Wrap an angle to (−π, π] — used for world-azimuth continuity residuals."""
    return math.atan2(math.sin(a), math.cos(a))


def image_row_thrust(
    cy_px: float,
    cy_rate_pxps: float,
    area_px2: float,
    bearing_rad: float,
    roll_rad: float,
    roll_trusted: bool,
    cfg: VisualHandoffConfig,
) -> float:
    """Collective thrust from the gate's image row (the reliable height cue).

    PD on ``cy`` toward :data:`CY_TARGET`. Preserves the legacy descent guard: a gate
    below centre (``cy > target``) may pull thrust down only when it is CLOSE (large
    area) and roughly on-axis — a far or off-axis low gate must not dive us into the
    floor. Only TRUSTED ESKF roll is tilt-compensated (``1/cos`` with a 0.85 floor); IMU
    and ESKF vertical velocity are deliberately not used.
    """
    if not _finite(cy_px, area_px2, bearing_rad):
        return cfg.thrust_hover
    err = (cy_px - CY_TARGET) / 180.0
    # Apply the PD servo when the gate is high (climb, always allowed) or when it is low
    # but CLOSE and roughly on-axis; otherwise hold hover (never dive at a far/off-axis gate).
    servo = err < 0 or (
        area_px2 >= cfg.thrust_descend_area_px2
        and abs(bearing_rad) <= cfg.thrust_descend_max_bearing_rad
    )
    if not servo:
        thr = cfg.thrust_hover
    else:
        rate = cy_rate_pxps if math.isfinite(cy_rate_pxps) else 0.0
        thr = cfg.thrust_hover - cfg.thrust_kp_px * err - cfg.thrust_kd_px * (rate / 180.0)
    if roll_trusted and math.isfinite(roll_rad):
        thr = thr / max(cfg.thrust_tilt_cos_floor, math.cos(roll_rad))
    return float(min(cfg.thrust_max, max(cfg.thrust_min, thr)))


class VisualHandoffController:
    """Deterministic visual gate-0→1 handoff state machine (Training-only authority).

    One instance per attempt; :meth:`reset` between attempts. Holds only the current
    state, per-state entry timestamps, the locked provisional identity, a two-point
    bearing/cy history for rate estimation, dropout bookkeeping, and the last selected
    command for slew-limited handover.
    """

    def __init__(self, config: VisualHandoffConfig | None = None) -> None:
        self.config = VisualHandoffConfig() if config is None else config
        self.reset()

    def reset(self) -> None:
        """Full per-attempt isolation."""
        self._state = VisualHandoffState.IDLE
        self._state_entered_ns = 0
        self._turn_entered_ns = 0
        self._commit_entered_ns = 0
        self._locked_track_id = -1
        self._capture_streak = 0
        self._succeeded = False
        self._safe_exit_reason = ""
        self._reason = "idle"
        # command handover
        self._last_selected: Command | None = None
        self._last_ns: int | None = None
        self._prev_source = "legacy"
        # bearing/cy history (last two UNIQUE structured frames, keyed by sequence)
        self._b_seq = -1
        self._b_prev: tuple[int, float] | None = None  # (arrival_ns, bearing)
        self._b_last: tuple[int, float] | None = None
        self._cy_seq = -1
        self._cy_prev: tuple[int, float] | None = None
        self._cy_last: tuple[int, float] | None = None
        # dropout / last-good
        self._last_seen_ns = 0
        self._last_bearing = float("nan")
        self._last_bearing_rate = 0.0
        self._last_roll_target = 0.0
        # gate-0 proximity latch + provisional-seen clock (PRETURN engagement window)
        self._gate0_close_ns = 0
        self._gate0_bearing = float("nan")
        self._prov_seen_ns = 0
        # feedforward commanded-bank integral (safety cap on the roll authority)
        self._cmd_bank_rad = 0.0
        self._last_roll_cmd = 0.0
        self._last_yaw_cmd = 0.0
        # REACQUIRE_ARC + SAFE_EXIT termination bookkeeping
        self._reacquire_entered_ns = 0
        self._safe_exit_entered_ns = 0
        self._terminate = False
        self._yaw_since_seen_rad = 0.0  # integrated commanded yaw since the locked target seen
        self._reacquire_result = ""
        self._reacquire_candidate_id = -1
        self._reacquire_candidate_hits = 0
        self._reacquire_residual = float("nan")
        self._reacquire_prev_id = -1
        # gate-0 exit-speed control state
        self._closure_samples: list[ClosureSample] = []
        self._last_agree_ns = 0
        self._speed_impulse_rad = 0.0  # cumulative commanded nose-up pitch (rad)
        self._speed_engaged = False  # brake engaged at least one tick this attempt
        self._exit_speed_mps = float("nan")
        self._corner_permitted = True
        self._last_closure_speed = float("nan")
        self._last_closure_n = 0
        self._last_valid_speed_mps = float("nan")
        self._speed_residual_mps = float("nan")
        self._prev_active_gate = -1
        self._speed_reason = "off"
        self._speed_active = False
        self._speed_pitch = 0.0
        self._speed_latched = False  # authority begins above engage speed, then holds
        # logical provisional gate-1 identity (Phase 1)
        self._logical_prov_id = -1
        self._logical_raw_id = -1
        self._logical_prev_raw = -1
        self._logical_azimuth = float("nan")
        self._logical_azimuth_ns = 0
        self._logical_streak = 0
        self._logical_reacq = 0
        self._logical_reason = "none"
        self._logical_residual = float("nan")
        # geometry-derived corner bank (Phase 3)
        self._corner_bank_speed = float("nan")
        self._corner_bank_raw = float("nan")
        self._corner_bank = float("nan")
        # bank observer + roll arbitration (Phase C/D)
        self._observer = _BankObserver(self.config.bank_observer)
        self._roll_valid_cached = False
        self._roll_meas_cached = float("nan")  # the arbitrated bank fed to the roll law
        self._roll_arbitration = ""
        self._shadow_roll_degraded = False
        self._active_roll_degraded = False
        self._roll_disagree_count = 0
        self._preturn_reject = ""  # first failing PRETURN eligibility gate (diagnostics)

    @property
    def state(self) -> VisualHandoffState:
        return self._state

    @property
    def locked_track_id(self) -> int:
        return self._locked_track_id

    @property
    def succeeded(self) -> bool:
        return self._succeeded

    # ------------------------------------------------------------------
    def step(self, inp: VisualHandoffInput) -> VisualHandoffDecision:
        """Advance the state machine one loop and select a command."""
        cfg = self.config
        # -- feedforward commanded-bank integral (safety cap; reset while legacy owns) ----
        if self._last_ns is not None and self._state in _ACTIVE_STATES:
            dt = max(0.0, (inp.now_ns - self._last_ns) * 1e-9)
            self._cmd_bank_rad += cfg.roll_command_actual_gain * self._last_roll_cmd * dt
            cap = cfg.bank_feedforward_cap_rad
            self._cmd_bank_rad = max(-cap, min(cap, self._cmd_bank_rad))
            self._yaw_since_seen_rad += self._last_yaw_cmd * dt  # for bearing-continuity guard
        else:
            self._cmd_bank_rad = 0.0
        # -- bank observer (seed → propagate → optional PnP correct) + roll arbitration --
        self._update_observer(inp)
        roll_valid = self._roll_valid(inp)
        self._update_gate0_latch(inp)
        # -- maintain the logical gate-1 identity (world-azimuth continuity) -------------
        if inp.enabled and inp.active_gate <= 0:
            self._track_logical_provisional(inp)
        # -- pick the relevant structured target for the current state ------------------
        target = self._relevant_target(inp)
        self._update_history(target, inp.now_ns)
        bearing_rate = self._bearing_rate()
        dropout_age = self._dropout_age(target, inp.now_ns)
        if dropout_age <= 0.0:  # locked target seen this tick → reset the arc-yaw accumulator
            self._yaw_since_seen_rad = 0.0
        self._reacquire_result = ""  # per-tick; set by _advance_reacquire

        # -- immediate hard aborts (bypass everything; hand back to legacy) -------------
        if inp.abort_reason:
            self._enter(VisualHandoffState.SAFE_EXIT, inp.now_ns, f"abort:{inp.abort_reason}")

        # -- disabled / not-yet-engaged: stay legacy ------------------------------------
        if not inp.enabled and self._state is not VisualHandoffState.SAFE_EXIT:
            self._state = VisualHandoffState.IDLE

        # -- gate-0 exit-speed brake (pitch-only overlay; anchored gate-0 approach window) --
        self._speed_pitch, self._speed_active, self._speed_reason = (0.0, False, "off")
        if inp.enabled and inp.anchor_gate == 0 and inp.active_gate <= 0:
            self._speed_pitch, self._speed_active, self._speed_reason = self._entry_speed_pitch(inp)
            self._speed_engaged = self._speed_engaged or self._speed_active
        # Capture the validated exit speed at the advance past gate 0 and decide corner permission.
        if (
            self._prev_active_gate <= 0
            and inp.active_gate >= 1
            and math.isnan(self._exit_speed_mps)
        ):
            self._exit_speed_mps = self._last_valid_speed_mps
            self._corner_permitted = not (
                math.isfinite(self._exit_speed_mps)
                and self._exit_speed_mps > cfg.entry_speed_refuse_corner_mps
            )

        # -- state transitions ----------------------------------------------------------
        if self._state is VisualHandoffState.SAFE_EXIT:
            # SAFE_EXIT holds a bounded level/brake for at most safe_exit_hold_s, then
            # signals the mission to terminate the attempt (it must NOT hold thrust forever).
            hold_s = (
                (inp.now_ns - self._safe_exit_entered_ns) * 1e-9
                if self._safe_exit_entered_ns > 0
                else 0.0
            )
            if hold_s >= cfg.safe_exit_hold_s:
                self._terminate = True
        else:
            self._advance(inp, target, roll_valid, dropout_age)

        # -- compute the requested command for the (possibly new) state -----------------
        requested, channels, roll_target = self._command(
            inp, target, roll_valid, bearing_rate, dropout_age
        )
        # Overlay the exit-speed brake onto the PITCH channel only (PRETURN keeps roll/yaw;
        # thrust stays legacy). Works during IDLE (legacy + pitch) and PRETURN (roll/yaw + pitch).
        if self._speed_active:
            base = list(requested) if requested is not None else list(inp.legacy_command)
            base[2] = self._speed_pitch
            requested = (base[0], base[1], base[2], base[3])
            channels = (channels[0], channels[1], True, channels[3])
        selected, source, slew_limited = self._select(inp, requested, channels)

        now_s_in_state = (inp.now_ns - self._state_entered_ns) * 1e-9
        transition_age = (
            (inp.now_ns - self._turn_entered_ns) * 1e-9 if self._turn_entered_ns > 0 else 0.0
        )
        status = VisualHandoffStatus(
            state=self._state,
            reason=self._reason,
            source=source,
            active=source == "visual",
            locked_track_id=self._locked_track_id,
            input_track_id=target.track_id if target.valid else -1,
            input_sequence=target.sequence if target.valid else -1,
            bearing_rad=target.bearing_rad if target.valid else float("nan"),
            bearing_rate_rps=bearing_rate,
            predicted_bearing_rad=self._predicted_bearing(dropout_age),
            cy_err_px=(target.cy_px - CY_TARGET) if target.valid else float("nan"),
            cy_rate_pxps=self._cy_rate(),
            dropout_age_s=dropout_age,
            roll_target_rad=roll_target,
            roll_estimate_rad=inp.eskf_roll_rad,
            att_cov_trace_rad2=inp.eskf_att_cov_trace_rad2,
            roll_valid=roll_valid,
            time_in_state_s=now_s_in_state,
            transition_age_s=transition_age,
            safe_exit_reason=self._safe_exit_reason,
            authoritative_channels=channels,
            entered=(source == "visual" and self._prev_source != "visual"),
            exited=(source != "visual" and self._prev_source == "visual"),
            slew_limited=slew_limited,
            succeeded=self._succeeded,
            terminate=self._terminate,
            reacquire_elapsed_s=(
                (inp.now_ns - self._reacquire_entered_ns) * 1e-9
                if self._reacquire_entered_ns > 0
                and self._state is VisualHandoffState.REACQUIRE_ARC
                else 0.0
            ),
            reacquire_yaw_since_seen_rad=self._yaw_since_seen_rad,
            reacquire_prev_track_id=self._reacquire_prev_id,
            reacquire_candidate_track_id=self._reacquire_candidate_id,
            reacquire_candidate_hits=self._reacquire_candidate_hits,
            reacquire_continuity_residual_rad=self._reacquire_residual,
            reacquire_result=self._reacquire_result,
            speed_active=self._speed_active,
            speed_reason=self._speed_reason,
            eskf_speed_mps=inp.eskf_speed_mps,
            vel_cov_trace=inp.eskf_vel_cov_trace,
            gate0_fix_age_s=inp.eskf_fix_age_s,
            pnp_closure_mps=self._last_closure_speed,
            pnp_closure_samples=self._last_closure_n,
            speed_residual_mps=self._speed_residual_mps,
            speed_error_mps=(
                inp.eskf_speed_mps - cfg.entry_speed_target_mps
                if math.isfinite(inp.eskf_speed_mps)
                else float("nan")
            ),
            speed_pitch_rate_rps=self._speed_pitch,
            speed_pitch_impulse_rad=self._speed_impulse_rad,
            exit_speed_mps=self._exit_speed_mps,
            corner_permitted=self._corner_permitted,
            logical_prov_id=self._logical_prov_id,
            logical_raw_id=self._logical_raw_id,
            logical_prev_raw_id=self._logical_prev_raw,
            logical_streak=self._logical_streak,
            logical_azimuth_rad=self._logical_azimuth,
            logical_azimuth_residual_rad=self._logical_residual,
            logical_gap_s=self._logical_gap_s(inp.now_ns),
            logical_reacquired=self._logical_reacq,
            logical_reason=self._logical_reason,
            control_yaw_rad=inp.control_yaw_rad,
            body_gyro_rps=inp.body_gyro_rps,
            corner_bank_speed_mps=self._corner_bank_speed,
            corner_bank_raw_rad=self._corner_bank_raw,
            corner_bank_rad=self._corner_bank,
            bank_authority_cap_rad=cfg.corner_bank_max_rad,
            observer_bank_rad=self._observer.bank_rad(),
            observer_source=self._observer.source,
            observer_age_s=self._observer.age_s(inp.now_ns),
            observer_pnp_accepted=self._observer.pnp_accepted,
            observer_pnp_rejected=self._observer.pnp_rejected,
            observer_pnp_innovation_rad=self._observer.last_innovation_rad,
            cmd_integral_bank_rad=self._cmd_bank_rad,
            roll_feedback_source=("observer" if math.isfinite(self._roll_meas_cached) else "none"),
            roll_arbitration=self._roll_arbitration,
            shadow_roll_degraded=self._shadow_roll_degraded,
            active_roll_degraded=self._active_roll_degraded,
            preturn_reject=self._preturn_reject,
        )
        # Integrate the commanded nose-up pitch impulse (bounded budget) using command dt.
        if self._speed_active and self._last_ns is not None:
            dt = max(0.0, (inp.now_ns - self._last_ns) * 1e-9)
            self._speed_impulse_rad += max(0.0, selected[2]) * dt
        self._prev_active_gate = inp.active_gate
        self._prev_source = source
        self._last_selected = selected
        self._last_roll_cmd = selected[1]  # feed the commanded-bank integral next tick
        self._last_yaw_cmd = selected[3]
        self._last_ns = inp.now_ns
        return VisualHandoffDecision(
            legacy=inp.legacy_command, requested=requested, selected=selected, status=status
        )

    # ------------------------------------------------------------------
    def _enter(self, state: VisualHandoffState, now_ns: int, reason: str) -> None:
        if state is not self._state:
            self._state = state
            self._state_entered_ns = now_ns
            if state is VisualHandoffState.TURN:
                self._turn_entered_ns = now_ns
                if not math.isfinite(self._corner_bank):
                    self._freeze_corner_bank()
            elif state is VisualHandoffState.COMMIT:
                self._commit_entered_ns = now_ns
            elif state is VisualHandoffState.REACQUIRE_ARC:
                self._reacquire_entered_ns = now_ns
            elif state is VisualHandoffState.SAFE_EXIT:
                self._safe_exit_reason = reason
                self._safe_exit_entered_ns = now_ns
            self._capture_streak = 0
        self._reason = reason

    def _advance(
        self,
        inp: VisualHandoffInput,
        target: StructuredTarget,
        roll_valid: bool,
        dropout_age: float,
    ) -> None:
        cfg = self.config
        self._reason = self._state.value
        state = self._state
        if state is VisualHandoffState.IDLE:
            if inp.enabled and self._preturn_eligible(inp, roll_valid):
                # lock the LOGICAL raw id (== the provisional's id at establishment); it will
                # follow guarded world-azimuth handovers through the raw-track churn.
                self._locked_track_id = self._preturn_lock_id(inp)
                self._prov_seen_ns = inp.now_ns
                self._enter(VisualHandoffState.PRETURN, inp.now_ns, "preturn_entry")
            return

        if state is VisualHandoffState.PRETURN:
            self._advance_preturn(inp, roll_valid)
            return

        if state is VisualHandoffState.REACQUIRE_ARC:
            self._advance_reacquire(inp, target, roll_valid)
            return

        # TURN / CAPTURE / COMMIT. Hard SAFETY (roll/bank) → SAFE_EXIT immediately.
        safety = self._safety_hard_exit(inp, roll_valid)
        if safety:
            self._enter(VisualHandoffState.SAFE_EXIT, inp.now_ns, safety)
            return
        if state is VisualHandoffState.COMMIT:
            self._advance_commit(inp, target, dropout_age)
            return
        # Is the LOCKED gate-1 target present and usable this tick?
        locked_present = (
            target.valid
            and target.track_id == self._locked_track_id
            and target.race_gate_id == 1
            and math.isfinite(target.bearing_rad)
        )
        if locked_present:
            if target.association_refused or target.wrong_turn or self._wrong_side(target):
                reason = "association_refused" if target.association_refused else "wrong_turn"
                self._enter(VisualHandoffState.SAFE_EXIT, inp.now_ns, reason)
                return
            if state is VisualHandoffState.TURN:
                if (inp.now_ns - self._turn_entered_ns) * 1e-9 > cfg.max_time_to_capture_s:
                    self._enter(VisualHandoffState.SAFE_EXIT, inp.now_ns, "capture_timeout")
                    return
                if self._capture_ready(inp, target):
                    self._enter(VisualHandoffState.CAPTURE, inp.now_ns, "capture")
            elif state is VisualHandoffState.CAPTURE and self._commit_ready(inp, target):
                self._enter(VisualHandoffState.COMMIT, inp.now_ns, "commit")
            return
        # Locked target temporarily lost → bounded right-arc reacquisition, else SAFE_EXIT.
        if self._reacquire_eligible(inp, roll_valid):
            self._enter(VisualHandoffState.REACQUIRE_ARC, inp.now_ns, "reacquire_arc")
        else:
            self._enter(VisualHandoffState.SAFE_EXIT, inp.now_ns, "target_lost")

    def _preturn_lock_id(self, inp: VisualHandoffInput) -> int:
        """Raw track id the pre-turn locks onto at entry.

        The logical identity once established (it follows guarded world-azimuth handovers
        through raw-track churn); otherwise the current provisional gate-1 candidate's raw
        id. The fallback preserves the pre-logical single-lock-at-entry behaviour and lets a
        recording that lacks the control-attitude yaw (needed to establish the logical id)
        still promote. If neither is available the existing lock is kept.
        """
        if self._logical_prov_id >= 0:
            return self._logical_raw_id
        prov = inp.provisional_target
        if prov is not None and prov.valid and prov.track_id >= 0:
            return prov.track_id
        return self._locked_track_id

    def _advance_preturn(self, inp: VisualHandoffInput, roll_valid: bool) -> None:
        cfg = self.config
        # Promotion: race counter advanced past gate 0. The worker's ACTIVE structured
        # target lags the race-counter advance by ~1 frame (it is momentarily unpublished on
        # the advance tick), so only a VALID, DIFFERENT gate-1 identity is a true mismatch;
        # a not-yet-published active target enters TURN and is governed by the per-frame
        # hard-exit identity check + the dropout ladder on the following frames.
        # The locked id follows the guarded LOGICAL handover (raw track 0→2 = same physical
        # gate); a discontinuous raw track (the imposter) is rejected by the logical tracker and
        # never becomes the locked id.
        if self._logical_prov_id >= 0:
            self._locked_track_id = self._logical_raw_id
        if inp.active_gate >= 1:
            if not self._corner_permitted:
                self._enter(VisualHandoffState.SAFE_EXIT, inp.now_ns, "exit_speed_too_high")
                return
            act = inp.active_target
            if act.valid and act.race_gate_id == 1 and act.track_id >= 0:
                if act.track_id == self._locked_track_id:
                    self._enter(VisualHandoffState.TURN, inp.now_ns, "promoted")
                else:
                    self._enter(VisualHandoffState.SAFE_EXIT, inp.now_ns, "identity_mismatch")
            else:
                self._enter(VisualHandoffState.TURN, inp.now_ns, "promoted_pending")
            return
        act = inp.active_target
        margin = cfg.gate0_bearing_limit_rad + _DEG(5.0)
        # Refresh the raw provisional-sighting clock so the lost-target grace measures the gap
        # since the gate-1 candidate was last actually seen (the logical clock is used once the
        # logical identity is established; this covers the pre-logical fallback).
        prov = inp.provisional_target
        if prov is not None and prov.valid and prov.race_gate_id == 1:
            self._prov_seen_ns = inp.now_ns
        revoke_gap = self._revoke_gap_s(inp)
        # Revoke to legacy IDLE ONLY on: lost roll; the pre-turn target absent longer than its
        # grace (world-azimuth continuity keeps it alive through raw-id churn); or gate 0 still
        # clearly close AND badly off-axis (a real mis-thread). A discontinuous imposter does
        # NOT revoke — the logical tracker simply refuses to hand off to it.
        if not roll_valid:
            self._enter(VisualHandoffState.IDLE, inp.now_ns, "revoke_roll_invalid")
        elif revoke_gap > cfg.logical_max_gap_s:
            self._enter(VisualHandoffState.IDLE, inp.now_ns, "revoke_lost")
        elif (
            act.valid
            and act.race_gate_id == 0
            and self._gate0_close(act)
            and math.isfinite(act.bearing_rad)
            and abs(act.bearing_rad) > margin
        ):
            self._enter(VisualHandoffState.IDLE, inp.now_ns, "revoke_gate0_misaligned")
        else:
            self._reason = "preturn"

    def _advance_commit(
        self, inp: VisualHandoffInput, target: StructuredTarget, dropout_age: float
    ) -> None:
        cfg = self.config
        if inp.active_gate >= 2:
            self._succeeded = True
            self._enter(VisualHandoffState.IDLE, inp.now_ns, "gate2_success")
            return
        if (inp.now_ns - self._commit_entered_ns) * 1e-9 > cfg.commit_max_s:
            self._enter(VisualHandoffState.SAFE_EXIT, inp.now_ns, "commit_timeout")
            return
        if dropout_age > cfg.dropout_recovery_s:
            self._enter(VisualHandoffState.SAFE_EXIT, inp.now_ns, "target_lost")
        self._reason = "commit"

    # ------------------------------------------------------------------
    def _gate0_close(self, act: StructuredTarget) -> bool:
        """Gate 0 is close enough to begin the pre-turn (large / clipped / near-metric)."""
        cfg = self.config
        return (
            (math.isfinite(act.area_px2) and act.area_px2 >= cfg.gate0_close_area_px2)
            or (
                math.isfinite(act.visible_fraction)
                and act.visible_fraction <= cfg.gate0_close_visible_fraction
            )
            or (
                act.has_metric_pose
                and math.isfinite(act.slant_range_m)
                and act.slant_range_m <= cfg.gate0_close_slant_m
            )
        )

    def _gate0_aligned_close(self, act: StructuredTarget) -> bool:
        cfg = self.config
        return bool(
            act.valid
            and act.race_gate_id == 0
            and self._gate0_close(act)
            and math.isfinite(act.bearing_rad)
            and abs(act.bearing_rad) <= cfg.gate0_bearing_limit_rad
        )

    def _update_gate0_latch(self, inp: VisualHandoffInput) -> None:
        """Remember the most recent close + aligned gate-0 sighting (proximity memory)."""
        act = inp.active_target
        if self._gate0_aligned_close(act):
            self._gate0_close_ns = inp.now_ns
            self._gate0_bearing = act.bearing_rad

    def _track_logical_provisional(self, inp: VisualHandoffInput) -> None:
        """Maintain a LOGICAL gate-1 identity above the churning raw tracker ids (Phase 1).

        Continuity uses the ESKF-independent world azimuth = control_yaw + bearing (a fixed
        gate holds it ~constant as the drone yaws). A different raw id is accepted as the SAME
        physical gate only if it continues the world azimuth (residual ≤ threshold, within the
        gap, confirmed, right side); the downstream imposter's large azimuth jump is rejected.
        """
        cfg = self.config
        prov = inp.provisional_target
        cy = inp.control_yaw_rad
        if not (
            prov.valid
            and prov.race_gate_id == 1
            and math.isfinite(prov.bearing_rad)
            and math.isfinite(cy)
        ):
            if self._logical_prov_id >= 0:
                self._logical_reason = "gap"
            return
        az = _wrap_angle(cy + prov.bearing_rad)
        raw = prov.track_id
        right_ok = bool(
            prov.expected_turn_sign == 1.0
            and not prov.wrong_turn
            and not prov.association_refused
            and prov.bearing_rad >= 0.0
            and abs(prov.bearing_rad) <= cfg.logical_fov_limit_rad
        )
        if self._logical_prov_id < 0:
            if (
                right_ok
                and raw >= 0
                and cfg.preturn_bearing_min_rad <= prov.bearing_rad <= cfg.preturn_bearing_max_rad
            ):
                self._logical_prov_id = raw
                self._logical_raw_id = raw
                self._logical_azimuth = az
                self._logical_azimuth_ns = inp.now_ns
                self._logical_streak = max(1, prov.track_hits)
                self._logical_reason = "established"
            return
        if raw == self._logical_raw_id:
            self._logical_azimuth = az
            self._logical_azimuth_ns = inp.now_ns
            self._logical_streak += 1
            self._logical_reason = "continue"
            return
        # different raw id → guarded world-azimuth-continuity handover
        gap = (inp.now_ns - self._logical_azimuth_ns) * 1e-9
        residual = abs(_wrap_angle(az - self._logical_azimuth))
        self._logical_residual = residual
        rate = residual / gap if gap > 1e-3 else 0.0
        if not right_ok:
            self._logical_reason = "rejected:wrong_side"
        elif raw < 0 or prov.track_hits < cfg.logical_min_hits:
            self._logical_reason = "rejected:unconfirmed"
        elif not (0.0 <= gap <= cfg.logical_max_gap_s):
            self._logical_reason = "rejected:gap"
        elif residual > cfg.logical_azimuth_residual_rad:
            self._logical_reason = "rejected:imposter"  # e.g. the downstream track's azimuth jump
        elif rate > cfg.logical_azimuth_rate_max_rps:
            self._logical_reason = "rejected:rate"
        else:
            self._logical_prev_raw = self._logical_raw_id
            self._logical_raw_id = raw
            self._logical_azimuth = az
            self._logical_azimuth_ns = inp.now_ns
            self._logical_reacq += 1
            self._logical_streak += 1  # preserve the logical confirmation across the handover
            self._logical_reason = "reacquired"

    def _logical_gap_s(self, now_ns: int) -> float:
        if self._logical_prov_id < 0 or self._logical_azimuth_ns <= 0:
            return 1e9
        return max(0.0, (now_ns - self._logical_azimuth_ns) * 1e-9)

    def _revoke_gap_s(self, inp: VisualHandoffInput) -> float:
        """Time since the pre-turn target was last observed, for the lost-target revocation.

        Uses the logical-azimuth age once the logical identity is established (it survives
        raw-track churn); otherwise the raw provisional-sighting age. Both are bounded by
        ``logical_max_gap_s`` before PRETURN reverts to legacy IDLE.
        """
        if self._logical_prov_id >= 0 and self._logical_azimuth_ns > 0:
            return max(0.0, (inp.now_ns - self._logical_azimuth_ns) * 1e-9)
        if self._prov_seen_ns > 0:
            return max(0.0, (inp.now_ns - self._prov_seen_ns) * 1e-9)
        return 1e9

    def _preturn_eligible(self, inp: VisualHandoffInput, roll_valid: bool) -> bool:
        """PRETURN entry test; records the FIRST failing gate in ``self._preturn_reject`` so a
        miss is diagnosable from the (mandatory) recording / end-of-attempt summary."""
        cfg = self.config
        prov = inp.provisional_target
        reason = ""
        if inp.active_gate != 0 or inp.anchor_gate != 0:
            reason = "not_gate0"
        elif not (prov.valid and prov.race_gate_id == 1):
            reason = "no_provisional"
        elif prov.expected_turn_sign != 1.0:
            reason = "wrong_turn_sign"
        elif prov.wrong_turn or prov.association_refused:
            reason = "refused"
        elif not (
            math.isfinite(prov.bearing_rad)
            and cfg.preturn_bearing_min_rad <= prov.bearing_rad <= cfg.preturn_bearing_max_rad
        ):
            reason = "bearing_band"
        elif (inp.now_ns - prov.arrival_ns) * 1e-9 > cfg.provisional_max_age_s:
            reason = "stale_provisional"
        elif prov.track_hits < cfg.provisional_min_hits or prov.track_id < 0:
            reason = "unconfirmed"
        else:
            # Gate-0 proximity: current target or a recently latched close+aligned sighting.
            current_ok = self._gate0_aligned_close(inp.active_target)
            recent_ok = (
                self._gate0_close_ns > 0
                and (inp.now_ns - self._gate0_close_ns) * 1e-9 <= cfg.gate0_proximity_memory_s
            )
            if not (current_ok or recent_ok):
                reason = "gate0_not_close"
            elif not roll_valid:
                reason = f"roll_invalid:{self._roll_arbitration}"
        self._preturn_reject = reason
        return reason == ""

    def _update_observer(self, inp: VisualHandoffInput) -> None:
        """Seed → propagate → optionally PnP-correct the dedicated bank observer, then arbitrate.

        Seeds from the control attitude while approaching/pre-turning (self-heals if a gyro gap
        invalidates it), propagates the raw body gyro (validated right-bank-positive sign), and
        computes the roll arbitration (Phase D): observer bank is primary, the command-integral
        bank is an independent envelope, the control attitude + shadow ESKF are cross-checks only.
        """
        obs = self._observer
        # (1) seed from the control attitude when not yet established and still pre-turn/approach.
        if (
            inp.enabled
            and not obs.established
            and inp.active_gate <= 1
            and math.isfinite(inp.control_roll_rad)
            and math.isfinite(inp.control_pitch_rad)
            and math.isfinite(inp.control_yaw_rad)
        ):
            obs.seed(inp.control_roll_rad, inp.control_pitch_rad, inp.control_yaw_rad, inp.now_ns)
        # (2) propagate with the raw body gyro.
        obs.propagate(inp.body_gyro_rps, inp.now_ns)
        # (3) optional conservative PnP roll correction from a full-pose, metric, confident gate.
        tgt = inp.active_target
        quality = bool(
            tgt.valid
            and tgt.output_mode == "full_pose"
            and tgt.has_metric_pose
            and math.isfinite(tgt.vision_roll_rad)
            and math.isfinite(tgt.confidence)
            and tgt.confidence >= 0.8
        )
        obs.correct_pnp(tgt.vision_roll_rad, quality)
        # (4) arbitrate.
        self._arbitrate_roll(inp)

    def _arbitrate_roll(self, inp: VisualHandoffInput) -> None:
        """Fail-closed multi-observer roll validity (see Phase D). Caches the result for the tick."""
        cfg = self.config
        obs = self._observer
        integral = self._cmd_bank_rad  # independent envelope
        shadow_corr = cfg.eskf_roll_feedback_sign * inp.eskf_roll_rad  # command-frame shadow roll
        active = inp.control_roll_rad
        ob = obs.bank_rad()
        self._shadow_roll_degraded = False
        self._active_roll_degraded = False
        self._roll_meas_cached = float("nan")

        # fail-closed conditions (in priority order)
        if not obs.established or not math.isfinite(ob):
            self._finish_arbitration(False, "fail:observer_unavailable")
            return
        if obs.source.startswith("stale") or not obs.warmed_up(inp.now_ns):
            self._finish_arbitration(False, "fail:observer_stale")
            return
        if abs(ob) > cfg.roll_abs_limit_rad:
            self._finish_arbitration(False, "fail:observer_over_limit")
            return
        if math.isfinite(integral) and abs(integral) > cfg.bank_feedforward_cap_rad + 1e-6:
            self._finish_arbitration(False, "fail:integral_over_cap")
            return
        # Persistent OVER-envelope disagreement → fail closed. Asymmetric on purpose: the danger
        # is an OVER-bank (observer reads MORE bank than the commanded-rate integral permits — a
        # runaway, or a spuriously-high observer). Under-reading (observer < integral) is not a
        # roll-safety emergency (less bank is safer) and, because the observer is gyro-tracked, it
        # cannot silently under-read a REAL over-bank the way the shadow ESKF did. The absolute
        # ``observer > 45°`` check above is the other over-bank backstop.
        if math.isfinite(integral) and abs(ob) > abs(integral) + cfg.roll_disagreement_rad:
            self._roll_disagree_count += 1
            if self._roll_disagree_count >= cfg.roll_disagreement_persist:
                self._finish_arbitration(False, "fail:observer_over_envelope")
                return
        else:
            self._roll_disagree_count = 0

        # degraded cross-checks: recorded, but a single degraded signal does NOT terminate.
        if math.isfinite(shadow_corr) and abs(shadow_corr - ob) > cfg.shadow_degraded_rad:
            self._shadow_roll_degraded = True
        if math.isfinite(active) and abs(active - ob) > cfg.shadow_degraded_rad:
            self._active_roll_degraded = True
        self._roll_meas_cached = ob
        self._finish_arbitration(True, "ok")

    def _finish_arbitration(self, valid: bool, reason: str) -> None:
        self._roll_valid_cached = valid
        self._roll_arbitration = reason

    def _roll_valid(self, inp: VisualHandoffInput) -> bool:
        # Arbitration is computed once per tick in _update_observer; return the cached verdict.
        return self._roll_valid_cached

    def _safety_hard_exit(self, inp: VisualHandoffInput, roll_valid: bool) -> str:
        """Immediate SAFE_EXIT reasons that apply in every engaged state (roll/bank only).

        Uses the arbitrated observer bank, NOT the sign-defective shadow ESKF roll. Identity/side
        handling is NOT here: a lost locked target routes to the bounded REACQUIRE_ARC, and a
        present locked target's wrong-side is checked at its call site.
        """
        cfg = self.config
        if not roll_valid:
            return "roll_invalid"
        ob = self._observer.bank_rad()
        if math.isfinite(ob) and abs(ob) > cfg.bank_abort_rad:
            return "excessive_bank"
        return ""

    def _reacquire_eligible(self, inp: VisualHandoffInput, roll_valid: bool) -> bool:
        """Entry test for a bounded right-arc continuation after a confirmed-target loss."""
        cfg = self.config
        if not roll_valid or inp.abort_reason:
            return False
        if inp.active_gate != 1 or self._locked_track_id < 0:
            return False
        lb = self._last_bearing
        return bool(
            math.isfinite(lb) and cfg.preturn_bearing_min_rad <= lb <= cfg.preturn_bearing_max_rad
        )

    def _advance_reacquire(
        self, inp: VisualHandoffInput, target: StructuredTarget, roll_valid: bool
    ) -> None:
        """Bounded (≤reacquire_max_s) right-arc: keep turning, re-lock an eligible gate 1."""
        cfg = self.config
        safety = self._safety_hard_exit(inp, roll_valid)
        if safety:
            self._enter(VisualHandoffState.SAFE_EXIT, inp.now_ns, safety)
            return
        result, retarget_id = self._evaluate_reacquisition(inp, target)
        self._reacquire_result = result
        if result in ("original", "new"):
            if result == "new":
                self._reacquire_prev_id = self._locked_track_id
                self._locked_track_id = retarget_id
            self._last_seen_ns = target.arrival_ns  # reset dropout timing
            self._yaw_since_seen_rad = 0.0
            if abs(target.bearing_rad) <= cfg.reacquire_capture_bearing_rad or self._in_band_metric(
                target
            ):
                self._enter(VisualHandoffState.CAPTURE, inp.now_ns, "reacquired_capture")
            else:
                self._enter(VisualHandoffState.TURN, inp.now_ns, "reacquired_turn")
            return
        if (inp.now_ns - self._reacquire_entered_ns) * 1e-9 > cfg.reacquire_max_s:
            self._enter(VisualHandoffState.SAFE_EXIT, inp.now_ns, "reacquire_timeout")
            return
        self._reason = "reacquire_arc"

    def _evaluate_reacquisition(
        self, inp: VisualHandoffInput, target: StructuredTarget
    ) -> tuple[str, int]:
        """Guarded reacquisition. Returns (result, track_id). result in
        original|new|rejected:<reason>|"" (no candidate this tick)."""
        cfg = self.config
        self._reacquire_candidate_id = target.track_id if target.valid else -1
        self._reacquire_candidate_hits = target.track_hits if target.valid else 0
        self._reacquire_residual = float("nan")
        if not target.valid or not math.isfinite(target.bearing_rad):
            return ("", -1)
        if target.race_gate_id != 1:
            return ("rejected:wrong_race", -1)
        if target.expected_turn_sign != 1.0:
            return ("rejected:turn_sign", -1)
        if target.association_refused:
            return ("rejected:assoc_refused", -1)
        if target.wrong_turn or self._wrong_side(target):
            return ("rejected:wrong_side", -1)
        if abs(target.bearing_rad) > cfg.reacquire_fov_limit_rad:
            return ("rejected:out_of_fov", -1)
        age = (inp.now_ns - target.arrival_ns) * 1e-9
        if age < 0.0 or age > cfg.reacquire_target_max_age_s:
            return ("rejected:stale", -1)
        if target.track_id == self._locked_track_id:
            return ("original", target.track_id)  # original id returned — preserve it
        # Guarded NEW id: confirmed streak + bearing continuity (imposter guard). A target
        # leaving the FOV can legitimately lose its tracker id, so accept a fresh right-side
        # gate-1 track only if it is confirmed and near the arc-predicted bearing.
        if target.track_id < 0 or target.track_hits < cfg.reacquire_min_hits:
            return ("rejected:unconfirmed", -1)
        predicted = self._last_bearing - self._yaw_since_seen_rad
        residual = abs(target.bearing_rad - predicted)
        self._reacquire_residual = residual
        if residual > cfg.reacquire_continuity_margin_rad:
            return ("rejected:imposter", -1)  # e.g. the downstream +42° gate
        return ("new", target.track_id)

    def _in_band_metric(self, target: StructuredTarget) -> bool:
        cfg = self.config
        lo, hi = cfg.slant_band_m
        return bool(
            target.has_metric_pose
            and math.isfinite(target.slant_range_m)
            and lo <= target.slant_range_m <= hi
            and math.isfinite(target.bearing_rad)
            and abs(target.bearing_rad) <= cfg.reacquire_capture_bearing_rad
        )

    def _wrong_side(self, target: StructuredTarget) -> bool:
        margin = self.config.wrong_side_bearing_margin_rad
        return bool(target.bearing_rad < 0.0 and abs(target.bearing_rad) > margin)

    def _capture_ready(self, inp: VisualHandoffInput, target: StructuredTarget) -> bool:
        cfg = self.config
        if not target.valid or not math.isfinite(target.bearing_rad):
            self._capture_streak = 0
            return False
        lo, hi = cfg.slant_band_m
        metric_ok = (
            target.has_metric_pose
            and math.isfinite(target.slant_range_m)
            and lo <= target.slant_range_m <= hi
            and abs(target.bearing_rad) <= cfg.capture_metric_bearing_rad
        )
        if abs(target.bearing_rad) <= cfg.capture_bearing_rad:
            self._capture_streak += 1
        else:
            self._capture_streak = 0
        return self._capture_streak >= cfg.capture_frames or metric_ok

    def _commit_ready(self, inp: VisualHandoffInput, target: StructuredTarget) -> bool:
        cfg = self.config
        if not target.valid or not math.isfinite(target.bearing_rad):
            return False
        if abs(target.bearing_rad) > cfg.commit_bearing_rad:
            return False
        if not (
            math.isfinite(target.cy_px) and abs(target.cy_px - CY_TARGET) <= cfg.commit_cy_err_px
        ):
            return False
        area_ok = math.isfinite(target.area_px2) and target.area_px2 >= cfg.commit_area_px2
        slant_ok = (
            math.isfinite(target.slant_range_m) and target.slant_range_m <= cfg.commit_slant_m
        )
        return area_ok or slant_ok

    # ------------------------------------------------------------------
    def _command(
        self,
        inp: VisualHandoffInput,
        target: StructuredTarget,
        roll_valid: bool,
        bearing_rate: float,
        dropout_age: float,
    ) -> tuple[Command | None, tuple[bool, bool, bool, bool], float]:
        """Return (requested command, authoritative channels, roll_target) for the state."""
        state = self._state
        if state is VisualHandoffState.IDLE:
            return None, _NO_AUTHORITY, float("nan")
        if state is VisualHandoffState.PRETURN:
            return self._preturn_command(inp, roll_valid)
        if state is VisualHandoffState.SAFE_EXIT:
            return self._safe_exit_command(inp), _FULL_AUTHORITY, 0.0
        if state is VisualHandoffState.REACQUIRE_ARC:
            return self._reacquire_command(inp, roll_valid)
        if state is VisualHandoffState.RECOVERY:
            return self._recovery_command(inp, roll_valid, dropout_age)
        # TURN / CAPTURE / COMMIT
        return self._turn_family_command(inp, target, roll_valid, dropout_age)

    def _roll_rate(self, roll_target: float, inp: VisualHandoffInput, cap: float) -> float:
        cfg = self.config
        # Command-frame bank from the arbitrated dedicated observer (Phase D). Falls back to 0
        # feedback if the observer is unavailable — the feedforward integral cap alone still
        # bounds the bank. (Previously the sign-defective shadow ESKF roll drove this and let the
        # bank overshoot; see the attempt-2 audit.)
        roll_meas = self._roll_meas_cached if math.isfinite(self._roll_meas_cached) else roll_target
        raw = cfg.roll_feedback_gain * (roll_target - roll_meas)
        raw = max(-cfg.roll_rate_clip_rps, min(cfg.roll_rate_clip_rps, raw))
        raw = max(-cap, min(cap, raw))
        # Feedforward SAFETY cap: the ESKF roll under-reads the fast turn, so never add MORE
        # bank once the integrated commanded bank has already reached the target magnitude
        # (this alone bounds the bank even if the ESKF roll never tracks it).
        if (
            roll_target >= 0.0
            and raw > 0.0
            and self._cmd_bank_rad >= roll_target
            or roll_target < 0.0
            and raw < 0.0
            and self._cmd_bank_rad <= roll_target
        ):
            raw = 0.0
        return raw

    def _reacquire_command(
        self, inp: VisualHandoffInput, roll_valid: bool
    ) -> tuple[Command, tuple[bool, bool, bool, bool], float]:
        """Bounded right-arc: hold an 18–22° right bank + steady +0.8 rad/s yaw, no decay.

        Continues the last confirmed right line-of-sight so the camera rotates through the
        gate's last +29–35° bearing. Roll uses the sign-corrected feedback + feedforward cap;
        yaw is NOT decayed merely because the target is briefly out of frame. Pitch zero,
        thrust bounded (also clamped downstream)."""
        cfg = self.config
        # Preserve the frozen geometry corner bank through the arc (keeps banking to continue
        # the coordinated turn while the target is momentarily lost); fall back to the fixed
        # reacquire bank if the geometry bank was never frozen. Both are ≤ 38°.
        roll_target = (
            self._corner_bank if math.isfinite(self._corner_bank) else cfg.reacquire_bank_rad
        )
        self._last_roll_target = roll_target
        rr = self._roll_rate(roll_target, inp, cfg.roll_rate_clip_rps) if roll_valid else 0.0
        pr = self._corner_brake_pitch(self._reacquire_entered_ns, self._last_bearing, inp.now_ns)
        return (cfg.reacquire_thrust, rr, pr, cfg.reacquire_yaw_rps), _FULL_AUTHORITY, roll_target

    def _entry_speed_pitch(self, inp: VisualHandoffInput) -> tuple[float, bool, str]:
        """Gate-0 exit-speed brake: a bounded nose-up PITCH rate from a PnP-validated ESKF
        speed. Returns (pitch_rate, engaged, reason). Owns pitch only; never used post-pass.
        """
        cfg = self.config
        # Runs during the whole gate-0 APPROACH: ESKF anchored to gate 0 and the race counter
        # not yet advanced past it. The metric window + the acceleration largely precede the
        # counter setting active_gate=0, so gate on the anchor, not active_gate.
        if inp.anchor_gate != 0 or inp.active_gate > 0:
            return 0.0, False, "not_gate0"
        act = inp.active_target
        # collect a genuine, near-centred gate-0 metric sample for the closure estimate
        if (
            act.valid
            and act.race_gate_id == 0
            and act.has_metric_pose
            and math.isfinite(act.slant_range_m)
            and math.isfinite(act.bearing_rad)
            and abs(act.bearing_rad) <= cfg.entry_speed_bearing_limit_rad
        ):
            self._closure_samples.append(
                ClosureSample(act.arrival_ns, act.track_id, 0, act.slant_range_m, act.bearing_rad)
            )
            cutoff = inp.now_ns - int(cfg.closure.max_window_s * 1e9)
            self._closure_samples = [s for s in self._closure_samples if s.arrival_ns >= cutoff][
                -24:
            ]
        closure = pnp_closure_estimate(self._closure_samples, now_ns=inp.now_ns, config=cfg.closure)
        self._last_closure_speed = closure.speed_mps
        self._last_closure_n = closure.n_samples
        agree_age = (
            (inp.now_ns - self._last_agree_ns) * 1e-9 if self._last_agree_ns else float("inf")
        )
        val = validate_gate0_speed(
            anchor_gate=inp.anchor_gate,
            initialised=inp.eskf_initialised,
            eskf_speed_mps=inp.eskf_speed_mps,
            vel_cov_trace=inp.eskf_vel_cov_trace,
            fix_age_s=inp.eskf_fix_age_s,
            agreement_age_s=agree_age,
            identity_jumped=False,  # the tracker identity-jump guard prevents a far gate here
            closure=closure,
            config=cfg.speed_validation,
        )
        self._speed_residual_mps = val.residual_mps
        # gate alignment / vertical safety band (never chase speed while off-axis / mis-height)
        aligned = (
            act.valid
            and act.race_gate_id == 0
            and math.isfinite(act.bearing_rad)
            and abs(act.bearing_rad) <= cfg.entry_speed_bearing_limit_rad
            and not (
                math.isfinite(act.cy_px) and abs(act.cy_px - CY_TARGET) > cfg.entry_speed_cy_band_px
            )
        )
        if not val.eligible or not aligned:
            self._speed_latched = False  # revoke pitch authority on any failure
            return 0.0, False, val.reason if not val.eligible else "misaligned"
        self._last_agree_ns = inp.now_ns
        # Trust the independent PnP CLOSURE speed, NOT the raw ESKF velocity: measured evidence
        # shows the ESKF over-reads ~2× on this approach (position-derived ~4 m/s vs ESKF ~9),
        # and it passes the agreement tolerance only because the tolerance is a few m/s. The
        # closure (range rate of the centred gate) is the reliable ground-speed estimate here.
        speed = closure.speed_mps if math.isfinite(closure.speed_mps) else val.speed_mps
        self._last_valid_speed_mps = speed
        # Authority BEGINS above the engage speed, then holds (commanding 0 if speed drops).
        if not self._speed_latched:
            if speed <= cfg.entry_speed_engage_mps:
                return 0.0, False, "below_engage"
            self._speed_latched = True
        if self._speed_impulse_rad >= cfg.entry_speed_impulse_cap_rad:
            return 0.0, True, "impulse_cap"
        err = speed - cfg.entry_speed_target_mps
        if err <= cfg.entry_speed_deadband_mps:
            return 0.0, True, "hold"
        pitch = max(0.0, min(cfg.entry_speed_pitch_cap_rps, cfg.entry_speed_kp * err))
        return pitch, True, "braking"

    def _corner_brake_pitch(self, entry_ns: int, bearing: float, now_ns: int) -> float:
        """Bounded nose-up (speed-bleed) pitch rate while off-axis (attempt-7 corner brake).

        Time-bounded from ``entry_ns`` (open-loop: no ESKF pitch/velocity signal is used),
        applied only while the gate bearing is off-axis so it never fights an aligned final
        approach. Sent through the unchanged rate clamps; the 60° tilt abort backstops it."""
        cfg = self.config
        if entry_ns <= 0:
            return 0.0
        # Avoid DOUBLE braking: if the pre-pass exit-speed controller already bled a material
        # nose-up impulse, the drone is already slow — skip the post-pass corner brake (which
        # otherwise pitches the camera off gate 1).
        if self._speed_engaged and self._speed_impulse_rad >= cfg.entry_speed_material_impulse_rad:
            return 0.0
        elapsed = (now_ns - entry_ns) * 1e-9
        off_axis = not math.isfinite(bearing) or abs(bearing) > cfg.corner_brake_bearing_rad
        if 0.0 <= elapsed <= cfg.corner_brake_max_s and off_axis:
            return cfg.corner_brake_pitch_rps
        return 0.0

    def _preturn_command(
        self, inp: VisualHandoffInput, roll_valid: bool
    ) -> tuple[Command, tuple[bool, bool, bool, bool], float]:
        cfg = self.config
        b = inp.provisional_target.bearing_rad if inp.provisional_target.valid else 0.0
        b = b if math.isfinite(b) else 0.0
        roll_target = min(
            cfg.preturn_bank_max_rad, max(cfg.preturn_bank_min_rad, cfg.preturn_bank_gain * b)
        )
        self._last_roll_target = roll_target
        rr = self._roll_rate(roll_target, inp, cfg.preturn_roll_rate_cap_rps) if roll_valid else 0.0
        yr = max(0.0, min(cfg.preturn_yaw_cap_rps, cfg.turn_yaw_gain * b))
        thr, pr = inp.legacy_command[0], inp.legacy_command[2]  # keep legacy thrust + pitch rate
        return (thr, rr, pr, yr), _ROLL_YAW_ONLY, roll_target

    def _turn_family_command(
        self,
        inp: VisualHandoffInput,
        target: StructuredTarget,
        roll_valid: bool,
        dropout_age: float,
    ) -> tuple[Command, tuple[bool, bool, bool, bool], float]:
        cfg = self.config
        state = self._state
        b = (
            target.bearing_rad
            if target.valid and math.isfinite(target.bearing_rad)
            else self._last_bearing
        )
        if not math.isfinite(b):
            b = 0.0
        if state is VisualHandoffState.COMMIT:
            taper = 1.0 - min(
                1.0, (inp.now_ns - self._commit_entered_ns) * 1e-9 / cfg.commit_taper_s
            )
            roll_target = self._last_roll_target * taper
            yr = 0.0
            pr = 0.0
        elif state is VisualHandoffState.CAPTURE:
            roll_target = max(0.0, min(cfg.capture_bank_cap_rad, cfg.capture_bank_gain * b))
            yr = max(0.0, min(cfg.capture_yaw_cap_rps, cfg.turn_yaw_gain * b))
            pr = 0.0
        else:  # TURN
            roll_target = self._turn_bank_target(b)
            yr = max(0.0, min(cfg.turn_yaw_cap_rps, cfg.turn_yaw_gain * b))
            pr = self._corner_brake_pitch(self._turn_entered_ns, b, inp.now_ns)
        self._last_roll_target = roll_target
        rr = self._roll_rate(roll_target, inp, cfg.roll_rate_clip_rps) if roll_valid else 0.0
        thr = self._vertical_thrust(inp, target, roll_valid)
        return (thr, rr, pr, yr), _FULL_AUTHORITY, roll_target

    def _freeze_corner_bank(self) -> None:
        """Freeze the coordinated-turn bank for the gate-0→1 corner at TURN entry.

        R = (x² + y²) / (2y) from the mined map offsets; bank = atan(v² / (g·R)) at the
        frozen conservative upper-CI pre-pass ground speed, clamped to the reviewed 38° max.
        Frozen once per TURN engagement so the bank ceiling is stable through the arc (and is
        preserved into REACQUIRE_ARC). Falls back to the bearing-scheduled ceiling on a
        degenerate geometry.
        """
        cfg = self.config
        x = cfg.corner_offset_fwd_m
        y = cfg.corner_offset_right_m
        v = cfg.corner_bank_speed_mps
        r = (x * x + y * y) / (2.0 * y) if y > 0.0 else float("nan")
        if math.isfinite(r) and r > 0.0 and math.isfinite(v):
            raw = math.atan((v * v) / (_G * r))
        else:
            raw = cfg.turn_bank_hi_rad
        self._corner_bank_speed = v
        self._corner_bank_raw = raw
        self._corner_bank = min(cfg.corner_bank_max_rad, raw)

    def _turn_bank_target(self, bearing: float) -> float:
        cfg = self.config
        # Ceiling is the frozen geometry bank (≤ 38°); the bearing schedule ramps up to it.
        hi = self._corner_bank if math.isfinite(self._corner_bank) else cfg.turn_bank_hi_rad
        b = abs(bearing)
        if b >= cfg.turn_bank_hi_bearing_rad:
            bank = hi
        elif b >= cfg.turn_bank_lo_bearing_rad:
            frac = (b - cfg.turn_bank_lo_bearing_rad) / (
                cfg.turn_bank_hi_bearing_rad - cfg.turn_bank_lo_bearing_rad
            )
            bank = cfg.turn_bank_lo_rad + frac * (hi - cfg.turn_bank_lo_rad)
        else:
            frac = b / cfg.turn_bank_lo_bearing_rad if cfg.turn_bank_lo_bearing_rad > 0 else 1.0
            bank = cfg.turn_bank_min_rad + frac * (cfg.turn_bank_lo_rad - cfg.turn_bank_min_rad)
        return min(cfg.corner_bank_max_rad, bank)

    def _recovery_command(
        self, inp: VisualHandoffInput, roll_valid: bool, dropout_age: float
    ) -> tuple[Command, tuple[bool, bool, bool, bool], float]:
        cfg = self.config
        # bounded right bank, decayed toward the last-good target, no pitch
        decay = math.exp(-max(0.0, dropout_age) / cfg.dropout_decay_tau_s)
        roll_target = min(cfg.recovery_bank_cap_rad, max(0.0, self._last_roll_target * decay))
        self._last_roll_target = roll_target
        rr = self._roll_rate(roll_target, inp, cfg.recovery_yaw_cap_rps) if roll_valid else 0.0
        pred = self._predicted_bearing(dropout_age)
        yr = (
            max(
                0.0,
                min(
                    cfg.recovery_yaw_cap_rps,
                    cfg.turn_yaw_gain * (pred if math.isfinite(pred) else 0.0),
                ),
            )
            * decay
        )
        return (cfg.recovery_thrust, rr, 0.0, yr), _FULL_AUTHORITY, roll_target

    def _safe_exit_command(self, inp: VisualHandoffInput) -> Command:
        # bounded leveling + gentle brake; never resume the blind SCAN. Roll rate levels
        # toward zero using trusted roll if available, else zero (no blind roll).
        cfg = self.config
        rr = self._roll_rate(0.0, inp, cfg.roll_rate_clip_rps) if self._roll_valid(inp) else 0.0
        return (cfg.recovery_thrust, rr, 0.0, 0.0)

    def _vertical_thrust(
        self, inp: VisualHandoffInput, target: StructuredTarget, roll_valid: bool
    ) -> float:
        cfg = self.config
        if not target.valid:
            return cfg.recovery_thrust
        return image_row_thrust(
            target.cy_px,
            self._cy_rate(),
            target.area_px2,
            target.bearing_rad,
            inp.eskf_roll_rad,
            roll_valid,
            cfg,
        )

    # ------------------------------------------------------------------
    def _select(
        self,
        inp: VisualHandoffInput,
        requested: Command | None,
        channels: tuple[bool, bool, bool, bool],
    ) -> tuple[Command, str, bool]:
        """Blend the requested command over the legacy one via a slew-limited handover.

        ``requested is None`` means no authority this tick (IDLE with no exit-speed overlay);
        otherwise at least one channel is overridden (an active state, or the gate-0
        exit-speed pitch overlay while still IDLE/PRETURN)."""
        legacy = inp.legacy_command
        if requested is None or channels == _NO_AUTHORITY:
            return legacy, "legacy", False
        desired = tuple(requested[i] if channels[i] else legacy[i] for i in range(4))
        source = "visual"
        # Abort bypasses the slew limiter — legacy/safety retains immediate priority is
        # handled upstream (abort routes us to SAFE_EXIT, still a bounded visual output).
        should_slew = source == "visual" or self._prev_source == "visual"
        if should_slew and self._last_selected is not None and self._last_ns is not None:
            selected, limited = self._slew(desired, inp.now_ns)  # type: ignore[arg-type]
        else:
            selected, limited = desired, False  # type: ignore[assignment]
        return selected, source, limited

    def _slew(self, desired: Command, now_ns: int) -> tuple[Command, bool]:
        cfg = self.config
        assert self._last_selected is not None and self._last_ns is not None
        dt = max(0.0, (now_ns - self._last_ns) * 1e-9)
        if dt == 0.0:
            return self._last_selected, self._last_selected != desired
        frac = min(dt / cfg.handover_s, 1.0)
        out = []
        limited = False
        for i in range(4):
            span = cfg.slew_spans[i] * frac
            delta = desired[i] - self._last_selected[i]
            if delta > span:
                out.append(self._last_selected[i] + span)
                limited = True
            elif delta < -span:
                out.append(self._last_selected[i] - span)
                limited = True
            else:
                out.append(desired[i])
        return (out[0], out[1], out[2], out[3]), limited

    # ------------------------------------------------------------------
    def _relevant_target(self, inp: VisualHandoffInput) -> StructuredTarget:
        """Track the provisional next gate before the race counter advances, the active gate
        after. Keyed on ``active_gate`` (not internal state) so the promotion tick already
        reads — and time-stamps — the promoted active target."""
        if inp.active_gate <= 0:
            return inp.provisional_target
        return inp.active_target

    def _update_history(self, target: StructuredTarget, now_ns: int) -> None:
        # Only the LOCKED identity feeds dropout/rate history (pre-lock we still seed rate
        # from the provisional so PRETURN has a slope). A wrong-id sighting never masks a
        # dropout — it is handled as an identity abort, not a fresh observation.
        if not target.valid:
            return
        if self._locked_track_id >= 0 and target.track_id != self._locked_track_id:
            return
        self._last_seen_ns = target.arrival_ns
        if math.isfinite(target.bearing_rad):
            self._last_bearing = target.bearing_rad
        if target.sequence != self._b_seq and math.isfinite(target.bearing_rad):
            self._b_seq = target.sequence
            self._b_prev = self._b_last
            self._b_last = (target.arrival_ns, target.bearing_rad)
            self._last_bearing_rate = self._bearing_rate()
        if target.sequence != self._cy_seq and math.isfinite(target.cy_px):
            self._cy_seq = target.sequence
            self._cy_prev = self._cy_last
            self._cy_last = (target.arrival_ns, target.cy_px)

    def _bearing_rate(self) -> float:
        if self._b_prev is None or self._b_last is None:
            return 0.0
        dt = (self._b_last[0] - self._b_prev[0]) * 1e-9
        if dt <= 0.0:
            return 0.0
        rate = (self._b_last[1] - self._b_prev[1]) / dt
        cap = self.config.dropout_bearing_rate_cap_rps
        return max(-cap, min(cap, rate))

    def _cy_rate(self) -> float:
        if self._cy_prev is None or self._cy_last is None:
            return 0.0
        dt = (self._cy_last[0] - self._cy_prev[0]) * 1e-9
        if dt <= 0.0:
            return 0.0
        return (self._cy_last[1] - self._cy_prev[1]) / dt

    def _dropout_age(self, target: StructuredTarget, now_ns: int) -> float:
        if target.valid and target.track_id == self._locked_track_id:
            return 0.0
        if self._last_seen_ns <= 0:
            return 0.0
        return max(0.0, (now_ns - self._last_seen_ns) * 1e-9)

    def _predicted_bearing(self, dropout_age: float) -> float:
        if not math.isfinite(self._last_bearing):
            return float("nan")
        if dropout_age <= 0.0:
            return self._last_bearing
        horizon = min(dropout_age, self.config.dropout_recovery_s)
        return self._last_bearing + self._last_bearing_rate * horizon


def default_controller() -> VisualHandoffController:
    """A conservative Training-only visual handoff controller (log-only until authorized)."""
    return VisualHandoffController(VisualHandoffConfig())
