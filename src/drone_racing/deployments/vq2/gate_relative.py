"""Gate-relative, velocity-aware shadow policy for the VQ2 deployment.

STRICT SHADOW MODE. This module has NO command authority. It never imports ``mission``,
``runtime``, ``calibration``, or ``DroneCommand``, never calls ``Flight.command``, and
never requires a flight-ready global course map. It computes a *hypothetical* desired
NED acceleration + heading for the currently-active race gate, converts it through the
single canonical :func:`follower.acceleration_to_rate_thrust` boundary into bounded
desired *actual* FRD body rates + collective thrust, and returns it for logging only.

Scope: gate approach / intercept (NOT a global-course planner). It keeps a positive,
bounded closure-speed objective while safely outside a near-gate handoff radius, damps
lateral/vertical velocity to centre the approach, turns toward the target, and fails
CLOSED (emits no setpoint, with a recorded reason) on any missing/unsafe input.

Frames: world/estimator NED, body FRD, VQ2 quaternion scalar-first ``[w,x,y,z]``.

ESKF sign contract (established from ``eskf.reanchor``: ``p = -R @ z`` with ``R``
body->world and ``z`` the gate-in-body measurement, so ``p = drone - gate`` in NED):
the vector FROM the drone TO the gate is therefore ``-p``. Regression-locked in the
tests; do not assume the sign.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import StrEnum

import numpy as np

from drone_racing.deployments.vq2.follower import (
    FollowerConfig,
    RateThrustSetpoint,
    StateQualityLimits,
    _readonly_array,
    acceleration_to_rate_thrust,
)


def rotation_body_to_ned(roll: float, pitch: float, yaw: float) -> np.ndarray:
    """Body->NED rotation for the ESKF ZYX (roll,pitch,yaw) convention: ``Rz@Ry@Rx``.

    Matches ``attitude.quat_to_euler`` (ZYX) so a state's euler round-trips to the same
    matrix ``attitude.quat_to_matrix`` produced (regression-locked in the tests).
    """
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    rx = np.array([[1.0, 0.0, 0.0], [0.0, cr, -sr], [0.0, sr, cr]])
    ry = np.array([[cp, 0.0, sp], [0.0, 1.0, 0.0], [-sp, 0.0, cp]])
    rz = np.array([[cy, -sy, 0.0], [sy, cy, 0.0], [0.0, 0.0, 1.0]])
    return rz @ ry @ rx


@dataclass(frozen=True, slots=True)
class GateApproachLimits:
    """Fail-closed quality/geometry gates an observation must pass before a setpoint."""

    max_fix_age_s: float = 0.5
    max_target_age_s: float = 0.5
    max_position_cov_trace_m2: float = 50.0
    max_velocity_cov_trace_m2ps2: float = 25.0
    max_current_tilt_rad: float = math.radians(45.0)
    # The reviewed gate-1 slant band (do not weaken). Below the lower bound the policy
    # hands off (avoids the line-of-sight singularity at zero range); above the upper
    # bound the metric target is not yet trustworthy as the active gate.
    slant_band_m: tuple[float, float] = (3.0, 16.0)
    handoff_distance_m: float = 3.0
    fov_bearing_limit_rad: float = math.radians(50.0)

    def __post_init__(self) -> None:
        values = (
            self.max_fix_age_s,
            self.max_target_age_s,
            self.max_position_cov_trace_m2,
            self.max_velocity_cov_trace_m2ps2,
            self.max_current_tilt_rad,
            self.handoff_distance_m,
            self.fov_bearing_limit_rad,
        )
        if any(v <= 0.0 or not math.isfinite(v) for v in values):
            raise ValueError("gate-approach limits must be finite and positive")
        lo, hi = self.slant_band_m
        if not 0.0 < lo < hi or not math.isfinite(hi):
            raise ValueError("slant_band_m must satisfy 0 < lo < hi")
        if self.handoff_distance_m > hi:
            raise ValueError("handoff_distance_m must lie within the slant band")


# Conservative SHADOW-ONLY speed ceiling. No documented VQ2 max-speed constant exists in
# the configs/tech-spec yet, so this is a named, reviewed shadow bound (NOT a hidden
# authority limit): it is high enough to PRESERVE an already-established gate-0 pass
# (~12 m/s observed in the Phase-4A capture) rather than brake it, and bounded so the
# transition law can never request runaway closure. Promotion to command authority must
# replace this with a spec/config-sourced limit under separate review.
SHADOW_SAFETY_SPEED_MAX_MPS = 12.0


@dataclass(frozen=True, slots=True)
class TransitionAcquireConfig:
    """LOS-only bridging law used after the race gate advances, before ESKF reanchor.

    Trusts target DIRECTION only (never far-range PnP scale). Tracks a bounded approach
    velocity along the confirmed line-of-sight, damps lateral/vertical velocity to centre,
    and PRESERVES useful existing closure instead of braking to the anchored cruise cap.
    """

    approach_speed_min_mps: float = 1.0  # positive floor while acquiring (never zero)
    safety_speed_max_mps: float = SHADOW_SAFETY_SPEED_MAX_MPS
    velocity_kd: tuple[float, float, float] = (2.2, 2.2, 1.8)
    min_track_hits: int = 2  # frames of continuous confirmed track before acting
    max_transition_age_s: float = 2.0  # strict timeout after the gate advance
    max_target_age_s: float = 0.5
    max_velocity_cov_trace_m2ps2: float = 25.0
    max_current_tilt_rad: float = math.radians(45.0)
    fov_bearing_limit_rad: float = math.radians(50.0)

    def __post_init__(self) -> None:
        positive = (
            self.approach_speed_min_mps,
            self.safety_speed_max_mps,
            *self.velocity_kd,
            self.max_transition_age_s,
            self.max_target_age_s,
            self.max_velocity_cov_trace_m2ps2,
            self.max_current_tilt_rad,
            self.fov_bearing_limit_rad,
        )
        if any(v <= 0.0 or not math.isfinite(v) for v in positive):
            raise ValueError("transition-acquire limits must be finite and positive")
        if self.approach_speed_min_mps > self.safety_speed_max_mps:
            raise ValueError("approach_speed_min must not exceed safety_speed_max")
        if self.min_track_hits < 1:
            raise ValueError("min_track_hits must be >= 1")


@dataclass(frozen=True, slots=True)
class GateApproachConfig:
    """Gate-relative approach law + the conversion config used at the command boundary."""

    conversion: FollowerConfig
    limits: GateApproachLimits = GateApproachLimits()
    transition: TransitionAcquireConfig = TransitionAcquireConfig()
    closure_gain_per_s: float = 0.6  # v_close = gain * distance (bounded below/above)
    closure_speed_min_mps: float = 1.0  # positive floor: never damp closure to zero
    closure_speed_max_mps: float = 4.0
    velocity_kd: tuple[float, float, float] = (2.2, 2.2, 1.8)
    # Optional map-derived turn guard: the sign of the expected gate bearing (rad>0 is a
    # right turn in the camera). 0.0 disables. Do not weaken the gate-1 right-turn guard.
    expected_turn_sign: float = 0.0
    wrong_turn_bearing_margin_rad: float = math.radians(12.0)

    def __post_init__(self) -> None:
        positive = (
            self.closure_gain_per_s,
            self.closure_speed_min_mps,
            self.closure_speed_max_mps,
            *self.velocity_kd,
            self.wrong_turn_bearing_margin_rad,
        )
        if any(v <= 0.0 or not math.isfinite(v) for v in positive):
            raise ValueError("gate-approach gains must be finite and positive")
        if self.closure_speed_min_mps > self.closure_speed_max_mps:
            raise ValueError("closure_speed_min must not exceed closure_speed_max")
        if not math.isfinite(self.expected_turn_sign):
            raise ValueError("expected_turn_sign must be finite")


@dataclass(frozen=True, slots=True)
class GateRelativeObservation:
    """One control-loop input from validated live information only. Immutable.

    ``position_drone_rel_gate_ned_m`` is the ESKF ``p`` (= drone - gate, NED). Perception
    disposition flags (``association_refused``/``wrong_turn``/``out_of_band``) come from
    the confirmed structured pipeline; the policy also enforces its own band/cov/tilt/
    freshness/handoff checks independently.
    """

    arrival_time_ns: int
    active_race_gate: int
    anchor_gate: int
    initialised: bool
    position_drone_rel_gate_ned_m: np.ndarray
    velocity_ned_mps: np.ndarray
    euler_rpy_rad: np.ndarray
    cov_diag: np.ndarray  # (>=6,) pos(3), vel(3), [att(3)...]
    fix_age_s: float
    target_mode: str  # "metric" | "bearing" | "none"
    target_bearing_rad: float
    target_slant_m: float
    target_age_s: float
    # Expected turn side for THIS race gate (+1 right / -1 left / 0 unconstrained), set by
    # the worker's course association. Gate 1 = +1; gate 0 = 0. Takes precedence over the
    # config default so the gate-1 right-turn guard is live, not just unit-testable.
    expected_turn_sign: float = 0.0
    association_refused: bool = False
    wrong_turn: bool = False
    out_of_band: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "position_drone_rel_gate_ned_m",
            _readonly_array(
                self.position_drone_rel_gate_ned_m, (3,), "position_drone_rel_gate_ned_m"
            ),
        )
        object.__setattr__(
            self,
            "velocity_ned_mps",
            _readonly_array(self.velocity_ned_mps, (3,), "velocity_ned_mps"),
        )
        object.__setattr__(
            self, "euler_rpy_rad", _readonly_array(self.euler_rpy_rad, (3,), "euler_rpy_rad")
        )
        cov = np.asarray(self.cov_diag, dtype=np.float64).ravel()
        cov.setflags(write=False)
        object.__setattr__(self, "cov_diag", cov)


@dataclass(frozen=True, slots=True)
class StructuredTarget:
    """Immutable snapshot of the worker's SELECTED GateTracker observation for one frame.

    Built and published by ``ShadowWorker`` from the confirmed structured pipeline (NOT
    the legacy ``gates.detect``). ``los_body_unit`` is the body-frame line-of-sight unit
    vector — from the metric pose DIRECTION when posed (scale untrusted), else the
    horizontal bearing. Disposition flags separate a genuine identity/wrong-side refusal
    (``association_refused``/``wrong_turn``) from a fusion-band ``out_of_band`` (right gate,
    right side, merely too far to fuse — usable as direction).
    """

    valid: bool  # a target was selected this frame
    frame_id: int
    frame_sensor_ns: int
    arrival_ns: int  # monotonic receipt clock (shared with IMU/ctrl)
    active_gate_at_receipt: int
    track_id: int
    race_gate_id: int
    track_hits: int  # continuous confirmed-track frame count
    output_mode: str  # "full_pose" | "position_only" | "bearing_only" | "none"
    bearing_rad: float
    los_body_unit: np.ndarray  # (3,) body-frame LOS unit vector (direction only)
    slant_range_m: float  # metric slant range if posed, else nan (NOT trusted in transition)
    has_metric_pose: bool
    confidence: float
    cov_trace: float
    expected_turn_sign: float  # +1/-1/0 for the active gate
    disposition: str
    association_refused: bool = False
    wrong_turn: bool = False
    out_of_band: bool = False
    sequence: int = 0  # worker publication identity; unchanged until a new frame arrives
    # -- image-space geometry for the visual handoff controller (schema-v9). Defaults keep
    # schema-v7/v8 replay and existing constructors valid; NaN means "not available". These
    # are diagnostic image quantities (never fused): the handoff uses them for the pre-turn
    # proximity trigger, image-row vertical control, and cy-convergence gating.
    cx_px: float = float("nan")  # visible-contour centroid x (px)
    cy_px: float = float("nan")  # visible-contour centroid y (px)
    area_px2: float = float("nan")  # outer red-contour area (px^2)
    bbox_xywh: tuple[int, int, int, int] = (-1, -1, -1, -1)  # outer contour x,y,w,h
    visible_fraction: float = float("nan")  # 0..1 fraction of the gate inside the frame
    vision_roll_rad: float = float("nan")  # absolute body roll from the upright gate (NaN if none)

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "los_body_unit", _readonly_array(self.los_body_unit, (3,), "los_body_unit")
        )


class GateApproachStatus(StrEnum):
    """Structured outcome. ``OK`` (anchored) and ``TRANSITION_ACQUIRE`` carry a setpoint;
    all others fail closed."""

    OK = "ok"
    # Bridging mode after the race gate advances but before the ESKF reanchors to it:
    # LOS-direction-only intercept (never trusts far-range PnP scale). Carries a setpoint.
    TRANSITION_ACQUIRE = "transition_acquire"
    NOT_INITIALISED = "not_initialised"
    ANCHOR_MISMATCH = "anchor_mismatch"
    INVALID_STATE = "invalid_state"
    EXCESSIVE_TILT = "excessive_tilt"
    NO_TARGET = "no_target"
    BEARING_ONLY = "bearing_only"
    ASSOCIATION_REFUSED = "association_refused"
    WRONG_TURN = "wrong_turn"
    OUT_OF_BAND = "out_of_band"
    STALE = "stale"
    UNCERTAIN = "uncertain"
    NEAR_GATE_HANDOFF = "near_gate_handoff"
    TRANSITION_TIMEOUT = "transition_timeout"
    INVALID_OUTPUT = "invalid_output"


@dataclass(frozen=True, slots=True)
class GateApproachResult:
    """Hypothetical (log-only) gate-relative decision for one control loop."""

    status: GateApproachStatus
    setpoint: RateThrustSetpoint | None = None
    to_gate_ned_m: np.ndarray | None = None
    distance_m: float | None = None
    desired_heading_rad: float | None = None
    closure_speed_mps: float | None = None
    velocity_error_ned_mps: np.ndarray | None = None
    requested_acceleration_ned_mps2: np.ndarray | None = None
    fov_margin_rad: float | None = None

    @property
    def eligible(self) -> bool:
        return self.status is GateApproachStatus.OK

    @property
    def commanding(self) -> bool:
        """True when a hypothetical setpoint is produced (anchored OK or transition)."""
        return self.setpoint is not None


@dataclass(frozen=True, slots=True)
class GuidanceDecision:
    """Immutable, atomically-published worker guidance snapshot. Log-only.

    The command loop only READS and RECORDS this; it performs no perception/ESKF/guidance.
    ``decision_mono_ns`` lets a reader compute decision age (stale worker → fail closed).
    """

    decision_mono_ns: int
    now_ns: int
    active_gate: int
    anchor_gate: int
    track_id: int
    target_arrival_ns: int
    mode: str  # "anchored" | "transition" | "none"
    result: GateApproachResult
    # Phase-4C replay identity. The worker fills these on publication; pure policy unit
    # calls leave the zero defaults. They make a recorded decision traceable to the exact
    # state/target snapshot that produced it instead of resampling mission-thread state.
    decision_sequence: int = 0
    state_sequence: int = 0
    state_mono_ns: int = 0
    target_sequence: int = 0
    input_mono_ns: int = 0
    initialised: bool = False
    fix_age_s: float = float("nan")
    transition_age_s: float = 0.0

    def age_s(self, read_now_ns: int) -> float:
        return (read_now_ns - self.decision_mono_ns) * 1e-9


def empty_structured_target(arrival_ns: int = 0) -> StructuredTarget:
    """An invalid (no-selection) structured target — the worker's default publish."""
    return StructuredTarget(
        valid=False,
        frame_id=-1,
        frame_sensor_ns=0,
        arrival_ns=arrival_ns,
        active_gate_at_receipt=-1,
        track_id=-1,
        race_gate_id=-1,
        track_hits=0,
        output_mode="none",
        bearing_rad=float("nan"),
        los_body_unit=np.zeros(3),
        slant_range_m=float("nan"),
        has_metric_pose=False,
        confidence=0.0,
        cov_trace=float("nan"),
        expected_turn_sign=0.0,
        disposition="none",
    )


def idle_decision(now_ns: int = 0) -> GuidanceDecision:
    """A fail-closed ``none``-mode decision — the worker's default before any tick."""
    return GuidanceDecision(
        decision_mono_ns=now_ns,
        now_ns=now_ns,
        active_gate=-1,
        anchor_gate=-1,
        track_id=-1,
        target_arrival_ns=-1,
        mode="none",
        result=GateApproachResult(status=GateApproachStatus.NOT_INITIALISED),
    )


class GateRelativePolicy:
    """Pure gate-relative, velocity-aware shadow policy. No command authority.

    Stateless across calls except for per-attempt isolation via :meth:`reset` (there is
    no integrator; :meth:`reset` exists so the runtime can assert attempt boundaries and
    deterministic replay). Every call is a deterministic function of the observation.
    """

    def __init__(self, config: GateApproachConfig) -> None:
        self.config = config
        self.reset()

    def reset(self) -> None:
        """Per-attempt reset. Present for attempt isolation + deterministic replay."""
        self._calls = 0

    def update(self, obs: GateRelativeObservation, *, now_ns: int) -> GateApproachResult:
        """Return a hypothetical setpoint, or a fail-closed result with a reason."""
        self._calls += 1
        cfg = self.config
        lim = cfg.limits
        fail = self._fail  # local alias

        if not obs.initialised:
            return fail(GateApproachStatus.NOT_INITIALISED)
        if (
            obs.anchor_gate < 0
            or obs.active_race_gate < 0
            or obs.anchor_gate != obs.active_race_gate
        ):
            return fail(GateApproachStatus.ANCHOR_MISMATCH)

        p = obs.position_drone_rel_gate_ned_m
        v = obs.velocity_ned_mps
        euler = obs.euler_rpy_rad
        if not (
            np.all(np.isfinite(p))
            and np.all(np.isfinite(v))
            and np.all(np.isfinite(euler))
            and np.all(np.isfinite(obs.cov_diag))
            and now_ns >= obs.arrival_time_ns >= 0
        ):
            return fail(GateApproachStatus.INVALID_STATE)

        rotation = rotation_body_to_ned(float(euler[0]), float(euler[1]), float(euler[2]))
        if not _is_rotation(rotation):
            return fail(GateApproachStatus.INVALID_STATE)
        current_tilt = math.acos(float(np.clip(rotation[2, 2], -1.0, 1.0)))
        if current_tilt > lim.max_current_tilt_rad:
            return fail(GateApproachStatus.EXCESSIVE_TILT)

        # -- structured-target gating (perception is authoritative on identity) --------
        if obs.target_mode == "none":
            return fail(GateApproachStatus.NO_TARGET)
        if obs.association_refused:
            return fail(GateApproachStatus.ASSOCIATION_REFUSED)
        if obs.target_mode == "bearing":
            return fail(GateApproachStatus.BEARING_ONLY)
        if obs.target_mode != "metric":
            return fail(GateApproachStatus.NO_TARGET)
        if obs.wrong_turn or self._wrong_turn(obs):
            return fail(GateApproachStatus.WRONG_TURN)
        slant = float(obs.target_slant_m)
        lo, hi = lim.slant_band_m
        if (
            obs.out_of_band
            or not math.isfinite(slant)
            or slant > hi
            or abs(obs.target_bearing_rad) > lim.fov_bearing_limit_rad
        ):
            return fail(GateApproachStatus.OUT_OF_BAND)

        # -- freshness / uncertainty ---------------------------------------------------
        now_s = (now_ns - obs.arrival_time_ns) * 1e-9
        if (
            not math.isfinite(obs.fix_age_s)
            or not math.isfinite(obs.target_age_s)
            or obs.fix_age_s + now_s > lim.max_fix_age_s
            or obs.target_age_s + now_s > lim.max_target_age_s
        ):
            return fail(GateApproachStatus.STALE)
        pos_trace = float(np.sum(obs.cov_diag[0:3]))
        vel_trace = float(np.sum(obs.cov_diag[3:6]))
        if (
            pos_trace > lim.max_position_cov_trace_m2
            or vel_trace > lim.max_velocity_cov_trace_m2ps2
        ):
            return fail(GateApproachStatus.UNCERTAIN)

        # -- geometry (ESKF gate-relative position; to-gate = -p) ----------------------
        to_gate = -np.asarray(p, dtype=np.float64)
        distance = float(np.linalg.norm(to_gate))
        fov_margin = float(lim.fov_bearing_limit_rad - abs(obs.target_bearing_rad))
        # Near-gate handoff: use the SMALLER of ESKF distance and perception slant so the
        # line-of-sight direction can never approach the zero-range singularity.
        if min(distance, slant) <= lim.handoff_distance_m:
            return self._fail(
                GateApproachStatus.NEAR_GATE_HANDOFF,
                to_gate=to_gate,
                distance=distance,
                fov_margin=fov_margin,
            )

        los = to_gate / distance
        # Positive, bounded closure objective while safely far; never zero (racing).
        closure = float(
            np.clip(
                cfg.closure_gain_per_s * distance,
                cfg.closure_speed_min_mps,
                cfg.closure_speed_max_mps,
            )
        )
        velocity_reference = los * closure
        velocity_error = velocity_reference - np.asarray(v, dtype=np.float64)
        requested_accel = np.asarray(cfg.velocity_kd, dtype=np.float64) * velocity_error
        # Heading points at the gate: turns TOWARD the target and keeps it in the FOV;
        # it does NOT zero forward approach merely because the bearing is large.
        heading = float(math.atan2(to_gate[1], to_gate[0]))

        if not np.all(np.isfinite(requested_accel)) or not math.isfinite(heading):
            return fail(GateApproachStatus.INVALID_OUTPUT)
        setpoint = acceleration_to_rate_thrust(requested_accel, heading, rotation, cfg.conversion)
        if not (
            np.all(np.isfinite(setpoint.body_rates_rps))
            and math.isfinite(setpoint.collective_thrust)
        ):
            return fail(GateApproachStatus.INVALID_OUTPUT)
        return GateApproachResult(
            status=GateApproachStatus.OK,
            setpoint=setpoint,
            to_gate_ned_m=to_gate,
            distance_m=distance,
            desired_heading_rad=heading,
            closure_speed_mps=closure,
            velocity_error_ned_mps=velocity_error,
            requested_acceleration_ned_mps2=requested_accel,
            fov_margin_rad=fov_margin,
        )

    def step(
        self,
        *,
        now_ns: int,
        active_gate: int,
        anchor_gate: int,
        initialised: bool,
        position_drone_rel_gate_ned_m: np.ndarray,
        velocity_ned_mps: np.ndarray,
        euler_rpy_rad: np.ndarray,
        cov_diag: np.ndarray,
        fix_age_s: float,
        target: StructuredTarget,
        transition_age_s: float,
    ) -> GuidanceDecision:
        """Dispatch to the anchored policy or the LOS-only TRANSITION_ACQUIRE bridge.

        The one entry point the worker calls each tick with the COHERENT (state, target)
        pair it owns. Deterministic. Returns an immutable :class:`GuidanceDecision`.
        """
        track_id = target.track_id if target.valid else -1
        if not initialised:
            result = self._fail(GateApproachStatus.NOT_INITIALISED)
            mode = "none"
        elif anchor_gate >= 0 and anchor_gate == active_gate:
            obs = self._anchored_obs(
                now_ns=now_ns,
                active_gate=active_gate,
                anchor_gate=anchor_gate,
                initialised=initialised,
                p=position_drone_rel_gate_ned_m,
                v=velocity_ned_mps,
                euler=euler_rpy_rad,
                cov_diag=cov_diag,
                fix_age_s=fix_age_s,
                target=target,
            )
            result = self.update(obs, now_ns=now_ns)
            mode = "anchored"
        elif anchor_gate >= 0 and active_gate == anchor_gate + 1:
            result = self._transition_acquire(
                now_ns=now_ns,
                active_gate=active_gate,
                velocity_ned_mps=velocity_ned_mps,
                euler_rpy_rad=euler_rpy_rad,
                cov_diag=cov_diag,
                target=target,
                transition_age_s=transition_age_s,
            )
            mode = "transition"
        else:
            result = self._fail(GateApproachStatus.ANCHOR_MISMATCH)
            mode = "none"
        return GuidanceDecision(
            decision_mono_ns=now_ns,
            now_ns=now_ns,
            active_gate=active_gate,
            anchor_gate=anchor_gate,
            track_id=track_id,
            target_arrival_ns=target.arrival_ns if target.valid else -1,
            mode=mode,
            result=result,
        )

    @staticmethod
    def _anchored_obs(
        *,
        now_ns: int,
        active_gate: int,
        anchor_gate: int,
        initialised: bool,
        p: np.ndarray,
        v: np.ndarray,
        euler: np.ndarray,
        cov_diag: np.ndarray,
        fix_age_s: float,
        target: StructuredTarget,
    ) -> GateRelativeObservation:
        if not target.valid or target.race_gate_id != active_gate:
            mode, bearing, slant, age = "none", float("nan"), float("nan"), 1e9
        else:
            mode = "metric" if target.has_metric_pose else "bearing"
            bearing = target.bearing_rad
            slant = target.slant_range_m
            age = (now_ns - target.arrival_ns) * 1e-9
        return GateRelativeObservation(
            arrival_time_ns=now_ns,
            active_race_gate=active_gate,
            anchor_gate=anchor_gate,
            initialised=initialised,
            position_drone_rel_gate_ned_m=np.asarray(p, float),
            velocity_ned_mps=np.asarray(v, float),
            euler_rpy_rad=np.asarray(euler, float),
            cov_diag=np.asarray(cov_diag, float),
            fix_age_s=fix_age_s if math.isfinite(fix_age_s) else 1e9,
            target_mode=mode,
            target_bearing_rad=bearing,
            target_slant_m=slant,
            target_age_s=age,
            expected_turn_sign=target.expected_turn_sign if target.valid else 0.0,
            association_refused=target.association_refused if target.valid else False,
            wrong_turn=target.wrong_turn if target.valid else False,
            out_of_band=target.out_of_band if target.valid else False,
        )

    def _transition_acquire(
        self,
        *,
        now_ns: int,
        active_gate: int,
        velocity_ned_mps: np.ndarray,
        euler_rpy_rad: np.ndarray,
        cov_diag: np.ndarray,
        target: StructuredTarget,
        transition_age_s: float,
    ) -> GateApproachResult:
        """LOS-only bridge: direction from the confirmed track, no far-range PnP scale."""
        tc = self.config.transition
        fail = self._fail
        # -- confirmed, continuous, correctly-associated track --------------------------
        if not target.valid or target.race_gate_id != active_gate or target.output_mode == "none":
            return fail(GateApproachStatus.NO_TARGET)
        if target.track_hits < tc.min_track_hits:
            return fail(GateApproachStatus.NO_TARGET)
        if target.association_refused:
            return fail(GateApproachStatus.ASSOCIATION_REFUSED)
        if target.wrong_turn or self._wrong_side(target):
            return fail(GateApproachStatus.WRONG_TURN)
        # -- finite state + valid attitude + tilt --------------------------------------
        v = np.asarray(velocity_ned_mps, float)
        euler = np.asarray(euler_rpy_rad, float)
        los_body = np.asarray(target.los_body_unit, float)
        if not (
            np.all(np.isfinite(v))
            and np.all(np.isfinite(euler))
            and np.all(np.isfinite(los_body))
            and np.all(np.isfinite(cov_diag))
            and math.isfinite(transition_age_s)
        ):
            return fail(GateApproachStatus.INVALID_STATE)
        rotation = rotation_body_to_ned(float(euler[0]), float(euler[1]), float(euler[2]))
        if not _is_rotation(rotation):
            return fail(GateApproachStatus.INVALID_STATE)
        if math.acos(float(np.clip(rotation[2, 2], -1.0, 1.0))) > tc.max_current_tilt_rad:
            return fail(GateApproachStatus.EXCESSIVE_TILT)
        # -- in FOV, not timed out, fresh, velocity certain ----------------------------
        if abs(target.bearing_rad) > tc.fov_bearing_limit_rad:
            return fail(GateApproachStatus.OUT_OF_BAND)
        if transition_age_s > tc.max_transition_age_s:
            return fail(GateApproachStatus.TRANSITION_TIMEOUT)
        target_age = (now_ns - target.arrival_ns) * 1e-9
        if target_age > tc.max_target_age_s or target_age < 0.0:
            return fail(GateApproachStatus.STALE)
        if float(np.sum(cov_diag[3:6])) > tc.max_velocity_cov_trace_m2ps2:
            return fail(GateApproachStatus.UNCERTAIN)

        # -- LOS-only intercept: preserve useful closure, damp lateral/vertical --------
        los_norm = float(np.linalg.norm(los_body))
        if los_norm <= 1e-9:
            return fail(GateApproachStatus.INVALID_STATE)
        los_ned = rotation @ (los_body / los_norm)
        los_ned /= float(np.linalg.norm(los_ned))
        closure_now = float(v @ los_ned)  # existing closure along the new LOS
        # PRESERVE useful closure (do not brake to the anchored cruise cap); floor so we
        # never damp closure to zero; ceiling at the reviewed shadow safety-speed limit.
        closure = float(
            np.clip(
                max(closure_now, tc.approach_speed_min_mps),
                tc.approach_speed_min_mps,
                tc.safety_speed_max_mps,
            )
        )
        velocity_reference = los_ned * closure
        velocity_error = velocity_reference - v
        requested_accel = np.asarray(tc.velocity_kd, float) * velocity_error
        heading = float(math.atan2(los_ned[1], los_ned[0]))
        fov_margin = float(tc.fov_bearing_limit_rad - abs(target.bearing_rad))
        if not np.all(np.isfinite(requested_accel)) or not math.isfinite(heading):
            return fail(GateApproachStatus.INVALID_OUTPUT)
        setpoint = acceleration_to_rate_thrust(
            requested_accel, heading, rotation, self.config.conversion
        )
        if not (
            np.all(np.isfinite(setpoint.body_rates_rps))
            and math.isfinite(setpoint.collective_thrust)
        ):
            return fail(GateApproachStatus.INVALID_OUTPUT)
        return GateApproachResult(
            status=GateApproachStatus.TRANSITION_ACQUIRE,
            setpoint=setpoint,
            to_gate_ned_m=los_ned,  # DIRECTION only (unit); range deliberately not trusted
            distance_m=None,
            desired_heading_rad=heading,
            closure_speed_mps=closure,
            velocity_error_ned_mps=velocity_error,
            requested_acceleration_ned_mps2=requested_accel,
            fov_margin_rad=fov_margin,
        )

    def _wrong_side(self, target: StructuredTarget) -> bool:
        sign = target.expected_turn_sign
        if sign == 0.0:
            return False
        margin = self.config.wrong_turn_bearing_margin_rad
        return bool(target.bearing_rad * sign < 0.0 and abs(target.bearing_rad) > margin)

    def _wrong_turn(self, obs: GateRelativeObservation) -> bool:
        # per-gate sign from the observation takes precedence over the config default
        sign = (
            obs.expected_turn_sign
            if obs.expected_turn_sign != 0.0
            else self.config.expected_turn_sign
        )
        if sign == 0.0:
            return False
        bearing = obs.target_bearing_rad
        margin = self.config.wrong_turn_bearing_margin_rad
        return bool(bearing * sign < 0.0 and abs(bearing) > margin)

    @staticmethod
    def _fail(
        status: GateApproachStatus,
        *,
        to_gate: np.ndarray | None = None,
        distance: float | None = None,
        fov_margin: float | None = None,
    ) -> GateApproachResult:
        return GateApproachResult(
            status=status,
            to_gate_ned_m=to_gate,
            distance_m=distance,
            fov_margin_rad=fov_margin,
        )


def default_policy() -> GateRelativePolicy:
    """A conservative VQ2 gate-relative shadow policy (log-only; no command authority).

    The conversion bounds mirror the reviewed follower limits; the approach law keeps a
    positive, bounded closure objective. Deliberately conservative — raising the closure
    ceiling toward racing speed is a separate, reviewed pre-authority tuning step.
    """
    conversion = FollowerConfig(
        quality=StateQualityLimits(
            max_age_s=0.5, max_position_cov_trace_m2=50.0, max_velocity_cov_trace_m2ps2=25.0
        )
    )
    return GateRelativePolicy(GateApproachConfig(conversion=conversion))


def _is_rotation(rotation: np.ndarray) -> bool:
    if rotation.shape != (3, 3) or not np.all(np.isfinite(rotation)):
        return False
    if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-6):
        return False
    return bool(math.isclose(float(np.linalg.det(rotation)), 1.0, abs_tol=1e-6))
