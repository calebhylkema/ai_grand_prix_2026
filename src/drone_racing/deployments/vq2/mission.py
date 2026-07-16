"""Legacy phased VQ2 mission runner retained as the flight regression baseline.

The 2026-07-09 pivot added gate-1 camera-depth and turn-window association constants
after the reactive controller hopped among several visible gates. Those constants came
from three legacy-pipeline co-visible pairs; they are useful historical evidence but
are not the typed, flight-ready course map introduced by ADR-0004. This mission does
not load ``course_map.py`` or ``reference.py`` and must not be described as trajectory
planning or ESKF guidance.

The camera extrinsic is now resolved at 20° UP. The original down-camera blindness
explanation and the 8.6 m pad distance are unresolved historical interpretations, not
facts to tune against. Current association below keeps its existing behavior but names
camera depth explicitly so it cannot be confused with slant range.

Phases:  STAB -> G0 (launch-assisted thread, proven) -> BRAKE -> SCAN -> THREAD
         -> (gate pass) -> BRAKE ...  until finish / timeout / failsafe.

Reuses the proven infrastructure in runtime.py unchanged: Flight (calibrated rates,
estimators, failsafes), vertical_thrust (CY servo + descend guards), _trace_frame,
TRACE_DIR archive scheme. Tunables load from ``configs/`` (training.toml /
qualification.toml) — the values and their tuning history are annotated there and
in the config-module docstrings.

Usage (sim in Training, run live)::

    uv run aigp-vq2 --config training --secs 45 --attempts 4
"""

from __future__ import annotations

import argparse
import contextlib
import math
import sys
import threading
import time
from collections import Counter
from typing import TYPE_CHECKING

import numpy as np
from pymavlink import mavutil

from drone_racing.deployments.vq2 import authority as auth
from drone_racing.deployments.vq2 import calibration as cc
from drone_racing.deployments.vq2 import config as vq2_config
from drone_racing.deployments.vq2 import perception as gates
from drone_racing.deployments.vq2 import runtime
from drone_racing.deployments.vq2 import visual_handoff as vh
from drone_racing.deployments.vq2.recorder import RunRecorder
from drone_racing.deployments.vq2.shadow import ShadowWorker
from drone_racing.deployments.vq2.timing import DeadlineScheduler, DeadlineTick
from drone_racing.utils.workspace import Workspace

if TYPE_CHECKING:
    from drone_racing.deployments.vq2.authority import AuthoritySelection
    from drone_racing.deployments.vq2.config import MissionParams, VQ2Config
    from drone_racing.deployments.vq2.gate_relative import GuidanceDecision, StructuredTarget
    from drone_racing.deployments.vq2.shadow import CoherentShadowSnapshot
    from drone_racing.deployments.vq2.visual_handoff import VisualHandoffDecision

cc.CALIBRATE = True

TRACE_DT = runtime.TRACE_DT
TRACE_DIR = runtime.TRACE_DIR


def _archive_traces() -> None:
    """Same keep-last-6 archive scheme as the reactive controller (shared TRACE_DIR)."""
    TRACE_DIR.mkdir(parents=True, exist_ok=True)
    olds = sorted(TRACE_DIR.glob("*.png"))
    if olds:
        arch = TRACE_DIR / time.strftime("prev_%Y%m%d_%H%M%S")
        arch.mkdir(exist_ok=True)
        for old in olds:
            old.rename(arch / old.name)
    arch_dirs = sorted(
        (d for d in TRACE_DIR.iterdir() if d.is_dir()), key=lambda d: d.stat().st_ctime
    )
    for d in arch_dirs[:-6]:
        for f in d.glob("*.png"):
            f.unlink()
        d.rmdir()


def map_accept(tgt: gates.GateTarget, scan_angle: float, m: MissionParams) -> bool:
    """Data association v0: is this detection the mapped next gate?

    Two map constraints, both required:
    - CAMERA DEPTH: a PnP pose inside the historical band. This is not slant range,
      and downstream gate identity is not inferred solely from a far depth.
    - DIRECTION: heading-turned-so-far + current bearing inside the mapped corner
      window (gate 1 is 35-70 deg RIGHT of the exit heading depending on overshoot).
      This rejects the left-side return-leg gates that pass the range test.
    """
    if tgt.pose_cam is None:
        return False
    # This legacy mission association is explicitly CAMERA DEPTH, not slant range.
    # The typed course-map/shadow pipeline uses a separate slant_range_m contract.
    if not m.accept_depth_m[0] <= tgt.pose_cam[2] <= m.accept_depth_m[1]:
        return False
    total = scan_angle + tgt.bearing  # rad right of the leg-exit heading
    return m.accept_turn[0] <= total <= m.accept_turn[1]


def _validate_transition_authority(cfg: VQ2Config, requested: bool) -> bool:
    """Return the enabled state or reject authority outside the explicit Training gate."""
    if not requested:
        return False
    if cfg.name != "training" or not cfg.run.permit_transition_authority:
        raise ValueError("experimental transition authority is permitted only in Training")
    return True


def _validate_visual_handoff(cfg: VQ2Config, requested: bool, transition_requested: bool) -> bool:
    """Return the enabled state or reject the visual handoff outside the Training gate.

    Both the Training TOML permission and the explicit CLI flag are required; Qualification
    rejects it. It is mutually exclusive with the older ESKF transition authority, which
    stays disabled whenever the visual handoff is engaged.
    """
    if not requested:
        return False
    if transition_requested:
        raise ValueError(
            "--experimental-visual-handoff and --experimental-transition-authority "
            "are mutually exclusive"
        )
    if cfg.name != "training" or not cfg.run.permit_visual_handoff:
        raise ValueError("experimental visual handoff is permitted only in Training")
    return True


def _record_control(
    fl: runtime.Flight,
    phase: str,
    active_gate: int,
    tgt: gates.GateTarget | None,
    fresh_s: float,
    failsafe: str = "",
    timing: DeadlineTick | None = None,
    loop_started_ns: int | None = None,
    coherent: CoherentShadowSnapshot | None = None,
    authority: AuthoritySelection | None = None,
    visual: VisualHandoffDecision | None = None,
) -> None:
    """Log one synchronized control-loop row (Phase-4A observability, log-only).

    Pulls the command-boundary telemetry command() just captured, the active attitude, and
    an ATOMIC shadow-ESKF snapshot (never the live filter), stamped with the shared
    ``time.monotonic_ns()`` receipt clock so it aligns with the IMU/frame/fix streams. No
    disk I/O, no shadow work — a handful of numbers appended to the recorder.
    """
    rec = fl.shadow.recorder if fl.shadow is not None else None
    if rec is None or fl.last_cmd is None:
        return
    now_ns = time.monotonic_ns()
    roll, pitch, yaw = fl.att.euler
    if tgt is not None:
        mode = "metric" if tgt.pose_cam is not None else "bearing"
        bearing = float(tgt.bearing)
        slant = float(np.linalg.norm(tgt.pose_cam)) if tgt.pose_cam is not None else float("nan")
    else:
        mode, bearing, slant = "none", float("nan"), float("nan")
    row: dict[str, object] = {
        "mono_ns": now_ns,
        "phase": phase,
        "active_gate": active_gate,
        "tgt_track": -1,  # legacy detect() has no persistent track id (shadow stream does)
        "tgt_mode": mode,
        "tgt_fresh_s": fresh_s,
        "tgt_bearing": bearing,
        "tgt_slant": slant,
        "req": fl.last_cmd["req"],
        "des": fl.last_cmd["des"],
        "cal": fl.last_cmd["cal"],
        "sat": fl.last_cmd["sat"],
        "att": (float(roll), float(pitch), float(yaw)),
        "vz": float(fl.vz),
        "failsafe": failsafe,
    }
    if timing is not None:
        recorded_ns = time.monotonic_ns()
        started_ns = timing.released_ns if loop_started_ns is None else loop_started_ns
        row.update(
            loop_scheduled_ns=timing.scheduled_ns,
            loop_released_ns=timing.released_ns,
            loop_lateness_ms=timing.lateness_ns * 1e-6,
            loop_missed=timing.missed_periods,
            loop_exec_ms=max(0, recorded_ns - started_ns) * 1e-6,
            loop_compute_ms=max(0, timing.released_ns - started_ns) * 1e-6,
            loop_post_ms=max(0, recorded_ns - timing.released_ns) * 1e-6,
        )
    if fl.shadow is not None:
        # ONE coherent worker-owned snapshot: estimator state + structured target +
        # guidance decision, all republished together each tick under the worker lock, so
        # the command loop can never pair a fresh state with an unrelated old target. The
        # command loop does NO perception/ESKF/guidance — it only reads and records.
        snapshot = fl.shadow.coherent_snapshot() if coherent is None else coherent
        s, target, decision = snapshot.state, snapshot.target, snapshot.decision
        fresh = (now_ns - s.last_fix_mono_ns) * 1e-9 if s.last_fix_mono_ns else float("nan")
        row.update(
            sh_anchor=s.anchor_gate,
            sh_init=int(s.initialised),
            sh_fresh_s=fresh,
            sh_p=s.p,
            sh_v=s.v,
            sh_euler=s.euler,
            sh_cov=s.cov_diag,
            sh_seq=s.sequence,
            sh_pub_ns=s.published_mono_ns,
        )
        _record_guidance(row, target, decision, now_ns)
        _record_next_target(row, snapshot.next_target)
    if authority is not None:
        row.update(
            auth_requested=int(authority.requested),
            auth_source=authority.source,
            auth_reason=authority.reason,
            auth_entered=int(authority.entered),
            auth_exited=int(authority.exited),
            auth_age=authority.decision_age_s,
            auth_elapsed=authority.authority_elapsed_s,
            auth_legacy=authority.legacy,
            auth_transition=(
                (float("nan"),) * 4 if authority.transition is None else authority.transition
            ),
            auth_selected=authority.selected,
            auth_slew=int(authority.slew_limited),
        )
    if visual is not None:
        _record_visual(row, visual)
    rec.add_control(**row)


def _record_visual(row: dict[str, object], visual: VisualHandoffDecision) -> None:
    """Record the visual-handoff decision + full status alongside each command (schema-v9).

    Log-only carrier: ``vh_selected`` is exactly what ``Flight.command`` was given, so live
    and replay decisions reproduce byte-for-byte from the recorded inputs."""
    s = visual.status
    row.update(
        vh_state=s.state.value,
        vh_reason=s.reason,
        vh_source=s.source,
        vh_active=int(s.active),
        vh_locked=s.locked_track_id,
        vh_in_track=s.input_track_id,
        vh_in_seq=s.input_sequence,
        vh_bearing=s.bearing_rad,
        vh_bearing_rate=s.bearing_rate_rps,
        vh_pred_bearing=s.predicted_bearing_rad,
        vh_cy_err=s.cy_err_px,
        vh_cy_rate=s.cy_rate_pxps,
        vh_dropout=s.dropout_age_s,
        vh_roll_tgt=s.roll_target_rad,
        vh_roll_est=s.roll_estimate_rad,
        vh_att_cov=s.att_cov_trace_rad2,
        vh_roll_valid=int(s.roll_valid),
        vh_in_state_s=s.time_in_state_s,
        vh_transition_s=s.transition_age_s,
        vh_safe_exit=s.safe_exit_reason,
        vh_chan="".join("1" if b else "0" for b in s.authoritative_channels),
        vh_entered=int(s.entered),
        vh_exited=int(s.exited),
        vh_slew=int(s.slew_limited),
        vh_succeeded=int(s.succeeded),
        vh_legacy=visual.legacy,
        vh_requested=(float("nan"),) * 4 if visual.requested is None else visual.requested,
        vh_selected=visual.selected,
        vh_terminate=int(s.terminate),
        vh_reacq_elapsed=s.reacquire_elapsed_s,
        vh_reacq_yaw=s.reacquire_yaw_since_seen_rad,
        vh_reacq_prev=s.reacquire_prev_track_id,
        vh_reacq_cand=s.reacquire_candidate_track_id,
        vh_reacq_hits=s.reacquire_candidate_hits,
        vh_reacq_resid=s.reacquire_continuity_residual_rad,
        vh_reacq_result=s.reacquire_result,
        vh_spd_active=int(s.speed_active),
        vh_spd_reason=s.speed_reason,
        vh_spd_eskf=s.eskf_speed_mps,
        vh_spd_velcov=s.vel_cov_trace,
        vh_spd_fixage=s.gate0_fix_age_s,
        vh_spd_closure=s.pnp_closure_mps,
        vh_spd_nsamp=s.pnp_closure_samples,
        vh_spd_resid=s.speed_residual_mps,
        vh_spd_err=s.speed_error_mps,
        vh_spd_pitch=s.speed_pitch_rate_rps,
        vh_spd_impulse=s.speed_pitch_impulse_rad,
        vh_spd_exit=s.exit_speed_mps,
        vh_spd_corner=int(s.corner_permitted),
        vh_ctrl_yaw=s.control_yaw_rad,
        vh_body_gyro=s.body_gyro_rps,
        vh_obs_bank=s.observer_bank_rad,
        vh_obs_src=s.observer_source,
        vh_obs_age=s.observer_age_s,
        vh_obs_pnp_acc=s.observer_pnp_accepted,
        vh_obs_pnp_rej=s.observer_pnp_rejected,
        vh_obs_pnp_innov=s.observer_pnp_innovation_rad,
        vh_cmd_integral=s.cmd_integral_bank_rad,
        vh_roll_fb_src=s.roll_feedback_source,
        vh_roll_arb=s.roll_arbitration,
        vh_shadow_degraded=int(s.shadow_roll_degraded),
        vh_active_degraded=int(s.active_roll_degraded),
        vh_bank_cap=s.bank_authority_cap_rad,
        vh_preturn_reject=s.preturn_reject,
        vh_lg_prov=s.logical_prov_id,
        vh_lg_raw=s.logical_raw_id,
        vh_lg_prev=s.logical_prev_raw_id,
        vh_lg_streak=s.logical_streak,
        vh_lg_az=s.logical_azimuth_rad,
        vh_lg_resid=s.logical_azimuth_residual_rad,
        vh_lg_gap=s.logical_gap_s,
        vh_lg_reacq=s.logical_reacquired,
        vh_lg_reason=s.logical_reason,
        vh_bank_speed=s.corner_bank_speed_mps,
        vh_bank_raw=s.corner_bank_raw_rad,
        vh_bank=s.corner_bank_rad,
    )


def _record_next_target(row: dict[str, object], target: StructuredTarget) -> None:
    """Record the worker's provisional next-gate identity (diagnostic-only)."""
    row.update(
        nt_valid=int(target.valid),
        nt_track=target.track_id,
        nt_race=target.race_gate_id,
        nt_hits=target.track_hits,
        nt_mode=target.output_mode,
        nt_bearing=target.bearing_rad,
        nt_slant=target.slant_range_m,
        nt_seq=target.sequence,
        nt_frame=target.frame_id,
        nt_sensor_ns=target.frame_sensor_ns,
        nt_arrival_ns=target.arrival_ns,
        nt_active=target.active_gate_at_receipt,
        nt_los=tuple(float(x) for x in target.los_body_unit),
        nt_conf=target.confidence,
        # schema-v9: identity/disposition + image geometry for exact visual-handoff replay
        nt_turn_sign=target.expected_turn_sign,
        nt_refused=int(target.association_refused),
        nt_wrong=int(target.wrong_turn),
        nt_oob=int(target.out_of_band),
        nt_metric=int(target.has_metric_pose),
        nt_cov=target.cov_trace,
        nt_disp=target.disposition,
        nt_cx=target.cx_px,
        nt_cy=target.cy_px,
        nt_area=target.area_px2,
        nt_visfrac=target.visible_fraction,
        nt_vroll=target.vision_roll_rad,
    )


def _record_guidance(
    row: dict[str, object],
    target: StructuredTarget,
    decision: GuidanceDecision,
    now_ns: int,
) -> None:
    """Record the worker's structured target + guidance decision beside the active command.

    STRICT SHADOW MODE: a pure READ of immutable worker-owned snapshots — no perception,
    ESKF, or guidance runs here. ``Flight.command`` output is byte-identical with or
    without this call (test-locked). ``sg_decage`` = decision age at this read, so a stalled
    worker's stale decision is visible (and treated as fail-closed in analysis)."""
    res = decision.result
    row["sg_status"] = res.status.value
    row["sg_mode"] = decision.mode
    row["sg_track"] = decision.track_id
    row["sg_active"] = decision.active_gate
    row["sg_anchor"] = decision.anchor_gate
    row["sg_decage"] = decision.age_s(now_ns)
    row["sg_seq"] = decision.decision_sequence
    row["sg_state_seq"] = decision.state_sequence
    row["sg_state_ns"] = decision.state_mono_ns
    row["sg_target_seq"] = decision.target_sequence
    row["sg_input_ns"] = decision.input_mono_ns
    row["sg_decision_ns"] = decision.decision_mono_ns
    row["sg_fix_age"] = decision.fix_age_s
    row["sg_transition_age"] = decision.transition_age_s
    row["sg_initialised"] = int(decision.initialised)
    # structured-target telemetry (separate from the legacy ctrl_tgt_* detector fields)
    row["st_valid"] = int(target.valid)
    row["st_track"] = target.track_id
    row["st_race"] = target.race_gate_id
    row["st_hits"] = target.track_hits
    row["st_mode"] = target.output_mode
    row["st_bearing"] = target.bearing_rad
    row["st_slant"] = target.slant_range_m
    row["st_disp"] = target.disposition
    row["st_oob"] = int(target.out_of_band)
    row["st_wrong"] = int(target.wrong_turn)
    row["st_refused"] = int(target.association_refused)
    row["st_seq"] = target.sequence
    row["st_frame"] = target.frame_id
    row["st_sensor_ns"] = target.frame_sensor_ns
    row["st_arrival_ns"] = target.arrival_ns
    row["st_active"] = target.active_gate_at_receipt
    row["st_los"] = tuple(float(x) for x in target.los_body_unit)
    row["st_metric"] = int(target.has_metric_pose)
    row["st_conf"] = target.confidence
    row["st_cov"] = target.cov_trace
    row["st_turn_sign"] = target.expected_turn_sign
    # schema-v9: image geometry for exact visual-handoff replay + vertical control
    row["st_cx"] = target.cx_px
    row["st_cy"] = target.cy_px
    row["st_area"] = target.area_px2
    row["st_visfrac"] = target.visible_fraction
    row["st_vroll"] = target.vision_roll_rad
    if res.setpoint is not None:
        thr = res.setpoint.collective_thrust
        rr, pr, yr = (float(x) for x in res.setpoint.body_rates_rps)
        row["sg_set"] = (thr, rr, pr, yr)
        row["sg_sat"] = "|".join(res.setpoint.saturation_flags)
    if res.requested_acceleration_ned_mps2 is not None:
        row["sg_acc"] = tuple(float(x) for x in res.requested_acceleration_ned_mps2)
    if res.velocity_error_ned_mps is not None:
        row["sg_verr"] = tuple(float(x) for x in res.velocity_error_ned_mps)
    if res.desired_heading_rad is not None:
        row["sg_heading"] = res.desired_heading_rad
    if res.closure_speed_mps is not None:
        row["sg_closure"] = res.closure_speed_mps
    if res.distance_m is not None:
        row["sg_dist"] = res.distance_m
    if res.fov_margin_rad is not None:
        row["sg_fov"] = res.fov_margin_rad


def _select_command(
    *,
    legacy: tuple[float, float, float, float],
    coherent: CoherentShadowSnapshot | None,
    released_ns: int,
    active_gate: int,
    transition_start_ns: int,
    authority_mux: auth.TransitionAuthorityMux,
    visual: vh.VisualHandoffController | None,
    training: bool,
    control_yaw_rad: float = float("nan"),
    control_roll_rad: float = float("nan"),
    control_pitch_rad: float = float("nan"),
    body_gyro_rps: tuple[float, float, float] = (float("nan"),) * 3,
    abort_reason: str = "",
) -> tuple[
    tuple[float, float, float, float],
    AuthoritySelection | None,
    VisualHandoffDecision | None,
]:
    """Pick the command to send: legacy, the old transition mux, or the visual handoff.

    Exactly one experimental path is ever active (CLI-enforced mutual exclusion). With
    neither engaged this returns the legacy command unchanged (byte-identical)."""
    if coherent is None:
        return legacy, None, None
    state = coherent.state
    if visual is not None:
        cov = state.cov_diag
        roll_cov = float(cov[6]) if len(cov) > 6 else float("nan")
        att_trace = float(sum(cov[6:9])) if len(cov) >= 9 else float("nan")
        vel_cov_trace = float(sum(cov[3:6])) if len(cov) >= 6 else float("nan")
        age = (released_ns - state.published_mono_ns) * 1e-9 if state.published_mono_ns else 1e9
        speed = math.hypot(float(state.v[0]), float(state.v[1]))
        fix_age = (released_ns - state.last_fix_mono_ns) * 1e-9 if state.last_fix_mono_ns else 1e9
        vis_input = vh.VisualHandoffInput(
            now_ns=released_ns,
            training=training,
            flag_enabled=True,
            active_gate=active_gate,
            anchor_gate=state.anchor_gate,
            active_target=coherent.target,
            provisional_target=coherent.next_target,
            eskf_roll_rad=float(state.euler[0]),
            eskf_roll_cov_rad2=roll_cov,
            eskf_att_cov_trace_rad2=att_trace,
            state_age_s=age,
            legacy_command=legacy,
            abort_reason=abort_reason,
            eskf_speed_mps=speed,
            eskf_vel_cov_trace=vel_cov_trace,
            eskf_fix_age_s=fix_age,
            eskf_initialised=bool(state.initialised),
            control_yaw_rad=control_yaw_rad,
            control_roll_rad=control_roll_rad,
            control_pitch_rad=control_pitch_rad,
            body_gyro_rps=body_gyro_rps,
        )
        vis_decision = visual.step(vis_input)
        return vis_decision.selected, None, vis_decision
    selection = authority_mux.select(
        now_ns=released_ns,
        transition_start_ns=transition_start_ns,
        active_gate=active_gate,
        anchor_gate=state.anchor_gate,
        state_sequence=state.sequence,
        target=coherent.target,
        decision=coherent.decision,
        legacy=legacy,
    )
    return selection.selected, selection, None


def run(
    fl: runtime.Flight,
    cfg: VQ2Config,
    *,
    transition_authority: bool = False,
    visual_handoff: bool = False,
) -> str:
    """One attempt of the phased mission. Returns the stop reason."""
    m = cfg.mission
    dt = 1.0 / m.cmd_hz
    t0_ns = time.monotonic_ns()
    t0 = t0_ns * 1e-9
    scheduler = DeadlineScheduler(m.cmd_hz, start_ns=t0_ns)
    authority_mux = auth.TransitionAuthorityMux(enabled=transition_authority)
    # Training-only visual handoff. Mutually exclusive with the transition authority mux
    # (CLI-enforced); when engaged the mux stays disabled and the visual controller owns
    # the (roll/yaw overlay in PRETURN, full) command selection in the gate-0→1 window.
    visual = vh.VisualHandoffController() if visual_handoff else None
    training = cfg.name == "training"
    transition_start_ns = 0
    phase = "G0"
    phase_t = 0.0  # phase entry time (absolute)
    gate_prev = -1
    held: gates.GateTarget | None = None
    held_at = 0.0
    prev_cy: float | None = None
    prev_cy_t = 0.0
    cy_rate = 0.0
    scan_angle = 0.0  # integrated |heading turned| since SCAN entry (rad)
    pass_until = 0.0
    trace_i = 0
    trace_next = 0.0
    reason = "timeout"
    if cfg.run.trace:
        _archive_traces()
    print(f"[fly2] STAB {m.stab_secs}s, then G0 thread -> brake/climb/scan corner plan")
    preturn_rejects: Counter[str] = Counter()  # why PRETURN did not engage (diagnostics)
    while True:
        loop_started_ns = time.monotonic_ns()
        now = loop_started_ns * 1e-9
        if now - t0 >= cfg.run.secs:
            break
        sensors = fl.step_sensors()
        if sensors is None:
            scheduler.wait_next()
            continue
        _, gyro = sensors
        fl.att.in_flight = True
        stop = fl.abort_reason()
        if stop:
            reason = stop
            break
        if now - t0 < m.stab_secs:  # settle the attitude filter before doing anything
            rr, pr = fl.level_rates(0.0)
            legacy = (runtime.HOVER_THRUST, rr, pr, 0.0)
            tick = scheduler.wait_next()
            coherent = fl.shadow.coherent_snapshot() if fl.shadow is not None else None
            selected, selection, vis_decision = _select_command(
                legacy=legacy,
                coherent=coherent,
                released_ns=tick.released_ns,
                active_gate=gate_prev,
                transition_start_ns=transition_start_ns,
                authority_mux=authority_mux,
                visual=visual,
                training=training,
                control_yaw_rad=float(fl.att.euler[2]),
                control_roll_rad=float(fl.att.euler[0]),
                control_pitch_rad=float(fl.att.euler[1]),
                body_gyro_rps=fl.body_gyro,
            )
            fl.command(*selected)
            _record_control(
                fl,
                "STAB",
                gate_prev,
                None,
                float("nan"),
                timing=tick,
                loop_started_ns=loop_started_ns,
                coherent=coherent,
                authority=selection,
                visual=vis_decision,
            )
            continue

        with fl.sh.lock:
            frame = fl.sh.frame
            gate_now = fl.sh.active_gate
        # NOTE: the shadow ESKF/perception runs entirely on ShadowWorker's own thread
        # (fed by the receiver queues); the command loop below does NO shadow work.
        if gate_now != gate_prev:
            print(f"  [gate] active_gate -> {gate_now}  t={now - t0:.1f}s  (phase {phase})")
            if gate_prev >= 0:  # real pass (not the initial latch): corner plan begins
                phase, phase_t = "BRAKE", now
                held = None
                prev_cy = None
                if gate_prev == 0 and gate_now == 1:
                    transition_start_ns = loop_started_ns
            gate_prev = gate_now

        # -- perception: one detect pass, phase-appropriate association --------
        held_fresh = held is not None and now - held_at < m.target_hold_s
        dets = gates.detect(frame) if frame is not None else []
        tgt: gates.GateTarget | None = None
        if phase in ("G0", "THREAD"):
            tgt = runtime.pick_target(dets, held, held_fresh)
            if tgt is not None and not runtime.target_switch_ok(held, held_fresh, tgt):
                tgt = None  # proven AV guard: a point-blank target does not teleport
        elif phase == "SCAN" and dets:
            cands = [t for t in dets if map_accept(t, scan_angle, m)]
            if cands:
                tgt = min(cands, key=lambda t: t.pose_cam[2])  # type: ignore[index]
                print(
                    f"  [scan] gate acquired t={now - t0:.1f}s z={tgt.pose_cam[2]:.1f}m "  # type: ignore[index]
                    f"b={math.degrees(tgt.bearing):+.0f}deg after {math.degrees(scan_angle):.0f}deg turn"
                )
                phase, phase_t = "THREAD", now
        if tgt is not None:
            if prev_cy is not None and now > prev_cy_t:
                cy_rate = (tgt.cy - prev_cy) / (now - prev_cy_t)
            prev_cy, prev_cy_t = tgt.cy, now
            held, held_at = tgt, now
            held_fresh = True

        # -- phase control laws -------------------------------------------------
        pitch_target = 0.0
        roll_target = 0.0
        yr = 0.0
        base_thr = runtime.HOVER_THRUST
        if phase == "G0":
            # launch-assisted thread: level + yaw/row centering; commit forward
            # pitch only when aligned in BOTH axes (proven aim-then-go).
            if held_fresh and held is not None:
                yr = float(np.clip(m.kp_yaw * held.bearing, -runtime.MAX_YAW, runtime.MAX_YAW))
                aligned = (
                    abs(held.bearing) < m.align_bear
                    and abs(held.cy - runtime.CY_TARGET) < m.cy_band
                )
                pitch_target = m.pitch_fwd if aligned else 0.0
                base_thr = runtime.vertical_thrust(held, cy_rate)
            else:
                base_thr = runtime.vertical_thrust(None)
        elif phase == "BRAKE":
            if now - phase_t < m.brake_arrest_s:  # blind flare: kill speed, arrest sink
                pitch_target = m.brake_pitch
                base_thr = runtime.HOVER_THRUST + m.brake_arrest_thr
            else:  # tail: nose down enough to see the ribbon; height goes closed-loop
                pitch_target = m.brake_tail_pitch
                line = gates.detect_line(frame) if frame is not None else None
                base_thr = (
                    runtime.line_thrust(line, 0.0) if line is not None else runtime.HOVER_THRUST
                )
            if now - phase_t > m.brake_secs:
                phase, phase_t = "SCAN", now
                scan_angle = 0.0
                print(f"  [brake] done t={now - t0:.1f}s -> SCAN right")
        elif phase == "SCAN":
            yr = m.scan_yaw  # map says next gate is RIGHT of every flown leg so far
            scan_angle += float(gyro[2]) * dt  # signed heading turned (+ = right)
            pitch_target = m.scan_hold_pitch
            line = gates.detect_line(frame) if frame is not None else None
            # the ribbon = world-position height reference (vz 0: fl.vz is fiction here)
            base_thr = runtime.line_thrust(line, 0.0) if line is not None else runtime.HOVER_THRUST
        elif phase == "THREAD":
            if now < pass_until:
                pitch_target, yr = m.pitch_fwd, 0.0  # hands-off through the gate plane
            elif held_fresh and held is not None:
                yr = float(np.clip(m.kp_yaw * held.bearing, -runtime.MAX_YAW, runtime.MAX_YAW))
                # bank toward the gate: bends the velocity vector (yaw cannot)
                roll_target = float(
                    np.clip(
                        m.kp_roll_bear * held.bearing,
                        -runtime.MAX_ROLL_TGT,
                        runtime.MAX_ROLL_TGT,
                    )
                )
                close = held.area > m.pass_area or (
                    held.pose_cam is not None and held.pose_cam[2] < m.pass_range_m
                )
                if close:
                    bear_px = math.atan2(held.cx - 320.0, 320.0)
                    if abs(bear_px) < m.align_bear and abs(held.cy - runtime.CY_TARGET) < m.cy_band:
                        pass_until = now + m.pass_commit_s  # centred: commit straight
                        pitch_target, yr, roll_target = m.pitch_fwd, 0.0, 0.0
                    else:
                        pitch_target = m.recenter_pitch  # off-centre: level brake,
                        #   keep the gate in frame, let yaw/roll re-centre
                else:
                    aligned = (
                        abs(held.bearing) < m.align_bear
                        and abs(held.cy - runtime.CY_TARGET) < m.cy_band
                    )
                    # far + off-centre: LEVEL, not the recenter brake — at ~10 m with
                    # near-zero closure the +0.15 nose-up just accelerates us backward
                    # and feeds the orbit (batch 7: area 9375 -> 7912 = receding while
                    # "recentering"). The brake belongs only in the CLOSE branch where
                    # there is real closure speed to kill.
                    pitch_target = m.pitch_fwd if aligned else 0.0
                base_thr = runtime.vertical_thrust(held, cy_rate)
            elif now - held_at < m.target_hold_s + m.lost_brake_s:
                pitch_target = m.scan_hold_pitch  # brief dropout: brake level, wait
            else:
                phase, phase_t = "SCAN", now  # lost it: resume the mapped scan
                scan_angle = 0.0
                print(f"  [thread] target lost t={now - t0:.1f}s -> re-SCAN")

        if phase == "THREAD" and now < pass_until:
            thr = fl.tilt_comp(runtime.HOVER_THRUST)  # frozen vertical through the plane
        else:
            thr = fl.tilt_comp(base_thr)
        if cfg.run.trace and now - t0 >= trace_next:
            runtime._trace_frame(
                fl, TRACE_DIR, now - t0, trace_i, held if held_fresh else None, gate_prev
            )
            trace_i += 1
            trace_next = (now - t0) + TRACE_DT
        rr, pr = fl.level_rates(pitch_target, roll_target)
        legacy = (thr, rr, pr, yr)
        # Release the command at its absolute deadline AFTER perception/control compute.
        # This keeps send spacing stable instead of allowing variable compute time to move
        # the command boundary around the nominal 60 Hz period.
        tick = scheduler.wait_next()
        coherent = fl.shadow.coherent_snapshot() if fl.shadow is not None else None
        selected, selection, vis_decision = _select_command(
            legacy=legacy,
            coherent=coherent,
            released_ns=tick.released_ns,
            active_gate=gate_now,
            transition_start_ns=transition_start_ns,
            authority_mux=authority_mux,
            visual=visual,
            training=training,
            control_yaw_rad=float(fl.att.euler[2]),
            control_roll_rad=float(fl.att.euler[0]),
            control_pitch_rad=float(fl.att.euler[1]),
            body_gyro_rps=fl.body_gyro,
        )
        fl.command(*selected)
        _record_control(
            fl,
            phase,
            gate_now,
            held if held_fresh else None,
            (now - held_at) if held is not None else float("nan"),
            timing=tick,
            loop_started_ns=loop_started_ns,
            coherent=coherent,
            authority=selection,
            visual=vis_decision,
        )
        # The visual handoff bounds SAFE_EXIT to safe_exit_hold_s and then asks the mission to
        # end the attempt (it must not keep sending SAFE_EXIT thrust for seconds). Existing
        # collision/tilt/feed-loss aborts above still take priority (they break earlier).
        if vis_decision is not None:
            st = vis_decision.status
            if st.preturn_reject:
                preturn_rejects[st.preturn_reject] += 1
            if st.terminate:
                reason = f"visual_handoff:{st.safe_exit_reason or 'safe_exit'}"
                break
    fl.command(0.05, 0.0, 0.0, 0.0)
    print(f"[fly2] stop: {reason} (phase {phase}); {fl.summary()}")
    if preturn_rejects:
        top = ", ".join(f"{k}={v}" for k, v in preturn_rejects.most_common(6))
        print(f"[fly2] PRETURN eligibility rejects: {top}")
    return reason


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="VQ2 phased mission runner.")
    ap.add_argument(
        "--config",
        default="training",
        help="deployment config: training | qualification | path to a TOML",
    )
    ap.add_argument("--secs", type=float, default=None, help="attempt duration cap (s)")
    ap.add_argument("--attempts", type=int, default=None)
    ap.add_argument(
        "--record",
        action="store_true",
        help="record the shadow IMU+PnP log (.npz) for offline ESKF replay (log-only)",
    )
    ap.add_argument(
        "--save-frames",
        action="store_true",
        help="also dump each shadow-processed RAW frame to .local/runs/vq2_shadow/frames_<ts>/ "
        "for offline perception analysis (diagnostic-only; implies --record)",
    )
    ap.add_argument(
        "--experimental-transition-authority",
        action="store_true",
        help="Training-only: allow at most 2 s of gate-1 TRANSITION_ACQUIRE authority",
    )
    ap.add_argument(
        "--experimental-visual-handoff",
        action="store_true",
        help="Training-only: enable the visual gate-0→1 handoff controller (PRETURN roll/yaw "
        "overlay + visual right-turn replacing BRAKE/SCAN). REQUIRES --record so every "
        "experimental-authority attempt leaves a replayable recording. Mutually exclusive with "
        "--experimental-transition-authority",
    )
    args = ap.parse_args(argv)
    # An experimental-authority attempt MUST be recorded (no un-diagnosable authority runs).
    if args.experimental_visual_handoff and not (args.record or args.save_frames):
        ap.error(
            "--experimental-visual-handoff requires --record (every authority attempt must be recorded)"
        )
    with contextlib.suppress(Exception):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[union-attr]

    cfg = vq2_config.load(args.config)
    try:
        transition_authority = _validate_transition_authority(
            cfg, args.experimental_transition_authority
        )
        visual_handoff = _validate_visual_handoff(
            cfg, args.experimental_visual_handoff, args.experimental_transition_authority
        )
    except ValueError as exc:
        ap.error(str(exc))
    overrides: dict[str, object] = {}
    if args.secs is not None:
        overrides["secs"] = args.secs
    if args.attempts is not None:
        overrides["attempts"] = args.attempts
    if overrides:
        cfg = vq2_config.replace_run(cfg, **overrides)
    print(f"[fly2] config '{cfg.name}' (trace={'on' if cfg.run.trace else 'off'})")

    print(f"Connecting to {cfg.run.mavlink_endpoint} … (calibrated rates: gain {cc.RATE_GAIN})")
    conn = mavutil.mavlink_connection(cfg.run.mavlink_endpoint)
    conn.wait_heartbeat()
    tgt_sys, tgt_comp = conn.target_system, conn.target_component
    print(f"Heartbeat OK. target_system={tgt_sys}")

    sh = cc.Shared()
    lock = threading.Lock()
    threading.Thread(target=cc.mav_rx, args=(conn, sh), daemon=True).start()
    threading.Thread(target=cc.vision_rx, args=(sh,), daemon=True).start()
    time.sleep(0.5)

    try:
        for i in range(cfg.run.attempts):
            if cfg.run.attempts > 1:
                print(f"\n=== attempt {i + 1}/{cfg.run.attempts} ===")
            if not cc.wait_fresh(sh, conn, tgt_sys, tgt_comp, lock):
                print("NO FRESH START — is a run live in the Training window?")
                return
            sh.clear_shadow_queues()  # explicit attempt boundary: drop pre-attempt IMU/frames
            fl = runtime.Flight(sh, conn, tgt_sys, tgt_comp, lock)
            recorder = RunRecorder() if (args.record or args.save_frames) else None
            ts_run = time.strftime("%Y%m%d_%H%M%S")
            frame_dir = (
                Workspace.locate().runs / "vq2_shadow" / f"frames_{ts_run}"
                if args.save_frames
                else None
            )
            worker = ShadowWorker(sh, recorder, frame_dir=frame_dir)  # log-only, own thread
            fl.shadow = worker
            worker.start()
            try:
                stop_reason = run(
                    fl,
                    cfg,
                    transition_authority=transition_authority,
                    visual_handoff=visual_handoff,
                )
            finally:
                worker.stop()  # request finish; wait_done confirms full termination below
            terminated = worker.wait_done()
            print(f"[fly2] {fl.summary()}")  # control estimators only
            if not terminated:
                # do NOT read worker state or save: the thread may still be mutating them
                print("[shadow] ERROR: worker did not terminate in time; recording NOT saved")
            else:
                print(f"[fly2] {worker.summary()}")  # shadow filter-health (log-only)
                if recorder is not None and recorder.n_imu:
                    ts = time.strftime("%Y%m%d_%H%M%S")
                    path = recorder.save(Workspace.locate().runs / "vq2_shadow" / f"run_{ts}.npz")
                    print(
                        f"[shadow] recorded {recorder.n_imu} IMU + {recorder.n_fix} fixes → {path}"
                    )
            if stop_reason == "FINISHED":
                print("🎉 course FINISHED")
                break
    finally:
        sh.stop.set()
        time.sleep(0.4)


if __name__ == "__main__":
    main()
