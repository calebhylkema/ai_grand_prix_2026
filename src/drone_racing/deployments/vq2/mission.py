"""Map-based mission executor (the 2026-07-09 strategic pivot) — the VQ2 runner.

The reactive controller grew into a branch stack: every loop it re-decided what to
chase from instantaneous detections, and the AX-batch forensics caught it latching
three DIFFERENT gates in 0.3 s (PnP z 27.2 -> 21.2 -> 17.1 m — an impossible 35 m/s
"closure") because nothing in the stack knows where the gates actually are. The
course is FIXED and attempts are unlimited, so fly the course from a MAP (mined
offline by mapping.py from saved traces), and use the map to gate data association.

Course map v0 (mapping.py over 6 trace archives, 3 independent co-visible pairs):

    gate 0:  8.6 m ahead of the pad hover, on the initial heading, up ~-0.1 m
    gate 1:  9.2 m fwd / 6.4 m right of gate 0 (34.8 deg right, 11.2 m horizontal),
             up +0.6 m — a MODEST climb, not the +2 m guessed from tilt-bad frames
    imposters: the "gate 1" sightings after crossing all read z 21-23 m -> that is
             gate 2 (second leg); REJECT any gate-1 candidate with z outside 3-16 m

Why the old stack never SAW gate 1 after crossing: the camera is 20 deg DOWN, so frame
row 0 is only ~9 deg above body-horizontal — a gate 0.6 m ABOVE us at 8-10 m is out of
frame (top). To put it at CY_TARGET (row 130) from 8 m we must be ~2 m ABOVE its
center. Hence the mission shape per corner: BRAKE hard -> CLIMB -> SCAN toward the
mapped bearing, accepting only detections inside the mapped range band -> THREAD with
the proven CY servo (which naturally descends onto the center as range shrinks).

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
from typing import TYPE_CHECKING

import numpy as np
from pymavlink import mavutil

from drone_racing.deployments.vq2 import calibration as cc
from drone_racing.deployments.vq2 import config as vq2_config
from drone_racing.deployments.vq2 import perception as gates
from drone_racing.deployments.vq2 import runtime

if TYPE_CHECKING:
    from drone_racing.deployments.vq2.config import MissionParams, VQ2Config

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
    - RANGE: a PnP pose inside the mapped band (post-crossing imposters read 21-23 m,
      the real gate 1 reads ~6-11 m).
    - DIRECTION: heading-turned-so-far + current bearing inside the mapped corner
      window (gate 1 is 35-70 deg RIGHT of the exit heading depending on overshoot).
      This rejects the left-side return-leg gates that pass the range test.
    """
    if tgt.pose_cam is None:
        return False
    if not m.accept_z[0] <= tgt.pose_cam[2] <= m.accept_z[1]:
        return False
    total = scan_angle + tgt.bearing  # rad right of the leg-exit heading
    return m.accept_turn[0] <= total <= m.accept_turn[1]


def run(fl: runtime.Flight, cfg: VQ2Config) -> str:
    """One attempt of the phased mission. Returns the stop reason."""
    m = cfg.mission
    dt = 1.0 / m.cmd_hz
    t0 = time.time()
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
    while time.time() - t0 < cfg.run.secs:
        sensors = fl.step_sensors()
        if sensors is None:
            time.sleep(0.002)
            continue
        _, gyro = sensors
        fl.att.in_flight = True
        stop = fl.abort_reason()
        if stop:
            reason = stop
            break
        now = time.time()
        if now - t0 < m.stab_secs:  # settle the attitude filter before doing anything
            rr, pr = fl.level_rates(0.0)
            fl.command(runtime.HOVER_THRUST, rr, pr, 0.0)
            time.sleep(dt)
            continue

        with fl.sh.lock:
            frame = fl.sh.frame
            gate_now = fl.sh.active_gate
        if gate_now != gate_prev:
            print(f"  [gate] active_gate -> {gate_now}  t={now - t0:.1f}s  (phase {phase})")
            if gate_prev >= 0:  # real pass (not the initial latch): corner plan begins
                phase, phase_t = "BRAKE", now
                held = None
                prev_cy = None
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
        fl.command(thr, rr, pr, yr)
        time.sleep(dt)
    fl.command(0.05, 0.0, 0.0, 0.0)
    print(f"[fly2] stop: {reason} (phase {phase}); {fl.summary()}")
    return reason


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="VQ2 map-and-plan mission runner.")
    ap.add_argument(
        "--config",
        default="training",
        help="deployment config: training | qualification | path to a TOML",
    )
    ap.add_argument("--secs", type=float, default=None, help="attempt duration cap (s)")
    ap.add_argument("--attempts", type=int, default=None)
    args = ap.parse_args(argv)
    with contextlib.suppress(Exception):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[union-attr]

    cfg = vq2_config.load(args.config)
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
            fl = runtime.Flight(sh, conn, tgt_sys, tgt_comp, lock)
            if run(fl, cfg) == "FINISHED":
                print("🎉 course FINISHED")
                break
    finally:
        sh.stop.set()
        time.sleep(0.4)


if __name__ == "__main__":
    main()
