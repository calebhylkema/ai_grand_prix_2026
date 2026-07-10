"""Map-based mission executor (the 2026-07-09 strategic pivot).

fly.py grew into a reactive branch stack: every loop it re-decides what to chase from
instantaneous detections, and the AX-batch forensics caught it latching three DIFFERENT
gates in 0.3 s (PnP z 27.2 -> 21.2 -> 17.1 m — an impossible 35 m/s "closure") because
nothing in the stack knows where the gates actually are. The course is FIXED and
attempts are unlimited, so fly the course from a MAP (mined offline by course_map.py
from saved traces), and use the map to gate data association.

Course map v0 (course_map.py over 6 trace archives, 3 independent co-visible pairs):

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

Reuses fly.py's proven infrastructure unchanged: Flight (calibrated rates, estimators,
failsafes), vertical_thrust (CY servo + descend guards), _trace_frame, TRACE_DIR
archive scheme. fly.py itself stays untouched as the fallback controller.

Usage (sim in Training, run live):
    uv run python scripts/vq2/fly2.py --secs 45 --attempts 4
"""

from __future__ import annotations

import argparse
import contextlib
import math
import sys
import threading
import time

import control_check as cc
import fly
import gates
import numpy as np
from pymavlink import mavutil

cc.CALIBRATE = True

CMD_HZ = 60.0
STAB_SECS = 1.5  # pre-launch attitude-filter settle (same as fly.py)

# --- G0: launch-assisted gate-0 thread (the platform catapults us at the gate;
#     the job is level attitude + yaw/row centering — proven 100% in fly.py) ---
KP_YAW = 1.5  # yaw rate per rad of bearing (sign: +bearing = gate right = +yaw)
PITCH_FWD = -0.05  # gentle forward commit when aligned
ALIGN_BEAR = 0.22  # rad — forward pitch only when roughly nose-on
CY_BAND = 60.0  # px — and vertically centred (aim-then-go, from fly.py)

# --- BRAKE: kill the launch momentum right after a gate pass (arrest-then-look) ---
BRAKE_SECS = 1.6
BRAKE_PITCH = +0.35  # rad nose-up ~ 3.6 m/s^2 decel (fly.py's +0.15 was too weak for
#                      the ~6 m/s catapult exit; unlike fly.py we are NOT trying to
#                      keep chasing anything while braking, so a deep flare is safe)
BRAKE_ARREST_S = 1.0  # blind-flare portion: the gate-0 crossing carries a REAL sink
#   (~-2 m/s: batches 5/6 ended the 1.6 s hover-thrust brake at FLOOR level, ~3 m
#   below the crossing). While the nose-up flare hides the world, spend a bounded
#   climb impulse to arrest it. 1.0 s at +0.04 ~ +2 m/s — sized to the observed sink,
#   short enough that an overcorrection stays ~1 m (batch 4's ceiling came from 1.6 s
#   of this UNDAMPED plus a 3 s feedback-free coast; the line servo now backstops).
BRAKE_ARREST_THR = +0.04  # thrust above hover during the arrest
BRAKE_TAIL_PITCH = +0.10  # brake tail: nose back down enough (camera 20deg-down sees
#   ~14deg below level again) that the cyan ribbon returns to view and line_thrust
#   owns the height CLOSED-LOOP for the last 0.6 s while still bleeding speed.
# --- SCAN: yaw toward the mapped next-gate bearing until a map-valid detection ---
SCAN_YAW = 0.5  # rad/s
SCAN_HOLD_PITCH = 0.0  # LEVEL while turning. Batch 7: +0.05 nose-up over the ~3.5 s
#   scan accelerated us BACKWARD ~1.7 m/s; after the ~70deg right turn that drift is
#   lateral to the acquired gate — the THREAD orbit signature (bearing +34 -> +42 at
#   MAX_YAW with the bank clamped) traced straight back to it. The brake already
#   killed the launch speed; there is no residual drift left worth bleeding.

# --- corner vertical: the RACING LINE is the height reference (batch 6) ---
# Batches 2-5 proved the corner's vertical cannot be flown open-loop OR on the IMU:
#   2/4: +0.04 brake climb undamped -> coasted to the CEILING grid;
#   3:   +0.02 never arrested the crossing sink -> floor clutter;
#   5:   vz-null on the vision-anchored integrator -> the launch had already pinned
#        vz at -3 (climb fiction), 3 posed sightings x gain 0.25 couldn't fix it, the
#        "honest" window blessed garbage and the brake dutifully FLEW US INTO THE FLOOR.
# The batch-5 death frames handed us the answer: the cyan racing-line ribbon is in
# view through the whole corner and threads gate 1. line_thrust (fly.py-proven: capped
# +/-0.06/0.04, far-low-path floor guard, aim-row not far-tip) is POSITION feedback
# from the world — no integrator to poison. vz arg stays 0.0: fl.vz is fiction here
# (the AU lesson: a poisoned -3 "climb" turns the damper into a -0.06 down-shove).

# --- THREAD: servo onto the acquired gate (gate-0 machinery, retargeted) ---
KP_ROLL_BEAR = 0.30  # bank toward the gate (fly.py-proven): yaw only points the
#   camera — banking is what bends the VELOCITY vector. Batch 2 orbit signature:
#   bearing grew +28 -> +35 deg while yawing right at 0.9 rad/s = lateral drift that
#   yaw could never cancel; arrived at the plane displaced and moving sideways.
RECENTER_PITCH = +0.15  # close-but-off-centre brake (fly.py TURN_BRAKE_PITCH). Batch
#   2 used the deep +0.35 flare here: pitching the camera up ~20 deg (~120 px) pushed
#   the low-in-frame gate out the BOTTOM of the image — self-inflicted target loss.
ACCEPT_Z = (3.0, 13.0)  # m — mapped gate-1 range band (reads ~6-11 m from the corner);
#   the second-leg imposters read 21-23 m. Batch 4 (2026-07-09): with the band at 16,
#   all 4 runs latched an ELEVATED far gate at z 15.1-15.6 / cy~61 — level with the
#   runaway-climb altitude, i.e. NOT gate 1 (which sits ~4 m below the ceiling grid) —
#   and the CY servo dragged us into the ceiling chasing it.
ACCEPT_TURN = (math.radians(25.0), math.radians(110.0))  # mapped corner window: the
#   candidate's absolute direction (heading turned since scan start + its current
#   bearing) must be RIGHT of the exit heading by roughly the mapped 35-70 deg.
#   Batch 1 (2026-07-09): a range-only gate latched a return-leg pair at -19 deg LEFT
#   (z 12-14 m, inside the band) on all 4 attempts and flew the pillar corridor into
#   ENV/threat2 — the real gate 1 sat just outside the right FOV edge at ~+55 deg.
TARGET_HOLD_S = 0.8  # sighting persistence (same as fly.py)
LOST_BRAKE_S = 0.8  # target stale mid-thread: brake level this long, then re-SCAN
PASS_AREA = 15000.0  # px^2 — close enough to commit straight through (fly.py value)
PASS_RANGE_M = 3.5
PASS_COMMIT_S = 1.2  # hands-off straight-and-level budget through the gate plane

TRACE_DT = fly.TRACE_DT
TRACE_DIR = fly.TRACE_DIR


def _archive_traces() -> None:
    """Same keep-last-6 archive scheme as fly.py (shared TRACE_DIR + tooling)."""
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


def map_accept(tgt: gates.GateTarget, scan_angle: float) -> bool:
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
    if not ACCEPT_Z[0] <= tgt.pose_cam[2] <= ACCEPT_Z[1]:
        return False
    total = scan_angle + tgt.bearing  # rad right of the leg-exit heading
    return ACCEPT_TURN[0] <= total <= ACCEPT_TURN[1]


def run(fl: fly.Flight, secs: float) -> str:
    """One attempt of the phased mission. Returns the stop reason."""
    dt = 1.0 / CMD_HZ
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
    _archive_traces()
    print(f"[fly2] STAB {STAB_SECS}s, then G0 thread -> brake/climb/scan corner plan")
    while time.time() - t0 < secs:
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
        if now - t0 < STAB_SECS:  # settle the attitude filter before doing anything
            rr, pr = fl.level_rates(0.0)
            fl.command(fly.HOVER_THRUST, rr, pr, 0.0)
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
        held_fresh = held is not None and now - held_at < TARGET_HOLD_S
        dets = gates.detect(frame) if frame is not None else []
        tgt: gates.GateTarget | None = None
        if phase in ("G0", "THREAD"):
            tgt = fly.pick_target(dets, held, held_fresh)
            if tgt is not None and not fly.target_switch_ok(held, held_fresh, tgt):
                tgt = None  # proven AV guard: a point-blank target does not teleport
        elif phase == "SCAN" and dets:
            cands = [t for t in dets if map_accept(t, scan_angle)]
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
        base_thr = fly.HOVER_THRUST
        if phase == "G0":
            # launch-assisted thread: level + yaw/row centering; commit forward
            # pitch only when aligned in BOTH axes (proven aim-then-go).
            if held_fresh and held is not None:
                yr = float(np.clip(KP_YAW * held.bearing, -fly.MAX_YAW, fly.MAX_YAW))
                aligned = abs(held.bearing) < ALIGN_BEAR and abs(held.cy - fly.CY_TARGET) < CY_BAND
                pitch_target = PITCH_FWD if aligned else 0.0
                base_thr = fly.vertical_thrust(held, cy_rate)
            else:
                base_thr = fly.vertical_thrust(None)
        elif phase == "BRAKE":
            if now - phase_t < BRAKE_ARREST_S:  # blind flare: kill speed, arrest sink
                pitch_target = BRAKE_PITCH
                base_thr = fly.HOVER_THRUST + BRAKE_ARREST_THR
            else:  # tail: nose down enough to see the ribbon; height goes closed-loop
                pitch_target = BRAKE_TAIL_PITCH
                line = gates.detect_line(frame) if frame is not None else None
                base_thr = fly.line_thrust(line, 0.0) if line is not None else fly.HOVER_THRUST
            if now - phase_t > BRAKE_SECS:
                phase, phase_t = "SCAN", now
                scan_angle = 0.0
                print(f"  [brake] done t={now - t0:.1f}s -> SCAN right")
        elif phase == "SCAN":
            yr = SCAN_YAW  # map says next gate is RIGHT of every flown leg so far
            scan_angle += float(gyro[2]) * dt  # signed heading turned (+ = right)
            pitch_target = SCAN_HOLD_PITCH
            line = gates.detect_line(frame) if frame is not None else None
            # the ribbon = world-position height reference (vz 0: fl.vz is fiction here)
            base_thr = fly.line_thrust(line, 0.0) if line is not None else fly.HOVER_THRUST
        elif phase == "THREAD":
            if now < pass_until:
                pitch_target, yr = PITCH_FWD, 0.0  # hands-off through the gate plane
            elif held_fresh and held is not None:
                yr = float(np.clip(KP_YAW * held.bearing, -fly.MAX_YAW, fly.MAX_YAW))
                # bank toward the gate: bends the velocity vector (yaw cannot)
                roll_target = float(
                    np.clip(KP_ROLL_BEAR * held.bearing, -fly.MAX_ROLL_TGT, fly.MAX_ROLL_TGT)
                )
                close = held.area > PASS_AREA or (
                    held.pose_cam is not None and held.pose_cam[2] < PASS_RANGE_M
                )
                if close:
                    bear_px = math.atan2(held.cx - 320.0, 320.0)
                    if abs(bear_px) < ALIGN_BEAR and abs(held.cy - fly.CY_TARGET) < CY_BAND:
                        pass_until = now + PASS_COMMIT_S  # centred: commit straight
                        pitch_target, yr, roll_target = PITCH_FWD, 0.0, 0.0
                    else:
                        pitch_target = RECENTER_PITCH  # off-centre: level brake,
                        #   keep the gate in frame, let yaw/roll re-centre
                else:
                    aligned = (
                        abs(held.bearing) < ALIGN_BEAR and abs(held.cy - fly.CY_TARGET) < CY_BAND
                    )
                    # far + off-centre: LEVEL, not the recenter brake — at ~10 m with
                    # near-zero closure the +0.15 nose-up just accelerates us backward
                    # and feeds the orbit (batch 7: area 9375 -> 7912 = receding while
                    # "recentering"). The brake belongs only in the CLOSE branch where
                    # there is real closure speed to kill.
                    pitch_target = PITCH_FWD if aligned else 0.0
                base_thr = fly.vertical_thrust(held, cy_rate)
            elif now - held_at < TARGET_HOLD_S + LOST_BRAKE_S:
                pitch_target = SCAN_HOLD_PITCH  # brief dropout: brake level, wait
            else:
                phase, phase_t = "SCAN", now  # lost it: resume the mapped scan
                scan_angle = 0.0
                print(f"  [thread] target lost t={now - t0:.1f}s -> re-SCAN")

        if phase == "THREAD" and now < pass_until:
            thr = fl.tilt_comp(fly.HOVER_THRUST)  # frozen vertical through the plane
        else:
            thr = fl.tilt_comp(base_thr)
        if now - t0 >= trace_next:
            fly._trace_frame(
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


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--secs", type=float, default=45.0, help="attempt duration cap (s)")
    ap.add_argument("--attempts", type=int, default=1)
    args = ap.parse_args()
    with contextlib.suppress(Exception):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    print(f"Connecting to {cc.MAVLINK_ENDPOINT} … (calibrated rates: gain {cc.RATE_GAIN})")
    conn = mavutil.mavlink_connection(cc.MAVLINK_ENDPOINT)
    conn.wait_heartbeat()
    tgt_sys, tgt_comp = conn.target_system, conn.target_component
    print(f"Heartbeat OK. target_system={tgt_sys}")

    sh = cc.Shared()
    lock = threading.Lock()
    threading.Thread(target=cc.mav_rx, args=(conn, sh), daemon=True).start()
    threading.Thread(target=cc.vision_rx, args=(sh,), daemon=True).start()
    time.sleep(0.5)

    try:
        for i in range(args.attempts):
            if args.attempts > 1:
                print(f"\n=== attempt {i + 1}/{args.attempts} ===")
            if not cc.wait_fresh(sh, conn, tgt_sys, tgt_comp, lock):
                print("NO FRESH START — is a run live in the Training window?")
                return
            fl = fly.Flight(sh, conn, tgt_sys, tgt_comp, lock)
            if run(fl, args.secs) == "FINISHED":
                print("🎉 course FINISHED")
                break
    finally:
        sh.stop.set()
        time.sleep(0.4)


if __name__ == "__main__":
    main()
