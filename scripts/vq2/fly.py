"""Fresh minimal VQ2 controller built on CALIBRATED rate commands.

This is the clean rebuild after the 2026-07-01 breakthrough: the sim's rate loop is
INVERTED and ~2.5x scaled (actual_gyro = RATE_GAIN * commanded, RATE_GAIN =
(-2.55, -2.55, -2.35)). Every desired body rate here passes through ONE boundary —
``control_check.send_att`` with CALIBRATE=True — so the physics-correct control math
(attitude P -> body rate) finally maps 1:1 onto what the drone actually does. No
sign-soup: if a sign is wrong here it is a real modelling error, not a compensation.

Stages (fly SLOW and reliable first, then speed — user directive):

    --stage hover     Level hover: null roll/pitch from the Mahony attitude filter at
                      hover thrust. Prove it sits stable (report tilt/gyro/collisions).
    --stage forward   Hover-stabilize, then gentle nose-down pitch + yaw servo toward
                      the nearest detected gate. Deliberately slow point-and-go.
    --stage race      forward, continued through gates with progression + finish
                      tracking and auto-retry attempts.

Reuses (imports, no copies): control primitives from ``control_check.py`` (Shared,
rx threads, calibrated send_att, patient reset) and the proven perception from
``race.py`` (red-gate blob + nearest-quad PnP + CAM_TO_BODY).

Usage (sim in Training, run live):
    uv run python scripts/vq2/fly.py --stage hover
    uv run python scripts/vq2/fly.py --stage forward --secs 12
    uv run python scripts/vq2/fly.py --stage race --attempts 3
"""

from __future__ import annotations

import argparse
import contextlib
import math
import sys
import threading
import time
from pathlib import Path

import control_check as cc
import cv2
import gates
import numpy as np
from estimator import AttitudeEstimator, VelocityEstimator
from pymavlink import mavutil

COLL_DUMP_DIR = Path(__file__).resolve().parents[2] / "data" / "vq2" / "fly_collisions"

# Calibration is ALWAYS on for this controller: cc.send_att maps desired->raw rates.
cc.CALIBRATE = True

CMD_HZ = 60.0
HOVER_THRUST = 0.30  # measured (control_check --test hover)

# --- IMU vertical-velocity estimate: DIAGNOSTIC ONLY (do not drive thrust with it) ---
# 2026-07-02: tried servoing thrust to hold this vz≈0 to stop the post-gate "floor"
# sink. It made things WORSE (broke gate-0 threading) because the estimate is badly
# biased through the violent sim launch — it read a CLIMB (vz≈-1 to -2) while the camera
# frames showed us skimming the floor. Integrated accel is untrustworthy over the launch
# window; the reliable height cue is the CAMERA (gate image row). vz stays logged only.
VZ_CLAMP = 3.0  # clamp the logged value so an impact spike doesn't dominate the range

# --- trajectory recorder (self-diagnosis without a human watching the sim) ---
# Dump a camera frame every TRACE_DT with attitude / vz / gate-detect baked into the
# filename, so the frame SEQUENCE into a collision can be read back to reconstruct the
# approach geometry. Cleared at the start of each race run.
TRACE_DIR = Path(__file__).resolve().parents[2] / "data" / "vq2" / "fly_trace"
TRACE_DT = 0.15  # seconds between trace frames

# --- outer attitude P loop (desired body rate from attitude error) ---
KP_ATT_ROLL = 4.0  # rad/s of desired rate per rad of roll error
KP_ATT_PITCH = 4.0
MAX_RATE = 2.0  # clamp desired roll/pitch rates (rad/s) — gentle by design

# --- forward flight (stage b) ---
PITCH_FWD = -0.05  # rad, gentle nose-down target => slow forward drift (SLOW first).
#                    Tried -0.03 (slower) to fix the gate-1 miss: NO help (bearing still
#                    ends -37°). The rightward miss is independent of forward speed — yaw/bank
#                    can't redirect the launch-direction momentum. Needs velocity guidance.
KP_YAW_BEAR = 1.5  # desired yaw rate per rad of bearing to the gate (body frame)
MAX_YAW = 1.0  # rad/s
SEARCH_YAW = 0.4  # rad/s slow scan when no gate visible
BRAKE_BEAR = 0.22  # rad (~13°) — taper forward pitch to 0 as |bearing| approaches this.
#                    Was 0.5 (28°). NECESSARY BUT NOT SUFFICIENT for the gate-1 miss:
#                    tightening it (turn-then-go) did not fix it — bearing still ends at
#                    -36..-39° (gate to our LEFT) because the real problem is uncontrolled
#                    LATERAL (rightward) momentum that forward-braking can't cancel and
#                    moderate bank can't null in time (aggressive bank tumbles). Detection
#                    is accurate (frame 025 t4.10: gate clearly left, on the racing line).
#                    The true fix needs a VELOCITY signal (optical flow / ESKF+guidance) to
#                    null lateral drift — see NEXT_SESSION step 3. Kept tight; it's sound.
TURN_BRAKE_PITCH = +0.15  # rad nose-up ACTIVE brake when the gate is far off-axis
#                           (|bearing| > BRAKE_BEAR). Gate 1 is a hard right turn; the
#                           launch momentum carried us PAST it (bearing pinned +35-40°)
#                           while yaw ran at max. Kill forward speed so the turn lands.
CY_ALIGN_BAND = 60.0  # px — "aim then go": only commit forward pitch when the gate is
#                       vertically centred too (|cy-CY_TARGET|<band). A high gate climbed
#                       out the TOP of frame (cy 33 -> lost) because we approached faster
#                       than we climbed and passed UNDER it — brake+climb to centre first.
KP_ROLL_BEAR = 0.30  # bank-toward-gate RE-ENABLED 2026-07-02: yaw alone rotates where we
#                      POINT, not where we GO — with ~6 m/s launch momentum the drone slid
#                      straight past gate 1's right-turn (bearing pinned +38°, fly_trace).
#                      Banking tilts the thrust vector to pull the velocity into the turn.
#                      The earlier "worse" test was pre-calibration + with the floor-sink;
#                      flight is stable+calibrated now. Roll sign follows the calibrated
#                      boundary (desired actual rate); watch fly_trace for the sign.
MAX_ROLL_TGT = 0.25  # rad (~14 deg) bank clamp — enough to actually turn, still bounded.
#                      Tried proximity-scaled bank (up to 3x) + 20° clamp to fix the gate-1
#                      right-miss: WORSE (roll excursions -80°, tumbled EARLIER, still missed).
# --- lateral drift damping (2026-07-02): the gate-1 miss is uncontrolled rightward drift.
# A STATIC gate's de-rotated bearing rate is a lateral-velocity proxy: drift = dβ/dt + gyro_z
# (rad/s). Validated OFFLINE from a gyro-logged trace — it is cleanly, monotonically negative
# (rightward drift) from the moment the gate centres. Bank to cancel it: roll += KD * drift
# (drift<0 => extra LEFT bank). Applied only during the approach; near the pass the bearing
# rate is parallax-dominated (would over-bank into a tumble), so gate it off by area.
KD_DRIFT_BANK = 0.0  # DISABLED by default. The drift SIGNAL is validated (offline: cleanly,
#                      monotonically negative through the gate-1 approach = rightward drift)
#                      and the sign is right (0.12 pulled bearing -37°→-11°). BUT bank alone
#                      can't thread gate 1: mild gain is neutral (bearing still ~-35°), strong
#                      gain tumbles (roll -103°). The approach is too FAST for the lateral
#                      window. Real fix = slower, velocity-nulled approach (ESKF + guidance);
#                      this drift proxy is the vision half of that velocity signal. Machinery
#                      stays wired (drift is computed + traced) so it's ready to turn back on.
DRIFT_TERM_CLAMP = 0.10  # rad — cap the drift term so it can't dominate/oscillate the bank
DRIFT_MAX_AREA = 3000.0  # only damp drift below this area (approach, not the parallax pass)
# vertical servo on thrust from the gate's image row (proven values from race.py —
# thrust was never part of the inversion bug; hover 0.30 re-confirmed on 3379)
CY_TARGET = 130.0
KP_THR_PX = 0.20  # was 0.10: too weak — a gate high in the frame (cy≈28) commanded only
#                  ~0.357 thrust, not enough to arrest the launch-arc descent to the floor
KD_THR_PX = 0.05  # visual D term: damp on d(cy)/dt (px/s, normalised). Climbing makes the
#                  gate move DOWN the frame (cy rises); without damping the P servo drove
#                  from floor to the CEILING trusses in ~1.5s (proven in fly_trace). This
#                  bleeds thrust as the gate approaches centre so we level off, not overshoot.
THR_MIN, THR_MAX = 0.25, 0.45  # was 0.36: give the climb servo real authority to recover
# The vertical servo now runs CONTINUOUSLY off the gate's image ROW (PnP range is always
# nan on 3379, so the old range gate meant it NEVER fired — the drone sank into the floor
# while the gate climbed out the top of the frame; proven in data/vq2/fly_trace).
# Asymmetric safety: a gate above center (cy<target) => CLIMB, always allowed (moves away
# from the floor). A gate below center (cy>target) => DESCEND, allowed only when the gate
# is clearly CLOSE (large area) so a far LOW gate can't pull us into the floor.
DESCEND_AREA = 2500.0  # px^2 — only descend toward a gate at least this big in frame
NO_GATE_CLIMB = 0.03  # thrust bias above hover when no gate is visible: forward flight
#                       bleeds vertical lift, so hold a mild climb rather than sink blind
TILT_COMP_MIN_COS = 0.5  # tilt-compensate thrust by 1/(cosR·cosP) so BANKING a turn (which
#                          tilts the thrust vector) doesn't bleed altitude — the sink
#                          recurred on the hard left turn past gate 1 (fly_trace). Uses the
#                          RELIABLE attitude estimate, not the biased IMU vz. Bounded so a
#                          near-90° tumble can't command infinite thrust.

# post-gate brake: the launch/cruise carries real momentum and we have no velocity
# estimate yet, so after each gate pass pitch UP briefly to bleed speed before the turn
# (live finding 2026-07-01: collisions near gate 1 happen with PERFECT attitude — the
# failure is entry speed, not stability).
BRAKE_SECS = 0.0  # OFF. Tried a 1.2 s post-gate-0 speed-kill brake (+0.15 nose-up) to slow
#                   the gate-1 approach: NO help (bearing still ends -37°, gate lost earlier).
#                   The gate-1 miss is not a forward-speed problem — needs velocity guidance.
BRAKE_PITCH = +0.15  # (retained; unused while BRAKE_SECS=0)

# post-gate OPEN-SPACE heading hold (2026-07-02): braking alone never changed the
# ~3.3s collision — the drone rides the launch momentum straight into a dark 3D
# structure that sits ON the line from the gate-0 exit toward gate 1. The straight-line
# turn toward gate 1's bearing is what clips it. Instead: after passing gate 0 keep the
# gate-0 EXIT heading (yaw held, wings level, no gate steering) for HOLD_HEADING_SECS so
# we coast through the open corridor, THEN acquire gate 1 from past the structure.
HOLD_HEADING_SECS = 0.0  # DISABLED with the brake (same disproven hypothesis): just
#                          point-and-go continuously; the vertical servo holds height.
HOLD_PITCH = -0.03  # (retained; unused while HOLD_HEADING_SECS=0)

# --- failsafes ---
TILT_ABORT = math.radians(60.0)  # attitude estimator says we're tumbling
COLL_ABORT = 3  # collision messages before aborting the attempt
STAB_SECS = 1.5  # initial window before point-and-go
STAB_PITCH = 0.0  # LEVEL through the sim's launch. Tried +0.10 nose-up to bleed the
#                   launch momentum before gate 0 (2026-07-02): the drone then hit the
#                   gate-0 FRAME instead of threading (3/3). Ride the launch level
#                   (gate 0 threads 100%), brake after.


# Perception: gates.nearest() returns ONLY ring-validated gates. The former blob
# fallback steered into glow-lit walls (root cause of the gate-1 ENV collisions —
# confirmed frame-by-frame in runs/vq2_collisions); glow is never a target now.
# A short persistence window bridges detection dropouts between valid sightings.
TARGET_HOLD_S = 0.8


class Flight:
    """One armed attempt; owns the estimator and the failsafe bookkeeping."""

    def __init__(self, sh: cc.Shared, conn, tgt_sys: int, tgt_comp: int, lock) -> None:
        self.sh, self.conn, self.lock = sh, conn, lock
        self.tgt = (tgt_sys, tgt_comp)
        self.att = AttitudeEstimator()
        self.vel = VelocityEstimator()
        self.vz = 0.0  # world-frame descent rate (m/s, +down), IMU-integrated
        self.gyro_z = 0.0  # latest body yaw rate (rad/s), for bearing de-rotation
        self.vz_min = 0.0  # most-negative (climb) seen
        self.vz_max = 0.0  # most-positive (sink) seen
        self.t_prev = time.time()
        self.last_arm = 0.0
        with sh.lock:
            self.coll0 = sh.collisions
        self.tilt_max = 0.0
        self.gyro_sq_sum = np.zeros(3)
        self.n = 0

    # -- per-loop primitives -------------------------------------------------
    def step_sensors(self) -> tuple[np.ndarray, np.ndarray] | None:
        """Re-arm, read IMU, advance the attitude filter. None if no IMU yet."""
        now = time.time()
        if now - self.last_arm > 0.4:
            cc.arm(self.conn, *self.tgt, self.lock)
            self.last_arm = now
        with self.sh.lock:
            imu = self.sh.imu
        if imu is None:
            return None
        dt = min(0.05, now - self.t_prev)
        self.t_prev = now
        acc = np.array(imu[:3])
        gyro = np.array(imu[3:])
        self.gyro_z = float(gyro[2])  # body yaw rate — de-rotates gate bearing drift
        self.att.update(gyro, acc, dt)
        self.vel.predict(acc, self.att.R, dt)
        self.vz = float(np.clip(self.vel.v[2], -VZ_CLAMP, VZ_CLAMP))
        self.vz_min = min(self.vz_min, self.vz)
        self.vz_max = max(self.vz_max, self.vz)
        roll, pitch, _ = self.att.euler
        self.tilt_max = max(self.tilt_max, abs(roll), abs(pitch))
        self.gyro_sq_sum += gyro**2
        self.n += 1
        return acc, gyro

    def command(self, thr: float, rr: float, pr: float, yr: float) -> None:
        """Single calibrated command boundary (desired ACTUAL body rates in, rad/s)."""
        rr = float(np.clip(rr, -MAX_RATE, MAX_RATE))
        pr = float(np.clip(pr, -MAX_RATE, MAX_RATE))
        yr = float(np.clip(yr, -MAX_YAW, MAX_YAW))
        cc.send_att(self.conn, *self.tgt, self.lock, thr, rr, pr, yr)

    def tilt_comp(self, thr: float) -> float:
        """Scale thrust by 1/(cosR·cosP) so a banked/pitched turn holds altitude."""
        roll, pitch, _ = self.att.euler
        c = math.cos(roll) * math.cos(pitch)
        return float(np.clip(thr / max(TILT_COMP_MIN_COS, c), THR_MIN, THR_MAX))

    def level_rates(
        self, pitch_target: float = 0.0, roll_target: float = 0.0
    ) -> tuple[float, float]:
        """Desired roll/pitch rates that P-track (roll_target, pitch_target) attitude."""
        roll, pitch, _ = self.att.euler
        return (KP_ATT_ROLL * (roll_target - roll), KP_ATT_PITCH * (pitch_target - pitch))

    # -- failsafes -----------------------------------------------------------
    def abort_reason(self) -> str | None:
        with self.sh.lock:
            coll = self.sh.collisions - self.coll0
            live = self.sh.is_live()
            fin = self.sh.race_finish_ns
        if fin >= 0:
            return "FINISHED"
        if not live:
            return "not live"
        if coll >= COLL_ABORT:
            return f"collisions ({coll})"
        roll, pitch, _ = self.att.euler
        if abs(roll) > TILT_ABORT or abs(pitch) > TILT_ABORT:
            return f"tilt abort (roll={math.degrees(roll):.0f} pitch={math.degrees(pitch):.0f} deg)"
        return None

    def summary(self) -> str:
        with self.sh.lock:
            coll = self.sh.collisions - self.coll0
            gate = self.sh.active_gate
            last = self.sh.last_collision
        rms = np.sqrt(self.gyro_sq_sum / max(1, self.n))
        what = ""
        if coll and last is not None:
            kind = {1001: "GATE", 1002: "ENV"}.get(last[0], str(last[0]))
            what = f" last_coll={kind}/threat{last[1]}"
        return (
            f"tilt_max={math.degrees(self.tilt_max):.1f}deg "
            f"gyro_rms=({rms[0]:.2f},{rms[1]:.2f},{rms[2]:.2f}) "
            f"vz=[{self.vz_min:+.1f},{self.vz_max:+.1f}]m/s "
            f"coll={coll}{what} active_gate={gate} loops={self.n}"
        )


def stage_hover(fl: Flight, secs: float) -> None:
    """Level hover on calibrated rates: prove the base layer is stable."""
    print(f"[hover] leveling at thrust {HOVER_THRUST} for {secs:.0f}s …")
    dt = 1.0 / CMD_HZ
    t0 = time.time()
    while time.time() - t0 < secs:
        if fl.step_sensors() is None:
            time.sleep(0.002)
            continue
        reason = fl.abort_reason()
        if reason:
            print(f"[hover] stop: {reason}")
            break
        rr, pr = fl.level_rates(0.0)
        fl.command(HOVER_THRUST, rr, pr, 0.0)
        time.sleep(dt)
    fl.command(0.05, 0.0, 0.0, 0.0)
    print(f"[hover] {fl.summary()}")


def vertical_thrust(held: gates.GateTarget | None, cy_rate: float = 0.0) -> float:
    """Thrust (PD on the gate's image row) that keeps the gate at CY_TARGET.

    Camera image row is the reliable height cue (IMU vz is too biased through the launch).
    ``cy_rate`` = d(cy)/dt in px/s (the visual climb-rate proxy) damps the overshoot.
    No gate visible => a mild climb bias, since forward flight bleeds vertical lift.
    """
    if held is None:
        return float(np.clip(HOVER_THRUST + NO_GATE_CLIMB, THR_MIN, THR_MAX))
    err = (held.cy - CY_TARGET) / 180.0  # <0: gate high => climb; >0: gate low => descend
    if err >= 0 and held.area < DESCEND_AREA:
        return HOVER_THRUST  # far low gate: hold, don't dive toward the floor
    thr = HOVER_THRUST - KP_THR_PX * err - KD_THR_PX * (cy_rate / 180.0)
    return float(np.clip(thr, THR_MIN, THR_MAX))


def _trace_frame(
    fl: Flight, tdir: Path, t_rel: float, idx: int, held: gates.GateTarget | None, gate: int
) -> None:
    """Save one trajectory frame with attitude / vz / gate-detect in the filename."""
    with fl.sh.lock:
        frame = fl.sh.frame
    if frame is None:
        return
    roll_d, pitch_d, _ = (math.degrees(v) for v in fl.att.euler)
    g = "none"
    if held is not None:
        g = f"cy{held.cy:.0f}a{held.area:.0f}b{math.degrees(held.bearing):+.1f}"
    name = (
        f"{idx:03d}_t{t_rel:.2f}_r{roll_d:+.0f}p{pitch_d:+.0f}"
        f"_vz{fl.vz:+.1f}_gz{fl.gyro_z:+.2f}_g{gate}_{g}.png"
    )
    cv2.imwrite(str(tdir / name), frame)


def stage_forward(fl: Flight, secs: float, race_mode: bool = False) -> str:
    """Hover-stabilize, then slow point-and-go toward the nearest gate.

    Returns the stop reason ("FINISHED" on a completed course in race mode)."""
    dt = 1.0 / CMD_HZ
    print(f"[forward] stabilizing {STAB_SECS}s, then slow point-and-go ({secs:.0f}s cap)")
    t0 = time.time()
    gate_prev = -1
    brake_until = 0.0
    hold_until = 0.0  # coast the gate-0 exit heading through the open corridor
    held: gates.GateTarget | None = None  # last validated target (persistence)
    held_at = 0.0
    prev_cy: float | None = None  # for the visual climb-rate D term
    prev_cy_t = 0.0
    cy_rate = 0.0
    prev_bear: float | None = None  # for the de-rotated lateral-drift proxy
    prev_bear_t = 0.0
    drift = 0.0  # dβ/dt + gyro_z (rad/s), + = drifting left, − = drifting right
    coll_seen = 0
    last_sight: tuple[float, float, float] | None = None  # (t_rel, bearing, range)
    reason = "timeout"
    trace_i = 0
    trace_next = 0.0
    if race_mode:  # fresh trajectory-frame corpus for this run
        TRACE_DIR.mkdir(parents=True, exist_ok=True)
        for old in TRACE_DIR.glob("*.png"):
            old.unlink()
    while time.time() - t0 < secs:
        if fl.step_sensors() is None:
            time.sleep(0.002)
            continue
        stop = fl.abort_reason()
        if stop:
            reason = stop
            break
        stabilizing = (time.time() - t0) < STAB_SECS
        pitch_target = STAB_PITCH if stabilizing else 0.0
        roll_target = 0.0
        yr = 0.0
        thr = HOVER_THRUST  # proven base (threads gate 0/1); gate-row servo adjusts below
        if not stabilizing:
            with fl.sh.lock:
                frame = fl.sh.frame
                gate_now = fl.sh.active_gate
            if gate_now != gate_prev:
                if race_mode:
                    print(f"  [gate] active_gate -> {gate_now}  t={time.time() - t0:.1f}s")
                if gate_prev >= 0:  # real gate pass (not the initial 0 latch)
                    # coast the exit heading through the open corridor (see HOLD_* note);
                    # brake first to shed launch momentum, then hold heading.
                    brake_until = time.time() + BRAKE_SECS
                    hold_until = time.time() + BRAKE_SECS + HOLD_HEADING_SECS
                gate_prev = gate_now
            with fl.sh.lock:
                coll_now = fl.sh.collisions - fl.coll0
            if coll_now > coll_seen and frame is not None:  # dump the collision view
                COLL_DUMP_DIR.mkdir(parents=True, exist_ok=True)
                roll_d, pitch_d, _ = (math.degrees(v) for v in fl.att.euler)
                name = f"c{coll_now}_t{time.time() - t0:.1f}s_r{roll_d:+.0f}p{pitch_d:+.0f}.png"
                cv2.imwrite(str(COLL_DUMP_DIR / name), frame)
                coll_seen = coll_now
            if frame is not None:
                tgt = gates.nearest(frame)
                if tgt is not None:
                    now = time.time()
                    if prev_cy is not None and now > prev_cy_t:  # visual climb-rate (px/s)
                        cy_rate = (tgt.cy - prev_cy) / (now - prev_cy_t)
                    if prev_bear is not None and now > prev_bear_t:  # de-rotated drift
                        drift = (tgt.bearing - prev_bear) / (now - prev_bear_t) + fl.gyro_z
                    prev_cy, prev_cy_t = tgt.cy, now
                    prev_bear, prev_bear_t = tgt.bearing, now
                    held, held_at = tgt, now
                    rng = tgt.range_m
                    last_sight = (
                        now - t0,
                        tgt.bearing,
                        rng if rng is not None else float("nan"),
                    )
            if held is not None and time.time() - held_at > TARGET_HOLD_S:
                held = None  # sighting stale: stop chasing it
                cy_rate, prev_cy = 0.0, None  # target dropped: reset the D terms
                drift, prev_bear = 0.0, None
            braking = time.time() < brake_until
            if held is None:
                yr = SEARCH_YAW  # scan; stay put (pitch level) until we see a REAL gate
            else:
                bearing = held.bearing
                yr = float(np.clip(KP_YAW_BEAR * bearing, -MAX_YAW, MAX_YAW))
                # "aim then go": commit forward pitch only when the gate is centred in
                # BOTH axes (bearing small AND vertically near CY_TARGET). Otherwise brake
                # (nose up) and let the yaw/bank turn + climb servo bring it to centre —
                # the vertical servo holds height meanwhile. Prevents both sliding past a
                # side gate and passing under a high gate that then exits frame-top.
                align = max(0.0, 1.0 - abs(bearing) / BRAKE_BEAR)
                elev_ok = abs(held.cy - CY_TARGET) < CY_ALIGN_BAND
                aligned = align > 0.0 and elev_ok
                pitch_target = PITCH_FWD * align if aligned else TURN_BRAKE_PITCH
                # bank toward the gate (P on bearing) PLUS lateral-drift damping (D on the
                # de-rotated bearing rate) to cancel the rightward drift that makes us miss
                # gate 1. Drift damping only during the approach — near the pass the bearing
                # rate is parallax, and reacting to it over-banks into a tumble.
                roll_bank = KP_ROLL_BEAR * bearing
                if held.area < DRIFT_MAX_AREA:
                    roll_bank += float(
                        np.clip(KD_DRIFT_BANK * drift, -DRIFT_TERM_CLAMP, DRIFT_TERM_CLAMP)
                    )
                roll_target = float(np.clip(roll_bank, -MAX_ROLL_TGT, MAX_ROLL_TGT))
            holding = time.time() < hold_until
            if braking:  # bleed post-gate momentum before chasing the next gate
                pitch_target = BRAKE_PITCH
                roll_target = 0.0
                yr = 0.0  # heading locked: do NOT yaw toward gate 1 yet (clips structure)
            elif holding:  # coast the exit heading through the open corridor past the
                #            dark structure, THEN resume gate acquisition
                pitch_target = HOLD_PITCH
                roll_target = 0.0
                yr = 0.0
            # thrust ALWAYS tracks the gate's image row (or a mild climb bias when blind)
            # — even while braking/holding, which is exactly when the drone was sinking.
            thr = fl.tilt_comp(vertical_thrust(held, cy_rate))
        if race_mode and time.time() - t0 >= trace_next:
            _trace_frame(fl, TRACE_DIR, time.time() - t0, trace_i, held, gate_prev)
            trace_i += 1
            trace_next = (time.time() - t0) + TRACE_DT
        rr, pr = fl.level_rates(pitch_target, roll_target)
        fl.command(thr, rr, pr, yr)
        time.sleep(dt)
    fl.command(0.05, 0.0, 0.0, 0.0)
    if last_sight is not None:
        print(
            f"  last gate sighting: t={last_sight[0]:.1f}s bearing={math.degrees(last_sight[1]):+.0f}deg "
            f"range={last_sight[2]:.1f}m (stop at t={time.time() - t0:.1f}s)"
        )
    print(f"[forward] stop: {reason}; {fl.summary()}")
    return reason


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["hover", "forward", "race"], required=True)
    ap.add_argument("--secs", type=float, default=8.0, help="stage duration cap (s)")
    ap.add_argument("--attempts", type=int, default=1, help="race attempts (auto-reset between)")
    args = ap.parse_args()
    with contextlib.suppress(Exception):  # Windows consoles default to cp1252
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
        attempts = args.attempts if args.stage == "race" else 1
        for i in range(attempts):
            if attempts > 1:
                print(f"\n=== attempt {i + 1}/{attempts} ===")
            if not cc.wait_fresh(sh, conn, tgt_sys, tgt_comp, lock):
                print("NO FRESH START — is a run live in the Training window?")
                return
            fl = Flight(sh, conn, tgt_sys, tgt_comp, lock)
            if args.stage == "hover":
                stage_hover(fl, args.secs)
            else:
                secs = args.secs if args.stage == "forward" else max(args.secs, 60.0)
                reason = stage_forward(fl, secs, race_mode=(args.stage == "race"))
                if reason == "FINISHED":
                    print("🎉 course FINISHED")
                    break
    finally:
        sh.stop.set()
        time.sleep(0.4)


if __name__ == "__main__":
    main()
