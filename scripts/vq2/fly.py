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
from estimator import AttitudeEstimator
from pymavlink import mavutil

COLL_DUMP_DIR = Path(__file__).resolve().parents[2] / "data" / "vq2" / "fly_collisions"

# Calibration is ALWAYS on for this controller: cc.send_att maps desired->raw rates.
cc.CALIBRATE = True

CMD_HZ = 60.0
HOVER_THRUST = 0.30  # measured (control_check --test hover)

# --- outer attitude P loop (desired body rate from attitude error) ---
KP_ATT_ROLL = 4.0  # rad/s of desired rate per rad of roll error
KP_ATT_PITCH = 4.0
MAX_RATE = 2.0  # clamp desired roll/pitch rates (rad/s) — gentle by design

# --- forward flight (stage b) ---
PITCH_FWD = -0.05  # rad, gentle nose-down target => slow forward drift (SLOW first)
KP_YAW_BEAR = 1.5  # desired yaw rate per rad of bearing to the gate (body frame)
MAX_YAW = 1.0  # rad/s
SEARCH_YAW = 0.4  # rad/s slow scan when no gate visible
BRAKE_BEAR = 0.5  # rad — taper forward pitch to 0 as |bearing| approaches this
#                   ("turn in place, then go": no momentum to fight at the turn)
KP_ROLL_BEAR = 0.0  # bank-toward-gate OFF: live test 2026-07-01 was WORSE with 0.25
#                     (tilt spikes near gate 1). Verify the roll->translation sign with
#                     control_check --test translate --calib before re-enabling.
MAX_ROLL_TGT = 0.12  # rad (~7 deg) bank clamp — gentle by design
# vertical servo on thrust from the gate's image row (proven values from race.py —
# thrust was never part of the inversion bug; hover 0.30 re-confirmed on 3379)
CY_TARGET = 130.0
KP_THR_PX = 0.10
THR_MIN, THR_MAX = 0.25, 0.36
VERT_RANGE_M = 7.0  # only run the descent servo when the gate is close: descending
#                     toward a far LOW gate drops the drone into intervening dark
#                     structures (confirmed in data/vq2/fly_collisions frame dumps)

# post-gate brake: the launch/cruise carries real momentum and we have no velocity
# estimate yet, so after each gate pass pitch UP briefly to bleed speed before the turn
# (live finding 2026-07-01: collisions near gate 1 happen with PERFECT attitude — the
# failure is entry speed, not stability).
BRAKE_SECS = 2.2  # 1.2s/+0.14 bled only ~1.6 m/s of an estimated 4-6 m/s of launch
BRAKE_PITCH = +0.22  # momentum -> still hit the dark structure right of gate 0; ~5 m/s now

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
        self.att.update(gyro, acc, dt)
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


def stage_forward(fl: Flight, secs: float, race_mode: bool = False) -> str:
    """Hover-stabilize, then slow point-and-go toward the nearest gate.

    Returns the stop reason ("FINISHED" on a completed course in race mode)."""
    dt = 1.0 / CMD_HZ
    print(f"[forward] stabilizing {STAB_SECS}s, then slow point-and-go ({secs:.0f}s cap)")
    t0 = time.time()
    gate_prev = -1
    brake_until = 0.0
    held: gates.GateTarget | None = None  # last validated target (persistence)
    held_at = 0.0
    coll_seen = 0
    last_sight: tuple[float, float, float] | None = None  # (t_rel, bearing, range)
    reason = "timeout"
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
        thr = HOVER_THRUST
        if not stabilizing:
            with fl.sh.lock:
                frame = fl.sh.frame
                gate_now = fl.sh.active_gate
            if gate_now != gate_prev:
                if race_mode:
                    print(f"  [gate] active_gate -> {gate_now}  t={time.time() - t0:.1f}s")
                if gate_prev >= 0:  # real gate pass (not the initial 0 latch): brake
                    brake_until = time.time() + BRAKE_SECS
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
                    held, held_at = tgt, time.time()
                    rng = tgt.range_m
                    last_sight = (
                        time.time() - t0,
                        tgt.bearing,
                        rng if rng is not None else float("nan"),
                    )
            if held is not None and time.time() - held_at > TARGET_HOLD_S:
                held = None  # sighting stale: stop chasing it
            braking = time.time() < brake_until
            if held is None:
                yr = SEARCH_YAW  # scan; stay put (pitch level) until we see a REAL gate
            else:
                bearing = held.bearing
                yr = float(np.clip(KP_YAW_BEAR * bearing, -MAX_YAW, MAX_YAW))
                # brake into the turn: forward pitch only when roughly aligned
                align = max(0.0, 1.0 - abs(bearing) / BRAKE_BEAR)
                pitch_target = PITCH_FWD * align
                # gentle bank toward the gate: rotates the velocity vector in turns
                roll_target = float(np.clip(KP_ROLL_BEAR * bearing, -MAX_ROLL_TGT, MAX_ROLL_TGT))
                # vertical: image-row servo on the VALIDATED gate only (the old blob
                # fallback followed floor/wall glow straight into the environment),
                # and only at close range — far low gates must not trigger an early
                # descent through whatever stands in between. Far/unknown range: hold.
                rng = held.range_m
                if rng is not None and rng < VERT_RANGE_M:
                    thr = float(
                        np.clip(
                            HOVER_THRUST - KP_THR_PX * ((held.cy - CY_TARGET) / 180.0),
                            THR_MIN,
                            THR_MAX,
                        )
                    )
            if braking:  # bleed post-gate momentum before chasing the next gate
                pitch_target = BRAKE_PITCH
                roll_target = 0.0
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
