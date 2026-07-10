"""Probe the OFFICIAL control modes from PyAIPilotExample-v2 against the live VQ2 sim.

The official example (VQ2/AI-GP Simulator v1.0.3379/PyAIPilotExample-v2) ships three
ways to command the drone:

  1. update_motor_control          — direct motor RPMs (SET_ACTUATOR_CONTROL_TARGET)
  2. update_attitude_flight_control — body rates + collective thrust (SET_ATTITUDE_TARGET)
  3. update_position_flight_control — position / velocity setpoint (SET_POSITION_TARGET_LOCAL_NED)

We already fly on (2) via a hand-rolled rate->attitude->velocity loop, which is where
our tumbles live. Mode (3) is the "tried and true" PX4-style path: you command a VELOCITY
VECTOR and the sim's OWN flight controller stabilizes attitude — it cannot tumble by
construction. This probe copies the official command functions VERBATIM and wraps them
with instrumentation (gyro = stability, optical flow = motion, collisions) so we can
measure, with numbers, whether the sim honors velocity setpoints on build 3379.

Reuses the proven rx/arm/reset plumbing from control_check.py.

Usage
-----
    uv run python scripts/vq2/example_probe.py --mode vel --frame body --vx 2 --secs 3
    uv run python scripts/vq2/example_probe.py --mode vel --frame ned  --vz -1 --secs 3
    uv run python scripts/vq2/example_probe.py --mode att   # sanity: official attitude cmd
"""

from __future__ import annotations

import argparse
import contextlib
import sys
import threading
import time

import cv2
import numpy as np
from control_check import (
    CMD_HZ,
    Shared,
    _flow_dir,
    arm,
    mav_rx,
    vision_rx,
    wait_fresh,
)
from pymavlink import mavutil

# --- official velocity/position mask (copied verbatim from controller.py) ---
VELOCITY_POSITION_MASK = (
    mavutil.mavlink.POSITION_TARGET_TYPEMASK_X_IGNORE
    | mavutil.mavlink.POSITION_TARGET_TYPEMASK_Y_IGNORE
    | mavutil.mavlink.POSITION_TARGET_TYPEMASK_Z_IGNORE
    | mavutil.mavlink.POSITION_TARGET_TYPEMASK_AX_IGNORE
    | mavutil.mavlink.POSITION_TARGET_TYPEMASK_AY_IGNORE
    | mavutil.mavlink.POSITION_TARGET_TYPEMASK_AZ_IGNORE
    | mavutil.mavlink.POSITION_TARGET_TYPEMASK_YAW_IGNORE
    | mavutil.mavlink.POSITION_TARGET_TYPEMASK_YAW_RATE_IGNORE
)

FRAMES = {
    "ned": mavutil.mavlink.MAV_FRAME_LOCAL_NED,  # world NED (example default)
    "body": mavutil.mavlink.MAV_FRAME_BODY_NED,  # forward/right/down relative to nose
    "body_offset": mavutil.mavlink.MAV_FRAME_BODY_OFFSET_NED,
}


def send_velocity(conn, tgt_sys, tgt_comp, lock, frame, vx, vy, vz, boot_ms) -> None:
    """Official update_position_flight_control, verbatim, but with a chosen frame + vel."""
    now_ms = int(time.time() * 1000)
    with lock:
        conn.mav.set_position_target_local_ned_send(
            (now_ms - boot_ms) & 0xFFFFFFFF,
            tgt_sys,
            tgt_comp,
            frame,
            VELOCITY_POSITION_MASK,
            0.0,
            0.0,
            0.0,  # position (ignored)
            float(vx),
            float(vy),
            float(vz),  # velocity setpoint
            0.0,
            0.0,
            0.0,  # accel (ignored)
            0.0,  # yaw (ignored)
            0.0,  # yaw rate (ignored)
        )


def send_attitude(conn, tgt_sys, tgt_comp, lock, rr, pr, yr, thrust, boot_ms) -> None:
    """Official update_attitude_flight_control, verbatim (rates + thrust)."""
    now_ms = int(time.time() * 1000)
    with lock:
        conn.mav.set_attitude_target_send(
            (now_ms - boot_ms) & 0xFFFFFFFF,
            tgt_sys,
            tgt_comp,
            mavutil.mavlink.ATTITUDE_TARGET_TYPEMASK_ATTITUDE_IGNORE,
            [1.0, 0.0, 0.0, 0.0],
            float(rr),
            float(pr),
            float(yr),
            float(thrust),
        )


def run_probe(sh, conn, tgt_sys, tgt_comp, lock, args, boot_ms) -> None:
    frame = FRAMES[args.frame]
    print(f"\n=== OFFICIAL {args.mode.upper()} PROBE ===")
    if args.mode == "vel":
        print(
            f"frame={args.frame} vel=({args.vx},{args.vy},{args.vz}) m/s "
            f"for {args.secs}s (NED: +x=N/fwd, +y=E/right, +z=DOWN)"
        )
    else:
        print(f"attitude rates=({args.rr},{args.pr},{args.yr}) thrust={args.thr} for {args.secs}s")

    if not wait_fresh(sh, conn, tgt_sys, tgt_comp, lock):
        print("  NO FRESH START — is a run live in the Training window?")
        return
    for _ in range(3):
        arm(conn, tgt_sys, tgt_comp, lock)
        time.sleep(0.05)

    dt = 1.0 / CMD_HZ
    if args.lift > 0:
        # Get airborne on attitude+thrust first, THEN hand over to velocity setpoints —
        # PX4-style controllers ignore velocity setpoints on the ground.
        print(f"  [lift] attitude thrust {args.thr} for {args.lift}s, then velocity…")
        t_lift = time.time()
        la = 0.0
        while time.time() - t_lift < args.lift and not sh.stop.is_set():
            if time.time() - la > 0.4:
                arm(conn, tgt_sys, tgt_comp, lock)
                la = time.time()
            send_attitude(conn, tgt_sys, tgt_comp, lock, 0.0, 0.0, 0.0, args.thr, boot_ms)
            time.sleep(dt)
    gyros: list[tuple] = []
    accs: list[tuple] = []
    us: list[float] = []
    vs: list[float] = []
    prev_gray: np.ndarray | None = None
    with sh.lock:
        c0 = sh.collisions
    last_arm = 0.0
    t0 = time.time()
    while time.time() - t0 < args.secs and not sh.stop.is_set():
        if time.time() - last_arm > 0.4:
            arm(conn, tgt_sys, tgt_comp, lock)
            last_arm = time.time()
        if args.mode == "vel":
            send_velocity(conn, tgt_sys, tgt_comp, lock, frame, args.vx, args.vy, args.vz, boot_ms)
        else:
            send_attitude(
                conn, tgt_sys, tgt_comp, lock, args.rr, args.pr, args.yr, args.thr, boot_ms
            )
        with sh.lock:
            imu, frame_img = sh.imu, sh.frame
        if imu is not None:
            accs.append(imu[:3])
            gyros.append(imu[3:])
        if frame_img is not None:
            gray = cv2.cvtColor(cv2.resize(frame_img, (160, 90)), cv2.COLOR_BGR2GRAY)
            if prev_gray is not None:
                u, v = _flow_dir(prev_gray, gray)
                us.append(u)
                vs.append(v)
            prev_gray = gray
        time.sleep(dt)

    with sh.lock:
        coll = sh.collisions - c0
        last_coll = sh.last_collision
    g = np.array(gyros) if gyros else np.zeros((1, 3))
    a = np.array(accs) if accs else np.zeros((1, 3))
    gyro_rms = np.sqrt(np.mean(g**2, axis=0))
    # last-half accel (skip lift transient)
    a_half = a[len(a) // 2 :]
    amag = float(np.median(np.linalg.norm(a_half, axis=1))) if len(a_half) else 0.0
    mu = float(np.median(us)) if us else 0.0
    mv = float(np.median(vs)) if vs else 0.0

    print("\n  --- result ---")
    print(
        f"  gyro_rms      = ({gyro_rms[0]:.2f},{gyro_rms[1]:.2f},{gyro_rms[2]:.2f}) rad/s  "
        f"[<0.3 = STABLE, sim is holding attitude; high = tumbling/uncontrolled]"
    )
    print(f"  accel |a| med = {amag:.2f} (g=9.81; ~g ⇒ holding altitude, >g climb, <g sink)")
    print(
        f"  optic flow    = (u={mu:+.2f}, v={mv:+.2f})  "
        f"[drone moves OPPOSITE scene flow: +u⇒moved LEFT, +v⇒moved UP]"
    )
    moved = []
    if abs(mu) > 0.08:
        moved.append("LEFT" if mu > 0 else "RIGHT")
    if abs(mv) > 0.08:
        moved.append("UP" if mv > 0 else "DOWN")
    print(f"  => translated : {'+'.join(moved) if moved else 'negligible'}")
    print(f"  collisions    = {coll}  last={last_coll}")
    verdict = (
        "SIM HONORS SETPOINT (stable + responded)"
        if (args.mode == "vel" and float(np.max(gyro_rms)) < 0.5 and (moved or coll))
        else "inconclusive/ignored — inspect numbers"
    )
    if args.mode == "vel":
        print(f"  VERDICT       : {verdict}")

    # settle / disarm-ish
    for _ in range(15):
        send_attitude(conn, tgt_sys, tgt_comp, lock, 0.0, 0.0, 0.0, 0.05, boot_ms)
        time.sleep(dt)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["vel", "att"], default="vel")
    ap.add_argument("--frame", choices=list(FRAMES), default="body")
    ap.add_argument("--vx", type=float, default=2.0, help="fwd (body) / north (ned) m/s")
    ap.add_argument("--vy", type=float, default=0.0, help="right (body) / east (ned) m/s")
    ap.add_argument("--vz", type=float, default=0.0, help="down m/s (negative = climb)")
    ap.add_argument("--rr", type=float, default=0.0)
    ap.add_argument("--pr", type=float, default=-0.3)
    ap.add_argument("--yr", type=float, default=0.0)
    ap.add_argument("--thr", type=float, default=0.6)
    ap.add_argument("--secs", type=float, default=3.0)
    ap.add_argument("--lift", type=float, default=0.0, help="sec of attitude-thrust liftoff first")
    args = ap.parse_args()

    with contextlib.suppress(Exception):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    print("Connecting to udpin:127.0.0.1:14550 …")
    conn = mavutil.mavlink_connection("udpin:127.0.0.1:14550")
    conn.wait_heartbeat()
    tgt_sys, tgt_comp = conn.target_system, conn.target_component
    boot_ms = int(time.time() * 1000)
    print(f"Heartbeat OK. target_system={tgt_sys}")

    sh = Shared()
    lock = threading.Lock()
    threading.Thread(target=mav_rx, args=(conn, sh), daemon=True).start()
    threading.Thread(target=vision_rx, args=(sh,), daemon=True).start()
    time.sleep(0.5)

    try:
        run_probe(sh, conn, tgt_sys, tgt_comp, lock, args, boot_ms)
    finally:
        sh.stop.set()
        time.sleep(0.4)


if __name__ == "__main__":
    main()
