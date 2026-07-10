"""Validate the VQ2 state estimator using only signals the drone still gives us.

Phase A (platform, no flight): the drone is truly at rest, so the estimated velocity
MUST stay ~0 — its drift directly measures attitude/gravity-compensation error. Also
checks the attitude reads the known rest pose (roll~0, pitch~-17.8deg).

Phase B (gentle armed hover/maneuver): checks attitude stays continuous through motion
(gravity-alignment error) and that IMU-integrated velocity matches the PnP-derived
velocity (the gate is a static landmark, so its relative motion = -drone motion).

Reuses the connection/reset/fly primitives from race.py.

    uv run python scripts/vq2/estimate_test.py            # phase A only (no flight)
    uv run python scripts/vq2/estimate_test.py --fly      # + phase B (arms, gentle hover)
"""

from __future__ import annotations

import argparse
import math
import threading
import time

import numpy as np
import race  # sibling module (scripts/vq2/race.py)
from estimator import AttitudeEstimator, VelocityEstimator


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fly", action="store_true")
    ap.add_argument("--secs", type=float, default=4.0)
    args = ap.parse_args()

    conn = race.mavutil.mavlink_connection(race.MAVLINK_ENDPOINT)
    conn.wait_heartbeat()
    tgt_sys, tgt_comp = conn.target_system, conn.target_component
    sh = race.Shared()
    lock = threading.Lock()
    threading.Thread(target=race.mav_rx, args=(conn, sh), daemon=True).start()
    threading.Thread(target=race.vision_rx, args=(sh,), daemon=True).start()
    time.sleep(0.5)

    print("Acquiring a fresh at-rest start...")
    if not race.wait_fresh_live(sh, conn, tgt_sys, tgt_comp, lock):
        print("no fresh start")
        return

    att = AttitudeEstimator()
    vel = VelocityEstimator()

    # ---------------- Phase A: stationary platform (no arming, no commands) ----------------
    print("\n[Phase A] platform rest — velocity should stay ~0 (validates attitude+gravity comp)")
    t0 = time.time()
    t_prev = t0
    vmax = 0.0
    n = 0
    while time.time() - t0 < args.secs:
        with sh.lock:
            imu = sh.imu
        if imu is not None:
            now = time.time()
            dt = min(0.05, now - t_prev)
            t_prev = now
            acc = np.array(imu[:3])
            gyro = np.array(imu[3:])
            att.update(gyro, acc, dt)
            vel.predict(acc, att.R, dt)
            vmax = max(vmax, vel.speed)
            n += 1
        time.sleep(0.005)
    roll, pitch, yaw = att.euler
    print(
        f"  attitude: roll={math.degrees(roll):+.1f}deg pitch={math.degrees(pitch):+.1f}deg "
        f"(expect roll~0, pitch~-17.8deg)"
    )
    print(f"  gravity-align err (mean): {att.mean_grav_err_deg:.2f}deg  (lower=better)")
    print(
        f"  velocity drift over {args.secs:.0f}s: |v|max={vmax:.3f} m/s, final |v|={vel.speed:.3f} m/s "
        f"(expect ~0; nonzero => attitude/gravity-comp error)"
    )

    if not args.fly:
        return

    # ---------------- Phase B: gentle armed hover, validate against PnP ----------------
    print("\n[Phase B] gentle hover — attitude continuity + IMU-vs-PnP velocity")
    vel.v = np.zeros(3)
    dt_cmd = 1.0 / 60.0
    t0 = time.time()
    t_prev = t0
    last_arm = 0.0
    prev_pose = None
    prev_pt = None
    pnp_res = []  # |v_imu - v_pnp| samples (world frame)
    grav_series = []
    while time.time() - t0 < 6.0:
        with sh.lock:
            imu = sh.imu
            frame = sh.frame
            coll = sh.collisions
        if time.time() - last_arm > 0.5:
            race.arm(conn, tgt_sys, tgt_comp, lock)
            last_arm = time.time()
        if imu is None:
            time.sleep(0.002)
            continue
        now = time.time()
        dt = min(0.05, now - t_prev)
        t_prev = now
        acc = np.array(imu[:3])
        gyro = np.array(imu[3:])
        att.update(gyro, acc, dt)
        vel.predict(acc, att.R, dt)
        grav_series.append(att.mean_grav_err_deg)

        # PnP-derived world velocity: gate is static, so v_drone_world ~ -d(R@p_cam)/dt
        if frame is not None:
            quad = race.detect_gate_quad(frame)
            pose = race.gate_relative_pose(quad) if quad is not None else None
            if pose is not None:
                p_world = att.R @ np.array(pose)  # gate offset in world frame
                if prev_pose is not None and now - prev_pt > 1e-3:
                    v_pnp = -(p_world - prev_pose) / (now - prev_pt)
                    pnp_res.append(float(np.linalg.norm(vel.v - v_pnp)))
                    # soft-anchor velocity to the PnP reference to bound drift
                    vel.anchor(v_pnp, 0.15)
                prev_pose, prev_pt = p_world, now

        # gentle level hover using RAW accel (independent of the estimator under test):
        # keep it near hover; never climb hard (low ceiling).
        roll_m = math.atan2(-acc[1], -acc[2])
        pitch_m = math.atan2(acc[0], math.hypot(acc[1], acc[2]))
        rr = -1.0 * roll_m - 0.2 * gyro[0]
        pr = -1.2 * (pitch_m - (-0.12)) - 0.05 * gyro[1]
        race.send_att(conn, tgt_sys, tgt_comp, lock, 0.30, rr, pr, 0.0)
        if coll - 0 > 4:
            print("  (collision during hover — stopping phase B)")
            break
        time.sleep(dt_cmd)

    race.send_att(conn, tgt_sys, tgt_comp, lock, 0.05, 0.0, 0.0, 0.0)
    print(f"  gravity-align err (mean over flight): {att.mean_grav_err_deg:.2f}deg")
    if pnp_res:
        arr = np.array(pnp_res)
        print(
            f"  IMU-vs-PnP velocity |d|: mean={arr.mean():.2f} median={np.median(arr):.2f} m/s "
            f"(n={len(arr)}; lower=better, they should agree)"
        )
    else:
        print("  no PnP velocity samples (gate not detected during hover)")
    race.reset_sim(conn, tgt_sys, tgt_comp, lock)


if __name__ == "__main__":
    main()
