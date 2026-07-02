"""VQ2 control-path probe — find out what actually moves the drone on build 3379.

Because VQ2 gives us no attitude/position telemetry, we infer the effect of each
command from the IMU (gyro = rate response, accel = gravity direction / thrust)
and from the camera (translation: gates grow/shift, ground slides).

The probe restarts the run itself via Backspace (keep the sim in the **Training**
window and FOCUSED, exactly like VQ1's --auto-restart), arms, waits for the race
to go live, then steps through a gentle scripted sequence, logging IMU per phase
and saving a camera frame every ~0.4 s tagged with the active phase.

Phases (each short + gentle; Training, so a tumble just gets Backspaced):
    settle   min throttle, let it sit
    hover    thrust 0.275, zero rates      -> does it hold / not tumble?
    climb    thrust 0.45, zero rates       -> accel net up, camera gates drop
    pitchfwd thrust 0.30, pitch_rate -0.3  -> gyro-y responds, camera moves fwd
    velfwd   SET_POSITION_TARGET body vx=2 -> does velocity-setpoint mode work now?

Usage
-----
    uv run python scripts/vq2/control_probe.py
    uv run python scripts/vq2/control_probe.py --no-restart   # don't send Backspace
"""

from __future__ import annotations

import argparse
import socket
import struct
import threading
import time
from pathlib import Path

import cv2
import numpy as np
from pymavlink import mavutil

MAVLINK_ENDPOINT = "udpin:127.0.0.1:14550"
VISION_PORT = 5600
VISION_HEADER = "<IHHIIQ"
FRAME_DIR = Path(__file__).resolve().parents[2] / "VQ2" / "probe_frames"
CMD_HZ = 50.0

# (name, seconds, thrust, roll_rate, pitch_rate, yaw_rate, mode)
# mode: "att" = SET_ATTITUDE_TARGET rate mode, "vel" = SET_POSITION_TARGET body vel
Phase = tuple[str, float, float, float, float, float, str]
SEQUENCE: list[Phase] = [
    ("settle", 1.5, 0.10, 0.0, 0.0, 0.0, "att"),
    ("hover", 3.0, 0.275, 0.0, 0.0, 0.0, "att"),
    ("climb", 2.0, 0.45, 0.0, 0.0, 0.0, "att"),
    ("pitchfwd", 2.0, 0.30, 0.0, -0.3, 0.0, "att"),
    ("velfwd", 3.0, 0.0, 0.0, 0.0, 0.0, "vel"),
]

# Isolated test: from a clean start send ONLY body-forward velocity setpoints
# (no attitude commands at all). If the drone flies forward under control, the
# sim honours velocity-setpoint mode and control becomes trivial.
SEQUENCE_VELONLY: list[Phase] = [
    ("vwait", 1.5, 0.0, 0.0, 0.0, 0.0, "none"),
    ("velfwd", 6.0, 0.0, 0.0, 0.0, 0.0, "vel"),
]


class Shared:
    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.imu: tuple[float, float, float, float, float, float] | None = None
        self.race_start_ms = -1
        self.race_finish_ns = -1
        self.sim_boot_ms = 0
        self.active_gate = -1
        self.collisions = 0
        self.stop = threading.Event()

    def is_live(self) -> bool:
        """True only after the countdown completes (green light) and before finish.

        The sim DQs us if we throttle up before this — ``race_start_boot_time_ms``
        is set to a FUTURE sim time the instant Race is clicked (countdown start),
        so ``sim_boot_ms >= race_start_ms`` is the real go signal.
        """
        if self.race_start_ms < 0 or self.race_finish_ns >= 0:
            return False
        return self.sim_boot_ms >= self.race_start_ms


def mav_rx(conn: mavutil.mavfile, sh: Shared) -> None:
    while not sh.stop.is_set():
        msg = conn.recv_match(blocking=False)
        if msg is None:
            time.sleep(0.0005)
            continue
        t = msg.get_type()
        if t == "HIGHRES_IMU":
            with sh.lock:
                sh.imu = (msg.xacc, msg.yacc, msg.zacc, msg.xgyro, msg.ygyro, msg.zgyro)
        elif t == "COLLISION":
            with sh.lock:
                sh.collisions += 1
        elif t == "ENCAPSULATED_DATA":
            raw = bytes(msg.data)
            if raw and raw[0] == 1:
                _, sim_ms, rs_ms, fin_ns, gate, _lg = struct.unpack_from("<BQqqIq", raw)
                with sh.lock:
                    sh.sim_boot_ms, sh.race_start_ms = sim_ms, rs_ms
                    sh.race_finish_ns, sh.active_gate = fin_ns, gate


def vision_rx(sh: Shared, phase_ref: list[str], saved: list[str]) -> None:
    header_sz = struct.calcsize(VISION_HEADER)
    frames: dict[int, dict] = {}
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.bind(("0.0.0.0", VISION_PORT))
    sock.settimeout(0.5)
    FRAME_DIR.mkdir(parents=True, exist_ok=True)
    last_save = 0.0
    while not sh.stop.is_set():
        try:
            packet, _ = sock.recvfrom(65536)
        except TimeoutError:
            continue
        h, payload = packet[:header_sz], packet[header_sz:]
        fid, cid, total, _js, _ps, _ts = struct.unpack(VISION_HEADER, h)
        f = frames.setdefault(fid, {"c": {}, "n": total})
        f["c"][cid] = payload
        if len(f["c"]) == f["n"] and all(i in f["c"] for i in range(f["n"])):
            buf = bytearray()
            for i in range(f["n"]):
                buf.extend(f["c"][i])
            img = cv2.imdecode(np.frombuffer(buf, np.uint8), cv2.IMREAD_COLOR)
            del frames[fid]
            now = time.time()
            if img is not None and now - last_save > 0.4 and phase_ref[0]:
                name = f"{len(saved):02d}_{phase_ref[0]}_{fid:06d}.png"
                cv2.imwrite(str(FRAME_DIR / name), img)
                saved.append(name)
                last_save = now


def send_att(conn, tgt_sys, tgt_comp, lock, thr, rr, pr, yr) -> None:
    with lock:
        conn.mav.set_attitude_target_send(
            int(time.monotonic() * 1000) & 0xFFFFFFFF,
            tgt_sys,
            tgt_comp,
            mavutil.mavlink.ATTITUDE_TARGET_TYPEMASK_ATTITUDE_IGNORE,
            [1.0, 0.0, 0.0, 0.0],
            float(rr),
            float(pr),
            float(yr),
            float(thr),
        )


def send_vel_body(conn, tgt_sys, tgt_comp, lock, vx, vy, vz) -> None:
    mask = (
        mavutil.mavlink.POSITION_TARGET_TYPEMASK_X_IGNORE
        | mavutil.mavlink.POSITION_TARGET_TYPEMASK_Y_IGNORE
        | mavutil.mavlink.POSITION_TARGET_TYPEMASK_Z_IGNORE
        | mavutil.mavlink.POSITION_TARGET_TYPEMASK_AX_IGNORE
        | mavutil.mavlink.POSITION_TARGET_TYPEMASK_AY_IGNORE
        | mavutil.mavlink.POSITION_TARGET_TYPEMASK_AZ_IGNORE
        | mavutil.mavlink.POSITION_TARGET_TYPEMASK_YAW_IGNORE
        | mavutil.mavlink.POSITION_TARGET_TYPEMASK_YAW_RATE_IGNORE
    )
    with lock:
        conn.mav.set_position_target_local_ned_send(
            int(time.monotonic() * 1000) & 0xFFFFFFFF,
            tgt_sys,
            tgt_comp,
            mavutil.mavlink.MAV_FRAME_BODY_NED,
            mask,
            0.0,
            0.0,
            0.0,
            float(vx),
            float(vy),
            float(vz),
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
        )


MAVLINK_CMD_SIM_RESET = 31000
STALE_RACE_MS = 8000


def reset_sim(conn, tgt_sys, tgt_comp, lock) -> None:
    """Reset the sim over the wire (no window focus needed), plus Backspace as backup."""
    with lock:
        conn.mav.command_long_send(
            tgt_sys,
            tgt_comp,
            MAVLINK_CMD_SIM_RESET,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
        )
    print("[reset] sent MAVLINK_CMD_SIM_RESET (31000)")
    try:
        import keyboard

        keyboard.send("backspace")
    except Exception:  # noqa: BLE001
        pass


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-restart", action="store_true")
    ap.add_argument(
        "--seq",
        choices=["velonly", "full"],
        default="velonly",
        help="velonly = body-velocity-setpoint test only (safe, no climb); full = att/thrust sweep",
    )
    ap.add_argument(
        "--coll-limit",
        type=int,
        default=3,
        help="Backspace-reset and abort once collisions exceed this",
    )
    args = ap.parse_args()
    sequence = SEQUENCE_VELONLY if args.seq == "velonly" else SEQUENCE

    print(f"Connecting to {MAVLINK_ENDPOINT} …")
    conn = mavutil.mavlink_connection(MAVLINK_ENDPOINT)
    conn.wait_heartbeat()
    tgt_sys, tgt_comp = conn.target_system, conn.target_component
    print(f"Heartbeat OK. target_system={tgt_sys}")

    sh = Shared()
    lock = threading.Lock()
    phase_ref = [""]
    saved: list[str] = []
    threading.Thread(target=mav_rx, args=(conn, sh), daemon=True).start()
    threading.Thread(target=vision_rx, args=(sh, phase_ref, saved), daemon=True).start()

    if not args.no_restart:
        reset_sim(conn, tgt_sys, tgt_comp, lock)

    # Wait for a FRESH live race (countdown finished AND not a stale leftover).
    # Commanding during the countdown DQs us; flying a stale race means the drone
    # already crashed. Keep resetting until a young live race appears.
    print("[race] waiting for a fresh green light…")
    t_wait = time.time()
    last_reset = time.time()
    while time.time() - t_wait < 40.0:
        with sh.lock:
            live = sh.is_live()
            age = sh.sim_boot_ms - sh.race_start_ms if sh.race_start_ms >= 0 else -1
        if live and 0 <= age < STALE_RACE_MS:
            break
        # Stale or not live: nudge a reset every few seconds.
        if live and age >= STALE_RACE_MS and time.time() - last_reset > 3.0:
            print(f"[race] stale race (age {age}ms) — resetting")
            reset_sim(conn, tgt_sys, tgt_comp, lock)
            last_reset = time.time()
        time.sleep(0.05)
    else:
        print(
            "[race] no fresh live race in 40s — click Race in the Training window (and keep it focused)."
        )
        sh.stop.set()
        return
    with sh.lock:
        base_coll = sh.collisions
        print(
            f"[race] LIVE & FRESH. sim_boot={sh.sim_boot_ms} start={sh.race_start_ms} "
            f"age={sh.sim_boot_ms - sh.race_start_ms}ms gate={sh.active_gate}"
        )

    # Arm only once live.
    with lock:
        conn.mav.command_long_send(
            tgt_sys,
            tgt_comp,
            mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM,
            0,
            1,
            0,
            0,
            0,
            0,
            0,
            0,
        )
    print("[arm] sent")

    dt = 1.0 / CMD_HZ
    results: list[str] = []
    aborted = False
    for name, secs, thr, rr, pr, yr, mode in sequence:
        phase_ref[0] = name
        accs: list[tuple[float, float, float]] = []
        gyros: list[tuple[float, float, float]] = []
        t_end = time.time() + secs
        while time.time() < t_end and not sh.stop.is_set():
            # Off-path / crash guard: reset the moment we start bouncing.
            with sh.lock:
                coll_now = sh.collisions - base_coll
            if coll_now > args.coll_limit:
                print(f"[abort] {coll_now} collisions during '{name}' — off path, resetting sim")
                reset_sim(conn, tgt_sys, tgt_comp, lock)
                aborted = True
                break
            if mode == "att":
                send_att(conn, tgt_sys, tgt_comp, lock, thr, rr, pr, yr)
            elif mode == "vel":
                send_vel_body(conn, tgt_sys, tgt_comp, lock, 2.0, 0.0, 0.0)
            # mode == "none": send nothing (isolates the velocity test)
            with sh.lock:
                if sh.imu is not None:
                    accs.append(sh.imu[:3])
                    gyros.append(sh.imu[3:])
            time.sleep(dt)
        if aborted:
            break
        a = np.array(accs) if accs else np.zeros((1, 3))
        g = np.array(gyros) if gyros else np.zeros((1, 3))
        with sh.lock:
            gate, coll = sh.active_gate, sh.collisions
        line = (
            f"  {name:9s} accel_mean=({a[:, 0].mean():+.2f},{a[:, 1].mean():+.2f},"
            f"{a[:, 2].mean():+.2f})  gyro_mean=({g[:, 0].mean():+.3f},"
            f"{g[:, 1].mean():+.3f},{g[:, 2].mean():+.3f})  gate={gate} coll={coll}"
        )
        print(line)
        results.append(line)

    # Cut throttle to settle
    phase_ref[0] = ""
    for _ in range(20):
        send_att(conn, tgt_sys, tgt_comp, lock, 0.05, 0.0, 0.0, 0.0)
        time.sleep(dt)

    sh.stop.set()
    time.sleep(0.6)
    print("\n=== SUMMARY ===")
    for line in results:
        print(line)
    print(f"frames saved: {len(saved)} -> {FRAME_DIR}")
    for n in saved:
        print("   ", n)


if __name__ == "__main__":
    main()
