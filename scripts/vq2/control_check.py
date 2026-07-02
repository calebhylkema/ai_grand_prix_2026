"""VQ2 control characterization — prove the primitives before building on them.

We have NO attitude/position/velocity telemetry, but we DO have HIGHRES_IMU. The
gyro is a direct, ground-truth measurement of body angular rate, and the accel is a
direct measurement of specific force (gravity direction at rest / near hover). The
camera optical flow tells us which way we are actually translating. This script uses
those to answer, with numbers, the questions every controller depends on:

  --test rates      For each body axis, command a +then- rate step and check the gyro
                    tracks it (SIGN + SCALE + lag). If commanded roll_rate=+0.8 does
                    not give gyro_x≈+0.8, our rate-mode sign is wrong — a root-cause bug.

  --test hover      Sweep collective thrust and report the vertical specific force, to
                    locate the hover thrust (where the drone neither climbs nor sinks).

  --test translate  Hold a small steady tilt on one axis and read the camera optical
                    flow to confirm which way the drone TRANSLATES (the tilt->motion
                    sign map — what steering actually relies on).

Run against the sim in the Training window with a run live. Each sub-test resets the
sim over the wire (MAVLINK_CMD_SIM_RESET) between maneuvers so every measurement starts
from the clean platform pose. Short bursts only (low-ceiling room).

Usage
-----
    uv run python scripts/vq2/control_check.py --test rates
    uv run python scripts/vq2/control_check.py --test hover
    uv run python scripts/vq2/control_check.py --test translate
"""

from __future__ import annotations

import argparse
import contextlib
import socket
import struct
import sys
import threading
import time

import cv2
import numpy as np
from pymavlink import mavutil

MAVLINK_ENDPOINT = "udpin:127.0.0.1:14550"
VISION_PORT = 5600
VISION_HEADER = "<IHHIIQ"
MAVLINK_CMD_SIM_RESET = 31000
CMD_HZ = 60.0
HOVER_THRUST = 0.30  # working estimate; the hover test refines it

# --- rate-command calibration (measured 2026-07-01 via --test rates) ---
# The sim's rate loop responds INVERTED and ~2.5x scaled: actual_gyro ≈ GAIN * commanded.
# To achieve a desired actual body rate, command desired / GAIN. Verified linear at 0.4/0.8.
RATE_GAIN = (-2.55, -2.55, -2.35)  # (roll, pitch, yaw): actual = GAIN * commanded (scale→~1.0)
CALIBRATE = False  # when True, send_att interprets its rates as DESIRED actual rates


def desired_to_cmd(rr: float, pr: float, yr: float) -> tuple[float, float, float]:
    """Map desired actual body rates -> raw command values via the measured inverse gain."""
    return (rr / RATE_GAIN[0], pr / RATE_GAIN[1], yr / RATE_GAIN[2])


class Shared:
    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.imu: tuple[float, float, float, float, float, float] | None = None
        self.frame: np.ndarray | None = None
        self.frame_id = -1
        self.race_start_ms = -1
        self.race_finish_ns = -1
        self.sim_boot_ms = 0
        self.active_gate = -1
        self.collisions = 0
        self.last_collision: tuple[int, int] | None = None  # (id: 1001=gate/1002=env, threat)
        self.stop = threading.Event()

    def is_live(self) -> bool:
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
                sh.last_collision = (int(msg.id), int(msg.threat_level))
        elif t == "ENCAPSULATED_DATA":
            raw = bytes(msg.data)
            if raw and raw[0] == 1:
                _, sim_ms, rs_ms, fin_ns, gate, _lg = struct.unpack_from("<BQqqIq", raw)
                with sh.lock:
                    sh.sim_boot_ms, sh.race_start_ms = sim_ms, rs_ms
                    sh.race_finish_ns, sh.active_gate = fin_ns, gate


def vision_rx(sh: Shared) -> None:
    header_sz = struct.calcsize(VISION_HEADER)
    frames: dict[int, dict] = {}
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.bind(("0.0.0.0", VISION_PORT))
    sock.settimeout(0.5)
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
            if img is not None:
                with sh.lock:
                    sh.frame, sh.frame_id = img, fid


def send_att(conn, tgt_sys, tgt_comp, lock, thr, rr, pr, yr) -> None:
    if CALIBRATE:
        rr, pr, yr = desired_to_cmd(rr, pr, yr)
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


def arm(conn, tgt_sys, tgt_comp, lock) -> None:
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


def reset_sim(conn, tgt_sys, tgt_comp, lock) -> None:
    with lock:
        conn.mav.command_long_send(tgt_sys, tgt_comp, MAVLINK_CMD_SIM_RESET, 0, 0, 0, 0, 0, 0, 0, 0)


def at_rest(sh: Shared) -> bool:
    """Drone parked on the start platform (accel≈(-3,0,-9.34), gyro≈0), race live, gate 0."""
    with sh.lock:
        live, gate, imu = sh.is_live(), sh.active_gate, sh.imu
    if not (live and gate <= 0 and imu is not None):
        return False
    ax, ay, az, gx, gy, gz = imu
    return (
        abs(ax + 3.0) < 0.4
        and abs(ay) < 0.4
        and abs(az + 9.34) < 0.4
        and abs(gx) < 0.05
        and abs(gy) < 0.05
        and abs(gz) < 0.05
    )


def wait_fresh(sh, conn, tgt_sys, tgt_comp, lock, timeout=40.0) -> bool:
    """Patiently get a clean platform start. Check FIRST (never reset a good state);
    else nudge ONE reset and wait PATIENTLY for the sim to come back — no spamming,
    which fights the sim's own restart."""

    def rest_held(seconds: float) -> bool:
        held = time.time()
        while time.time() - held < seconds:
            if not at_rest(sh):
                return False
            time.sleep(0.05)
        return True

    t0 = time.time()
    while time.time() - t0 < timeout:
        if rest_held(0.3):
            return True
        reset_sim(conn, tgt_sys, tgt_comp, lock)
        print("[reset] sent — waiting for a clean platform start…")
        wt = time.time()
        while time.time() - wt < 8.0:
            if rest_held(0.3):
                return True
            time.sleep(0.2)
    return False


def _arm_hold(sh, conn, tgt_sys, tgt_comp, lock) -> None:
    """Arm and keep armed briefly (a single arm may not latch)."""
    for _ in range(3):
        arm(conn, tgt_sys, tgt_comp, lock)
        time.sleep(0.05)


def _burst(
    sh, conn, tgt_sys, tgt_comp, lock, thr, rr, pr, yr, secs
) -> tuple[np.ndarray, np.ndarray, int]:
    """Send a fixed command for `secs`, re-arming each loop; collect accel + gyro."""
    dt = 1.0 / CMD_HZ
    accs: list[tuple] = []
    gyros: list[tuple] = []
    with sh.lock:
        c0 = sh.collisions
    last_arm = 0.0
    t_end = time.time() + secs
    while time.time() < t_end and not sh.stop.is_set():
        if time.time() - last_arm > 0.4:
            arm(conn, tgt_sys, tgt_comp, lock)
            last_arm = time.time()
        send_att(conn, tgt_sys, tgt_comp, lock, thr, rr, pr, yr)
        with sh.lock:
            if sh.imu is not None:
                accs.append(sh.imu[:3])
                gyros.append(sh.imu[3:])
        time.sleep(dt)
    with sh.lock:
        coll = sh.collisions - c0
    a = np.array(accs) if accs else np.zeros((1, 3))
    g = np.array(gyros) if gyros else np.zeros((1, 3))
    return a, g, coll


# ---------------------------------------------------------------------------
def test_rates(sh, conn, tgt_sys, tgt_comp, lock, rate: float = 0.8) -> None:
    """Command a rate step on each axis; verify the gyro tracks it (sign + scale).

    Expected (MAVLink body FRD, rate-mode SET_ATTITUDE_TARGET): a commanded body rate
    on an axis should appear as the SAME-SIGN gyro reading on that axis. A sign flip or
    a scale far from 1.0 is a control-mapping bug that dooms every downstream controller.
    """
    # ONE continuous flight (one reset only): roll±, pitch±, yaw±, each a short rate
    # burst then a brief level settle. The gyro responds in <0.1 s, so even as the drone
    # drifts forward off the pitched platform we capture a clean per-axis response before
    # anything crashes. + and - back-to-back per axis partly cancel the drift.
    seq = [
        ("roll", 0, +1.0),
        ("roll", 0, -1.0),
        ("pitch", 1, +1.0),
        ("pitch", 1, -1.0),
        ("yaw", 2, +1.0),
        ("yaw", 2, -1.0),
    ]
    print("\n=== RATE CONTROL CHECK (gyro = ground truth) ===")
    print(f"commanding ±{rate} rad/s per axis at hover thrust {HOVER_THRUST}")
    if not wait_fresh(sh, conn, tgt_sys, tgt_comp, lock):
        print("  NO FRESH START — is a run live in the Training window?")
        return
    _arm_hold(sh, conn, tgt_sys, tgt_comp, lock)
    _burst(sh, conn, tgt_sys, tgt_comp, lock, HOVER_THRUST, 0.0, 0.0, 0.0, 0.4)  # lift
    lines: list[str] = []
    for name, idx, sign in seq:
        cmd = [0.0, 0.0, 0.0]
        cmd[idx] = sign * rate
        _a, g, coll = _burst(
            sh, conn, tgt_sys, tgt_comp, lock, HOVER_THRUST, cmd[0], cmd[1], cmd[2], 0.35
        )
        # median of the second half of the burst (skip the ramp-in transient)
        half = g[len(g) // 2 :] if len(g) > 3 else g
        meas = float(np.median(half[:, idx]))
        other = [float(np.median(half[:, j])) for j in range(3) if j != idx]
        scale = meas / (sign * rate) if rate else 0.0
        ok = "OK" if (meas * sign > 0 and abs(meas) > 0.2) else "?? CHECK SIGN/SCALE"
        lines.append(
            f"  {name:5s} cmd={sign * rate:+.2f}  gyro[{idx}]={meas:+.3f} "
            f"(scale {scale:+.2f})  cross={other[0]:+.2f},{other[1]:+.2f}  coll={coll}  {ok}"
        )
        _burst(sh, conn, tgt_sys, tgt_comp, lock, HOVER_THRUST, 0.0, 0.0, 0.0, 0.25)  # settle
    print()
    for line in lines:
        print(line)
    # settle
    for _ in range(15):
        send_att(conn, tgt_sys, tgt_comp, lock, 0.05, 0.0, 0.0, 0.0)
        time.sleep(1.0 / CMD_HZ)


# ---------------------------------------------------------------------------
def test_hover(sh, conn, tgt_sys, tgt_comp, lock) -> None:
    """Sweep thrust; report vertical specific force to locate hover.

    At true hover the accel magnitude ≈ g (9.81) and stays steady. Below hover the
    drone sinks (|accel|<g net); above it climbs (|accel|>g). We report |accel| and
    the body-z accel per thrust so hover is where |accel|≈g with the least motion.
    """
    thrusts = [0.24, 0.27, 0.30, 0.33, 0.36]
    print("\n=== HOVER THRUST SWEEP (accel |a|≈g at hover) ===")
    print("specific force |a| = thrust/mass; hover is where |a|≈g=9.81\n")
    lines: list[str] = []
    for thr in thrusts:
        # Clean reset per level; measure |a| in the FIRST 0.2s (before the drone moves
        # or hits anything — a collision spikes accel and would corrupt the reading).
        if not wait_fresh(sh, conn, tgt_sys, tgt_comp, lock):
            lines.append(f"  thr={thr:.2f}: NO FRESH START (skip)")
            continue
        _arm_hold(sh, conn, tgt_sys, tgt_comp, lock)
        a, _g, coll = _burst(sh, conn, tgt_sys, tgt_comp, lock, thr, 0.0, 0.0, 0.0, 0.25)
        amag = float(np.median(np.linalg.norm(a, axis=1)))
        note = " <- near hover" if abs(amag - 9.81) < 1.5 else ""
        lines.append(f"  thr={thr:.2f}  |a|={amag:6.2f} (g=9.81)  coll={coll}{note}")
    print()
    for line in lines:
        print(line)
    for _ in range(15):
        send_att(conn, tgt_sys, tgt_comp, lock, 0.05, 0.0, 0.0, 0.0)
        time.sleep(1.0 / CMD_HZ)


# ---------------------------------------------------------------------------
def _flow_dir(prev: np.ndarray, cur: np.ndarray) -> tuple[float, float]:
    """Mean optical-flow vector (px). +u = scene moves right, +v = scene moves down.
    Scene moving right ⇒ drone translating LEFT (and vice versa)."""
    flow = cv2.calcOpticalFlowFarneback(prev, cur, None, 0.5, 2, 15, 3, 5, 1.2, 0)
    return float(flow[..., 0].mean()), float(flow[..., 1].mean())


def test_translate(sh, conn, tgt_sys, tgt_comp, lock) -> None:
    """Hold a small steady tilt on roll / pitch and read optical flow to learn which
    way the drone actually TRANSLATES (the tilt→motion sign map steering relies on)."""
    print("\n=== TRANSLATION DIRECTION CHECK (optical flow = motion truth) ===")
    print("scene-flow +u=right,+v=down  => drone moves OPPOSITE the scene flow\n")
    # hold a rate briefly to build a tilt, then hold ~level and glide; watch the flow
    trials = [
        ("roll+", 0, +0.6),
        ("roll-", 0, -0.6),
        ("pitch+", 1, +0.6),
        ("pitch-", 1, -0.6),
    ]
    for name, idx, rate in trials:
        if not wait_fresh(sh, conn, tgt_sys, tgt_comp, lock):
            print(f"  {name}: NO FRESH START (skip)")
            continue
        _arm_hold(sh, conn, tgt_sys, tgt_comp, lock)
        _burst(sh, conn, tgt_sys, tgt_comp, lock, HOVER_THRUST, 0.0, 0.0, 0.0, 0.3)
        cmd = [0.0, 0.0, 0.0]
        cmd[idx] = rate
        # tip in for a short time to establish a tilt
        _burst(sh, conn, tgt_sys, tgt_comp, lock, HOVER_THRUST, cmd[0], cmd[1], cmd[2], 0.25)
        # now glide (hold attitude) and sample optical flow
        dt = 1.0 / CMD_HZ
        us: list[float] = []
        vs: list[float] = []
        prev_gray: np.ndarray | None = None
        t_end = time.time() + 0.6
        last_arm = 0.0
        with sh.lock:
            c0 = sh.collisions
        while time.time() < t_end and not sh.stop.is_set():
            if time.time() - last_arm > 0.4:
                arm(conn, tgt_sys, tgt_comp, lock)
                last_arm = time.time()
            send_att(conn, tgt_sys, tgt_comp, lock, HOVER_THRUST, 0.0, 0.0, 0.0)
            with sh.lock:
                frame = sh.frame
            if frame is not None:
                gray = cv2.cvtColor(cv2.resize(frame, (160, 90)), cv2.COLOR_BGR2GRAY)
                if prev_gray is not None:
                    u, v = _flow_dir(prev_gray, gray)
                    us.append(u)
                    vs.append(v)
                prev_gray = gray
            time.sleep(dt)
        with sh.lock:
            coll = sh.collisions - c0
        mu = float(np.median(us)) if us else 0.0
        mv = float(np.median(vs)) if vs else 0.0
        moves = []
        if abs(mu) > 0.05:
            moves.append("LEFT" if mu > 0 else "RIGHT")
        if abs(mv) > 0.05:
            moves.append("UP" if mv > 0 else "DOWN")
        print(
            f"  {name:7s} scene_flow=(u={mu:+.2f},v={mv:+.2f}) "
            f"=> drone moves {'+'.join(moves) if moves else 'negligible'}  coll={coll}"
        )
    for _ in range(15):
        send_att(conn, tgt_sys, tgt_comp, lock, 0.05, 0.0, 0.0, 0.0)
        time.sleep(1.0 / CMD_HZ)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--test", choices=["rates", "hover", "translate"], required=True)
    ap.add_argument("--rate", type=float, default=0.8, help="rate-step magnitude (rad/s)")
    ap.add_argument(
        "--calib", action="store_true", help="apply measured rate calibration (expect scale ~+1.0)"
    )
    args = ap.parse_args()
    global CALIBRATE
    CALIBRATE = args.calib
    with contextlib.suppress(Exception):  # Windows consoles default to cp1252
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    print(f"Connecting to {MAVLINK_ENDPOINT} …")
    conn = mavutil.mavlink_connection(MAVLINK_ENDPOINT)
    conn.wait_heartbeat()
    tgt_sys, tgt_comp = conn.target_system, conn.target_component
    print(f"Heartbeat OK. target_system={tgt_sys}")

    sh = Shared()
    lock = threading.Lock()
    threading.Thread(target=mav_rx, args=(conn, sh), daemon=True).start()
    threading.Thread(target=vision_rx, args=(sh,), daemon=True).start()
    time.sleep(0.5)

    try:
        if args.test == "rates":
            test_rates(sh, conn, tgt_sys, tgt_comp, lock, args.rate)
        elif args.test == "hover":
            test_hover(sh, conn, tgt_sys, tgt_comp, lock)
        elif args.test == "translate":
            test_translate(sh, conn, tgt_sys, tgt_comp, lock)
    finally:
        sh.stop.set()
        time.sleep(0.4)


if __name__ == "__main__":
    main()
