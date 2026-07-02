"""VQ2 closed-loop race runner (vision + IMU only, no ground truth).

Control architecture (established by scripts/vq2/control_probe.py on build 3379):
- Reset over the wire with MAVLINK_CMD_SIM_RESET (31000) — no window focus needed.
- Wait for TRUE is_live (sim_boot_ms >= race_start_ms) before arming/commanding, else DQ.
- Fly with rate-mode SET_ATTITUDE_TARGET (body rates + collective thrust); velocity/position
  setpoints are ignored by the sim.

State we actually have: HIGHRES_IMU (accel+gyro), camera (30 Hz, 640x360), RACE_STATUS
(active_gate_index, timing), COLLISION. No attitude/position/velocity telemetry.

Controller (minimum viable):
- Attitude: roll/pitch from the accelerometer (low-pass), rates from the gyro.
- Leveling: PD body-rate command to hold roll~0 and a modest nose-down pitch (the 17.8-deg
  nose-down start already drives the drone forward under thrust).
- Steering: detect the bright-red gate (HSV), servo yaw to center it in the image.
- Thrust: modest near-hover constant (~0.28); the room has a LOW CEILING so we never climb hard.

Failsafes (all reset via SIM_RESET):
- collision watchdog (off path / crashing),
- standstill watchdog (frames stop changing => run is effectively done, per user),
- lost-live / finished.

Usage
-----
    uv run python scripts/vq2/race.py --attempts 3
    uv run python scripts/vq2/race.py --attempts 1 --record runs/vq2_0NN
"""

from __future__ import annotations

import argparse
import math
import socket
import struct
import threading
import time
from collections import deque
from pathlib import Path

import cv2
import numpy as np
from eskf import ESKF
from guidance import geometric_control
from pymavlink import mavutil

MAVLINK_ENDPOINT = "udpin:127.0.0.1:14550"
VISION_PORT = 5600
VISION_HEADER = "<IHHIIQ"
MAVLINK_CMD_SIM_RESET = 31000
STALE_RACE_MS = 8000

# ---- camera intrinsics (VQ2 spec) ----
IMG_W, IMG_H = 640, 360
CX, CY = 320.0, 180.0

# ---- control gains (defaults; override via CLI). Start from the proven-stable
# probe config: constant thrust + zero attitude push (sim self-damps) + gentle yaw. ----
CMD_HZ = 60.0
THRUST = 0.30  # near-hover collective (commands only act while ARMED — we re-arm each loop)
PITCH_DES_RAD = -0.15  # gentle nose-down to drive forward; accel-gated leveling holds it
KP_ROLL = 1.0  # accel-gated leveling (only trusts accel when |a|≈g, i.e. not mid-launch)
KP_PITCH = 1.2
KD_RATE = 0.05
KD_ROLL = 0.2  # STRONG roll-rate damping: accel-leveling gates off during the turn's
#                lateral accel, so without this the drone banks freely → floor. Roll-only
#                (not pitch) so the vertical/approach response stays crisp.
# Gyro-integrated complementary filter: keeps roll/pitch CONTINUOUS through maneuvers
# (accel-only froze at −22° during the gate-1 turn → drone flew blind & banked into the
# floor). accel-roll = −(body roll), so the gyro integrates with a sign flip.
GYRO_INT_SIGN = -1.0
ACC_CORR = 0.02  # weak pull toward the accel tilt when |a|≈g (mostly trust gyro short-term)
KP_YAW_PX = 1.2  # yaw_rate from normalized horizontal gate error (blob fallback)
KP_LEAD = 0.0  # yaw-lead from gate's image-space horizontal velocity (off by default)
# Metric PnP steering (preferred when the gate quad is cleanly detected):
KP_YAW_BEAR = 2.2  # yaw_rate per rad of bearing to the gate center (metric, ≈ blob centering)
KP_LEAD_M = 0.0  # yaw-lead per m/s metric lateral vel — OFF: hurt threading; needs metric
#                  vertical control + sign/tuning work before it pays off (turns).
SLOW_RANGE_M = 9.0  # start the close-range slowdown below this PnP range (m)
# Roll cross-track: yaw only POINTS at the gate; banking TRANSLATES the drone onto
# the gate centerline (fixes lateral offset + turns). Uses the metric lateral offset x.
# gate-1+ roll cross-track (ESKF lateral, PD) — OFF: even PD-damped banking won't converge the
# moving-turn lateral reactively. Needs a trajectory planner (task 9). Kept for future use.
KP_ROLL_GATE1 = 0.0
KD_ROLL_X = 0.0
KP_ROLL_X = 0.0  # roll cross-track OFF: it over-banked into the floor on the turn; yaw-only
#                  (bearing) tracked gate 1 far better. Re-enable only with strong roll damping.
ROLL_X_SIGN = -1.0  # VQ1 found roll sign inverted; verify empirically (watchdog protects)
MAX_BANK = 0.25  # clamp commanded bank angle (rad, ~14deg)
YAW_SIGN = 1.0  # flip if it steers the wrong way (watchdog protects)
SEARCH_YAW = 0.4  # slow rotate when no gate visible
MAX_YAW = 1.2  # clamp yaw rate (rad/s)
MAX_RATE = 3.0  # clamp roll/pitch body-rate commands (rad/s)

# Vertical visual servo on THRUST (no altitude sensor): keep the gate at a target
# image row. Accel is useless for attitude during the high-thrust launch, so we
# do NOT level from accel — the sim self-damps attitude when rates are ~0.
KP_THR_PX = 0.10  # thrust delta per normalized vertical gate error (blob fallback)
THR_SIGN = -1.0  # gate low in image (cy big) => too high => less thrust
CY_TARGET = 130.0  # target gate image row: tuned so PnP vertical y→0 at gate 0 (4/4 threads;
#                    95 hit the floor, 150 clipped the top bar)
THR_MIN, THR_MAX = 0.25, 0.36  # wide enough to actually descend to gate level
# Metric vertical (preferred): the 20deg cam up-tilt + forward pitch means a gate on
# the flight path sits BELOW the optical axis by ~z*K_TILT metres. Drive thrust to
# null (y_cam - z*K_TILT).
KP_THR_M = 0.05  # thrust delta per metre of vertical error
K_TILT = 0.5  # y/z slope of the flight path in the camera frame (tan(~26deg); tune)

# Close-range slowdown: taper forward pitch as the gate blob area grows (fine
# alignment at the 1.5 m opening + brake into turns).
AREA_SLOW = 45000.0  # px^2 at which the gate is "close" (frame is 640x360=230k)
PITCH_SLOW_FRAC = 0.4  # reduce forward pitch by up to this fraction when close
# Close-range metric vertical servo (arrests the sink into the floor at a low gate's base).
METRIC_VERT_RANGE = 3.5  # m — below this PnP range, use metric-y vertical instead of blob cy
KP_THR_Y = 0.06  # thrust delta per metre of vertical error at close range
KD_THR_Y = 0.05  # thrust delta per (m/s) of vertical closing rate — arrests the descent
K_TILT_CLOSE = 0.3  # y/z slope (gate sits ~this*z below the axis when on the flight path)
KV_THR_VZ = 0.06  # thrust per (m/s) of ESKF world-down velocity beyond the deadband
VZ_DEADBAND = 1.0  # only damp descents faster than this (m/s) — leaves gate-0 alone
KP_GATE_DOWN = 0.05  # thrust per metre of ESKF gate-below-body offset (gate-1+ vertical guidance)
GATE_DOWN_TARGET = 0.8  # aim the drone higher (+ve => fly higher); fixes gate-1 low arrival

USE_GUIDANCE = True  # gate 1+: use the geometric trajectory controller on the ESKF state
THROUGH_DIST = 2.5  # aim this far beyond the gate so the drone punches THROUGH the plane
LOOM_THRESH = 0.6  # optical-flow expansion above which we treat something as looming
OBS_AVOID_GAIN = 0.5  # metres of lateral aim-shift per unit looming (reactive avoidance)
GUIDE_K_RATE = 6.0  # body-rate gain for the guidance controller (offline-validated value)
GUIDE_MIN_GATE = 1  # apply guidance from this gate index (gate 0 needs the gentle reactive launch;
#                     the full geometric controller flips the drone from the pitched on-platform start)
PNP_LOG = False  # print PnP relative pose periodically (validation)

# constant-rate sign-probe (set via CLI)
CONST_MODE = False
CONST_ROLL = 0.0
CONST_PITCH = 0.0
CONST_YAW = 0.0

# ---- failsafe thresholds ----
COLL_LIMIT = 3
STANDSTILL_S = 4.0  # frames ~unchanged this long => run done (user heuristic)
STANDSTILL_DIFF = 2.0  # mean abs pixel diff below this = "not moving"
MAX_ATTEMPT_S = 90.0

# ---- HSV red gate ----
# Red wraps hue 0/180; the AI-GP gates are bright/saturated red-orange.
RED_LO1 = np.array([0, 90, 90])
RED_HI1 = np.array([12, 255, 255])
RED_LO2 = np.array([165, 90, 90])
RED_HI2 = np.array([180, 255, 255])
MIN_GATE_AREA = 250  # px^2
# Temporal-consistency filter: reject a PnP detection whose range jumps FARTHER than this
# from the tracked value in one step — kills spurious far background-gate detections that
# otherwise bank the guidance into the flanking pillars (gate-1 killer). Approach range
# only shrinks smoothly; a sudden jump to a larger range is a wrong-gate lock.
POSE_Z_JUMP = 5.0  # m
POSE_Z_MAX = 25.0  # m — a targeted gate is never this far in this course; reject outright


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
        self.last_coll_id = 0
        self.stop = threading.Event()

    def is_live(self) -> bool:
        if self.race_start_ms < 0 or self.race_finish_ns >= 0:
            return False
        return self.sim_boot_ms >= self.race_start_ms

    def race_age_ms(self) -> int:
        return self.sim_boot_ms - self.race_start_ms if self.race_start_ms >= 0 else -1


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
                sh.last_coll_id = int(getattr(msg, "id", 0))  # 1001 gate, 1002 environment
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


def _red_mask(img: np.ndarray) -> np.ndarray:
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, RED_LO1, RED_HI1) | cv2.inRange(hsv, RED_LO2, RED_HI2)
    return cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))


def detect_gate(img: np.ndarray) -> tuple[float, float, float] | None:
    """Return (cx, cy, area) of the largest red gate blob, or None."""
    mask = _red_mask(img)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    c = max(contours, key=cv2.contourArea)
    area = float(cv2.contourArea(c))
    if area < MIN_GATE_AREA:
        return None
    m = cv2.moments(c)
    if m["m00"] == 0:
        return None
    return (m["m10"] / m["m00"], m["m01"] / m["m00"], area)


# --- PnP relative pose (metric gate position from the 4 corners) ---
GATE_OUTER_M = 2.7  # red frame outer edge length (spec §3.7)
CAMERA_K = np.array([[320.0, 0, 320.0], [0, 320.0, 180.0], [0, 0, 1.0]], np.float64)  # VQ2 spec
_H = GATE_OUTER_M / 2.0
# SOLVEPNP_IPPE_SQUARE convention: TL, TR, BR, BL with +y UP (must match image order).
GATE_OBJ = np.array([[-_H, _H, 0], [_H, _H, 0], [_H, -_H, 0], [-_H, -_H, 0]], np.float64)


def _order_quad(pts: np.ndarray) -> np.ndarray:
    """Order 4 image points as TL, TR, BR, BL (for SOLVEPNP_IPPE_SQUARE)."""
    s = pts.sum(axis=1)
    d = np.diff(pts, axis=1).ravel()  # x - y
    return np.array(
        [pts[np.argmin(s)], pts[np.argmax(d)], pts[np.argmax(s)], pts[np.argmin(d)]],
        np.float64,
    )


def detect_gate_quad(img: np.ndarray) -> np.ndarray | None:
    """Return the gate's 4 outer corners (4,2) ordered TL,TR,BR,BL, or None."""
    mask = _red_mask(img)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    c = max(contours, key=cv2.contourArea)
    if cv2.contourArea(c) < MIN_GATE_AREA:
        return None
    peri = cv2.arcLength(c, True)
    approx = cv2.approxPolyDP(c, 0.04 * peri, True)
    if len(approx) != 4 or not cv2.isContourConvex(approx):
        return None
    return _order_quad(approx.reshape(4, 2).astype(np.float64))


# Camera(OpenCV: x-right,y-down,z-fwd, tilted 20deg UP) -> body FRD(x-fwd,y-right,z-down).
_M_AXIS = np.array([[0, 0, 1.0], [1.0, 0, 0], [0, 1.0, 0]])  # z_c->fwd, x_c->right, y_c->down
_TILT = math.radians(-20.0)  # camera looks 20deg up => rotate -20 about body-Y (nose-up)
_R_TILT = np.array(
    [[math.cos(_TILT), 0, math.sin(_TILT)], [0, 1, 0], [-math.sin(_TILT), 0, math.cos(_TILT)]]
)
CAM_TO_BODY = _R_TILT @ _M_AXIS


def euler_to_R(roll: float, pitch: float, yaw: float) -> np.ndarray:  # noqa: N802
    """Body->world rotation from ZYX Euler (roll, pitch, yaw), radians.

    Used to build a RELIABLE control-frame attitude for gate-1+ guidance from the
    accel-gated roll/pitch (which stay sane — see runs CSV) plus a gyro-integrated
    yaw, deliberately BYPASSING the ESKF attitude, which diverges under sustained
    flight + jumping PnP (est_pitch drifted −18°→−60° during gate 0)."""
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    rz = np.array([[cy, -sy, 0.0], [sy, cy, 0.0], [0.0, 0.0, 1.0]])
    ry = np.array([[cp, 0.0, sp], [0.0, 1.0, 0.0], [-sp, 0.0, cp]])
    rx = np.array([[1.0, 0.0, 0.0], [0.0, cr, -sr], [0.0, sr, cr]])
    return rz @ ry @ rx


def gate_relative_pose(corners: np.ndarray) -> tuple[float, float, float] | None:
    """Solve PnP → gate center in CAMERA frame (x right, y down, z forward), metres."""
    try:
        ok, _rvec, tvec = cv2.solvePnP(
            GATE_OBJ, corners, CAMERA_K, None, flags=cv2.SOLVEPNP_IPPE_SQUARE
        )
    except cv2.error:
        return None
    if not ok:
        return None
    x, y, z = tvec.ravel()
    if z <= 0:
        return None
    return (float(x), float(y), float(z))


def compute_looming(prev_gray: np.ndarray, gray: np.ndarray) -> tuple[float, float]:
    """Classical monocular obstacle cue from optical-flow EXPANSION (time-to-contact).
    Returns (looming, lateral_bias): looming>0 => something expanding ahead (approaching);
    lateral_bias>0 => more flow on the LEFT (obstacle left → steer right). No learning;
    a reactive SAFETY net for genuine looming obstacles (research: racing line avoids the
    rest). Weak on head-on/low-texture — used only as a gentle bias, not primary nav."""
    flow = cv2.calcOpticalFlowFarneback(prev_gray, gray, None, 0.5, 2, 13, 3, 5, 1.2, 0)
    h, w = flow.shape[:2]
    band = flow[h // 3 : 2 * h // 3]  # central horizontal band = flight path
    fx = band[..., 0]
    left, right = fx[:, : w // 2], fx[:, w // 2 :]
    looming = float(right.mean() - left.mean())  # outward expansion => approaching
    mag = np.abs(band).sum(axis=2)
    lat = float(mag[:, : w // 3].mean() - mag[:, 2 * w // 3 :].mean())  # closer side = faster flow
    return looming, lat


def closest_gate_pose(img: np.ndarray) -> tuple[float, float, float] | None:
    """PnP pose of the NEAREST red gate quad (data association: track the target gate,
    not the largest blob — prevents the drone chasing a far/wrong gate = fly-offs)."""
    mask = _red_mask(img)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    best: tuple[float, float, float] | None = None
    for c in contours:
        if cv2.contourArea(c) < MIN_GATE_AREA:
            continue
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.04 * peri, True)
        if len(approx) != 4 or not cv2.isContourConvex(approx):
            continue
        pose = gate_relative_pose(_order_quad(approx.reshape(4, 2).astype(np.float64)))
        if pose is not None and (best is None or pose[2] < best[2]):
            best = pose
    return best


def reset_sim(conn, tgt_sys, tgt_comp, lock) -> None:
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
    try:
        import keyboard

        keyboard.send("backspace")
    except Exception:  # noqa: BLE001
        pass


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


def send_att(conn, tgt_sys, tgt_comp, lock, thr, rr, pr, yr) -> None:
    rr = max(-MAX_RATE, min(MAX_RATE, rr))
    pr = max(-MAX_RATE, min(MAX_RATE, pr))
    yr = max(-MAX_RATE, min(MAX_RATE, yr))
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


def wait_fresh_live(sh, conn, tgt_sys, tgt_comp, lock, timeout=45.0) -> bool:
    """Reset and wait for a genuinely fresh race: live, young, gate 0, and the
    drone sitting still on the platform (no collisions accruing)."""

    def at_rest() -> bool:
        with sh.lock:
            live, gate, imu = sh.is_live(), sh.active_gate, sh.imu
        # race_start is a fixed ~boot constant, so "age" = sim uptime, NOT staleness.
        # Freshness = live + gate 0 + drone parked at the start pose (IMU ≈ gravity,
        # gyro ≈ 0). A crashed/embedded drone reads non-rest accel or nonzero gyro.
        if not (live and gate <= 0 and imu is not None):
            return False
        ax, ay, az, gx, gy, gz = imu
        # Tight: the platform rest pose is exactly (-3.0, 0, -9.34), gyro 0. A drone
        # stuck against a wall reads only *near*-rest — reject it so we keep resetting.
        return (
            abs(ax + 3.0) < 0.4
            and abs(ay) < 0.4
            and abs(az + 9.34) < 0.4
            and abs(gx) < 0.05
            and abs(gy) < 0.05
            and abs(gz) < 0.05
        )

    def rest_held(seconds: float) -> bool:
        held = time.time()
        while time.time() - held < seconds:
            if not at_rest():
                return False
            time.sleep(0.05)
        return True

    t0 = time.time()
    while time.time() - t0 < timeout:
        # CHECK FIRST: if already parked + live, go (never reset a good state — each
        # 31000 re-teleports into a transient).
        if rest_held(0.3):
            return True
        # Post-crash the race briefly goes finished (live=False) then the sim
        # auto-restarts. Nudge ONE reset, then wait PATIENTLY (no spamming, which
        # interferes with the auto-restart) for it to come back fresh.
        reset_sim(conn, tgt_sys, tgt_comp, lock)
        wt = time.time()
        while time.time() - wt < 8.0:
            if rest_held(0.3):
                return True
            time.sleep(0.2)
    with sh.lock:
        print(
            f"[wait_fresh_live] timeout — last imu={sh.imu} gate={sh.active_gate} live={sh.is_live()}"
        )
    return False


def run_attempt(sh, conn, tgt_sys, tgt_comp, lock, rec: Path | None) -> dict:
    # Acquire a genuinely clean start: fresh+live AND frames flowing (not a frozen
    # post-crash sim) AND no collisions accruing. Retry with resets otherwise.
    for _ in range(5):
        if not wait_fresh_live(sh, conn, tgt_sys, tgt_comp, lock):
            return {"ok": False, "reason": "no_fresh_live", "max_gate": -1, "dur": 0.0}
        with sh.lock:
            c0, f0 = sh.collisions, sh.frame_id
        time.sleep(0.5)
        with sh.lock:
            clean = sh.collisions == c0 and sh.frame_id != f0 and sh.is_live()
        if clean:
            break
        for _ in range(3):  # frozen/bad state → force spaced resets
            reset_sim(conn, tgt_sys, tgt_comp, lock)
            time.sleep(0.5)
    else:
        return {"ok": False, "reason": "no_clean_start", "max_gate": -1, "dur": 0.0}
    with sh.lock:
        base_coll = sh.collisions
        start_gate = sh.active_gate
    arm(conn, tgt_sys, tgt_comp, lock)
    time.sleep(0.1)

    dt = 1.0 / CMD_HZ
    t0 = time.time()
    last_arm = 0.0
    max_gate = start_gate
    roll_f: float | None = None  # init to first measurement to avoid a startup transient
    pitch_f: float | None = None
    prev_gray: np.ndarray | None = None
    prev_cx: float | None = None
    prev_x: float | None = None  # metric lateral offset (for lead-pursuit)
    xdot_s = 0.0  # smoothed metric lateral velocity
    prev_eym: float | None = None  # metric vertical error (for descent damping)
    eym_dot = 0.0
    # --- decoupled gate-1+ guidance state (bypasses the diverging ESKF) ---
    yaw_int = 0.0  # gyro-integrated heading for a self-consistent control world frame
    gb_body: np.ndarray | None = None  # latest PnP gate-in-body vector (FRD)
    gb_gate = -1  # gate index gb_body was captured for
    gb_t = 0.0  # capture time (freshness)
    guide_prel_prev: np.ndarray | None = None  # last gate-relative world position
    guide_prel_t = 0.0
    guide_prel_gate = -1  # gate index guide_prel_prev belongs to (don't diff across gates)
    guide_v = np.zeros(3)  # smoothed drone velocity (world), from PnP finite-diff
    track_z: float | None = None  # tracked gate range for temporal-consistency filtering
    track_gate = -1  # gate index track_z belongs to
    kf = ESKF()  # validated 15-state visual-inertial ESKF
    kf_gate: int | None = None  # gate the ESKF origin is anchored to
    pw_hist: deque = deque(maxlen=8)  # (t, gate_world_pos) for a smoothed PnP velocity baseline
    vfwd_res: list[float] = []  # validation: |est fwd speed - PnP closing speed|
    innov_res: list[float] = []  # validation: ESKF gate-position innovation (m)
    still_since = time.time()
    reason = "timeout"
    last_det: tuple[float, float, float] | None = None
    last_pose: tuple[float, float, float] | None = None  # PnP (x,y,z) at last detection
    ring: deque = deque(maxlen=6)  # recent frames for collision post-mortem
    traj: list[tuple] = []  # per-tick trajectory for offline analysis
    t_prev = time.time()  # for real-dt gyro integration
    n = 0
    while time.time() - t0 < MAX_ATTEMPT_S and not sh.stop.is_set():
        with sh.lock:
            imu = sh.imu
            frame = sh.frame
            gate = sh.active_gate
            coll = sh.collisions - base_coll
            finished = sh.race_finish_ns >= 0
        max_gate = max(max_gate, gate)

        # Keep the drone armed — a single arm after reset may not latch.
        if time.time() - last_arm > 0.5:
            arm(conn, tgt_sys, tgt_comp, lock)
            last_arm = time.time()

        if frame is not None:
            ring.append((frame, last_pose, gate))

        if finished:
            reason = "finished"
            break
        if coll > COLL_LIMIT:
            with sh.lock:
                cid = sh.last_coll_id
            kind = {1001: "GATE", 1002: "ENV"}.get(cid, str(cid))
            reason = f"collision({coll},{kind})"
            break

        # --- attitude: gyro-integrated complementary filter. Gyro keeps roll/pitch
        # continuous THROUGH maneuvers; accel corrects the DC only when |accel|≈g
        # (during the launch/turns accel reads thrust, not gravity, so it's gated). ---
        now = time.time()
        tick_dt = min(0.1, now - t_prev)
        t_prev = now
        gx = gy = 0.0
        if imu is not None:
            ax, ay, az, gx, gy, gz = imu
            roll_acc = math.atan2(-ay, -az)
            pitch_acc = math.atan2(ax, math.hypot(ay, az))
            amag = math.sqrt(ax * ax + ay * ay + az * az)
            acc_valid = 8.5 < amag < 11.0
            # ROLL: gyro-integrated complementary filter — stays CONTINUOUS through the
            # turn's lateral accel (accel-only froze at −22° → banked into the floor).
            if roll_f is None:
                roll_f = roll_acc
            else:
                roll_f += GYRO_INT_SIGN * gx * tick_dt
                if acc_valid:
                    roll_f += ACC_CORR * (roll_acc - roll_f)
            # PITCH: accel-gated (unchanged) — this drove the working 6/6 gate-0 flight;
            # gyro-integrating it changed the forward/descent balance and broke gate 0.
            if acc_valid:
                pitch_f = pitch_acc if pitch_f is None else pitch_f + 0.25 * (pitch_acc - pitch_f)
            # Gyro-integrated heading for the decoupled guidance world frame. Absolute
            # value is arbitrary (short gate-1 approach); only self-consistency matters.
            yaw_int += gz * tick_dt
            # Validated ESKF prediction + gravity attitude correction (runs alongside
            # for logging/validation; NO LONGER feeds gate-1 control — it diverges).
            kf.predict(np.array([ax, ay, az]), np.array([gx, gy, gz]), tick_dt)
            kf.update_gravity(np.array([ax, ay, az]))

        # --- vision servo: yaw centers gate horizontally, thrust centers it vertically ---
        yaw_rate = SEARCH_YAW * YAW_SIGN
        thrust = THRUST
        pitch_des_eff = PITCH_DES_RAD
        roll_target = 0.0
        if frame is not None:
            det = detect_gate(frame)
            pose = closest_gate_pose(frame)  # NEAREST gate (data association, anti-fly-off)
            # Temporal-consistency gate: drop spurious far/jumpy detections (a background
            # red quad seen when the near gate isn't a clean convex quad) — these banked
            # the guidance into the flanking pillars. Range only shrinks on approach.
            if pose is not None:
                pz = pose[2]
                if pz > POSE_Z_MAX:
                    pose = None  # implausibly far: never a targeted gate here
                elif track_gate != gate:  # new target gate: accept + reset tracker
                    track_z, track_gate = pz, gate
                elif track_z is not None and pz > track_z + POSE_Z_JUMP:
                    pose = None  # sudden far jump = wrong-gate lock; hold last good state
                else:
                    track_z = 0.5 * pz + 0.5 * track_z if track_z is not None else pz
            if pose is not None:
                last_pose = pose
                # Feed the gate (static landmark) to the ESKF. Re-anchor the world origin
                # when the target gate changes (or first sight); otherwise fuse.
                gate_body = CAM_TO_BODY @ np.array(pose)
                # Capture the PnP gate-in-body for the decoupled gate-1+ guidance.
                gb_body, gb_gate, gb_t = gate_body, gate, time.time()
                if kf_gate != gate or not kf.initialised:
                    kf.reanchor(gate_body)
                    kf_gate = gate
                else:
                    innov = kf.update_gate(gate_body)
                    # Re-anchor if the filter has clearly lost the gate — keeps innov ~1 m
                    # and the filter locked through gate transitions.
                    if innov > 3.0:
                        kf.reanchor(gate_body)
                    elif coll == 0:
                        innov_res.append(innov)
                # Validation: est velocity vs a smoothed PnP-derived velocity (independent).
                p_world = kf.R @ gate_body
                tnow = time.time()
                base = next((h for h in pw_hist if tnow - h[0] > 0.15), None)
                pw_hist.append((tnow, p_world))
                if base is not None:
                    v_pnp = -(p_world - base[1]) / (tnow - base[0])
                    if float(np.linalg.norm(v_pnp)) < 12.0 and coll == 0:
                        los = p_world / (np.linalg.norm(p_world) + 1e-6)
                        vfwd_res.append(abs(float((kf.v - v_pnp) @ los)))
            if det is not None:
                last_det = det
                cx, cy, area = det
                # Vertical thrust servo. FAR: blob-row cy (metric PnP-y is tilt-coupled at
                # range). CLOSE (z<METRIC_VERT_RANGE): metric PnP-y, where the tilt offset
                # z*K_TILT_CLOSE is small & reliable — climb/descend to center on the gate
                # so we don't sink into the floor at a low gate's base (gate-1 failure).
                if pose is not None and pose[2] < METRIC_VERT_RANGE:
                    ey_m = pose[1] - pose[2] * K_TILT_CLOSE  # +ve => gate below center => too high
                    ey_dot_raw = (ey_m - prev_eym) / max(dt, 1e-3) if prev_eym is not None else 0.0
                    eym_dot = 0.4 * ey_dot_raw + 0.6 * eym_dot  # smoothed vertical rate
                    prev_eym = ey_m
                    thrust = THRUST - KP_THR_Y * ey_m - KD_THR_Y * eym_dot
                else:
                    prev_eym = None
                    ey = (cy - CY_TARGET) / CY
                    thrust = THRUST + THR_SIGN * KP_THR_PX * ey
                # Lateral: prefer METRIC PnP bearing (cleaner than the blob centroid);
                # blob-centroid fallback when the quad isn't cleanly detected.
                if pose is not None:
                    x, _y, z = pose
                    bearing = math.atan2(x, z)
                    xdot_raw = (x - prev_x) / max(dt, 1e-3) if prev_x is not None else 0.0
                    xdot_s = 0.4 * xdot_raw + 0.6 * xdot_s
                    prev_x, prev_cx = x, None
                    yaw_rate = YAW_SIGN * (KP_YAW_BEAR * bearing + KP_LEAD_M * xdot_s)
                    # Cross-track: gate 1+ (turns) bank toward the gate centerline using the
                    # ESKF lateral offset (drift-free) — yaw alone can't translate the drone
                    # off its momentum line. Gate 0 keeps roll level (proven).
                    if gate >= 1 and kf.initialised and kf_gate == gate:
                        gate_right = float((-kf.R.T @ kf.p)[1])  # +ve => gate to the right
                        v_right = float((kf.R.T @ kf.v)[1])  # ESKF body-right velocity
                        # PD cross-track: bank toward the gate but DAMP with lateral velocity
                        # (P-only banking oscillated; the ESKF velocity term stabilises it).
                        cmd = KP_ROLL_GATE1 * gate_right - KD_ROLL_X * v_right
                        roll_target = ROLL_X_SIGN * max(-MAX_BANK, min(MAX_BANK, cmd))
                    else:
                        roll_target = ROLL_X_SIGN * max(-MAX_BANK, min(MAX_BANK, KP_ROLL_X * x))
                    close = max(0.0, min(1.0, 1.0 - z / SLOW_RANGE_M))
                    if PNP_LOG and n % 20 == 0:
                        print(
                            f"  t={time.time() - t0:4.1f}s PnP x={x:+.2f} z={z:.2f}m "
                            f"bear={math.degrees(bearing):+.0f} cy={cy:.0f}"
                        )
                else:
                    ex = (cx - CX) / CX
                    cx_dot = (cx - prev_cx) / max(dt, 1e-3) / CX if prev_cx is not None else 0.0
                    prev_cx, prev_x = cx, None
                    yaw_rate = YAW_SIGN * (KP_YAW_PX * ex + KP_LEAD * cx_dot)
                    close = min(1.0, area / AREA_SLOW)
                pitch_des_eff = PITCH_DES_RAD * (1.0 - PITCH_SLOW_FRAC * close)
            else:
                prev_cx = prev_x = None
        # ESKF vertical-velocity damping (real D-term), deadbanded: only arrest FAST
        # descents (>VZ_DEADBAND m/s) so the drone doesn't overshoot into the floor at a
        # low gate's base (gate-1 failure), without disturbing gate-0's gentle descent.
        if kf.initialised:
            vz = float(kf.v[2])  # world-down; +ve = descending
            if vz > VZ_DEADBAND:
                thrust += KV_THR_VZ * (vz - VZ_DEADBAND)
        thrust = max(THR_MIN, min(THR_MAX, thrust))
        yaw_rate = max(-MAX_YAW, min(MAX_YAW, yaw_rate))
        looming = lat_bias = 0.0
        if frame is not None:
            # standstill detection via frame diff
            gray = cv2.cvtColor(cv2.resize(frame, (160, 90)), cv2.COLOR_BGR2GRAY)
            if prev_gray is not None:
                diff = float(np.mean(np.abs(gray.astype(np.int16) - prev_gray.astype(np.int16))))
                if diff > STANDSTILL_DIFF:
                    still_since = time.time()
                looming, lat_bias = compute_looming(prev_gray, gray)  # reactive obstacle cue
            prev_gray = gray
            if time.time() - still_since > STANDSTILL_S:
                reason = "standstill"
                break

        # --- leveling PD (body rates); 0 gains => don't fight the sim's self-damping ---
        r = roll_f if roll_f is not None else 0.0
        p = pitch_f if pitch_f is not None else pitch_des_eff
        roll_rate = -KP_ROLL * (r - roll_target) - KD_ROLL * gx
        pitch_rate = -KP_PITCH * (p - pitch_des_eff) - KD_RATE * gy

        # GATE 1+ : geometric trajectory controller fed with DECOUPLED reliable state —
        # accel-gated attitude (roll_f/pitch_f) + PnP-direct relative position + PnP-diff
        # velocity — NOT the ESKF, which diverges under sustained flight + jumping PnP
        # (est_pitch −18°→−60° through gate 0) and tumbled the drone at gate 1. Flies a
        # smooth intercept arriving centered, which reactive yaw-only can't on the turn.
        # Gate 0 keeps its proven reactive path (no regression).
        gb_fresh = gb_body is not None and gb_gate == gate and (time.time() - gb_t) < 0.4
        if USE_GUIDANCE and gate >= GUIDE_MIN_GATE and roll_f is not None and gb_fresh:
            pitch_ctrl = pitch_f if pitch_f is not None else PITCH_DES_RAD
            r_ctrl = euler_to_R(roll_f, pitch_ctrl, yaw_int)  # reliable body->world
            p_rel = r_ctrl @ gb_body  # gate relative to drone, world (per-frame, drift-free)
            # Drone velocity from PnP finite-diff: gate is static so v_drone = -d(p_rel)/dt.
            tnow = time.time()
            if (
                guide_prel_prev is not None
                and guide_prel_gate == gate
                and tnow - guide_prel_t > 1e-3
            ):
                v_raw = -(p_rel - guide_prel_prev) / (tnow - guide_prel_t)
                if float(np.linalg.norm(v_raw)) < 15.0:  # reject PnP-jump spikes
                    guide_v = 0.5 * v_raw + 0.5 * guide_v  # EMA smooth
            guide_prel_prev, guide_prel_t, guide_prel_gate = p_rel, tnow, gate
            # Aim THROUGH_DIST beyond the gate so the drone carries velocity THROUGH the
            # plane (punch through) instead of decelerating to the center.
            d = float(np.linalg.norm(p_rel))
            target = p_rel + THROUGH_DIST * (p_rel / d) if d > 0.5 else p_rel
            # Reactive obstacle avoidance: shift the aim-point away from a looming obstacle.
            if looming > LOOM_THRESH and d > 2.0:
                shift = OBS_AVOID_GAIN * min(looming, 5.0) * (1.0 if lat_bias > 0 else -1.0)
                target = target + r_ctrl @ np.array([0.0, shift, 0.0])
            coll_thr, omega, _ = geometric_control(
                target, guide_v, r_ctrl, hover_thrust=THRUST, k_rate=GUIDE_K_RATE
            )
            thrust = coll_thr
            roll_rate, pitch_rate, yaw_rate = float(omega[0]), float(omega[1]), float(omega[2])

        # Constant-rate sign-probe mode: bypass the controller, send fixed rates.
        if CONST_MODE:
            roll_rate, pitch_rate, yaw_rate, thrust = CONST_ROLL, CONST_PITCH, CONST_YAW, THRUST

        send_att(conn, tgt_sys, tgt_comp, lock, thrust, roll_rate, pitch_rate, yaw_rate)

        px = py = pz = None
        if last_pose is not None:
            px, py, pz = last_pose
        est_roll, est_pitch, _est_yaw = kf.euler
        v_body = kf.R.T @ kf.v  # world vel -> body (fwd, right, down)
        traj.append(
            (
                round(time.time() - t0, 2),
                gate,
                coll,
                round(px, 2) if px is not None else "",
                round(py, 2) if py is not None else "",
                round(pz, 2) if pz is not None else "",
                round(thrust, 3),
                round(math.degrees(r), 1),  # roll (accel-gated, drives control)
                round(math.degrees(p), 1),  # pitch (accel-gated, drives control)
                round(yaw_rate, 2),
                round(math.degrees(est_roll), 1),  # roll (quaternion estimator)
                round(math.degrees(est_pitch), 1),  # pitch (quaternion estimator)
                round(float(v_body[0]), 2),  # est forward speed (m/s)
                round(float(v_body[2]), 2),  # est down speed (m/s)
            )
        )

        if rec is not None and frame is not None and n % 15 == 0:
            cv2.imwrite(str(rec / f"{int((time.time() - t0) * 1000):06d}_g{gate}.png"), frame)
        n += 1
        time.sleep(dt)

    # settle
    for _ in range(15):
        send_att(conn, tgt_sys, tgt_comp, lock, 0.05, 0.0, 0.0, 0.0)
        time.sleep(dt)
    ld = None
    if last_det is not None:
        ld = (round(last_det[0]), round(last_det[1]), round(last_det[2]))
    lp = None
    if last_pose is not None:
        lp = (round(last_pose[0], 2), round(last_pose[1], 2), round(last_pose[2], 2))
    # Dump the full trajectory for offline analysis.
    tdir = Path("runs")
    tdir.mkdir(parents=True, exist_ok=True)
    with (tdir / "vq2_traj_last.csv").open("w") as f:
        f.write(
            "t,gate,coll,x,y,z,thrust,roll_deg,pitch_deg,yaw_rate,"
            "est_roll,est_pitch,est_vfwd,est_vdown\n"
        )
        for row in traj:
            f.write(",".join(str(v) for v in row) + "\n")
    # Dump the pre-collision frames for post-mortem (top vs bottom vs side).
    if "collision" in reason:
        cdir = Path("runs/vq2_collisions")
        cdir.mkdir(parents=True, exist_ok=True)
        for i, (fr, ps, gt) in enumerate(ring):
            ptag = f"x{ps[0]:+.1f}y{ps[1]:+.1f}z{ps[2]:.1f}" if ps else "noPnP"
            cv2.imwrite(str(cdir / f"g{gt}_{i}_{ptag}.png"), fr)
    return {
        "ok": True,
        "reason": reason,
        "max_gate": max_gate,
        "start_gate": start_gate,
        "dur": round(time.time() - t0, 1),
        "last_gate_px": ld,  # (cx, cy, area) at last detection before end
        "pnp_xyz": lp,  # PnP (x=lat, y=vert-down, z=range) at last detection
        "vel_innov": round(float(np.mean(vfwd_res)), 2) if vfwd_res else None,  # est-vs-PnP m/s
        "eskf_innov": round(float(np.mean(innov_res)), 2)
        if innov_res
        else None,  # ESKF gate innov m
    }


def main() -> None:
    global THRUST, KP_YAW_PX, YAW_SIGN, KP_ROLL, KP_PITCH, KD_RATE, PITCH_DES_RAD, MAX_ATTEMPT_S
    global CONST_MODE, CONST_ROLL, CONST_PITCH, CONST_YAW, KP_THR_PX, CY_TARGET, PNP_LOG
    global KP_YAW_BEAR, KP_LEAD_M, KP_THR_M, K_TILT, KP_ROLL_X, ROLL_X_SIGN, KD_ROLL, USE_GUIDANCE
    ap = argparse.ArgumentParser()
    ap.add_argument("--attempts", type=int, default=1)
    ap.add_argument("--record", type=str, default=None)
    ap.add_argument("--thrust", type=float, default=THRUST)
    ap.add_argument("--kp-yaw", type=float, default=KP_YAW_PX)
    ap.add_argument("--yaw-sign", type=float, default=YAW_SIGN)
    ap.add_argument("--kp-roll", type=float, default=KP_ROLL)
    ap.add_argument("--kp-pitch", type=float, default=KP_PITCH)
    ap.add_argument("--kd", type=float, default=KD_RATE)
    ap.add_argument("--pitch-des", type=float, default=PITCH_DES_RAD)
    ap.add_argument("--kp-thr", type=float, default=KP_THR_PX)
    ap.add_argument("--cy-target", type=float, default=CY_TARGET)
    ap.add_argument("--max-s", type=float, default=MAX_ATTEMPT_S)
    ap.add_argument("--const-roll", type=float, default=None)
    ap.add_argument("--const-pitch", type=float, default=None)
    ap.add_argument("--const-yaw", type=float, default=None)
    ap.add_argument("--pnp-log", action="store_true", help="print PnP relative pose")
    ap.add_argument("--no-guidance", action="store_true", help="disable gate-1+ geometric guidance")
    ap.add_argument("--kp-bear", type=float, default=KP_YAW_BEAR)
    ap.add_argument("--kp-lead-m", type=float, default=KP_LEAD_M)
    ap.add_argument("--kp-thr-m", type=float, default=KP_THR_M)
    ap.add_argument("--k-tilt", type=float, default=K_TILT)
    ap.add_argument("--kp-roll-x", type=float, default=KP_ROLL_X)
    ap.add_argument("--roll-x-sign", type=float, default=ROLL_X_SIGN)
    ap.add_argument("--kd-roll", type=float, default=KD_ROLL)
    args = ap.parse_args()
    KD_ROLL = args.kd_roll
    PNP_LOG = args.pnp_log
    USE_GUIDANCE = not args.no_guidance
    KP_YAW_BEAR = args.kp_bear
    KP_LEAD_M = args.kp_lead_m
    KP_THR_M = args.kp_thr_m
    K_TILT = args.k_tilt
    KP_ROLL_X = args.kp_roll_x
    ROLL_X_SIGN = args.roll_x_sign

    if args.const_roll is not None or args.const_pitch is not None or args.const_yaw is not None:
        CONST_MODE = True
        CONST_ROLL = args.const_roll or 0.0
        CONST_PITCH = args.const_pitch or 0.0
        CONST_YAW = args.const_yaw or 0.0
    THRUST = args.thrust
    KP_YAW_PX = args.kp_yaw
    YAW_SIGN = args.yaw_sign
    KP_ROLL = args.kp_roll
    KP_PITCH = args.kp_pitch
    KD_RATE = args.kd
    PITCH_DES_RAD = args.pitch_des
    KP_THR_PX = args.kp_thr
    CY_TARGET = args.cy_target
    MAX_ATTEMPT_S = args.max_s

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

    print(
        f"config: THRUST={THRUST} PITCH_DES={PITCH_DES_RAD} KP_YAW={KP_YAW_PX} "
        f"YAW_SIGN={YAW_SIGN} KP_ROLL={KP_ROLL} KP_PITCH={KP_PITCH}"
    )
    for i in range(args.attempts):
        rec = None
        if args.record:
            rec = Path(args.record + f"_{i:02d}")
            rec.mkdir(parents=True, exist_ok=True)
        res = run_attempt(sh, conn, tgt_sys, tgt_comp, lock, rec)
        print(f"[attempt {i}] {res}")

    sh.stop.set()
    time.sleep(0.4)


if __name__ == "__main__":
    main()
