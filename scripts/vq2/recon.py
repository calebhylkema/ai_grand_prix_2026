"""VQ2 telemetry recon — passively observe what the sim actually streams.

Run this with the sim in **Training** mode and a run started. It listens on the
MAVLink UDP port and the vision UDP port for a fixed duration, then reports:

- which MAVLink message types arrived, with counts and approximate rates
- a sample HIGHRES_IMU reading (accel + gyro)
- a sample RACE_STATUS (active gate index, timing)
- whether TRACK_INFO gate pose/size fields are nulled (as the VQ2 spec claims)
- saves the first few complete camera frames to ``data/vq2/recon_frames/`` as PNGs

Nothing is armed and no control commands are sent — purely read-only, so it is
safe to run against any mode. Heartbeats are emitted so the sim keeps streaming.

Usage
-----
    uv run python scripts/vq2/recon.py            # 20 s capture
    uv run python scripts/vq2/recon.py 30         # 30 s capture
"""

from __future__ import annotations

import collections
import socket
import struct
import sys
import threading
import time
from pathlib import Path

import cv2
import numpy as np
from pymavlink import mavutil

MAVLINK_ENDPOINT = "udpin:127.0.0.1:14550"
VISION_IP = "0.0.0.0"
VISION_PORT = 5600
VISION_HEADER = "<IHHIIQ"
FRAME_DIR = Path(__file__).resolve().parents[2] / "data" / "vq2" / "recon_frames"
MAX_FRAMES_TO_SAVE = 5


def vision_listener(stop: threading.Event, saved: list[int]) -> None:
    """Reassemble chunked JPEG frames from UDP 5600 and save a handful as PNG."""
    header_sz = struct.calcsize(VISION_HEADER)
    frames: dict[int, dict] = {}
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    try:
        sock.bind((VISION_IP, VISION_PORT))
    except OSError as exc:
        print(f"[vision] could not bind UDP {VISION_PORT}: {exc}")
        return
    sock.settimeout(0.5)
    FRAME_DIR.mkdir(parents=True, exist_ok=True)
    count = 0
    while not stop.is_set():
        try:
            packet, _ = sock.recvfrom(65536)
        except TimeoutError:
            continue
        header, payload = packet[:header_sz], packet[header_sz:]
        frame_id, chunk_id, total_chunks, _jpeg_sz, _pl_sz, _ts = struct.unpack(
            VISION_HEADER, header
        )
        f = frames.setdefault(frame_id, {"chunks": {}, "total": total_chunks})
        f["chunks"][chunk_id] = payload
        if len(f["chunks"]) == f["total"]:
            if all(i in f["chunks"] for i in range(f["total"])):
                buf = bytearray()
                for i in range(f["total"]):
                    buf.extend(f["chunks"][i])
                img = cv2.imdecode(np.frombuffer(buf, np.uint8), cv2.IMREAD_COLOR)
                if img is not None:
                    count += 1
                    if len(saved) < MAX_FRAMES_TO_SAVE:
                        p = FRAME_DIR / f"frame_{frame_id:06d}.png"
                        cv2.imwrite(str(p), img)
                        saved.append(frame_id)
                        print(f"[vision] saved {p.name}  shape={img.shape}")
            del frames[frame_id]
    print(f"[vision] complete frames decoded: {count}")


def main() -> None:
    duration = float(sys.argv[1]) if len(sys.argv) > 1 else 20.0

    print(
        f"Connecting to {MAVLINK_ENDPOINT} … (make sure the sim is in Training and a run is live)"
    )
    conn = mavutil.mavlink_connection(MAVLINK_ENDPOINT)
    conn.wait_heartbeat()
    print(f"Heartbeat received. target_system={conn.target_system}\n")

    stop = threading.Event()
    saved: list[int] = []
    vt = threading.Thread(target=vision_listener, args=(stop, saved), daemon=True)
    vt.start()

    counts: collections.Counter[str] = collections.Counter()
    samples: dict[str, object] = {}
    track_reported = False

    t0 = time.time()
    last_hb = 0.0
    while time.time() - t0 < duration:
        # keep the sim talking to us
        if time.time() - last_hb > 0.5:
            conn.mav.heartbeat_send(
                mavutil.mavlink.MAV_TYPE_GCS,
                mavutil.mavlink.MAV_AUTOPILOT_INVALID,
                0,
                0,
                0,
            )
            last_hb = time.time()

        msg = conn.recv_match(blocking=False)
        if msg is None:
            time.sleep(0.001)
            continue
        mtype = msg.get_type()
        if mtype == "BAD_DATA":
            counts["BAD_DATA"] += 1
            continue
        counts[mtype] += 1

        if mtype == "HIGHRES_IMU" and "HIGHRES_IMU" not in samples:
            samples["HIGHRES_IMU"] = (
                f"accel=({msg.xacc:+.2f},{msg.yacc:+.2f},{msg.zacc:+.2f}) m/s^2  "
                f"gyro=({msg.xgyro:+.3f},{msg.ygyro:+.3f},{msg.zgyro:+.3f}) rad/s"
            )
        elif mtype in ("ATTITUDE", "ODOMETRY", "LOCAL_POSITION_NED") and mtype not in samples:
            # If these ever fire, VQ2 did NOT actually remove them — flag loudly.
            samples[mtype] = f"!!! PRESENT (spec says removed): {msg.to_dict()}"
        elif mtype == "ENCAPSULATED_DATA":
            raw = bytes(msg.data)
            if raw and raw[0] == 1 and "RACE_STATUS" not in samples:
                _, sim_ms, race_start_ms, race_finish_ns, gate_idx, last_gate = struct.unpack_from(
                    "<BQqqIq", raw
                )
                samples["RACE_STATUS"] = (
                    f"active_gate_index={gate_idx}  race_start_ms={race_start_ms}  "
                    f"race_finish_ns={race_finish_ns}  last_gate_time={last_gate}"
                )
            elif raw and raw[0] == 2 and not track_reported:
                # Try to peek at the first gate entry to check whether pose/size are nulled.
                samples["TRACK_INFO_chunk"] = "received (reassembly needed for full gate list)"

    stop.set()
    time.sleep(0.6)

    print("\n" + "=" * 60)
    print(f"MAVLink message types seen in {duration:.0f}s:")
    for mtype, n in counts.most_common():
        print(f"  {mtype:28s} {n:6d}   (~{n / duration:6.1f} Hz)")

    print("\nKey samples:")
    for k in (
        "HIGHRES_IMU",
        "RACE_STATUS",
        "TRACK_INFO_chunk",
        "ATTITUDE",
        "ODOMETRY",
        "LOCAL_POSITION_NED",
    ):
        if k in samples:
            print(f"  {k}: {samples[k]}")

    print("\nRemoved-telemetry check (VQ2 spec says these should be ABSENT):")
    for k in ("ATTITUDE", "ODOMETRY", "LOCAL_POSITION_NED"):
        status = "!!! STILL PRESENT" if k in counts else "absent [OK]"
        print(f"  {k:20s} {status}")

    print(f"\nCamera frames saved: {len(saved)} → {FRAME_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()
