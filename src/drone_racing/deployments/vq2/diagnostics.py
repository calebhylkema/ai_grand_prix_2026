"""VQ2 diagnostics: passive telemetry recon + offline perception lab.

Two subcommands:

``recon`` — passively observe what the sim actually streams. Run with the sim in
**Training** mode and a run started. Listens on the MAVLink UDP port and the
vision UDP port for a fixed duration, then reports:

- which MAVLink message types arrived, with counts and approximate rates
- a sample HIGHRES_IMU reading (accel + gyro)
- a sample RACE_STATUS (active gate index, timing)
- whether TRACK_INFO gate pose/size fields are nulled (as the VQ2 spec claims)
- saves the first few complete camera frames to ``.local/datasets/vq2/recon_frames/``

Nothing is armed and no control commands are sent — purely read-only, so it is
safe to run against any mode. Heartbeats are emitted so the sim keeps streaming.

``lab`` — run the CURRENT perception module (``perception.detect`` +
``perception.detect_line``) offline over a directory of saved frames (collision
dumps, trace archives) and report per-frame detections; optionally write
annotated copies for eyeballing. Replaces the archived race_v1 detect_lab.

Usage
-----
    uv run python -m drone_racing.deployments.vq2.diagnostics recon           # 20 s
    uv run python -m drone_racing.deployments.vq2.diagnostics recon --secs 30
    uv run python -m drone_racing.deployments.vq2.diagnostics lab             # collision dumps
    uv run python -m drone_racing.deployments.vq2.diagnostics lab DIR --annotate 20
"""

from __future__ import annotations

import argparse
import collections
import math
import socket
import struct
import threading
import time
from pathlib import Path

import cv2
import numpy as np
from pymavlink import mavutil

from drone_racing.deployments.vq2 import perception
from drone_racing.deployments.vq2.vision import FrameReassembler
from drone_racing.utils.workspace import Workspace

MAVLINK_ENDPOINT = "udpin:127.0.0.1:14550"
VISION_IP = "0.0.0.0"
VISION_PORT = 5600
MAX_FRAMES_TO_SAVE = 15  # first N unique consecutive frames saved for offline PnP

_WS = Workspace.locate()
FRAME_DIR = _WS.datasets / "vq2" / "recon_frames"
LAB_DEFAULT_DIR = _WS.runs / "vq2_collisions"
LAB_OUT_DIR = _WS.datasets / "vq2" / "lab"


# ---------------------------------------------------------------------------
# recon
# ---------------------------------------------------------------------------
def vision_listener(stop: threading.Event, reasm: FrameReassembler, saved: list[int]) -> None:
    """Reassemble chunked JPEG frames from UDP 5600 (dedup-aware) and save the first
    ``MAX_FRAMES_TO_SAVE`` UNIQUE frames as PNG. Stream health accumulates in ``reasm.stats``."""
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    try:
        sock.bind((VISION_IP, VISION_PORT))
    except OSError as exc:
        print(f"[vision] could not bind UDP {VISION_PORT}: {exc}")
        return
    sock.settimeout(0.5)
    FRAME_DIR.mkdir(parents=True, exist_ok=True)
    while not stop.is_set():
        try:
            packet, _ = sock.recvfrom(65536)
        except TimeoutError:
            continue
        cf = reasm.push(packet)
        if cf is not None and len(saved) < MAX_FRAMES_TO_SAVE:
            p = FRAME_DIR / f"frame_{cf.frame_id:06d}.png"
            cv2.imwrite(str(p), cf.image)
            saved.append(cf.frame_id)


def _report_imu_time_usec(times: list[int], duration: float) -> None:
    """Summarize HIGHRES_IMU.time_usec: is it real, monotonic, usable for per-sample dt?"""
    if not times:
        print("  no HIGHRES_IMU samples received")
        return
    n = len(times)
    nonzero = sum(1 for t in times if t > 0)
    distinct = len(set(times))
    print(
        f"  samples={n} (~{n / duration:.0f} Hz wall)  nonzero={nonzero}/{n}  distinct={distinct}"
    )
    if nonzero < n or distinct <= 1:
        print("  -> ZERO/CONSTANT: NOT usable; the ESKF must fall back to arrival-clock dt")
        return
    t = np.asarray(times, float)
    steps = np.diff(t)
    mono = bool(np.all(steps >= 0))
    diffs = steps[steps > 0] * 1e-6  # positive inter-sample dt (seconds)
    span_s = (t[-1] - t[0]) * 1e-6
    med = float(np.median(diffs))
    print(
        f"  timestamp span={span_s:.2f}s  rate_from_span={(n - 1) / span_s:.1f} Hz  monotonic={mono}"
    )
    print(
        f"  dt(ms): mean={diffs.mean() * 1e3:.2f}  median={med * 1e3:.2f}  "
        f"p95={np.quantile(diffs, 0.95) * 1e3:.2f}  max={diffs.max() * 1e3:.2f}"
    )
    gaps2x = int(np.count_nonzero(diffs >= 2.0 * med))
    print(f"  gaps >= 2x median dt: {gaps2x}  (median rate ~{1.0 / med:.0f} Hz)")
    ok = mono and distinct > n * 0.5
    print(f"  -> {'USABLE for per-sample dt' if ok else 'SUSPECT; verify before trusting'}")


def _archive_recon_frames() -> None:
    """Move any existing recon frames into a timestamped subdir so a new capture starts
    clean (old frames must not enter the geometry report)."""
    existing = sorted(FRAME_DIR.glob("frame_*.png"))
    if not existing:
        return
    dest = FRAME_DIR / f"prev_{time.strftime('%Y%m%d_%H%M%S')}"
    dest.mkdir(parents=True, exist_ok=True)
    for p in existing:
        p.rename(dest / p.name)
    print(f"[recon] archived {len(existing)} stale frames → {dest.name}")


def _report_stream(reasm: FrameReassembler, duration: float) -> None:
    """Report vision-stream freshness: unique FPS, dup/stale, invalid, expiries, gaps."""
    s = reasm.stats
    print("\nVision stream (dedup-aware reassembly):")
    print(
        f"  unique frames decoded={s.unique_decoded}  "
        f"(~{s.unique_decoded / duration:.1f} Hz wall, ~{s.fps():.1f} Hz by sim-clock)"
    )
    print(
        f"  duplicate/stale chunks={s.dup_stale_chunks}  invalid={s.invalid_chunks}  "
        f"incomplete expired={s.incomplete_expired}  decode failures={s.decode_failures}"
    )
    print(
        f"  frame-id gaps={s.frame_id_gaps}  timestamp gaps(>1.5x)={s.ts_gaps}  resets={s.resets}"
    )


def recon(duration: float) -> None:
    """Passive MAVLink + vision capture; report message types and key samples."""
    print(
        f"Connecting to {MAVLINK_ENDPOINT} … (make sure the sim is in Training and a run is live)"
    )
    conn = mavutil.mavlink_connection(MAVLINK_ENDPOINT)
    conn.wait_heartbeat()
    print(f"Heartbeat received. target_system={conn.target_system}\n")

    _archive_recon_frames()  # so stale frames cannot contaminate this capture
    stop = threading.Event()
    saved: list[int] = []
    reasm = FrameReassembler()
    vt = threading.Thread(target=vision_listener, args=(stop, reasm, saved), daemon=True)
    vt.start()

    counts: collections.Counter[str] = collections.Counter()
    samples: dict[str, object] = {}
    imu_times: list[int] = []  # HIGHRES_IMU.time_usec — gates the ESKF timing design
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

        if mtype == "HIGHRES_IMU":
            imu_times.append(int(getattr(msg, "time_usec", 0)))
            if "HIGHRES_IMU" not in samples:
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

    print("\nHIGHRES_IMU.time_usec (gates the per-sample ESKF timing design):")
    _report_imu_time_usec(imu_times, duration)

    _report_stream(reasm, duration)
    print(f"\nCamera frames saved: {len(saved)} unique → {FRAME_DIR}")
    print("=" * 60)


# ---------------------------------------------------------------------------
# lab
# ---------------------------------------------------------------------------
def _draw_structured(
    img: np.ndarray,
    cands: list[perception.GateObservation],
    selected: perception.GateObservation | None,
) -> None:
    """Overlay candidates, corners, the selected active target, and PnP info."""
    for o in cands:
        x, y, w, h = o.bbox
        is_sel = selected is not None and o.track_id == selected.track_id
        full = o.visibility is perception.Visibility.FULL
        color = (0, 0, 255) if is_sel else ((0, 255, 0) if full else (0, 180, 255))
        cv2.rectangle(img, (x, y), (x + w, y + h), color, 3 if is_sel else 1)
        for corners, ccol in ((o.outer_corners, (255, 255, 255)), (o.inner_corners, (255, 255, 0))):
            if corners is not None:
                for px, py in corners:
                    cv2.circle(img, (int(px), int(py)), 3, ccol, -1)
        label = o.output_mode.value
        if o.pose_cam is not None:
            label += f" z{o.pose_cam[2]:.1f}m r{o.reproj_rms:.1f}"
        elif o.reject_reason:
            label += f" ({o.reject_reason})"
        cv2.putText(img, label, (x, max(12, y - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)


def _lab_structured(files: list[Path], annotate: int) -> None:
    """Run detect_candidates + GateTracker + solve_gate_pose over frames (temporal order)."""
    tracker = perception.GateTracker()
    n_cand = n_full = n_partial = n_posed = n_selected = 0
    annotated = 0
    if annotate:
        LAB_OUT_DIR.mkdir(parents=True, exist_ok=True)
    for f in files:
        img = cv2.imread(str(f))
        if img is None:
            continue
        cands = perception.detect_candidates(img)
        for obs in cands:
            perception.solve_gate_pose(obs)
        selected = tracker.step(cands)  # active_gate index is unknown offline
        n_cand += len(cands)
        n_full += sum(o.visibility is perception.Visibility.FULL for o in cands)
        n_partial += sum(o.visibility is perception.Visibility.PARTIAL for o in cands)
        n_posed += sum(o.pose_cam is not None for o in cands)
        n_selected += selected is not None
        parts = []
        for o in cands:
            tag = "F" if o.visibility is perception.Visibility.FULL else "P"
            sel = "*" if (selected is not None and o.track_id == selected.track_id) else ""
            z = f" z{o.pose_cam[2]:.1f}" if o.pose_cam is not None else ""
            parts.append(f"{sel}{tag}[{o.output_mode.value}] b{math.degrees(o.bearing):+.0f}{z}")
        print(f"  {f.name}: {'; '.join(parts) if parts else '(none)'}")
        if annotate and annotated < annotate:
            vis = img.copy()
            _draw_structured(vis, cands, selected)
            cv2.imwrite(str(LAB_OUT_DIR / f.name), vis)
            annotated += 1
    print(
        f"\nSummary: {n_cand} candidates over {len(files)} frames "
        f"(FULL={n_full} PARTIAL={n_partial} posed={n_posed}); active target in {n_selected} frames"
    )
    if annotated:
        print(f"Annotated {annotated} frames → {LAB_OUT_DIR}")


def lab(frames_dir: Path, annotate: int, structured: bool = False) -> None:
    """Run the perception stack offline over saved frames and report.

    ``structured=True`` runs the new pipeline (``detect_candidates`` + ``GateTracker`` +
    ``solve_gate_pose``) with temporal continuity and richer overlays; otherwise the
    legacy ``detect``/``detect_line`` path.
    """
    files = sorted(p for p in frames_dir.glob("*.png")) + sorted(frames_dir.glob("*.jpg"))
    if not files:
        print(f"No frames (*.png / *.jpg) in {frames_dir}")
        return
    print(f"{len(files)} frames from {frames_dir}")
    if structured:
        _lab_structured(files, annotate)
        return
    n_gate = n_posed = n_line = 0
    annotated = 0
    if annotate:
        LAB_OUT_DIR.mkdir(parents=True, exist_ok=True)
    for f in files:
        img = cv2.imread(str(f))
        if img is None:
            continue
        dets = perception.detect(img)
        line = perception.detect_line(img)
        if dets:
            n_gate += 1
            n_posed += any(t.pose_cam is not None for t in dets)
        if line is not None:
            n_line += 1
        parts = []
        for t in dets:
            z = f" z{t.pose_cam[2]:.1f}m" if t.pose_cam is not None else ""
            parts.append(f"gate cy{t.cy:.0f} a{t.area:.0f} b{math.degrees(t.bearing):+.0f}deg{z}")
        if line is not None:
            parts.append(f"line aim b{math.degrees(line.aim_bearing):+.0f}deg y{line.aim_cy:.0f}")
        print(f"  {f.name}: {'; '.join(parts) if parts else '(nothing)'}")
        if annotate and annotated < annotate:
            vis = img.copy()
            for t in dets:
                cv2.circle(vis, (int(t.cx), int(t.cy)), 6, (0, 255, 0), 2)
            if line is not None:
                cv2.circle(vis, (int(line.aim_cx), int(line.aim_cy)), 5, (255, 255, 0), 2)
            cv2.imwrite(str(LAB_OUT_DIR / f.name), vis)
            annotated += 1
    print(
        f"\nSummary: gates in {n_gate}/{len(files)} frames ({n_posed} with PnP pose), "
        f"racing line in {n_line}/{len(files)}"
    )
    if annotated:
        print(f"Annotated {annotated} frames → {LAB_OUT_DIR}")


# ---------------------------------------------------------------------------
# replay
# ---------------------------------------------------------------------------
def replay(run_path: Path) -> None:
    """Re-run the ESKF over a recorded ``.npz`` (ARRIVAL-order) and report diagnostics.

    Arrival-order replay reproduces live fuse-in-receive-order behaviour (incl. camera
    transport/decode latency); it is NOT exact capture-time delayed fusion.
    """
    from drone_racing.deployments.vq2.recorder import load_run, replay_arrival_order

    res = replay_arrival_order(load_run(run_path))
    print(f"Arrival-order replay of {run_path}")
    print(f"  IMU: predicts={res.n_predict}  dup(dt<=0)={res.n_dup}  drop(gap)={res.n_drop}")
    n_updates = res.n_fix_seen - res.reanchors
    rate = res.n_fix_accepted / n_updates if n_updates else 1.0
    print(
        f"  fixes: seen={res.n_fix_seen}  reanchors={res.reanchors}  "
        f"accepted={res.n_fix_accepted}/{n_updates} ({rate:.0%})"
    )
    if res.nis:
        nis = np.array(res.nis)
        print(f"  NIS: median={np.median(nis):.2f}  p95={np.quantile(nis, 0.95):.2f}")
    if len(res.pos):
        hs = res.horizontal_speed
        print(
            f"  final pos={np.round(res.pos[-1], 2)}  vel={np.round(res.vel[-1], 2)} m/s  "
            f"euler(deg)={np.round(np.degrees(res.euler[-1]), 1)}"
        )
        print(f"  horizontal speed: max={hs.max():.2f}  final={hs[-1]:.2f} m/s")
    for gate, vel in sorted(res.vel_at_gate_change.items()):
        v = np.array(vel)
        print(
            f"  @ active_gate={gate}: world v={np.round(v, 2)} m/s  "
            f"|v_horiz|={np.hypot(v[0], v[1]):.2f}"
        )


# ---------------------------------------------------------------------------
def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="VQ2 diagnostics: telemetry recon + perception lab.")
    sub = ap.add_subparsers(dest="cmd", required=True)

    ap_recon = sub.add_parser("recon", help="passively observe what the sim streams")
    ap_recon.add_argument("--secs", type=float, default=20.0, help="capture duration (s)")

    ap_lab = sub.add_parser("lab", help="run perception offline over saved frames")
    ap_lab.add_argument(
        "frames_dir",
        nargs="?",
        type=Path,
        default=LAB_DEFAULT_DIR,
        help=f"directory of frames (default: {LAB_DEFAULT_DIR})",
    )
    ap_lab.add_argument(
        "--annotate", type=int, default=0, metavar="N", help="write N annotated frames to lab dir"
    )
    ap_lab.add_argument(
        "--structured",
        action="store_true",
        help="use the new detect_candidates+tracker+PnP pipeline",
    )

    ap_replay = sub.add_parser("replay", help="re-run the ESKF over a recorded .npz run")
    ap_replay.add_argument("run", type=Path, help="path to a recorded run .npz")

    args = ap.parse_args(argv)
    if args.cmd == "recon":
        recon(args.secs)
    elif args.cmd == "replay":
        replay(args.run)
    else:
        lab(args.frames_dir, args.annotate, args.structured)


if __name__ == "__main__":
    main()
