"""Course-map miner: reconstruct gate geometry from saved fly_trace frames.

The strategic pivot (2026-07-09, user-approved): stop flying the course from
instantaneous detections and reactive branches — the course is FIXED and attempts are
unlimited, so build a map once and fly the map. The AX-batch forensics showed the
reactive controller latching three DIFFERENT gates (PnP z 27.2 -> 21.2 -> 17.1 m in
0.3 s, an impossible 35 m/s "closure") because nothing in the stack knows where the
gates actually are.

Method: any frame with TWO PnP-valid gates gives their relative position directly
(no odometry, no velocity estimate needed):

    d_cam  = pose_j - pose_i                      # camera frame
    d_body = CAM_TO_BODY @ d_cam                  # FRD body frame
    d_lvl  = Ry(pitch) @ Rx(roll) @ d_body        # gravity-leveled (yaw-free up/fwd)

The leveled vertical component (up = -z in FRD/NED) and horizontal distance are
yaw-invariant; the horizontal direction is expressed in the drone's heading frame,
which is consistent across runs on the same course segment (every run flies pad ->
gate 0 on nearly the same heading). Single-gate sightings are also dumped (with the
active_gate context from the trace filename) for identity/cluster analysis.

Usage:
    uv run python -m drone_racing.deployments.vq2.mapping [trace_dir ...]
        (default: every prev_* under .local/datasets/vq2/fly_trace)

Writes .local/datasets/vq2/map/observations.csv and prints the co-visible pair table.
"""

from __future__ import annotations

import argparse
import csv
import math
import re
from pathlib import Path

import cv2
import numpy as np

from drone_racing.deployments.vq2 import perception as gates
from drone_racing.utils.workspace import Workspace

_WS = Workspace.locate()
TRACE_ROOT = _WS.datasets / "vq2" / "fly_trace"
OUT_DIR = _WS.datasets / "vq2" / "map"

# trace filename telemetry: NNN_tX.XX_r{roll}p{pitch}_vz{vz}_gz{gz}_g{active_gate}_...
_NAME_RE = re.compile(
    r"^(\d+)_t([\d.]+)_r([+-]?\d+)p([+-]?\d+)_vz([+-]?[\d.]+)_gz([+-]?[\d.]+)_g(-?\d+)_"
)


def _level_rot(roll: float, pitch: float) -> np.ndarray:
    """Body->leveled-world rotation (ZYX with yaw=0), radians. Matches estimator ZYX."""
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    ry = np.array([[cp, 0.0, sp], [0.0, 1.0, 0.0], [-sp, 0.0, cp]])
    rx = np.array([[1.0, 0.0, 0.0], [0.0, cr, -sr], [0.0, sr, cr]])
    return ry @ rx


def scan(dirs: list[Path]) -> list[dict[str, object]]:
    """All PnP-valid gate observations (one row per gate per frame), leveled."""
    rows: list[dict[str, object]] = []
    for d in dirs:
        for f in sorted(d.glob("*.png")):
            m = _NAME_RE.match(f.name)
            if m is None:
                continue
            t_rel = float(m.group(2))
            roll = math.radians(float(m.group(3)))
            pitch = math.radians(float(m.group(4)))
            active_gate = int(m.group(7))
            img = cv2.imread(str(f))
            if img is None:
                continue
            r_lvl = _level_rot(roll, pitch)
            for g in gates.detect(img):
                if g.pose_cam is None:
                    continue
                p_lvl = r_lvl @ (gates.CAM_TO_BODY @ np.array(g.pose_cam))
                rows.append(
                    {
                        "run": d.name,
                        "frame": f.name.split("_", 1)[0],
                        "t": t_rel,
                        "active_gate": active_gate,
                        "area": g.area,
                        "cy": round(g.cy, 1),
                        "z_cam": round(g.pose_cam[2], 2),
                        "bearing_deg": round(math.degrees(g.bearing), 1),
                        "fwd": round(float(p_lvl[0]), 2),
                        "right": round(float(p_lvl[1]), 2),
                        "up": round(float(-p_lvl[2]), 2),  # FRD: up = -z
                    }
                )
    return rows


def pair_table(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    """Co-visible pair deltas (near gate -> far gate), the yaw-free map constraints."""
    pairs: list[dict[str, object]] = []
    by_frame: dict[tuple[object, object], list[dict[str, object]]] = {}
    for r in rows:
        by_frame.setdefault((r["run"], r["frame"]), []).append(r)
    for (run, frame), obs in by_frame.items():
        if len(obs) < 2:
            continue
        obs = sorted(obs, key=lambda o: float(o["z_cam"]))  # type: ignore[arg-type]
        for i in range(len(obs) - 1):
            a, b = obs[i], obs[i + 1]
            dfwd = float(b["fwd"]) - float(a["fwd"])  # type: ignore[arg-type]
            drgt = float(b["right"]) - float(a["right"])  # type: ignore[arg-type]
            dup = float(b["up"]) - float(a["up"])  # type: ignore[arg-type]
            pairs.append(
                {
                    "run": run,
                    "frame": frame,
                    "t": a["t"],
                    "active_gate": a["active_gate"],
                    "z_near": a["z_cam"],
                    "z_far": b["z_cam"],
                    "d_fwd": round(dfwd, 2),
                    "d_right": round(drgt, 2),
                    "d_up": round(dup, 2),
                    "d_horiz": round(math.hypot(dfwd, drgt), 2),
                }
            )
    return pairs


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="Mine gate geometry from saved fly_trace frames.")
    ap.add_argument(
        "trace_dirs",
        nargs="*",
        type=Path,
        help=f"trace archive dirs (default: every prev_* under {TRACE_ROOT})",
    )
    args = ap.parse_args(argv)
    dirs = args.trace_dirs if args.trace_dirs else sorted(TRACE_ROOT.glob("prev_*"))
    rows = scan(dirs)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    if rows:
        with (OUT_DIR / "observations.csv").open("w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
    print(f"{len(rows)} posed-gate observations from {len(dirs)} runs")
    print("\n--- single sightings (identity clues: t + active_gate + z) ---")
    for r in rows:
        print(
            f"  {r['run']}/{r['frame']} t{r['t']:<5} g{r['active_gate']} "
            f"a{r['area']:<6.0f} z{r['z_cam']:<6} b{r['bearing_deg']:+6.1f}deg "
            f"fwd{r['fwd']:+7.2f} right{r['right']:+7.2f} up{r['up']:+6.2f}"
        )
    pairs = pair_table(rows)
    print(f"\n--- co-visible pairs ({len(pairs)}): near gate -> far gate ---")
    for p in pairs:
        print(
            f"  {p['run']}/{p['frame']} t{p['t']:<5} g{p['active_gate']} "
            f"z {p['z_near']}->{p['z_far']}: dfwd {p['d_fwd']:+6.2f} "
            f"dright {p['d_right']:+6.2f} dup {p['d_up']:+6.2f} (horiz {p['d_horiz']})"
        )


if __name__ == "__main__":
    main()
