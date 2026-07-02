"""One-off: evaluate gates.detect/nearest over the collision frame dumps."""

from __future__ import annotations

from pathlib import Path

import cv2
import gates
import race as rc

FRAMES = Path(__file__).resolve().parents[2] / "runs" / "vq2_collisions"

for pat in ("g0_*.png", "g1_*.png"):
    files = sorted(FRAMES.glob(pat))
    n = 0
    stats = {"any_target": 0, "with_pose": 0, "blob_v1": 0, "pose_v1": 0}
    for f in files:
        img = cv2.imread(str(f))
        if img is None:
            continue
        n += 1
        t = gates.nearest(img)
        stats["any_target"] += t is not None
        stats["with_pose"] += t is not None and t.pose_cam is not None
        stats["blob_v1"] += rc.detect_gate(img) is not None
        stats["pose_v1"] += rc.closest_gate_pose(img) is not None
    print(f"--- {pat}: {n} frames ---")
    for k, v in stats.items():
        print(f"  {k:11s}: {v:4d} ({100.0 * v / max(1, n):.0f}%)")
