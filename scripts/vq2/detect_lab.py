"""Offline gate-detection lab — iterate perception on REAL captured frames, no sim.

The live gate-1 failure (2026-07-01) is perceptual: near gates the red frame blooms
into a huge diffuse glow (incl. floor reflections), the PnP quad stops resolving
(`range=nan`), and blob-centroid steering drags the drone into the environment.
This script runs the detectors over the collision frame dumps
(``runs/vq2_collisions/*.png``, filenames encode race.py's PnP pose at dump time)
and reports, per frame and in aggregate:

- current blob detection (race.detect_gate) and quad/PnP (race.closest_gate_pose)
- candidate improved detector (ring test: a real gate is a red RING around a
  non-red interior; glow/reflection is a SOLID red mass)

Usage
-----
    uv run python scripts/vq2/detect_lab.py                    # aggregate stats
    uv run python scripts/vq2/detect_lab.py --annotate 12      # write N annotated
                                                               # frames to data/vq2/lab/
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
import race as rc

FRAMES = Path(__file__).resolve().parents[2] / "runs" / "vq2_collisions"
OUT = Path(__file__).resolve().parents[2] / "data" / "vq2" / "lab"


# --- candidate red/orange mask ----------------------------------------------
# The AI-GP gate face is ORANGE-red (hue ~5..22 in OpenCV H<=180), but race.py's
# band stops at 12 — half the gate pixels get masked out, fragmenting the contour.
# The glow/reflection trap is DESATURATED dark red, so we raise the S/V floors.
ORANGE_LO = np.array([0, 120, 120])
ORANGE_HI = np.array([22, 255, 255])
WRAP_LO = np.array([170, 120, 120])
WRAP_HI = np.array([180, 255, 255])
MIN_POSE_AREA = 800.0  # px^2 — below this a quad is too small for a trustworthy PnP


def red_mask_v2(img: np.ndarray) -> np.ndarray:
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, ORANGE_LO, ORANGE_HI) | cv2.inRange(hsv, WRAP_LO, WRAP_HI)
    return cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))


def gate_quad_v2(img: np.ndarray) -> tuple[np.ndarray | None, str]:
    """Improved gate quad: red mask -> contour -> RING check -> quad fit.

    Returns (quad TL,TR,BR,BL or None, reject-reason). Differences vs v1:
    - ring test: reject solid red masses (glow/reflection) — a gate frame's
      contour interior is mostly NOT red;
    - accepts 4..8-vertex convex approximations and falls back to minAreaRect
      (the AI-GP banner and perspective often break the strict 4-vertex fit);
    - rejects contours touching the image border (partial gates can't give
      4 true corners -> PnP would be garbage).
    """
    mask = rc._red_mask(img)  # proven band; v2's gains are structural, not chromatic
    h, w = mask.shape
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    best: tuple[float, np.ndarray] | None = None
    reason = "no-red"
    for c in contours:
        area = cv2.contourArea(c)
        if area < MIN_POSE_AREA:
            continue
        x, y, bw, bh = cv2.boundingRect(c)
        if x <= 1 or y <= 1 or x + bw >= w - 1 or y + bh >= h - 1:
            reason = "border"
            continue  # touches border: partial gate
        # RING test: fraction of red inside the contour's bounding box core.
        core = mask[y + bh // 4 : y + 3 * bh // 4, x + bw // 4 : x + 3 * bw // 4]
        if core.size == 0:
            continue
        fill = float(np.count_nonzero(core)) / core.size
        # A gate is a hollow ring (opening 1.5/2.7 of the outer edge => core mostly
        # empty), but a TILTED bounding box leaks frame pixels into the core, so the
        # threshold must be loose: solid glow/reflection masses read ~0.9-1.0.
        if fill > 0.55:
            reason = "solid"
            continue
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.04 * peri, True)
        if len(approx) == 4 and cv2.isContourConvex(approx):
            quad = rc._order_quad(approx.reshape(4, 2).astype(np.float64))
        else:
            # banner/perspective broke the 4-vertex fit: use the min-area rectangle
            # if it explains the contour well (area ratio close to 1)
            rect = cv2.minAreaRect(c)
            rect_area = rect[1][0] * rect[1][1]
            if rect_area <= 0 or area / rect_area < 0.5:
                reason = "rect-fit"
                continue
            quad = rc._order_quad(cv2.boxPoints(rect).astype(np.float64))
        if best is None or area > best[0]:
            best = (float(area), quad)
    if best is None:
        return None, reason
    return best[1], "ok"


def pose_v2(img: np.ndarray) -> tuple[float, float, float] | None:
    quad, _ = gate_quad_v2(img)
    return None if quad is None else rc.gate_relative_pose(quad)


# --- harness -----------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--annotate", type=int, default=0, help="write N annotated frames")
    ap.add_argument("--glob", default="g1_*.png", help="frame filename filter")
    args = ap.parse_args()

    files = sorted(FRAMES.glob(args.glob))
    if not files:
        print(f"no frames match {args.glob} in {FRAMES}")
        return
    n = len(files)
    stats = {"blob": 0, "quad_v1": 0, "quad_v2": 0}
    v2_reasons: dict[str, int] = {}
    annotated = 0
    OUT.mkdir(parents=True, exist_ok=True)

    for f in files:
        img = cv2.imread(str(f))
        if img is None:
            n -= 1
            continue
        blob = rc.detect_gate(img)
        v1 = rc.closest_gate_pose(img)
        quad2, why = gate_quad_v2(img)
        v2 = rc.gate_relative_pose(quad2) if quad2 is not None else None
        stats["blob"] += blob is not None
        stats["quad_v1"] += v1 is not None
        stats["quad_v2"] += v2 is not None
        if v2 is None:
            v2_reasons[why] = v2_reasons.get(why, 0) + 1

        if args.annotate and annotated < args.annotate and (v2 is not None) != (v1 is not None):
            vis = img.copy()
            if blob is not None:
                cv2.circle(vis, (int(blob[0]), int(blob[1])), 6, (255, 255, 0), 2)
            if quad2 is not None:
                cv2.polylines(vis, [quad2.astype(np.int32)], True, (0, 255, 0), 2)
            label = f"v1={'Y' if v1 else 'n'} v2={'Y' if v2 else 'n'}"
            if v2 is not None:
                label += f" z2={v2[2]:.1f}m"
            cv2.putText(vis, label, (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 1)
            cv2.imwrite(str(OUT / f"lab_{f.name}"), vis)
            annotated += 1

    print(f"--- detection over {n} frames ({args.glob}) ---")
    for k, v in stats.items():
        print(f"  {k:8s}: {v:4d}  ({100.0 * v / max(1, n):.0f}%)")
    if v2_reasons:
        print(f"  v2 rejects: {v2_reasons}")
    if annotated:
        print(f"  wrote {annotated} annotated frames to {OUT}")


if __name__ == "__main__":
    main()
