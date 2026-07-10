"""Offline PnP forensics on saved fly_trace frames (phantom gate-1 range hunt).

Every corner run sights gate 1 at b+31..36 deg with PnP range 19-31 m while the
kinematics say ~10-13 m — the metric branch rejects the fix (FIX_RANGE_MAX) and the
corner goes blind. This annotates what the detector actually saw: red-mask contour,
fitted quad, per-candidate PnP z, plus a bbox-height sanity range (864/bh: gates are
world-upright so the projected height is yaw-foreshortening-free).

Usage: uv run python scripts/vq2/_pnp_probe.py <frame.png> [...]
Writes annotated copies to data/vq2/lab/probe_<name>.
"""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import gates
import numpy as np

OUT = Path(__file__).resolve().parents[2] / "data" / "vq2" / "lab"


def probe(path: str) -> None:
    img = cv2.imread(path)
    if img is None:
        print(f"{path}: unreadable")
        return
    mask = gates.red_mask(img)
    h, w = mask.shape
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    name = path.rsplit("\\", 1)[-1]
    print(f"--- {name}  ({w}x{h})")
    vis = img.copy()
    for c in contours:
        area = float(cv2.contourArea(c))
        if area < gates.MIN_AREA:
            continue
        x, y, bw, bh = cv2.boundingRect(c)
        border = x <= 1 or y <= 1 or x + bw >= w - 1 or y + bh >= h - 1
        core = mask[y + bh // 4 : y + 3 * bh // 4, x + bw // 4 : x + 3 * bw // 4]
        fill = float(np.count_nonzero(core)) / core.size if core.size else 1.0
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.04 * peri, True)
        quad = None
        path_used = "none"
        if len(approx) == 4 and cv2.isContourConvex(approx):
            quad = gates._order_quad(approx.reshape(4, 2).astype(np.float64))
            path_used = "approx4"
        else:
            rect = cv2.minAreaRect(c)
            rect_area = rect[1][0] * rect[1][1]
            if rect_area > 0 and area / rect_area >= 0.5:
                quad = gates._order_quad(cv2.boxPoints(rect).astype(np.float64))
                path_used = "rect"
        z = None
        if quad is not None:
            fit = gates._pnp(quad)
            if fit is not None:
                z = fit[0][2]
        z_h = 320.0 * gates.GATE_OUTER_M / bh  # upright-gate height range check
        cv2.drawContours(vis, [c], -1, (0, 255, 255), 1)
        if quad is not None:
            cv2.polylines(vis, [quad.astype(np.int32)], True, (0, 255, 0), 1)
            for i, pt in enumerate(quad.astype(int)):
                cv2.circle(vis, tuple(pt), 3, (255, 0, 255), -1)
                cv2.putText(
                    vis, str(i), (pt[0] + 4, pt[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 255), 1
                )
        zs = "nan" if z is None else f"{z:.1f}"
        cv2.putText(
            vis,
            f"a{area:.0f} {path_used} z={zs} zh={z_h:.0f}",
            (x, max(12, y - 4)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (0, 255, 0),
            1,
        )
        print(
            f"  a={area:6.0f} bbox={bw}x{bh}@({x},{y}) nverts={len(approx)} "
            f"border={border} fill={fill:.2f} path={path_used} z={zs}m zh={z_h:.1f}m"
        )
    OUT.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(OUT / f"probe_{name}"), vis)


if __name__ == "__main__":
    for p in sys.argv[1:]:
        probe(p)
