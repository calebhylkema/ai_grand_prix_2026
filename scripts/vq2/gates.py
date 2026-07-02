"""Glow-resistant gate perception for VQ2 (validated offline in detect_lab.py).

The gate-1 ENV collisions (2026-07-01/02) were traced to a perception trap: the
gates cast a huge diffuse red GLOW on nearby walls/pillars/floor. Both the blob
centroid and a naive quad fit lock onto the glow and steer the drone INTO the
environment (confirmed frame-by-frame in runs/vq2_collisions).

A real gate is a bright red RING around a non-red opening; glow is a SOLID mass,
partial gates touch the image border. ``detect()`` therefore only returns
ring-validated candidates:

- border-touching contours rejected (partial gate -> PnP would be garbage),
- solid masses rejected (core fill > 0.55; tilted boxes leak ~0.4, glow ~0.9+),
- 4-vertex convex fit, else minAreaRect fallback (banner/perspective break the
  strict fit) when the rect explains >= 50% of the contour,
- pose only from quads with area >= MIN_POSE_AREA (small/far quads = noisy PnP;
  bearing is still returned for them).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import cv2
import numpy as np

# --- red mask (same band race.py flew gate 0 with) ---
RED_LO1 = np.array([0, 90, 90])
RED_HI1 = np.array([12, 255, 255])
RED_LO2 = np.array([165, 90, 90])
RED_HI2 = np.array([180, 255, 255])
MIN_AREA = 250.0  # px^2 — smallest contour considered at all
MIN_POSE_AREA = 800.0  # px^2 — smallest quad trusted for PnP
MAX_CORE_FILL = 0.55  # ring test: solid glow reads ~0.9+, tilted gates ~0.4

# --- camera + gate geometry (VQ2 spec) ---
CAMERA_K = np.array([[320.0, 0, 320.0], [0, 320.0, 180.0], [0, 0, 1.0]], np.float64)
GATE_OUTER_M = 2.7
_H = GATE_OUTER_M / 2.0
GATE_OBJ = np.array([[-_H, _H, 0], [_H, _H, 0], [_H, -_H, 0], [-_H, -_H, 0]], np.float64)

# Camera (OpenCV: x-right, y-down, z-fwd; tilted 20 deg UP) -> body FRD.
_M_AXIS = np.array([[0, 0, 1.0], [1.0, 0, 0], [0, 1.0, 0]])
_TILT = math.radians(-20.0)
_R_TILT = np.array(
    [[math.cos(_TILT), 0, math.sin(_TILT)], [0, 1, 0], [-math.sin(_TILT), 0, math.cos(_TILT)]]
)
CAM_TO_BODY = _R_TILT @ _M_AXIS


@dataclass
class GateTarget:
    """One ring-validated gate candidate."""

    cx: float  # image centroid x (px)
    cy: float  # image centroid y (px)
    area: float  # contour area (px^2)
    bearing: float  # rad, body frame (+ = gate to the right)
    pose_cam: tuple[float, float, float] | None  # PnP (x,y,z) camera frame, m (None if small)

    @property
    def range_m(self) -> float | None:
        if self.pose_cam is None:
            return None
        return float(np.linalg.norm(CAM_TO_BODY @ np.array(self.pose_cam)))


def red_mask(img: np.ndarray) -> np.ndarray:
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, RED_LO1, RED_HI1) | cv2.inRange(hsv, RED_LO2, RED_HI2)
    return cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))


def _order_quad(pts: np.ndarray) -> np.ndarray:
    s = pts.sum(axis=1)
    d = np.diff(pts, axis=1).ravel()
    return np.array(
        [pts[np.argmin(s)], pts[np.argmax(d)], pts[np.argmax(s)], pts[np.argmin(d)]],
        np.float64,
    )


def _pnp(quad: np.ndarray) -> tuple[float, float, float] | None:
    try:
        ok, _rvec, tvec = cv2.solvePnP(
            GATE_OBJ, quad, CAMERA_K, None, flags=cv2.SOLVEPNP_IPPE_SQUARE
        )
    except cv2.error:
        return None
    if not ok:
        return None
    x, y, z = tvec.ravel()
    if z <= 0:
        return None
    return (float(x), float(y), float(z))


def detect(img: np.ndarray) -> list[GateTarget]:
    """All ring-validated gates in the frame (may be empty — glow is NOT a gate)."""
    mask = red_mask(img)
    h, w = mask.shape
    out: list[GateTarget] = []
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for c in contours:
        area = float(cv2.contourArea(c))
        if area < MIN_AREA:
            continue
        x, y, bw, bh = cv2.boundingRect(c)
        if x <= 1 or y <= 1 or x + bw >= w - 1 or y + bh >= h - 1:
            continue  # partial gate / glow spilling over the border
        core = mask[y + bh // 4 : y + 3 * bh // 4, x + bw // 4 : x + 3 * bw // 4]
        if core.size == 0:
            continue
        if float(np.count_nonzero(core)) / core.size > MAX_CORE_FILL:
            continue  # solid mass = glow/reflection, not a hollow gate ring
        m = cv2.moments(c)
        if m["m00"] == 0:
            continue
        cx, cy = m["m10"] / m["m00"], m["m01"] / m["m00"]
        pose: tuple[float, float, float] | None = None
        if area >= MIN_POSE_AREA:
            peri = cv2.arcLength(c, True)
            approx = cv2.approxPolyDP(c, 0.04 * peri, True)
            if len(approx) == 4 and cv2.isContourConvex(approx):
                pose = _pnp(_order_quad(approx.reshape(4, 2).astype(np.float64)))
            else:
                rect = cv2.minAreaRect(c)
                rect_area = rect[1][0] * rect[1][1]
                if rect_area > 0 and area / rect_area >= 0.5:
                    pose = _pnp(_order_quad(cv2.boxPoints(rect).astype(np.float64)))
        if pose is not None:
            gate_body = CAM_TO_BODY @ np.array(pose)
            bearing = float(math.atan2(gate_body[1], gate_body[0]))
        else:
            bearing = float(math.atan2(cx - 320.0, 320.0))  # image-x through fx=320
        out.append(GateTarget(float(cx), float(cy), area, bearing, pose))
    return out


def nearest(img: np.ndarray) -> GateTarget | None:
    """The best steering target: nearest by PnP range, else largest validated ring."""
    targets = detect(img)
    if not targets:
        return None
    with_pose = [t for t in targets if t.pose_cam is not None]
    if with_pose:
        return min(with_pose, key=lambda t: t.pose_cam[2])  # type: ignore[index]
    return max(targets, key=lambda t: t.area)
