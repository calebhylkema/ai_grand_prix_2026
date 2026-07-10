"""Glow-resistant gate perception for VQ2 (validated offline; see ``diagnostics.py lab``).

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

# Camera (OpenCV: x-right, y-down, z-fwd; tilted 20 deg DOWN) -> body FRD.
# 2026-07-03: the sim camera points ~20° DOWN, not up — verified from live imagery
# (a Mahony-level hover puts the horizon at y≈64 with floor filling the frame, and
# far gates are detected right at that line). The spec/config note claiming "20° UP"
# is wrong for build 3379. The matrix below already implements DOWN (an on-axis
# point maps to body z>0 = below), so PnP body poses have been correct all along —
# only the comment lied.
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
    up_body: tuple[float, float, float] | None = None  # world-up in body frame (from rvec)

    @property
    def range_m(self) -> float | None:
        if self.pose_cam is None:
            return None
        return float(np.linalg.norm(CAM_TO_BODY @ np.array(self.pose_cam)))

    @property
    def vision_roll(self) -> float | None:
        """Absolute body roll (rad, + = right wing down) from the gate's uprightness.

        Level body has world-up = (0,0,-1) in FRD. Only the in-plane component is
        trustworthy (planar-PnP tilt ambiguity corrupts pitch) — validated 2026-07-03
        against synthetic renders: sign + magnitude exact to ~1° out to 40° of bank.
        """
        if self.up_body is None:
            return None
        return float(math.atan2(-self.up_body[1], -self.up_body[2]))


# --- cyan racing-line ribbon (spec §3.4 "course guidance structures") ---
# The sim renders an elevated cyan ribbon threading the gates at gate height — a
# camera-visible, obstacle-free path (fly_trace 2026-07-03 frame 017: it arcs LEFT
# around the pillar field straight to gate 1). Segments may be broken up by pillars
# occluding the ribbon, so detection merges all elongated cyan components.
LINE_LO = np.array([78, 40, 80])  # far ribbon is desaturated/dim (S~50 V~107 at 20m+),
LINE_HI = np.array([115, 255, 255])  # and the gate HOLOGRAM shares the hue (H med 99,
#                     S 189) — hue cannot separate them. Shape does: the ribbon is the
#                     only long+thin cyan thing; require strong elongation instead.
LINE_MIN_PX = 100.0  # px^2 — smallest ribbon segment considered
LINE_MIN_ELONG = 3.0  # ribbon segments are long+thin; holograms/signage are blocky
LINE_NEAR_BLOB_PX = 800.0  # px^2 — EXCEPTION to the elongation test: at close range the
#                            tube is FAT in frame, and Y3 frame 013 rejected the entire
#                            near tube (thousands of px) as "blocky" exactly when it broke
#                            right at our shoulder to climb to gate 1 — the follower then
#                            steered on the distant course line instead. A fat blob IS the
#                            near tube when it is big and anchored to the frame's bottom
#                            or lower sides (holograms/signage float mid-frame).
LINE_WALK_STEP = 24.0  # px — ridge-walk row step from the near point toward the aim row
LINE_WALK_LAT = 90.0  # px — max lateral jump per step that still counts as OUR branch


@dataclass
class LineTarget:
    """The visible cyan racing-line ribbon (union of its unoccluded segments)."""

    cx: float  # centroid x over all accepted segments (px)
    cy: float  # centroid y (px)
    area: float  # total accepted segment area (px^2)
    bearing: float  # rad, + = ribbon centroid to the right (image-x through fx)
    far_cx: float  # far end of the ribbon = highest accepted point (smallest y)
    far_cy: float
    aim_cx: float  # pure-pursuit lookahead = mid-path point (median row of the ribbon)
    aim_cy: float
    near_cx: float = 320.0  # nearest visible ribbon point (deepest row) — the lateral
    near_cy: float = 360.0  # cross-track cue: where the path is right beside our nose

    @property
    def near_bearing(self) -> float:
        """Bearing to the path's NEAREST visible point — the cross-track error cue.

        Pure pursuit on the mid-path aim point converges the HEADING to the path but
        cuts corners (runs R1–R3, 2026-07-04: the arc consistently sliced the inside
        of the gate-0→1 right corner, through the Station-19 parked jet). The near
        point says where the path is just ahead of us: bank toward it to null the
        lateral offset while yaw follows the aim point.
        """
        return float(math.atan2(self.near_cx - 320.0, 320.0))

    @property
    def far_bearing(self) -> float:
        """Bearing to the path's farthest visible point (diagnostic; do NOT steer on it).

        Steering at the far tip flies the CHORD across the path's arc — live 07-03 the
        tip near gate 1 was the post-gate segment at +32..44° right, and chasing it cut
        straight into the pillar field the arc exists to avoid.
        """
        return float(math.atan2(self.far_cx - 320.0, 320.0))

    @property
    def aim_bearing(self) -> float:
        """Bearing to the mid-path lookahead — the pure-pursuit steering point.

        The median-row point of the ribbon: far enough ahead to make progress, near
        enough that following it TRACKS the arc instead of cutting its chord.
        """
        return float(math.atan2(self.aim_cx - 320.0, 320.0))


def cyan_mask(img: np.ndarray) -> np.ndarray:
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, LINE_LO, LINE_HI)
    return cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))


def detect_line(img: np.ndarray) -> LineTarget | None:
    """The cyan ribbon if visible: merged centroid + far point of elongated segments."""
    mask = cyan_mask(img)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    m_sum = np.zeros(3)  # [sum(a), sum(a*cx), sum(a*cy)] over accepted segments
    far: tuple[float, float] | None = None
    accepted = np.zeros_like(mask)  # dense pixels of accepted segments (for the lookahead)
    h, w = mask.shape
    for c in contours:
        area = float(cv2.contourArea(c))
        if area < LINE_MIN_PX:
            continue
        (_, _), (rw, rh), _ = cv2.minAreaRect(c)
        if min(rw, rh) <= 0 or max(rw, rh) / min(rw, rh) < LINE_MIN_ELONG:
            # blocky = gate hologram / signage — UNLESS it is the near tube: big and
            # anchored to the frame's bottom (or a side edge in the lower half). At
            # close range the tube is fat, and rejecting it left only the distant
            # course line in the aim (Y3 frame 013: steered "dead ahead" while the
            # racing line broke hard right at our shoulder). See LINE_NEAR_BLOB_PX.
            bx, by, bw2, bh2 = cv2.boundingRect(c)
            anchored = by + bh2 >= h - 2 or ((bx <= 2 or bx + bw2 >= w - 2) and by + bh2 > h / 2)
            if not (area >= LINE_NEAR_BLOB_PX and anchored):
                continue
        m = cv2.moments(c)
        if m["m00"] == 0:
            continue
        m_sum += (area, area * m["m10"] / m["m00"], area * m["m01"] / m["m00"])
        cv2.drawContours(accepted, [c], -1, 255, -1)
        pts = c.reshape(-1, 2)
        top = pts[np.argmin(pts[:, 1])]
        if far is None or top[1] < far[1]:
            far = (float(top[0]), float(top[1]))
    if m_sum[0] == 0 or far is None:
        return None
    cx, cy = m_sum[1] / m_sum[0], m_sum[2] / m_sum[0]
    bearing = float(math.atan2(cx - 320.0, 320.0))
    # pure-pursuit lookahead: the mid-path point (image row ≈ distance with the
    # down-tilted camera, so the median row splits near/far halves of the path).
    # Dense mask pixels, not contour vertices — CHAIN_APPROX_SIMPLE vertices cluster
    # at segment ends and leave the median row empty when pillars split the ribbon.
    ys, xs = np.nonzero(accepted)
    y_aim = float(np.quantile(ys, 0.5))
    # near point: the deepest visible rows (95th percentile guards single-pixel noise)
    y_near = float(np.quantile(ys, 0.95))
    nband = np.abs(ys - y_near) < 12.0
    if not nband.any():
        nband = np.abs(ys - y_near) <= float(np.min(np.abs(ys - y_near)))
    near_cx, near_cy = float(xs[nband].mean()), float(ys[nband].mean())
    # pure-pursuit lookahead: RIDGE-WALK from the near point up toward the median row,
    # each step keeping only the pixels laterally continuous with the previous step.
    # One image row can hold TWO branches of the course at once — Y3 frame 013: the
    # near tube broke hard RIGHT at our shoulder to climb to gate 1 while the
    # post-gate-1 course ran across the frame top, and the plain band-mean mixed the
    # branches into aim=+8° ("dead ahead") for the whole first second of the corner —
    # the wide-low overshoot into the Station-19 bays began as a PERCEPTION error.
    # Rows holding only the other branch (or an occlusion gap) keep the last
    # on-branch point; if our branch ends early, we aim at its visible end.
    aim_cx, aim_cy = near_cx, near_cy
    row = near_cy
    while row - LINE_WALK_STEP >= y_aim:
        row -= LINE_WALK_STEP
        band = np.abs(ys - row) < LINE_WALK_STEP / 2
        if not band.any():
            continue
        bx = xs[band]
        same = np.abs(bx - aim_cx) < LINE_WALK_LAT
        if same.any():
            aim_cx, aim_cy = float(bx[same].mean()), row
    return LineTarget(
        float(cx),
        float(cy),
        float(m_sum[0]),
        bearing,
        far[0],
        far[1],
        aim_cx,
        aim_cy,
        near_cx,
        near_cy,
    )


def red_mask(img: np.ndarray) -> np.ndarray:
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, RED_LO1, RED_HI1) | cv2.inRange(  # type: ignore[operator]
        hsv, RED_LO2, RED_HI2
    )
    return cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))


def _order_quad(pts: np.ndarray) -> np.ndarray:
    """Order corners tl,tr,br,bl to match GATE_OBJ.

    2026-07-03: fixed a tr/bl SWAP (argmax(y-x) is bottom-left, not top-right). The old
    mirrored correspondence left tvec roughly usable on the symmetric square (all prior
    ranges/heights still valid) but made rvec garbage — and rvec is now used as an
    absolute roll reference (see up_body).
    """
    s = pts.sum(axis=1)
    d = np.diff(pts, axis=1).ravel()  # y - x: min at top-right, max at bottom-left
    return np.array(
        [pts[np.argmin(s)], pts[np.argmin(d)], pts[np.argmax(s)], pts[np.argmax(d)]],
        np.float64,
    )


def _pnp(
    quad: np.ndarray,
) -> tuple[tuple[float, float, float], tuple[float, float, float]] | None:
    """PnP pose of an ordered gate quad: (tvec camera-frame, world-up in BODY frame).

    Gates are world-upright squares, so the rotation's up-axis (gate +Y in camera
    frame, mapped to body) is an ABSOLUTE gravity reference — the only one we have in
    flight, where the accelerometer just reads body-z thrust ("level" at any bank).
    The in-plane (roll) component is unambiguous; the out-of-plane component carries
    the planar-PnP tilt ambiguity, so consumers should trust roll only.
    """
    try:
        ok, rvec, tvec = cv2.solvePnP(
            GATE_OBJ, quad, CAMERA_K, None, flags=cv2.SOLVEPNP_IPPE_SQUARE
        )
    except cv2.error:
        return None
    if not ok:
        return None
    x, y, z = tvec.ravel()
    if z <= 0:
        return None
    r_cg, _ = cv2.Rodrigues(rvec)
    up_body = CAM_TO_BODY @ (r_cg @ np.array([0.0, 1.0, 0.0]))  # gate +Y = world up
    return (float(x), float(y), float(z)), (
        float(up_body[0]),
        float(up_body[1]),
        float(up_body[2]),
    )


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
        up_body: tuple[float, float, float] | None = None
        if area >= MIN_POSE_AREA:
            peri = cv2.arcLength(c, True)
            approx = cv2.approxPolyDP(c, 0.04 * peri, True)
            fit: tuple[tuple[float, float, float], tuple[float, float, float]] | None = None
            if len(approx) == 4 and cv2.isContourConvex(approx):
                fit = _pnp(_order_quad(approx.reshape(4, 2).astype(np.float64)))
            else:
                rect = cv2.minAreaRect(c)
                rect_area = rect[1][0] * rect[1][1]
                if rect_area > 0 and area / rect_area >= 0.5:
                    fit = _pnp(_order_quad(cv2.boxPoints(rect).astype(np.float64)))
            if fit is not None:
                pose, up_body = fit
        if pose is not None:
            gate_body = CAM_TO_BODY @ np.array(pose)
            bearing = float(math.atan2(gate_body[1], gate_body[0]))
        else:
            bearing = float(math.atan2(cx - 320.0, 320.0))  # image-x through fx=320
        out.append(GateTarget(float(cx), float(cy), area, bearing, pose, up_body))
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
