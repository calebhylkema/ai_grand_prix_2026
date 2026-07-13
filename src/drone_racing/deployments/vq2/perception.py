"""Glow-resistant gate perception for VQ2 (validated offline; see ``diagnostics.py lab``).

The gate-1 ENV collisions (2026-07-01/02) were traced to a perception trap: the
gates cast a huge diffuse red GLOW on nearby walls/pillars/floor. Both the blob
centroid and a naive quad fit lock onto the glow and steer the drone INTO the
environment (confirmed frame-by-frame in runs/vq2_collisions).

A real gate is a bright red RING around a non-red opening; glow is a SOLID mass,
partial gates touch the image border.

Two surfaces live here:

- ``detect()``/``nearest()`` — the LEGACY live-flight path, kept byte-for-byte while
  the structured pipeline is validated. Returns ring-validated ``GateTarget``s only:
  border-touching contours rejected, solid masses rejected (core fill > 0.55), a
  4-vertex convex fit else ``minAreaRect`` fallback, pose only for area >= MIN_POSE_AREA.
- ``detect_candidates()`` + ``GateTracker`` — the STRUCTURED pipeline (additive, not yet
  flown). Separates detection from target selection: every plausible candidate is
  returned as a ``GateObservation`` (FULL or PARTIAL, with inner/outer corners and
  component scores) and a persistent tracker owns the active-gate identity so a
  downstream gate cannot steal it. Metric PnP is a separate, quality-gated stage.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from enum import Enum

import cv2
import numpy as np

logger = logging.getLogger(__name__)

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
# 2026-07-10: the camera is tilted 20° UP, per spec VADR-TS-003 §3.8 ("tilted upwards
# by 20°") and flight physics (forward flight pitches the nose down, so a 20°-up mount
# then looks straight ahead). The old "20° DOWN, verified from live imagery" reading
# (2026-07-03) was CONFOUNDED by the rest pitch: the VQ2 hover pose is already nose-down
# ~18° (attitude filter reads -17.8° at the start accel (-3,0,-9.34)), so a 20°-up
# camera on an 18°-down body points ~horizontal and the floor fills the lower frame —
# which looked like a down-camera but is not. The optical axis (camera +z) maps to body
# (cosθ, 0, -sinθ): θ=+20° => body z<0 = UP. Flipping this inverts all body-frame
# outputs (bearing/range/up_body); camera-frame pose_cam (raw tvec) is unaffected.
_M_AXIS = np.array([[0, 0, 1.0], [1.0, 0, 0], [0, 1.0, 0]])
CAMERA_TILT_DEG = 20.0  # camera pitched UP from body-forward (spec VADR-TS-003 §3.8)


def cam_to_body(tilt_deg: float = CAMERA_TILT_DEG) -> np.ndarray:
    """Rotation mapping a camera-frame vector to body FRD.

    Camera is OpenCV convention (x-right, y-down, z-forward); body is FRD (x-forward,
    y-right, z-down). ``tilt_deg`` > 0 pitches the camera UP, so the optical axis
    (camera +z) maps to body ``(cos t, 0, -sin t)`` — i.e. body z < 0 (up). See the
    block comment above for the full sign derivation.

    Parameters
    ----------
    tilt_deg : float
        Camera up-tilt in degrees (positive = up).

    Returns
    -------
    np.ndarray
        (3, 3) rotation matrix, camera-frame -> body FRD.
    """
    t = math.radians(tilt_deg)
    r_tilt = np.array([[math.cos(t), 0, math.sin(t)], [0, 1.0, 0], [-math.sin(t), 0, math.cos(t)]])
    return r_tilt @ _M_AXIS


CAM_TO_BODY = cam_to_body()


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


# Specular-recovery band: a bright top/edge bar can glare to near-white — S drops below
# the base 90 floor and hue drifts toward magenta (measured on stationary frames 959-973:
# S 115→38, H 172→150, V pinned ~250) — while staying physically contiguous with the
# solid-red ring. These bright, partly-desaturated red/magenta pixels are RECOVERED only
# where connected to solid red (see red_mask_specular), so glare is bridged without
# admitting isolated pink clutter.
_SPEC_LO_A = np.array([0, 25, 200])
_SPEC_HI_A = np.array([18, 255, 255])
_SPEC_LO_B = np.array([145, 25, 200])
_SPEC_HI_B = np.array([180, 255, 255])


def red_mask_specular(img: np.ndarray) -> np.ndarray:
    """Red mask with connectivity-gated specular recovery (STRUCTURED pipeline only).

    Geodesic reconstruction of the strict base red mask under ``base | specular``: the
    mask grows into bright, partly-desaturated red/magenta pixels ONLY where contiguous
    with solid red, recovering a glared gate bar without admitting isolated clutter (a
    corpus check showed candidate counts unchanged/slightly lower vs the base mask).
    The legacy ``detect()`` keeps the stricter :func:`red_mask` until a replay
    comparison justifies changing it.
    """
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    base = cv2.inRange(hsv, RED_LO1, RED_HI1) | cv2.inRange(hsv, RED_LO2, RED_HI2)  # type: ignore[operator]
    spec = cv2.inRange(hsv, _SPEC_LO_A, _SPEC_HI_A) | cv2.inRange(  # type: ignore[operator]
        hsv, _SPEC_LO_B, _SPEC_HI_B
    )
    allowed = cv2.morphologyEx(base | spec, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    marker = base.copy()
    kernel = np.ones((5, 5), np.uint8)
    for _ in range(25):  # geodesic reconstruction; converges well before the cap
        grown = cv2.dilate(marker, kernel) & allowed  # type: ignore[operator]
        if np.array_equal(grown, marker):
            break
        marker = grown
    return cv2.morphologyEx(marker, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))


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
    """The best steering target: nearest by PnP range, else largest validated ring.

    NOTE: this is the *legacy* live-flight surface, kept byte-for-byte while the
    structured pipeline below (``detect_candidates``) is validated offline/in shadow.
    """
    targets = detect(img)
    if not targets:
        return None
    with_pose = [t for t in targets if t.pose_cam is not None]
    if with_pose:
        return min(with_pose, key=lambda t: t.pose_cam[2])  # type: ignore[index]
    return max(targets, key=lambda t: t.area)


# ===========================================================================
# Structured perception pipeline (additive; not yet on the live flight path).
#
# Separates DETECTION (every plausible candidate, full or partial) from TARGET
# SELECTION (a persistent association layer — see tracker below). The detector no
# longer decides which candidate is the active race gate. See the perception plan
# and docs/reference for the sign/geometry conventions.
# ===========================================================================


class Visibility(Enum):
    """Whether the gate is fully in-frame or clipped by an image border."""

    FULL = "full"
    PARTIAL = "partial"


class OutputMode(Enum):
    """How much of the observation downstream may trust as a metric measurement."""

    NONE = "none"  # nothing trustworthy
    BEARING_ONLY = "bearing_only"  # direction only (partial / far / no PnP)
    POSITION_ONLY = "position_only"  # translation trustworthy, orientation ambiguous
    FULL_POSE = "full_pose"  # translation + orientation trustworthy


# Gate geometry (VQ2 spec §3.7). Object-point scale MUST match the pixel edge used.
GATE_INNER_M = 1.5  # inner opening (sharp, glow-immune — preferred for PnP)
GATE_FRAME_DEPTH_M = 0.26
_INNER_H = GATE_INNER_M / 2.0
# ideal inner-hole / outer-boundary area ratio for the ring test
_IDEAL_INNER_OUTER_AREA = (GATE_INNER_M / GATE_OUTER_M) ** 2  # ≈ 0.309

# Object points (gate-local, z=0 plane, y-up) ordered tl,tr,br,bl to match _order_quad.
GATE_OBJ_INNER = np.array(
    [
        [-_INNER_H, _INNER_H, 0],
        [_INNER_H, _INNER_H, 0],
        [_INNER_H, -_INNER_H, 0],
        [-_INNER_H, -_INNER_H, 0],
    ],
    np.float64,
)

# PnP acceptance/validation thresholds.
PNP_MAX_REPROJ_PX = 2.5  # accepted-pose reprojection RMSE ceiling
PNP_MIN_RANGE_M = 0.5
PNP_MAX_RANGE_M = 30.0
PNP_AMBIG_MARGIN_PX = 0.5  # if the two planar solutions differ by less, orientation is
#                            ambiguous -> degrade to position_only unless a prior breaks it
_CORNER_PX_SIGMA = 1.0  # baseline per-corner localization noise (px)
_COMBINED_INNER_MARGIN_PX = 0.8  # if 8-corner reproj exceeds inner-only by more than this,
#                                  the outer edge (glow-inflated) is inconsistent — use inner
# --- geometry guards (never let a low reprojection error validate a degenerate contour) ---
_MIN_INNER_SIDE_PX = 8.0  # collapsed inner openings (e.g. the 6px glare artifact) aren't real
_INNER_OUTER_AREA_RANGE = (0.12, 0.75)  # plausible inner/outer area ratio (spec ≈ 0.31)
_MIN_QUAD_SIDE_PX = 8.0  # a quad with any side shorter than this is degenerate — no metric pose
_RANGE_DISAGREE_FRAC = 0.30  # inner-only vs outer-only range mismatch beyond this = bad ring


@dataclass
class GateObservation:
    """One gate candidate from a single frame — detection output, pre-selection.

    Metric pose fields (``pose_cam``/``up_body``/``reproj_rms``) are populated by the
    PnP stage; until then ``output_mode`` is at most ``BEARING_ONLY``.
    """

    cx: float  # visible-contour centroid x (px)
    cy: float  # visible-contour centroid y (px)
    area: float  # outer red-contour area (px^2)
    bbox: tuple[int, int, int, int]  # x, y, w, h of the outer contour
    bearing: float  # rad, body frame (+ = right); image-based until PnP refines it
    visibility: Visibility
    visible_fraction: float  # 0..1 estimate of the gate inside the frame
    outer_corners: np.ndarray | None  # (4,2) tl,tr,br,bl or None (partial/no clean fit)
    inner_corners: np.ndarray | None  # (4,2) opening corners or None
    confidence: float  # 0..1 aggregate detection confidence
    scores: dict[str, float] = field(default_factory=dict)  # component scores
    pose_cam: tuple[float, float, float] | None = None  # PnP tvec, camera frame (m)
    up_body: tuple[float, float, float] | None = None  # world-up in body (from rvec)
    reproj_rms: float | None = None
    pose_cov: np.ndarray | None = None  # (3,3) camera-frame position covariance (m^2)
    ambiguity_margin: float | None = None  # reproj-px gap between the two planar solutions
    solver: str | None = None  # which correspondence set/solver produced the pose
    output_mode: OutputMode = OutputMode.BEARING_ONLY
    reject_reason: str | None = None
    frame_id: int | None = None
    timestamp_ns: int | None = None
    track_id: int | None = None  # visual track identity (tracker-assigned)
    race_gate_id: int | None = None  # expected race-gate index (tracker-assigned)

    @property
    def range_cam(self) -> float | None:
        """Camera-frame depth (m), i.e. PnP z; None until posed."""
        return None if self.pose_cam is None else float(self.pose_cam[2])

    @property
    def slant_range(self) -> float | None:
        """Euclidean range (m) = ‖pose_cam‖; None until posed.

        Use this (not ``range_cam``) for range bands: at large bearings the camera-z
        depth is materially shorter than the true slant range (~18% at 35°).
        """
        return None if self.pose_cam is None else float(np.linalg.norm(self.pose_cam))


def _quad_from_contour(c: np.ndarray) -> np.ndarray | None:
    """Ordered (4,2) tl,tr,br,bl quad if the contour fits a convex 4-gon, else None."""
    peri = cv2.arcLength(c, True)
    approx = cv2.approxPolyDP(c, 0.04 * peri, True)
    if len(approx) == 4 and cv2.isContourConvex(approx):
        return _order_quad(approx.reshape(4, 2).astype(np.float64))
    return None


def _visible_fraction(
    bbox: tuple[int, int, int, int], w_img: int, h_img: int
) -> tuple[Visibility, float]:
    """Classify FULL/PARTIAL and estimate the in-frame fraction of a ~square gate.

    Assumes the gate is roughly square in the image: if only one axis is clipped, the
    unclipped axis gives the true side length and the ratio is the visible fraction.
    """
    x, y, w, h = bbox
    touch_h = x <= 1 or x + w >= w_img - 1
    touch_v = y <= 1 or y + h >= h_img - 1
    if not (touch_h or touch_v):
        return Visibility.FULL, 1.0
    if touch_h and not touch_v:
        frac = w / h if h > 0 else 0.0
    elif touch_v and not touch_h:
        frac = h / w if w > 0 else 0.0
    else:  # clipped on both axes — heavily partial
        frac = 0.5 * (min(w, h) / max(w, h)) if max(w, h) > 0 else 0.0
    return Visibility.PARTIAL, float(np.clip(frac, 0.05, 1.0))


def _candidate_scores(
    contour: np.ndarray,
    area: float,
    inner_area: float,
    outer_corners: np.ndarray | None,
    core_fill: float,
) -> dict[str, float]:
    """Per-candidate geometric evidence scores in [0, 1] (higher = more gate-like)."""
    hull_area = float(cv2.contourArea(cv2.convexHull(contour)))
    convexity = float(np.clip(area / hull_area, 0.0, 1.0)) if hull_area > 0 else 0.0
    # ring: a real gate is a hollow band; glow is solid. Prefer a present hole whose
    # area ratio matches the spec 1.5/2.7 opening; fall back to (1 - core_fill).
    if inner_area > 0 and area > 0:
        ratio = inner_area / area
        ring = float(math.exp(-((ratio - _IDEAL_INNER_OUTER_AREA) ** 2) / (2 * 0.12**2)))
    else:
        ring = float(np.clip(1.0 - core_fill / MAX_CORE_FILL, 0.0, 1.0))
    quad = 1.0 if outer_corners is not None else 0.4
    return {"convexity": convexity, "ring": ring, "quad": quad}


def detect_candidates(img: np.ndarray) -> list[GateObservation]:
    """Every plausible gate candidate in the frame — FULL and PARTIAL, no selection.

    Uses a two-level contour hierarchy (``RETR_CCOMP``) so the outer red boundary and
    its enclosed opening are recovered together. Border-touching gates are retained as
    PARTIAL (bearing/tracking only); solid red masses (glow, no opening) are rejected.
    Metric PnP is left to the PnP stage — observations here are at most BEARING_ONLY.

    Parameters
    ----------
    img : np.ndarray
        BGR frame (OpenCV order).

    Returns
    -------
    list[GateObservation]
        Candidates, unordered. Empty if nothing gate-like is present.
    """
    mask = red_mask_specular(img)  # structured pipeline uses the glare-robust mask
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    h_img, w_img = mask.shape
    contours, hierarchy = cv2.findContours(mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    out: list[GateObservation] = []
    if hierarchy is None:
        return out
    hier = hierarchy[0]  # (N, 4): next, prev, first_child, parent
    for i, c in enumerate(contours):
        if hier[i][3] != -1:
            continue  # a hole/child — handled via its parent
        area = float(cv2.contourArea(c))
        if area < MIN_AREA:
            continue
        x, y, bw, bh = cv2.boundingRect(c)
        bbox = (x, y, bw, bh)

        # largest child hole = the gate opening (if any)
        inner_contour: np.ndarray | None = None
        inner_area = 0.0
        j = hier[i][2]
        while j != -1:
            ca = float(cv2.contourArea(contours[j]))
            if ca > inner_area:
                inner_area, inner_contour = ca, contours[j]
            j = hier[j][0]

        # ring test: a real gate is hollow. With no child hole, fall back to the
        # core-fill test (glow reads solid) exactly as the legacy detector.
        core = mask[y + bh // 4 : y + 3 * bh // 4, x + bw // 4 : x + 3 * bw // 4]
        core_fill = float(np.count_nonzero(core)) / core.size if core.size else 1.0
        if inner_contour is None and core_fill > MAX_CORE_FILL:
            continue  # solid mass = glow/reflection, not a gate ring

        m = cv2.moments(c)
        if m["m00"] == 0:
            continue
        cx, cy = m["m10"] / m["m00"], m["m01"] / m["m00"]

        visibility, vis_frac = _visible_fraction(bbox, w_img, h_img)
        # only trust a clean 4-gon fit for FULL gates; a clipped contour's fitted
        # corners include the image edge and are not gate corners.
        outer_corners = _quad_from_contour(c) if visibility is Visibility.FULL else None
        if outer_corners is not None:
            outer_corners = _refine_corners(gray, outer_corners)
        inner_corners = (
            _quad_from_contour(inner_contour)
            if inner_contour is not None and visibility is Visibility.FULL
            else None
        )
        if inner_corners is not None and inner_contour is not None:
            iw2, ih2 = cv2.boundingRect(inner_contour)[2:]
            ratio = inner_area / area if area > 0 else 0.0
            if min(iw2, ih2) < _MIN_INNER_SIDE_PX or not (
                _INNER_OUTER_AREA_RANGE[0] <= ratio <= _INNER_OUTER_AREA_RANGE[1]
            ):
                inner_corners = None  # collapsed/implausible opening (e.g. a glare artifact)
            else:
                inner_corners = _refine_corners(gray, inner_corners)
        scores = _candidate_scores(c, area, inner_area, outer_corners, core_fill)
        conf = float(np.mean([scores["convexity"], scores["ring"], scores["quad"]]))
        if visibility is Visibility.PARTIAL:
            conf *= vis_frac  # partial gates are inherently less certain

        out.append(
            GateObservation(
                cx=float(cx),
                cy=float(cy),
                area=area,
                bbox=bbox,
                bearing=float(math.atan2(cx - 320.0, 320.0)),
                visibility=visibility,
                visible_fraction=vis_frac,
                outer_corners=outer_corners,
                inner_corners=inner_corners,
                confidence=conf,
                scores=scores,
                output_mode=OutputMode.BEARING_ONLY,
            )
        )
    return out


# --- persistent target identity (association layer) ---------------------------
#
# The AX-batch forensics caught the old detector latching 3 different physical gates
# within 0.3 s (PnP z 27→21→17 m — a 35 m/s "closure"). Instantaneous nearest/largest
# selection cannot hold identity through the turn, when the active gate is partial at
# the right edge while a downstream gate is fully visible. This layer keeps a track
# per candidate, associates by motion continuity, and only rebinds the ACTIVE target
# after sustained evidence — never on a single large/near blob.

_ASSOC_CENTROID_NORM = 160.0  # px; centroid distance scale in the association cost
_ASSOC_MAX_COST = 1.2  # reject matches above this normalized cost
_SWITCH_CONFIRM = 3  # consecutive frames before the active identity may move
_MAX_MISSES = 6  # drop a track after this many un-matched frames
_PASSED_WINDOW_FRAMES = 30  # exclude the passed gate's image region for this many frames
_PASSED_EXCL_PX = 80.0  # radius (px) around a passed gate's last position to exclude
_JUMP_CENTROID_PX = 260.0  # impossible per-frame image jump
_JUMP_SCALE_RATIO = 2.5  # impossible per-frame apparent-size ratio


@dataclass
class GateTrack:
    """A single tracked gate across frames (internal visual identity)."""

    track_id: int
    cx: float
    cy: float
    area: float
    bearing: float
    vx: float = 0.0
    vy: float = 0.0
    hits: int = 1
    misses: int = 0
    age: int = 0
    race_gate_id: int | None = None
    last_obs: GateObservation | None = None

    def predict(self) -> tuple[float, float]:
        """Constant-velocity image-space prediction of the next centroid."""
        return self.cx + self.vx, self.cy + self.vy

    def update(self, obs: GateObservation) -> None:
        pcx, pcy = self.predict()
        self.vx = 0.6 * self.vx + 0.4 * (obs.cx - self.cx)
        self.vy = 0.6 * self.vy + 0.4 * (obs.cy - self.cy)
        self.cx, self.cy, self.area, self.bearing = obs.cx, obs.cy, obs.area, obs.bearing
        self.hits += 1
        self.misses = 0
        self.age += 1
        self.last_obs = obs


def _assoc_cost(track: GateTrack, obs: GateObservation) -> float:
    """Normalized association cost (lower = better); ``inf`` if physically impossible."""
    pcx, pcy = track.predict()
    dist = math.hypot(obs.cx - pcx, obs.cy - pcy)
    scale = math.sqrt(max(obs.area, 1.0) / max(track.area, 1.0))
    scale_ratio = max(scale, 1.0 / scale)
    if dist > _JUMP_CENTROID_PX or scale_ratio > _JUMP_SCALE_RATIO:
        return math.inf
    d_cost = dist / _ASSOC_CENTROID_NORM
    s_cost = abs(math.log(scale))  # 0 when unchanged
    b_cost = abs(obs.bearing - track.bearing)
    return d_cost + s_cost + b_cost


class GateTracker:
    """Associates detections to persistent tracks and owns the active-gate identity.

    ``step`` returns the observation currently believed to be the active race gate, or
    None when it is lost/only predicted. Metric pose is never fabricated for a track
    that was not matched this frame.
    """

    def __init__(self) -> None:
        self._tracks: list[GateTrack] = []
        self._next_id = 0
        self._active_id: int | None = None
        self._pending_id: int | None = None
        self._pending_count = 0
        self._active_gate_index: int | None = None
        self._retired: set[int] = set()  # track ids of already-passed gates (never re-select)
        self._turn_sign: float | None = None  # expected bearing sign for the new target
        self._range_band: tuple[float, float] | None = None  # expected slant-range window (m)
        self._frame = 0  # frame counter for time-boxed exclusion zones
        self._passed_zones: list[tuple[float, float, int]] = []  # (cx, cy, expire_frame)

    @property
    def tracks(self) -> list[GateTrack]:
        return self._tracks

    def retire_active(self) -> None:
        """Blacklist + unbind the current active track so it must be re-acquired.

        Used on a race-gate advance AND when an active track resolves to an out-of-band
        metric pose (a wrong-gate lock): the old id is permanently blacklisted and its
        last image region is excluded for a window so it cannot immediately re-win.
        """
        if self._active_id is not None:
            self._retired.add(self._active_id)
            prev = next((t for t in self._tracks if t.track_id == self._active_id), None)
            if prev is not None:
                self._passed_zones.append((prev.cx, prev.cy, self._frame + _PASSED_WINDOW_FRAMES))
        self._active_id = None
        self._pending_id = None
        self._pending_count = 0

    def _associate(self, obs_list: list[GateObservation]) -> dict[int, int]:
        """Greedy min-cost matching of tracks→observations. Returns {track_id: obs_idx}."""
        pairs: list[tuple[float, int, int]] = []
        for t in self._tracks:
            for oi, obs in enumerate(obs_list):
                c = _assoc_cost(t, obs)
                if c <= _ASSOC_MAX_COST:
                    pairs.append((c, t.track_id, oi))
        pairs.sort(key=lambda p: p[0])
        used_tracks: set[int] = set()
        used_obs: set[int] = set()
        matches: dict[int, int] = {}
        for _c, tid, oi in pairs:
            if tid in used_tracks or oi in used_obs:
                continue
            matches[tid] = oi
            used_tracks.add(tid)
            used_obs.add(oi)
        return matches

    def step(
        self,
        observations: list[GateObservation],
        active_gate_index: int | None = None,
        expected_turn_sign: float | None = None,
        range_band: tuple[float, float] | None = None,
    ) -> GateObservation | None:
        """Advance the tracker one frame and return the active-gate observation.

        Parameters
        ----------
        observations : list[GateObservation]
            This frame's candidates (from ``detect_candidates``).
        active_gate_index : int | None
            The race-status active gate index, if known. A change **retires** the
            previous active track (it can never be re-selected) and forces re-acquisition.
        expected_turn_sign : float | None
            Expected bearing sign of the next gate (+1 = right, -1 = left). A new active
            candidate must lie on that side — stops the passed/oncoming gate stealing it.
        range_band : tuple[float, float] | None
            Expected (min, max) range (m) for the next gate; metric candidates outside
            it are not eligible to become active.

        Returns
        -------
        GateObservation | None
            The observation bound to the active identity this frame, or None if the
            active gate is unmatched/lost (identity is retained internally).
        """
        self._turn_sign = expected_turn_sign
        self._range_band = range_band
        self._frame += 1
        self._passed_zones = [z for z in self._passed_zones if z[2] > self._frame]  # prune
        # a race-gate advance retires the old active gate and forces re-acquisition
        if active_gate_index is not None and active_gate_index != self._active_gate_index:
            self.retire_active()
            self._active_gate_index = active_gate_index

        matches = self._associate(observations)
        matched_obs: set[int] = set()
        for t in self._tracks:
            if t.track_id in matches:
                oi = matches[t.track_id]
                obs = observations[oi]
                obs.track_id = t.track_id
                obs.race_gate_id = t.race_gate_id
                t.update(obs)
                matched_obs.add(oi)
            else:
                t.misses += 1
                t.age += 1
        # spawn tracks for unmatched observations
        for oi, obs in enumerate(observations):
            if oi in matched_obs:
                continue
            t = GateTrack(self._next_id, obs.cx, obs.cy, obs.area, obs.bearing)
            obs.track_id = t.track_id
            self._tracks.append(t)
            self._next_id += 1
        # retire stale tracks
        self._tracks = [t for t in self._tracks if t.misses <= _MAX_MISSES]

        return self._select_active(observations, matches)

    def _select_active(
        self, observations: list[GateObservation], matches: dict[int, int]
    ) -> GateObservation | None:
        """Hold the active identity; rebind only after sustained, consistent evidence."""
        live = {t.track_id: t for t in self._tracks}

        # keep the current active track if it is still alive
        if self._active_id is not None and self._active_id in live:
            active = live[self._active_id]
            if active.race_gate_id is None:
                active.race_gate_id = self._active_gate_index
            if self._active_id in matches:
                obs = observations[matches[self._active_id]]
                return obs
            return None  # active track predicted-only this frame — no fresh measurement

        # no active track: choose the best ELIGIBLE matched candidate, confirmed over N
        # frames. Retired (passed) gates and candidates failing the expected turn
        # direction / range band are never eligible.
        candidates = [
            (t, observations[matches[t.track_id]])
            for t in self._tracks
            if t.track_id in matches
            and t.track_id not in self._retired
            and self._eligible_new(observations[matches[t.track_id]])
        ]
        if not candidates:
            self._pending_id, self._pending_count = None, 0
            return None
        best_track, best_obs = max(candidates, key=lambda c: c[1].confidence)
        best_tid = best_track.track_id
        if best_tid == self._pending_id:
            self._pending_count += 1
        else:
            self._pending_id, self._pending_count = best_tid, 1
        if self._pending_count >= _SWITCH_CONFIRM:
            self._active_id = best_tid
            live[best_tid].race_gate_id = self._active_gate_index
            best_obs.race_gate_id = self._active_gate_index
            return best_obs
        return None

    def _eligible_new(self, obs: GateObservation) -> bool:
        """Whether a candidate may become the new active target under the expected
        course geometry (turn direction + slant-range band) and the passed-gate
        exclusion zones. Bearing-only candidates pass the range test (no metric range to
        disqualify) but still obey the turn direction and exclusion zones."""
        if self._turn_sign is not None and obs.bearing * self._turn_sign <= 0.0:
            return False
        if self._range_band is not None and obs.slant_range is not None:
            lo, hi = self._range_band
            if not (lo <= obs.slant_range <= hi):
                return False
        for zx, zy, _exp in self._passed_zones:
            if math.hypot(obs.cx - zx, obs.cy - zy) < _PASSED_EXCL_PX:
                return False  # too close to a recently-passed gate's image region
        return True


# --- metric PnP (quality-gated) ------------------------------------------------
#
# The gate is a planar square, so PnP is inherently two-fold ambiguous. We solve for
# BOTH planar solutions, score them, and only emit a confident 6-DOF pose when the
# geometry is unambiguous and the reprojection is tight — otherwise we degrade to
# position_only / bearing_only rather than commit to a possibly-flipped orientation.
# Object-point scale is matched to the pixel edge used (inner 1.5 m vs outer 2.7 m).


def _refine_corners(gray: np.ndarray, quad: np.ndarray) -> np.ndarray:
    """Subpixel-refine four corners against the grayscale image; identity on failure."""
    pts = quad.reshape(-1, 1, 2).astype(np.float32)
    crit = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.01)
    try:
        cv2.cornerSubPix(gray, pts, (5, 5), (-1, -1), crit)
    except cv2.error:
        return quad
    return pts.reshape(-1, 2).astype(np.float64)


def _reproj_rms(obj: np.ndarray, img: np.ndarray, rvec: np.ndarray, tvec: np.ndarray) -> float:
    proj, _ = cv2.projectPoints(obj, rvec, tvec, CAMERA_K, None)
    d = proj.reshape(-1, 2) - img
    return float(np.sqrt(np.mean(np.sum(d * d, axis=1))))


def _all_in_front(obj: np.ndarray, r_cg: np.ndarray, tvec: np.ndarray) -> bool:
    """True iff every object point is in front of the camera (positive depth)."""
    pc = (r_cg @ obj.T).T + tvec
    return bool(np.all(pc[:, 2] > 0))


def _quad_min_side(quad: np.ndarray) -> float:
    """Shortest edge length (px) of an ordered 4-corner quad — a degeneracy measure."""
    return float(min(np.linalg.norm(quad[i] - quad[(i + 1) % 4]) for i in range(4)))


def _normal_is_physical(r_cg: np.ndarray) -> bool:
    """True if the gate's normal points along the viewing direction (physical solution).

    Convention validated empirically against the object-point ordering in ``GATE_OBJ``:
    the correct PnP solution for a gate ahead has its gate-local +Z axis mapping to a
    camera-frame vector with ``z > 0`` (down-range). The planar mirror solution that
    tilts the gate toward the camera has ``z < 0`` and is non-physical here.
    """
    return bool((r_cg @ np.array([0.0, 0.0, 1.0]))[2] > 0.0)


def pose_cov_body(cov_cam: np.ndarray, tilt_deg: float = CAMERA_TILT_DEG) -> np.ndarray:
    """Rotate a camera-frame position covariance into body FRD: R·Σ·Rᵀ.

    The ESKF gate update lives in the body frame, so the camera-frame anisotropic
    covariance (weak depth axis along camera +z) must be rotated before fusion.
    """
    r = cam_to_body(tilt_deg)
    return r @ np.asarray(cov_cam, float) @ r.T


def _pnp_position_cov(
    gate_scale_m: float, rng: float, reproj_rms: float, vis_frac: float
) -> np.ndarray:
    """Anisotropic camera-frame position covariance (m^2).

    From a monocular planar square, lateral precision scales like ``z/f`` per pixel of
    corner noise while depth precision scales like ``z^2/(f*S)`` — so depth is ``z/S``
    times worse than lateral (monocular range is the weak axis). Inflated by corner
    noise and partial visibility.
    """
    f = float(CAMERA_K[0, 0])
    sigma_px = max(_CORNER_PX_SIGMA, reproj_rms)
    lateral = rng / f * sigma_px
    depth = rng * rng / (f * gate_scale_m) * sigma_px
    infl = 1.0 / max(vis_frac, 0.2)
    return np.diag([(lateral * infl) ** 2, (lateral * infl) ** 2, (depth * infl) ** 2])


def _solve_planar(obj: np.ndarray, img: np.ndarray) -> list[tuple[np.ndarray, np.ndarray, float]]:
    """All planar PnP solutions as (R_cg, tvec, reproj_rms), sorted by reprojection."""
    flag = cv2.SOLVEPNP_IPPE_SQUARE if len(obj) == 4 else cv2.SOLVEPNP_IPPE
    try:
        n, rvecs, tvecs, _errs = cv2.solvePnPGeneric(obj, img, CAMERA_K, None, flags=flag)
    except cv2.error:
        return []
    sols: list[tuple[np.ndarray, np.ndarray, float]] = []
    for i in range(n):
        r_cg, _ = cv2.Rodrigues(rvecs[i])
        t = tvecs[i].ravel()
        if not np.all(np.isfinite(t)) or not _all_in_front(obj, r_cg, t):
            continue
        sols.append((r_cg, t, _reproj_rms(obj, img, rvecs[i], tvecs[i])))
    sols.sort(key=lambda s: s[2])
    return sols


def solve_gate_pose(obs: GateObservation, prior_range: float | None = None) -> GateObservation:
    """Fill ``obs`` metric-pose fields from its corners (in place); return ``obs``.

    Prefers the inner opening; uses all eight corners when both rings are available.
    Evaluates both planar solutions, scoring by reprojection (+ optional range prior)
    and rejecting/downgrading untrustworthy geometry. Sets ``output_mode`` to FULL_POSE,
    POSITION_ONLY (orientation ambiguous), or BEARING_ONLY/NONE and records the reason.

    Parameters
    ----------
    obs : GateObservation
        Candidate with ``inner_corners``/``outer_corners`` populated (or not).
    prior_range : float | None
        Optional expected range (m) from the tracker/map/ESKF used to break the planar
        ambiguity and reject implausible depths. ``None`` = no prior.
    """
    inner_c, outer_c = obs.inner_corners, obs.outer_corners
    ref = outer_c if outer_c is not None else inner_c
    if ref is None:
        obs.output_mode = OutputMode.BEARING_ONLY  # no corners — direction only
        return obs
    if _quad_min_side(ref) < _MIN_QUAD_SIDE_PX:
        # tiny/degenerate quad: a low reprojection here is meaningless — never metric
        obs.reject_reason = "degenerate"
        obs.output_mode = OutputMode.BEARING_ONLY
        return obs

    if inner_c is not None and outer_c is not None:
        # Solve the 8-corner (outer+inner) set, the inner-only set, and the outer-only
        # set. The inner opening is sharper and glow-immune, so if combining with the
        # outer edge materially worsens the reprojection, trust the clean inner solution.
        img8 = np.ascontiguousarray(np.vstack([outer_c, inner_c]), np.float64)
        sols8 = _solve_planar(np.vstack([GATE_OBJ, GATE_OBJ_INNER]), img8)
        sols_in = _solve_planar(GATE_OBJ_INNER, np.ascontiguousarray(inner_c, np.float64))
        sols_out = _solve_planar(GATE_OBJ, np.ascontiguousarray(outer_c, np.float64))
        # ring consistency: inner-only and outer-only ranges must agree (a glared/partial
        # ring makes them diverge — e.g. a collapsed inner reads far while the outer reads near)
        if sols_in and sols_out:
            ri = float(np.linalg.norm(sols_in[0][1]))
            ro = float(np.linalg.norm(sols_out[0][1]))
            if abs(ri - ro) / min(ri, ro) > _RANGE_DISAGREE_FRAC:
                obs.reject_reason = "ring_range_disagree"
                obs.output_mode = OutputMode.BEARING_ONLY
                return obs
        if sols_in and (not sols8 or sols8[0][2] - sols_in[0][2] > _COMBINED_INNER_MARGIN_PX):
            sols, solver, gate_scale = sols_in, "inner4", GATE_INNER_M
            if sols8:
                logger.debug(
                    "PnP: 8-corner rms %.2f >> inner %.2f px — using inner opening",
                    sols8[0][2],
                    sols_in[0][2],
                )
        else:
            sols, solver, gate_scale = sols8, "outer+inner8", GATE_OUTER_M
    elif inner_c is not None:
        sols = _solve_planar(GATE_OBJ_INNER, np.ascontiguousarray(inner_c, np.float64))
        solver, gate_scale = "inner4", GATE_INNER_M
    else:  # outer-only (ref is not None ⇒ outer_c is not None here)
        sols = _solve_planar(GATE_OBJ, np.ascontiguousarray(outer_c, np.float64))
        solver, gate_scale = "outer4", GATE_OUTER_M

    if not sols:  # positive-depth solutions, sorted by reprojection
        obs.reject_reason = "pnp_failed"
        obs.output_mode = OutputMode.BEARING_ONLY
        return obs

    # Reject/validate on the best-reprojection solution (its translation is the most
    # reliable regardless of the orientation ambiguity).
    best = sols[0]
    rng0 = float(np.linalg.norm(best[1]))
    if rng0 < PNP_MIN_RANGE_M or rng0 > PNP_MAX_RANGE_M:
        obs.reject_reason = "range"
        obs.output_mode = OutputMode.BEARING_ONLY
        return obs
    if best[2] > PNP_MAX_REPROJ_PX:
        obs.reject_reason = "reproj"
        obs.output_mode = OutputMode.BEARING_ONLY
        return obs

    # Orientation trust. Reprojection ambiguity is kept SEPARATE from any prior: the
    # margin is the raw reprojection gap between the two planar solutions. A prior only
    # promotes to FULL_POSE when it MEANINGFULLY separates them.
    top = sols[:2]
    reproj_margin = top[1][2] - top[0][2] if len(top) == 2 else math.inf
    faces = [_normal_is_physical(s[0]) for s in top]
    chosen = best
    reason: str | None = None
    if reproj_margin >= PNP_AMBIG_MARGIN_PX:
        mode = OutputMode.FULL_POSE  # reprojection alone distinguishes the solutions
    elif len(top) == 2 and faces[0] != faces[1]:
        chosen = top[0] if faces[0] else top[1]  # exactly one is physically oriented
        mode = OutputMode.FULL_POSE
    elif prior_range is not None:
        d = [abs(float(np.linalg.norm(s[1])) - prior_range) for s in top]
        if len(d) == 2 and abs(d[0] - d[1]) > 0.15 * prior_range:
            chosen = top[int(np.argmin(d))]  # prior clearly prefers one
            mode = OutputMode.FULL_POSE
        else:
            mode, reason = OutputMode.POSITION_ONLY, "ambiguous_orientation"  # prior can't separate
    else:
        mode, reason = OutputMode.POSITION_ONLY, "ambiguous_orientation"

    # never claim a full pose from a non-physical orientation
    if mode is OutputMode.FULL_POSE and not _normal_is_physical(chosen[0]):
        mode, reason = OutputMode.POSITION_ONLY, "normal_nonphysical"

    r_cg, t, rms = chosen
    rng = float(np.linalg.norm(t))
    up_body = CAM_TO_BODY @ (r_cg @ np.array([0.0, 1.0, 0.0]))
    gate_body = CAM_TO_BODY @ t
    obs.pose_cam = (float(t[0]), float(t[1]), float(t[2]))
    obs.up_body = (float(up_body[0]), float(up_body[1]), float(up_body[2]))
    obs.reproj_rms = rms
    obs.ambiguity_margin = float(reproj_margin) if math.isfinite(reproj_margin) else None
    obs.solver = solver
    obs.pose_cov = _pnp_position_cov(gate_scale, rng, rms, obs.visible_fraction)
    obs.bearing = float(math.atan2(gate_body[1], gate_body[0]))
    obs.reject_reason = reason
    obs.output_mode = mode
    return obs
