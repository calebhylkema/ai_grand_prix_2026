"""Unit tests for the active VQ2 gate perception (deployments/vq2/perception.py).

Run: uv run pytest tests/unit/vq2/test_perception.py -q

Currently locks in the camera extrinsic sign (the 2026-07-10 resolution: the camera
is tilted 20 deg UP, not down). Detection/tracking/PnP tests are added alongside those
features in later steps of the perception plan.
"""

from __future__ import annotations

import math

import cv2
import numpy as np

from drone_racing.deployments.vq2 import perception as gates


def _ring_img(cx: int = 320, cy: int = 180, half: int = 80) -> np.ndarray:
    """A red gate ring (outer square with a concentric opening) centred at (cx, cy)."""
    img = np.zeros((360, 640, 3), np.uint8)
    inner = int(round(half * gates.GATE_INNER_M / gates.GATE_OUTER_M))
    cv2.rectangle(img, (cx - half, cy - half), (cx + half, cy + half), (0, 0, 255), -1)
    cv2.rectangle(img, (cx - inner, cy - inner), (cx + inner, cy + inner), (0, 0, 0), -1)
    return img


def _glare_top_bar(img: np.ndarray, cx: int = 320, cy: int = 180, half: int = 80) -> np.ndarray:
    """Blow out the ring's top bar to the measured glare appearance (H155 S40 V250):
    bright, desaturated, hue drifted to magenta — below the base red S/hue thresholds."""
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    hsv[cy - half : cy - half + 18, cx - half : cx + half] = (155, 40, 250)
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)


def _project(p_body: np.ndarray, tilt_deg: float = gates.CAMERA_TILT_DEG) -> tuple[float, float]:
    """Project a body-FRD point to pixels via the camera extrinsic + intrinsics."""
    r_bc = gates.cam_to_body(tilt_deg).T  # body -> camera
    x_c, y_c, z_c = r_bc @ p_body
    assert z_c > 0, "point must be in front of the camera"
    u = gates.CAMERA_K[0, 0] * x_c / z_c + gates.CAMERA_K[0, 2]
    v = gates.CAMERA_K[1, 1] * y_c / z_c + gates.CAMERA_K[1, 2]
    return float(u), float(v)


def test_cam_to_body_is_a_proper_rotation() -> None:
    r = gates.CAM_TO_BODY
    assert np.allclose(r @ r.T, np.eye(3), atol=1e-9)
    assert abs(float(np.linalg.det(r)) - 1.0) < 1e-9
    assert np.allclose(r, gates.cam_to_body(gates.CAMERA_TILT_DEG))
    assert gates.CAMERA_TILT_DEG == 20.0  # spec §3.8: tilted UP by 20 deg


def test_optical_axis_points_up_in_body() -> None:
    """Camera +z (optical axis) must map to body z < 0 (UP) for the +20 deg mount."""
    axis_up = gates.cam_to_body(20.0) @ np.array([0.0, 0.0, 1.0])
    assert axis_up[0] > 0.0, "optical axis should point forward"
    assert axis_up[2] < 0.0, "optical axis should point UP (body z<0)"
    assert axis_up[1] == 0.0
    assert abs(axis_up[0] - math.cos(math.radians(20.0))) < 1e-9
    assert abs(axis_up[2] + math.sin(math.radians(20.0))) < 1e-9


def test_wrong_tilt_sign_inverts_optical_axis() -> None:
    """The old (wrong) -20 deg sign points the optical axis DOWN — the guard that
    would have caught 10 days of inverted PnP geometry."""
    axis_down = gates.cam_to_body(-20.0) @ np.array([0.0, 0.0, 1.0])
    assert axis_down[2] > 0.0, "the -20 deg (DOWN) sign must fail the up-axis property"


def test_camera_right_maps_to_body_right_tilt_invariant() -> None:
    """Camera x (right) maps to body +y (right); a pitch tilt about y leaves it fixed."""
    for tilt in (-20.0, 0.0, 20.0):
        right = gates.cam_to_body(tilt) @ np.array([1.0, 0.0, 0.0])
        assert np.allclose(right, [0.0, 1.0, 0.0], atol=1e-9)


def test_projection_signs_body_camera_image() -> None:
    """End-to-end sign chain: with a 20 deg up-tilt a level-forward gate sits in the
    LOWER frame; climbing raises it; moving right shifts it right."""
    u0, v0 = _project(np.array([8.0, 0.0, 0.0]))  # 8 m ahead, level
    assert abs(u0 - 320.0) < 1e-6, "level-forward gate is horizontally centred"
    assert v0 > 180.0, "camera looks up, so a level-forward point is below image centre"

    _, v_up = _project(np.array([8.0, 0.0, -2.0]))  # 2 m higher
    assert v_up < v0, "climbing moves the gate up the frame (smaller row)"

    u_right, _ = _project(np.array([8.0, 2.0, 0.0]))  # 2 m to the right
    assert u_right > u0, "a gate to the right projects right of centre"


# --- structured detection: detect_candidates ----------------------------------


def test_full_ring_detected_with_inner_and_outer_corners() -> None:
    cands = gates.detect_candidates(_ring_img())
    assert len(cands) == 1
    obs = cands[0]
    assert obs.visibility is gates.Visibility.FULL
    assert obs.visible_fraction == 1.0
    assert obs.outer_corners is not None and obs.outer_corners.shape == (4, 2)
    assert obs.inner_corners is not None and obs.inner_corners.shape == (4, 2)
    assert obs.scores["ring"] > 0.6, "hollow ring with spec proportions scores high"
    assert obs.confidence > 0.6
    assert abs(math.degrees(obs.bearing)) < 2.0  # centred gate ≈ zero bearing


def test_solid_red_blob_rejected_as_glow() -> None:
    img = np.zeros((360, 640, 3), np.uint8)
    cv2.rectangle(img, (270, 130), (370, 230), (0, 0, 255), -1)  # solid, no opening
    assert gates.detect_candidates(img) == []


def test_partial_gate_at_right_border_kept_as_bearing_only() -> None:
    # ring centred past the right edge → clipped, opening open to the border
    cands = gates.detect_candidates(_ring_img(cx=600, cy=180, half=80))
    assert len(cands) == 1
    obs = cands[0]
    assert obs.visibility is gates.Visibility.PARTIAL
    assert obs.visible_fraction < 1.0
    assert obs.outer_corners is None, "clipped contour corners are not gate corners"
    assert obs.output_mode is gates.OutputMode.BEARING_ONLY
    assert obs.pose_cam is None
    assert math.degrees(obs.bearing) > 15.0, "gate to the right → positive bearing"


# --- persistent identity: GateTracker -----------------------------------------


def _obs(
    cx: float,
    cy: float = 180.0,
    area: float = 15000.0,
    conf: float = 0.9,
    vis: gates.Visibility = gates.Visibility.FULL,
) -> gates.GateObservation:
    return gates.GateObservation(
        cx=cx,
        cy=cy,
        area=area,
        bbox=(int(cx - 60), int(cy - 60), 120, 120),
        bearing=math.atan2(cx - 320.0, 320.0),
        visibility=vis,
        visible_fraction=1.0 if vis is gates.Visibility.FULL else 0.5,
        outer_corners=None,
        inner_corners=None,
        confidence=conf,
    )


def _establish(tr: gates.GateTracker, cx: float, n: int = 5) -> gates.GateObservation:
    """Drive a steady detection until it is confirmed as the active target."""
    sel = None
    for _ in range(n):
        sel = tr.step([_obs(cx)], active_gate_index=0)
    assert sel is not None, "target should confirm as active"
    return sel


def test_tracker_confirms_and_holds_identity() -> None:
    tr = gates.GateTracker()
    sel = _establish(tr, 200.0)
    tid = sel.track_id
    for dx in (4, 8, 12):
        sel = tr.step([_obs(200.0 + dx)], active_gate_index=0)
        assert sel is not None and sel.track_id == tid
    assert len(tr.tracks) == 1


def test_tracker_does_not_let_downstream_gate_steal_identity() -> None:
    tr = gates.GateTracker()
    a_tid = _establish(tr, 200.0).track_id
    # a bigger, fully-visible downstream gate appears far right while the active gate
    # continues (now partial, lower confidence). Active identity must NOT jump to it.
    for _ in range(5):
        sel = tr.step(
            [
                _obs(205.0, conf=0.5, vis=gates.Visibility.PARTIAL),
                _obs(520.0, area=32000.0, conf=0.98),
            ],
            active_gate_index=0,
        )
        assert sel is not None and sel.track_id == a_tid
    assert len(tr.tracks) == 2  # both tracked, but identity stayed with the active gate


def test_tracker_survives_brief_dropout() -> None:
    tr = gates.GateTracker()
    a_tid = _establish(tr, 200.0).track_id
    for _ in range(2):  # detector drops out — predicted only, no fresh measurement
        assert tr.step([], active_gate_index=0) is None
    sel = tr.step([_obs(205.0)], active_gate_index=0)  # reacquired
    assert sel is not None and sel.track_id == a_tid


def test_tracker_reacquire_blacklists_passed_gate_and_uses_geometry() -> None:
    """After advancing, the just-passed gate must NOT be reacquired; the new target
    must match the expected turn direction."""
    tr = gates.GateTracker()
    passed_tid = _establish(tr, 320.0).track_id  # gate 0 dead ahead
    # advance to gate 1 (a right turn). The passed gate lingers at centre while a new
    # gate appears to the right. The centred passed gate is blacklisted AND off-side.
    sel = None
    for _ in range(4):
        sel = tr.step(
            [_obs(320.0), _obs(520.0)],
            active_gate_index=1,
            expected_turn_sign=+1.0,  # right turn
        )
    assert sel is not None
    assert sel.track_id != passed_tid, "must not reacquire the passed gate"
    assert sel.cx > 400, "new target is the right-hand gate, matching the turn direction"


def test_tracker_advance_returns_none_until_confirmed() -> None:
    tr = gates.GateTracker()
    _establish(tr, 320.0)
    # first frame after advance cannot immediately re-confirm a new active target
    assert tr.step([_obs(520.0)], active_gate_index=1, expected_turn_sign=+1.0) is None


def test_range_band_uses_slant_range_not_depth() -> None:
    tr = gates.GateTracker()
    obs = _obs(500.0)
    obs.pose_cam = (5.0, 0.0, 5.0)  # camera depth z=5 m, slant range ≈ 7.07 m
    assert abs(obs.slant_range - 7.071) < 1e-2 and obs.range_cam == 5.0
    tr._range_band = (6.0, 8.0)  # accepts by slant (would reject on depth=5)
    assert tr._eligible_new(obs)
    tr._range_band = (3.0, 5.5)  # a depth-based band would accept; slant correctly rejects
    assert not tr._eligible_new(obs)


def test_tracker_passed_gate_excluded_after_track_deletion() -> None:
    """A passed gate that disappears, has its track deleted, then reappears as a NEW
    track id must stay excluded during the acquisition window (not just ID-blacklisted)."""
    tr = gates.GateTracker()
    _establish(tr, 320.0)  # gate 0 confirmed active at centre
    tr.step([_obs(320.0)], active_gate_index=1)  # advance → retire + set exclusion zone
    for _ in range(gates._MAX_MISSES + 2):
        tr.step([], active_gate_index=1)  # gate 0 vanishes → its original track is deleted
    # gate 0 reappears at centre as a fresh track within the exclusion window
    for _ in range(5):
        assert tr.step([_obs(320.0)], active_gate_index=1) is None
    # after the window expires the region is eligible again
    sel = None
    for _ in range(gates._PASSED_WINDOW_FRAMES):
        sel = tr.step([_obs(320.0)], active_gate_index=1)
    assert sel is not None


def test_tracker_rejects_impossible_jump() -> None:
    tr = gates.GateTracker()
    tr.step([_obs(100.0)], active_gate_index=0)
    tr.step([_obs(520.0)], active_gate_index=0)  # 420 px jump — not the same gate
    assert len(tr.tracks) == 2


# --- metric PnP: solve_gate_pose ----------------------------------------------


def _project_obj(obj: np.ndarray, r_cg: np.ndarray, t: np.ndarray) -> np.ndarray:
    rvec, _ = cv2.Rodrigues(r_cg)
    proj, _ = cv2.projectPoints(obj, rvec, t.reshape(3, 1), gates.CAMERA_K, None)
    return proj.reshape(-1, 2).astype(np.float64)


def _posed_obs(
    t: np.ndarray, r_cg: np.ndarray, *, inner: bool = True, outer: bool = True
) -> gates.GateObservation:
    return gates.GateObservation(
        cx=320.0,
        cy=180.0,
        area=15000.0,
        bbox=(280, 140, 80, 80),
        bearing=0.0,
        visibility=gates.Visibility.FULL,
        visible_fraction=1.0,
        outer_corners=_project_obj(gates.GATE_OBJ, r_cg, t) if outer else None,
        inner_corners=_project_obj(gates.GATE_OBJ_INNER, r_cg, t) if inner else None,
        confidence=0.9,
    )


def test_pnp_recovers_known_pose() -> None:
    t_true = np.array([0.3, -0.2, 6.0])
    r_cg, _ = cv2.Rodrigues(np.array([0.0, 0.35, 0.0]))  # ~20 deg yaw — unambiguous
    obs = gates.solve_gate_pose(_posed_obs(t_true, r_cg))
    assert obs.output_mode is gates.OutputMode.FULL_POSE
    assert obs.pose_cam is not None
    assert np.allclose(np.array(obs.pose_cam), t_true, atol=0.05)
    assert obs.reproj_rms is not None and obs.reproj_rms < 1.0
    assert obs.solver == "outer+inner8"
    # depth is the weak monocular axis
    assert obs.pose_cov[2, 2] > obs.pose_cov[0, 0]


def test_pnp_head_on_orientation_ambiguity_not_faked_by_prior() -> None:
    """A head-on square is orientation-ambiguous; both planar solutions share the same
    range, so a range prior CANNOT separate them and must not fake a full pose."""
    t_true = np.array([0.0, 0.0, 6.0])
    r_cg = np.eye(3)  # perfectly head-on square — maximally ambiguous
    obs = gates.solve_gate_pose(_posed_obs(t_true, r_cg, outer=False))
    assert obs.output_mode is gates.OutputMode.POSITION_ONLY
    assert obs.reject_reason == "ambiguous_orientation"
    assert obs.pose_cam is not None and abs(obs.pose_cam[2] - 6.0) < 0.1
    # a prior at the same range as both solutions does not distinguish them
    obs2 = gates.solve_gate_pose(_posed_obs(t_true, r_cg, outer=False), prior_range=6.0)
    assert obs2.output_mode is gates.OutputMode.POSITION_ONLY


def test_gate_normal_convention_is_physical() -> None:
    """The physical solution (gate ahead) has camera-z normal > 0 in the GATE_OBJ
    convention; solve_gate_pose selects a physically-oriented solution for FULL_POSE."""
    r_true, _ = cv2.Rodrigues(np.array([0.0, 0.35, 0.0]))  # a real forward-facing gate
    assert gates._normal_is_physical(r_true)
    obs = gates.solve_gate_pose(_posed_obs(np.array([0.2, 0.0, 7.0]), r_true))
    assert obs.output_mode is gates.OutputMode.FULL_POSE
    # the recovered orientation matches truth (chosen the physical, not the mirror)
    expected_up = gates.CAM_TO_BODY @ (r_true @ np.array([0.0, 1.0, 0.0]))
    assert np.allclose(np.array(obs.up_body), expected_up, atol=0.1)


def test_pose_cov_rotates_to_body_weak_axis() -> None:
    """The weak depth axis (camera +z) must rotate to the correct body direction."""
    cov_cam = np.diag([0.02, 0.02, 4.0])  # large uncertainty along camera depth (z)
    cov_body = gates.pose_cov_body(cov_cam)
    assert np.allclose(cov_body, cov_body.T)  # stays symmetric
    # the max-variance eigenvector should align with CAM_TO_BODY @ (camera +z)
    w, v = np.linalg.eigh(cov_body)
    weak_axis = v[:, int(np.argmax(w))]
    expected = gates.CAM_TO_BODY @ np.array([0.0, 0.0, 1.0])
    assert abs(abs(float(weak_axis @ expected)) - 1.0) < 1e-6
    assert abs(w.max() - 4.0) < 1e-9  # eigenvalues (variances) preserved by rotation


def test_pnp_covariance_grows_with_range() -> None:
    r_cg, _ = cv2.Rodrigues(np.array([0.0, 0.30, 0.0]))
    near = gates.solve_gate_pose(_posed_obs(np.array([0.0, 0.0, 4.0]), r_cg))
    far = gates.solve_gate_pose(_posed_obs(np.array([0.0, 0.0, 12.0]), r_cg))
    assert far.pose_cov[2, 2] > near.pose_cov[2, 2]


def test_pnp_rejects_out_of_range() -> None:
    r_cg, _ = cv2.Rodrigues(np.array([0.0, 0.30, 0.0]))
    obs = gates.solve_gate_pose(_posed_obs(np.array([0.0, 0.0, 40.0]), r_cg))
    assert obs.output_mode is gates.OutputMode.BEARING_ONLY
    assert obs.reject_reason == "range"
    assert obs.pose_cam is None


def test_pnp_no_corners_is_bearing_only() -> None:
    obs = _obs(500.0, vis=gates.Visibility.PARTIAL)  # no corners
    solved = gates.solve_gate_pose(obs)
    assert solved.output_mode is gates.OutputMode.BEARING_ONLY
    assert solved.pose_cam is None


def test_specular_mask_recovers_glared_top_bar() -> None:
    """The base red mask loses a glared top bar; the specular mask recovers it (only
    because it is contiguous with the solid-red ring)."""
    img = _glare_top_bar(_ring_img())
    band = (slice(100, 118), slice(240, 400))  # the glared top-bar region
    base_cov = gates.red_mask(img)[band].mean()
    spec_cov = gates.red_mask_specular(img)[band].mean()
    assert base_cov < 60.0, "base mask should mostly miss the desaturated top bar"
    assert spec_cov > 200.0, "specular mask should recover the contiguous glared bar"


def test_detect_recovers_glared_ring_as_full() -> None:
    """A ring whose top bar is glared is still detected as a FULL ring with corners."""
    cands = gates.detect_candidates(_glare_top_bar(_ring_img()))
    assert len(cands) == 1
    assert cands[0].visibility is gates.Visibility.FULL
    assert cands[0].outer_corners is not None


def test_pnp_rejects_degenerate_quad() -> None:
    """A tiny/degenerate quad must not yield a metric pose even if PnP reprojects low."""
    tiny = np.array([[318.0, 178.0], [323.0, 178.0], [323.0, 183.0], [318.0, 183.0]])  # 5 px
    obs = gates.GateObservation(
        cx=320.0,
        cy=180.0,
        area=25.0,
        bbox=(318, 178, 5, 5),
        bearing=0.0,
        visibility=gates.Visibility.FULL,
        visible_fraction=1.0,
        outer_corners=tiny,
        inner_corners=None,
        confidence=0.5,
    )
    gates.solve_gate_pose(obs)
    assert obs.output_mode is gates.OutputMode.BEARING_ONLY
    assert obs.reject_reason == "degenerate"
    assert obs.pose_cam is None


def test_pnp_rejects_inner_outer_range_disagreement() -> None:
    """When inner-only and outer-only ranges disagree materially (a glared/partial ring),
    the observation is downgraded to bearing-only rather than emitting a wrong pose."""
    r_cg = np.eye(3)
    outer = _project_obj(gates.GATE_OBJ, r_cg, np.array([0.0, 0.0, 7.0]))  # outer says 7 m
    inner = _project_obj(gates.GATE_OBJ_INNER, r_cg, np.array([0.0, 0.0, 11.0]))  # inner says 11 m
    obs = gates.GateObservation(
        cx=320.0,
        cy=180.0,
        area=15000.0,
        bbox=(280, 140, 80, 80),
        bearing=0.0,
        visibility=gates.Visibility.FULL,
        visible_fraction=1.0,
        outer_corners=outer,
        inner_corners=inner,
        confidence=0.9,
    )
    gates.solve_gate_pose(obs)
    assert obs.output_mode is gates.OutputMode.BEARING_ONLY
    assert obs.reject_reason == "ring_range_disagree"


def test_pnp_covariance_grows_with_corner_noise() -> None:
    """Corner jitter (a proxy for inner/outer disagreement / temporal range jitter) raises
    reprojection RMS, which must inflate the reported pose covariance."""
    t = np.array([0.0, 0.0, 6.0])
    r_cg, _ = cv2.Rodrigues(np.array([0.0, 0.3, 0.0]))
    clean = _posed_obs(t, r_cg)
    gates.solve_gate_pose(clean)
    noisy = _posed_obs(t, r_cg)
    rng = np.random.default_rng(0)
    noisy.inner_corners = noisy.inner_corners + rng.normal(0, 1.5, (4, 2))
    noisy.outer_corners = noisy.outer_corners + rng.normal(0, 1.5, (4, 2))
    gates.solve_gate_pose(noisy)
    assert clean.pose_cov is not None and noisy.pose_cov is not None
    assert np.trace(noisy.pose_cov) > np.trace(clean.pose_cov)


def test_pnp_prefers_inner_when_outer_edge_inconsistent() -> None:
    """When the (glow-inflated) outer edge disagrees with the clean inner opening, the
    solver uses inner-only rather than a worse 8-corner fit."""
    t = np.array([0.0, 0.0, 7.0])
    r_cg, _ = cv2.Rodrigues(np.array([0.0, 0.3, 0.0]))
    inner = _project_obj(gates.GATE_OBJ_INNER, r_cg, t)
    outer = _project_obj(gates.GATE_OBJ, r_cg, t)
    c = outer.mean(axis=0)
    outer = c + (outer - c) * 1.12  # ~12% glow inflation of the outer boundary
    obs = gates.GateObservation(
        cx=320.0,
        cy=180.0,
        area=15000.0,
        bbox=(280, 140, 80, 80),
        bearing=0.0,
        visibility=gates.Visibility.FULL,
        visible_fraction=1.0,
        outer_corners=outer,
        inner_corners=inner,
        confidence=0.9,
    )
    gates.solve_gate_pose(obs)
    assert obs.solver == "inner4"
    assert obs.pose_cam is not None and abs(obs.pose_cam[2] - 7.0) < 0.2  # inner → correct range
