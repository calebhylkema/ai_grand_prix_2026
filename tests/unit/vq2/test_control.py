"""Unit tests for the VQ2 runtime's pure control math.

Run: uv run pytest tests/unit/vq2/test_control.py -q

Locks in the 2026-07-02 control logic that got the drone airborne past the floor
sink: the camera-row vertical PD servo (``vertical_thrust``) and the tilt-
compensated thrust (``Flight.tilt_comp``). These are the pieces that must not
silently regress — a sign flip or a broken clamp here re-sinks the drone.
"""

from __future__ import annotations

import math
import threading

import cv2
import numpy as np

from drone_racing.deployments.vq2 import calibration as cc
from drone_racing.deployments.vq2 import perception as gates
from drone_racing.deployments.vq2 import runtime as fly
from drone_racing.deployments.vq2.perception import GateTarget, LineTarget


def _gate(cy: float, area: float, bearing: float = 0.0) -> GateTarget:
    """A GateTarget with just the fields the vertical servo reads."""
    return GateTarget(cx=160.0, cy=cy, area=area, bearing=bearing, pose_cam=None)


# --- vertical_thrust: camera-row PD height servo -------------------------------


def test_vertical_thrust_no_gate_is_mild_climb() -> None:
    """No gate visible => a mild climb bias above hover (counters forward sink)."""
    thr = fly.vertical_thrust(None)
    expected = min(fly.THR_MAX, fly.HOVER_THRUST + fly.NO_GATE_CLIMB)
    assert thr == expected
    assert thr > fly.HOVER_THRUST


def test_vertical_thrust_high_gate_climbs() -> None:
    """Gate above centre (cy < CY_TARGET) => climb (thrust above hover)."""
    thr = fly.vertical_thrust(_gate(cy=30.0, area=1000.0))
    assert thr > fly.HOVER_THRUST


def test_vertical_thrust_low_far_gate_holds() -> None:
    """Gate below centre but small/far => hold hover, do NOT dive at the floor."""
    thr = fly.vertical_thrust(_gate(cy=250.0, area=fly.DESCEND_AREA - 1.0))
    assert thr == fly.HOVER_THRUST


def test_vertical_thrust_low_close_gate_descends() -> None:
    """Gate below centre AND close (big area) => descend (thrust below hover)."""
    thr = fly.vertical_thrust(_gate(cy=250.0, area=fly.DESCEND_AREA + 2000.0))
    assert thr < fly.HOVER_THRUST


def test_vertical_thrust_cy_rate_damps_climb() -> None:
    """Rising cy (climbing, gate moving down the frame) bleeds thrust vs. no rate."""
    gate = _gate(cy=30.0, area=1000.0)
    thr_static = fly.vertical_thrust(gate, cy_rate=0.0)
    thr_damped = fly.vertical_thrust(gate, cy_rate=200.0)  # climbing fast
    assert thr_damped < thr_static


def test_vertical_thrust_clamped_to_bounds() -> None:
    """An extreme error saturates at THR_MAX, never beyond."""
    thr = fly.vertical_thrust(_gate(cy=-5000.0, area=1000.0))
    assert thr == fly.THR_MAX


# --- Flight.tilt_comp: hold altitude through banked/pitched turns --------------


def _flight() -> fly.Flight:
    sh = cc.Shared()
    return fly.Flight(sh, conn=None, tgt_sys=1, tgt_comp=1, lock=threading.Lock())


def _set_pitch(fl: fly.Flight, pitch_rad: float) -> None:
    """Force the attitude filter to a pure pitch (rotation about body y)."""
    fl.att.q = np.array([math.cos(pitch_rad / 2), 0.0, math.sin(pitch_rad / 2), 0.0])


def test_command_captures_boundary_telemetry_without_changing_output(monkeypatch) -> None:
    """Phase-4A: command() records requested / post-clamp desired / calibrated levels and
    per-axis saturation, and still SENDS exactly the clamped rates (behaviour unchanged)."""
    sent: dict[str, float] = {}

    def _fake_send(conn, ts, tc, lock, thr, rr, pr, yr) -> None:  # noqa: ANN001
        sent.update(thr=thr, rr=rr, pr=pr, yr=yr)

    monkeypatch.setattr(cc, "send_att", _fake_send)
    fl = _flight()
    fl.command(0.4, 9.9, 0.0, 5.0)  # roll far over MAX_RATE, yaw over MAX_YAW
    assert fl.last_cmd is not None
    assert fl.last_cmd["req"] == (0.4, 9.9, 0.0, 5.0)
    # desired = clamped ACTUAL rates; the SAME values are what got sent
    assert fl.last_cmd["des"][1] == fly.MAX_RATE and fl.last_cmd["des"][3] == fly.MAX_YAW
    assert sent["rr"] == fly.MAX_RATE and sent["yr"] == fly.MAX_YAW and sent["thr"] == 0.4
    # calibrated = desired / RATE_GAIN (applied ONCE, here)
    assert abs(fl.last_cmd["cal"][1] - fly.MAX_RATE / cc.RATE_GAIN[0]) < 1e-9
    assert fl.last_cmd["sat"] == (0.0, 1.0, 0.0, 1.0)  # roll+yaw clamped, pitch not


def test_command_no_saturation_marks_clear(monkeypatch) -> None:
    monkeypatch.setattr(cc, "send_att", lambda *a, **k: None)
    fl = _flight()
    fl.command(0.30, 0.5, -0.5, 0.5)  # all within limits
    assert fl.last_cmd is not None and fl.last_cmd["sat"] == (0.0, 0.0, 0.0, 0.0)
    assert fl.last_cmd["des"] == (0.30, 0.5, -0.5, 0.5)


def test_command_output_is_shadow_independent(monkeypatch) -> None:
    """Phase-4B strict shadow mode: guidance now lives ENTIRELY in ShadowWorker; the
    command boundary has no reference to it, so command() output is deterministic and
    cannot be perturbed by any shadow state. Flight has no ``shadow_policy`` attribute."""
    sent: list[tuple[float, float, float, float]] = []
    monkeypatch.setattr(
        cc, "send_att", lambda c, ts, tc, lk, thr, rr, pr, yr: sent.append((thr, rr, pr, yr))
    )
    fl = _flight()
    assert not hasattr(fl, "shadow_policy")  # guidance is not on the command path
    fl.command(0.31, 0.4, -0.3, 0.2)
    first_send, first_cmd = sent[-1], dict(fl.last_cmd)  # type: ignore[arg-type]
    fl.command(0.31, 0.4, -0.3, 0.2)
    assert sent[-1] == first_send  # deterministic sent rates
    assert dict(fl.last_cmd) == first_cmd  # type: ignore[arg-type]  # identical boundary telemetry


def test_tilt_comp_level_is_identity() -> None:
    """Level (cos=1) leaves an in-range thrust unchanged."""
    fl = _flight()
    _set_pitch(fl, 0.0)
    assert fl.tilt_comp(0.30) == 0.30


def test_tilt_comp_boosts_when_pitched() -> None:
    """A 30° pitch scales thrust up by ~1/cos(30°) to hold vertical force."""
    fl = _flight()
    _set_pitch(fl, math.radians(30.0))
    assert fl.tilt_comp(0.30) == float(
        np.clip(0.30 / math.cos(math.radians(30.0)), fly.THR_MIN, fly.THR_MAX)
    )
    assert fl.tilt_comp(0.30) > 0.30


def test_tilt_comp_bounded_at_extreme_tilt() -> None:
    """Near-90° tilt cannot command runaway thrust: floored cos + THR_MAX clamp."""
    fl = _flight()
    _set_pitch(fl, math.radians(85.0))
    assert fl.tilt_comp(0.40) == fly.THR_MAX


# --- accel_guidance: accel→attitude outer loop (yaw does NOT steer) -------------
# World frame here: x=north(fwd at yaw=0), y=east(right at yaw=0), z=down (NED).


def test_accel_gate_ahead_pitches_forward() -> None:
    """Gate straight ahead, no velocity => nose-down pitch, wings level."""
    pt, rt = fly.accel_guidance(np.array([10.0, 0.0, 0.0]), np.zeros(3), yaw=0.0)
    assert pt < 0.0  # nose down = forward acceleration (body FRD)
    assert rt == 0.0


def test_accel_gate_right_banks_right() -> None:
    """Gate to the right, no velocity => right bank (positive roll), no fwd pitch."""
    pt, rt = fly.accel_guidance(np.array([0.0, 5.0, 0.0]), np.zeros(3), yaw=0.0)
    assert rt > 0.0
    assert pt == 0.0


def test_accel_rightward_drift_past_gate_banks_left_and_brakes() -> None:
    """THE gate-1 scenario: gate ahead-right but drifting right fast => the damping
    term dominates: bank LEFT to null the drift and pitch UP to brake."""
    p_rel = np.array([10.0, 2.0, 0.0])  # gate 10m ahead, 2m right
    v_drone = np.array([6.0, 1.5, 0.0])  # flying fast, drifting right
    v_rel = -v_drone  # gate static: d(p_rel)/dt = -v
    pt, rt = fly.accel_guidance(p_rel, v_rel, yaw=0.0)
    assert rt < 0.0  # left bank against the rightward drift
    assert pt > 0.0  # nose up: brake the too-fast closure


def test_accel_yaw_rotates_world_to_heading_frame() -> None:
    """Facing east (yaw=90°), a gate to the east is straight AHEAD => pitch fwd."""
    pt, rt = fly.accel_guidance(np.array([0.0, 8.0, 0.0]), np.zeros(3), yaw=math.pi / 2)
    assert pt < 0.0
    assert abs(rt) < 1e-9


def test_accel_overspeed_brakes_even_toward_gate() -> None:
    """Flying at the gate but ABOVE V_MAX => brake (the raw PD could never do this)."""
    p_rel = np.array([20.0, 0.0, 0.0])  # gate far ahead
    v_rel = np.array([-6.0, 0.0, 0.0])  # closing at 6 m/s > V_MAX
    pt, _rt = fly.accel_guidance(p_rel, v_rel, yaw=0.0)
    assert pt > 0.0  # nose up: shed speed down to the cap


def test_accel_tilt_clamped() -> None:
    """A huge position error saturates at MAX_TILT_CMD, never beyond."""
    pt, _rt = fly.accel_guidance(np.array([100.0, 0.0, 0.0]), np.zeros(3), yaw=0.0)
    assert pt == -fly.MAX_TILT_CMD


def test_accel_elev_gate_blocks_forward_only() -> None:
    """elev_ok=False forbids forward acceleration (climb first) but keeps braking."""
    pt_fwd, _ = fly.accel_guidance(np.array([10.0, 0.0, 0.0]), np.zeros(3), yaw=0.0, elev_ok=False)
    assert pt_fwd == 0.0  # forward accel suppressed
    pt_brake, _ = fly.accel_guidance(
        np.array([2.0, 0.0, 0.0]), np.array([-5.0, 0.0, 0.0]), yaw=0.0, elev_ok=False
    )
    assert pt_brake > 0.0  # braking still allowed


# --- vision roll: absolute bank from the gate's world-uprightness ---------------


def _render_gate(phi_deg: float) -> np.ndarray:
    """A world-upright 2.7m gate 8m ahead as seen by a body banked ``phi_deg`` right.

    World is NED-ish (x fwd, y right, z down); the gate centre sits where the
    20-deg-UP camera looks (above body-horizontal), so the ring lands mid-frame.
    """
    phi = math.radians(phi_deg)
    r_wb = np.array(  # body->world: pure roll about x
        [[1, 0, 0], [0, math.cos(phi), -math.sin(phi)], [0, math.sin(phi), math.cos(phi)]]
    )
    ctr = np.array([8.0, 0.0, -8.0 * math.tan(math.radians(20.0))])
    h = gates.GATE_OUTER_M / 2
    corners = [
        ctr + np.array([0, sy * h, sz * h]) for sy, sz in ((-1, -1), (1, -1), (1, 1), (-1, 1))
    ]
    img = np.zeros((360, 640, 3), np.uint8)
    pts = []
    for cw in corners:
        cc = gates.CAM_TO_BODY.T @ (r_wb.T @ cw)  # world -> body -> camera
        pts.append(
            (int(round(320.0 * cc[0] / cc[2] + 320.0)), int(round(320.0 * cc[1] / cc[2] + 180.0)))
        )
    cv2.polylines(img, [np.array(pts)], True, (0, 0, 255), 12)
    return img


def test_vision_roll_level_reads_zero() -> None:
    t = gates.nearest(_render_gate(0.0))
    assert t is not None and t.vision_roll is not None
    assert abs(math.degrees(t.vision_roll)) < 2.0


def test_vision_roll_sign_and_magnitude() -> None:
    """THE run-3 lesson: a hidden 40 deg bank must be measurable from the gate's
    in-plane rotation — correct SIGN (right bank => positive) and magnitude."""
    for phi in (15.0, -15.0, 40.0, -40.0):
        t = gates.nearest(_render_gate(phi))
        assert t is not None and t.vision_roll is not None
        assert abs(math.degrees(t.vision_roll) - phi) < 2.5, f"phi={phi}"


# --- cyan racing-line detection + follower thrust -------------------------------


def _cyan_img(draw) -> np.ndarray:
    """A black 640x360 frame with sim-ribbon cyan (BGR ~ (255,229,0)) painted by draw()."""
    img = np.zeros((360, 640, 3), np.uint8)
    draw(img)
    return img


def test_detect_line_finds_diagonal_ribbon() -> None:
    """A thin diagonal cyan stripe is the ribbon: detected, far point at its high end."""
    img = _cyan_img(lambda im: cv2.line(im, (150, 300), (500, 100), (255, 229, 0), 6))
    line = gates.detect_line(img)
    assert line is not None
    assert abs(line.far_cy - 100) < 15  # far point = the stripe's upper end
    assert abs(line.far_cx - 500) < 15
    assert line.far_bearing > 0.0  # upper end right of centre => steer right


def test_detect_line_rejects_blocky_hologram() -> None:
    """A solid cyan square (the gate hologram shares the hue) must NOT read as the path."""
    img = _cyan_img(lambda im: cv2.rectangle(im, (280, 140), (360, 220), (255, 229, 0), -1))
    assert gates.detect_line(img) is None


def test_detect_line_merges_occluded_segments() -> None:
    """Pillars break the ribbon into segments; detection must merge them (far = highest)."""
    img = _cyan_img(
        lambda im: (
            cv2.line(im, (150, 300), (280, 230), (255, 229, 0), 6),
            cv2.line(im, (360, 180), (500, 100), (255, 229, 0), 6),
        )
    )
    line = gates.detect_line(img)
    assert line is not None
    assert abs(line.far_cy - 100) < 15  # far point comes from the far segment


def test_detect_line_aim_point_tracks_arc_not_chord() -> None:
    """THE run-3 lesson: on a curved ribbon the pure-pursuit aim point must sit ON
    the arc (mid-path), not at the far tip — steering at the tip cuts the chord
    straight through the pillar field the arc exists to avoid."""
    # ribbon arcs LEFT then its far end swings right (like the gate-1 approach),
    # drawn as pillar-occluded segments so each stays thin/elongated:
    img = _cyan_img(
        lambda im: (
            cv2.line(im, (320, 340), (240, 240), (255, 229, 0), 6),  # near: up-left
            cv2.line(im, (195, 210), (185, 110), (255, 229, 0), 6),  # mid: left side
            cv2.line(im, (210, 95), (420, 60), (255, 229, 0), 6),  # far: hooks RIGHT
        )
    )
    line = gates.detect_line(img)
    assert line is not None
    assert line.far_bearing > 0.0  # the far TIP is right of centre (the trap)
    assert line.aim_bearing < 0.0  # ...but mid-path is LEFT: follow the arc
    assert line.aim_cy > line.far_cy  # aim point is nearer than the tip


def test_detect_line_ignores_specks() -> None:
    """Tiny cyan noise below LINE_MIN_PX is not a path."""
    img = _cyan_img(lambda im: cv2.circle(im, (320, 180), 3, (255, 229, 0), -1))
    assert gates.detect_line(img) is None


def _line(aim_cy: float, area: float) -> LineTarget:
    return LineTarget(
        cx=320.0,
        cy=200.0,
        area=area,
        bearing=0.0,
        far_cx=320.0,
        far_cy=max(0.0, aim_cy - 100.0),
        aim_cx=320.0,
        aim_cy=aim_cy,
    )


def test_line_thrust_high_path_climbs() -> None:
    """Path aim row above LINE_CY_TARGET => climb (thrust above hover)."""
    assert fly.line_thrust(_line(aim_cy=30.0, area=500.0)) > fly.HOVER_THRUST


def test_line_thrust_low_far_path_holds() -> None:
    """Path below target but small/far => hold hover, never dive at the floor."""
    assert fly.line_thrust(_line(aim_cy=250.0, area=fly.LINE_DESCEND_AREA - 1)) == fly.HOVER_THRUST


def test_line_thrust_low_close_path_descends() -> None:
    """Path below target AND big/near => gentle descend (thrust below hover)."""
    assert fly.line_thrust(_line(aim_cy=250.0, area=fly.LINE_DESCEND_AREA + 500)) < fly.HOVER_THRUST


def test_line_thrust_climb_is_capped() -> None:
    """RUN-5 lesson: the far tip rides the horizon at any height, so the old far-row
    servo pegged max climb all crossing (vz -3 m/s) and overflew gate 1 by 3m+.
    Even a maximal aim-row error must not exceed the soft climb cap."""
    thr = fly.line_thrust(_line(aim_cy=0.0, area=500.0))
    assert thr <= fly.HOVER_THRUST + fly.LINE_CLIMB_CAP


def test_line_thrust_vz_damps_climb() -> None:
    """Already climbing fast (vz<0, NED) => less thrust than static for the same error."""
    line = _line(aim_cy=30.0, area=500.0)
    assert fly.line_thrust(line, vz=-3.0) < fly.line_thrust(line, vz=0.0)


# --- blind_arc_cmd: the metric (velocity-matched) blind corner arc -------------


def test_blind_arc_is_coordinated() -> None:
    """Bank supplies exactly the centripetal accel for the commanded yaw rate:
    g*tan(phi) == v * psidot  (uncapped regime) => the velocity vector actually
    follows the arc the nose traces (AH5: fixed 20-deg bank at run-dependent speed
    put the arc anywhere within metres in a ~2-3m corridor)."""
    v = 2.0  # below both caps at R=2.8
    bank, rate = fly.blind_arc_cmd(v)
    assert math.isclose(9.81 * math.tan(bank), v * rate, rel_tol=1e-6)


def test_blind_arc_pins_radius() -> None:
    """Different entry speeds carve the SAME radius R = v^2 / (g*tan(phi))."""
    for v in (1.0, 1.5, 2.0, 2.5):
        bank, _ = fly.blind_arc_cmd(v)
        assert math.isclose(v * v / (9.81 * math.tan(bank)), fly.TURN_REF_R, rel_tol=1e-6)


def test_blind_arc_caps() -> None:
    """Fast entries cap at the latch bank/yaw clips instead of demanding a tumble."""
    bank, rate = fly.blind_arc_cmd(8.0)
    assert bank <= fly.TURN_BANK and rate <= fly.MAX_YAW


def test_descend_requires_alignment() -> None:
    """Config AO (07-05): the t6.2 run reached gate 1 at a15159 but b+32 deg and the
    row servo dove at THR_MIN onto the jet parked under the gate. A below-centre gate
    must NOT command descent while off-axis, however big it is."""
    off_axis = gates.GateTarget(cx=500.0, cy=279.0, area=15159.0, bearing=0.56, pose_cam=None)
    assert fly.vertical_thrust(off_axis) == fly.HOVER_THRUST
    aligned = gates.GateTarget(cx=330.0, cy=279.0, area=15159.0, bearing=0.03, pose_cam=None)
    assert fly.vertical_thrust(aligned) < fly.HOVER_THRUST  # aligned: descent allowed


def test_metric_fix_rejects_far_pnp() -> None:
    """Config AP (07-05): post-gate-0, gate 1 shows as a tiny frame-top blob that PnP
    ranged at ~28m (true ~12-15m); those fixes reanchored the kf 28m out and jumped
    the velocity integrator 1.6->3.3 m/s in one frame. Far fixes are bearing-only."""
    assert not fly.metric_fix_ok(None)
    assert not fly.metric_fix_ok((2.0, -1.0, 28.5))  # the batch-2 phantom range
    assert fly.metric_fix_ok((1.0, -0.5, 12.1))  # AE1's genuine re-acquisition
    assert fly.metric_fix_ok((0.5, 0.0, 6.0))  # close corner fix


def test_metric_fix_rejects_tiny_gates() -> None:
    """Config AS (07-08): a278-660 px^2 far gates PnP-ranged UNDER 20m around the
    gate-0 pass; differencing their metre-noisy poses poisoned the velocity anchor
    (fl.vz pinned -3 "climbing" while the frames showed the floor 2m below), so the
    climb servos cut thrust and the corner arrived at jet height. Below
    STEER_MIN_AREA the pose is bearing-only-grade."""
    assert not fly.metric_fix_ok((0.5, 0.0, 18.0), area=278.0)  # AR-batch poison fix
    assert not fly.metric_fix_ok((0.5, 0.0, 15.0), area=660.0)
    assert fly.metric_fix_ok((1.0, -0.5, 12.1), area=908.0)  # genuine gate-1 lock
    assert fly.metric_fix_ok((0.5, 0.0, 6.0), area=4464.0)  # close corner fix


def test_target_switch_guard() -> None:
    """Config AV (07-09): a fresh point-blank held gate may not be replaced by a
    much-smaller sighting far away in bearing — that is a DIFFERENT physical gate
    (AU attempt 3: gate 1 a7565 b+32deg clipped out at the gate plane and the
    detector re-locked gate 2 a832 b-9deg in one frame; guidance then dove and
    steered for gate 2 while crossing gate 1's plane outside the hoop)."""
    g1_close = _gate(72.0, 7565.0, bearing=math.radians(32.0))
    g2_far = _gate(133.0, 832.0, bearing=math.radians(-9.0))
    assert not fly.target_switch_ok(g1_close, True, g2_far)  # the AU-3 kill switch
    assert fly.target_switch_ok(None, False, g2_far)  # nothing held: adopt freely
    assert fly.target_switch_ok(g1_close, False, g2_far)  # held stale: adopt
    # same gate collapsing from border clipping: area drops, bearing continuous
    g1_clip = _gate(60.0, 2200.0, bearing=math.radians(38.0))
    assert fly.target_switch_ok(g1_close, True, g1_clip)
    # gate-0 pass flip (a3010 -> a288, delta-bearing 12deg): stays ALLOWED — the
    # AR/AS/AT stack owns that case, the bearing cut keeps AV out of it
    g0 = _gate(46.0, 3010.0, bearing=math.radians(23.0))
    wrong = _gate(64.0, 288.0, bearing=math.radians(11.0))
    assert fly.target_switch_ok(g0, True, wrong)
    # held not point-blank: guard never applies
    far_held = _gate(50.0, 1300.0, bearing=math.radians(30.0))
    assert fly.target_switch_ok(far_held, True, g2_far)
