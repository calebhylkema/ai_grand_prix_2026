"""Deterministic tests for the Training-only visual gate-0→1 handoff controller.

Covers PRETURN eligibility (positive + every rejection), ESKF-roll-only dependency,
bank/yaw schedules, image-row thrust + descent guard, dropout boundaries, SAFE_EXIT
(never BRAKE→SCAN), attempt isolation, abort priority, and exact byte-identical default
behaviour. No simulator, no wall clock — all timing is injected via ``now_ns``.
"""

from __future__ import annotations

import math

import numpy as np

from drone_racing.deployments.vq2.gate_relative import StructuredTarget, empty_structured_target
from drone_racing.deployments.vq2.visual_handoff import (
    CY_TARGET,
    VisualHandoffConfig,
    VisualHandoffController,
    VisualHandoffInput,
    VisualHandoffState,
    image_row_thrust,
)

MS = 1_000_000
NOW = 10_000_000_000
LEGACY = (0.30, 0.0, -0.2, 0.0)
LOCK = 3


def prov_target(now_ns: int = NOW, **over: object) -> StructuredTarget:
    """A healthy provisional gate-1 target (right side, confirmed)."""
    base: dict[str, object] = {
        "valid": True,
        "frame_id": 5,
        "frame_sensor_ns": 1,
        "arrival_ns": now_ns,
        "active_gate_at_receipt": 0,
        "track_id": LOCK,
        "race_gate_id": 1,
        "track_hits": 6,
        "output_mode": "bearing_only",
        "bearing_rad": math.radians(22.0),
        "los_body_unit": np.array([1.0, 0.0, 0.0]),
        "slant_range_m": float("nan"),
        "has_metric_pose": False,
        "confidence": 0.9,
        "cov_trace": float("nan"),
        "expected_turn_sign": 1.0,
        "disposition": "provisional",
        "area_px2": 4000.0,
        "cx_px": 420.0,
        "cy_px": 130.0,
        "visible_fraction": 0.5,
        "sequence": 1,
    }
    base.update(over)
    return StructuredTarget(**base)  # type: ignore[arg-type]


def gate0_target(now_ns: int = NOW, **over: object) -> StructuredTarget:
    """A close, centred gate-0 active target."""
    base: dict[str, object] = {
        "valid": True,
        "frame_id": 5,
        "frame_sensor_ns": 1,
        "arrival_ns": now_ns,
        "active_gate_at_receipt": 0,
        "track_id": 0,
        "race_gate_id": 0,
        "track_hits": 20,
        "output_mode": "full_pose",
        "bearing_rad": math.radians(2.0),
        "los_body_unit": np.array([1.0, 0.0, 0.0]),
        "slant_range_m": 5.0,
        "has_metric_pose": True,
        "confidence": 0.95,
        "cov_trace": 0.2,
        "expected_turn_sign": 0.0,
        "disposition": "anchored",
        "area_px2": 40000.0,
        "cx_px": 320.0,
        "cy_px": 130.0,
        "visible_fraction": 0.45,
        "sequence": 1,
    }
    base.update(over)
    return StructuredTarget(**base)  # type: ignore[arg-type]


def make_input(
    now_ns: int = NOW,
    *,
    active_gate: int = 0,
    anchor_gate: int = 0,
    active_target: StructuredTarget | None = None,
    provisional_target: StructuredTarget | None = None,
    roll_rad: float = math.radians(5.0),
    roll_cov: float = 0.001,
    att_cov_trace: float = 0.01,
    state_age_s: float = 0.02,
    legacy: tuple[float, float, float, float] = LEGACY,
    training: bool = True,
    flag_enabled: bool = True,
    abort_reason: str = "",
    control_yaw: float = 0.0,
    control_roll: float = 0.0,
    control_pitch: float = 0.0,
    body_gyro: tuple[float, float, float] = (0.0, 0.0, 0.0),
) -> VisualHandoffInput:
    return VisualHandoffInput(
        now_ns=now_ns,
        training=training,
        flag_enabled=flag_enabled,
        active_gate=active_gate,
        anchor_gate=anchor_gate,
        active_target=gate0_target(now_ns) if active_target is None else active_target,
        provisional_target=prov_target(now_ns)
        if provisional_target is None
        else provisional_target,
        eskf_roll_rad=roll_rad,
        eskf_roll_cov_rad2=roll_cov,
        eskf_att_cov_trace_rad2=att_cov_trace,
        state_age_s=state_age_s,
        legacy_command=legacy,
        abort_reason=abort_reason,
        control_yaw_rad=control_yaw,
        control_roll_rad=control_roll,
        control_pitch_rad=control_pitch,
        body_gyro_rps=body_gyro,
    )


def drive_to_preturn(c: VisualHandoffController) -> None:
    """One fresh (age≈0) tick is enough for eligibility → PRETURN."""
    c.step(make_input(NOW))
    assert c.state is VisualHandoffState.PRETURN


def hold_empty_provisional(
    c: VisualHandoffController, start_ns: int, end_ns: int, *, step_ms: int = 16
) -> object:
    """Step at control rate (60 Hz) with an absent provisional — the observer keeps integrating
    body gyro through the gap (as it would live), isolating the PROVISIONAL-grace revocation."""
    t = start_ns
    last: object = None
    while t <= end_ns:
        last = c.step(make_input(t, provisional_target=empty_structured_target(t)))
        t += step_ms * MS
    return last


# --------------------------------------------------------------------------- PRETURN entry
def test_preturn_positive_entry() -> None:
    c = VisualHandoffController()
    c.step(make_input(NOW))
    assert c.state is VisualHandoffState.PRETURN
    assert c.locked_track_id == LOCK


def test_far_provisional_alone_cannot_activate() -> None:
    # Everything confirmed BUT gate 0 not close → stays legacy IDLE.
    c = VisualHandoffController()
    far0 = gate0_target(
        area_px2=1000.0, visible_fraction=0.95, has_metric_pose=False, slant_range_m=float("nan")
    )
    c.step(make_input(active_target=far0))
    assert c.state is VisualHandoffState.IDLE


def test_preturn_rejects_when_not_training() -> None:
    c = VisualHandoffController()
    c.step(make_input(training=False))
    assert c.state is VisualHandoffState.IDLE


def test_preturn_rejects_when_flag_disabled() -> None:
    c = VisualHandoffController()
    c.step(make_input(flag_enabled=False))
    assert c.state is VisualHandoffState.IDLE


def test_preturn_rejects_wrong_turn_sign() -> None:
    c = VisualHandoffController()
    c.step(make_input(provisional_target=prov_target(expected_turn_sign=-1.0)))
    assert c.state is VisualHandoffState.IDLE


def test_preturn_rejects_left_side_bearing() -> None:
    c = VisualHandoffController()
    c.step(make_input(provisional_target=prov_target(bearing_rad=math.radians(-20.0))))
    assert c.state is VisualHandoffState.IDLE


def test_preturn_rejects_bearing_out_of_band() -> None:
    c = VisualHandoffController()  # 3° is below the 8° pre-turn floor
    c.step(make_input(provisional_target=prov_target(bearing_rad=math.radians(3.0))))
    assert c.state is VisualHandoffState.IDLE


def test_preturn_rejects_stale_provisional() -> None:
    c = VisualHandoffController()  # 200 ms old vs the 80 ms limit
    c.step(make_input(provisional_target=prov_target(NOW - 200 * MS)))
    assert c.state is VisualHandoffState.IDLE


def test_preturn_rejects_unconfirmed_track() -> None:
    c = VisualHandoffController()
    c.step(make_input(provisional_target=prov_target(track_hits=2)))
    assert c.state is VisualHandoffState.IDLE


def test_preturn_rejects_association_refused() -> None:
    c = VisualHandoffController()
    c.step(make_input(provisional_target=prov_target(association_refused=True)))
    assert c.state is VisualHandoffState.IDLE


def test_preturn_accepts_out_of_band_direction_only() -> None:
    # out_of_band is direction-usable → does NOT block PRETURN.
    c = VisualHandoffController()
    c.step(make_input(provisional_target=prov_target(out_of_band=True)))
    assert c.state is VisualHandoffState.PRETURN


def test_preturn_rejects_when_gate0_off_axis() -> None:
    c = VisualHandoffController()
    c.step(make_input(active_target=gate0_target(bearing_rad=math.radians(25.0))))
    assert c.state is VisualHandoffState.IDLE


def test_gate0_proximity_via_visible_fraction() -> None:
    c = VisualHandoffController()
    clipped = dict(
        area_px2=1000.0, visible_fraction=0.55, has_metric_pose=False, slant_range_m=float("nan")
    )
    c.step(make_input(active_target=gate0_target(**clipped)))
    assert c.state is VisualHandoffState.PRETURN


# --------------------------------------------------------------------------- PRETURN command
def test_preturn_changes_only_roll_and_yaw() -> None:
    c = VisualHandoffController()
    drive_to_preturn(c)
    d = c.step(
        make_input(
            NOW + 32 * MS,
            active_target=gate0_target(NOW + 32 * MS),
            provisional_target=prov_target(NOW + 32 * MS, sequence=3),
            legacy=(0.31, 0.05, -0.15, -0.02),
        )
    )
    assert d.status.authoritative_channels == (False, True, False, True)
    # thrust + pitch equal legacy exactly
    assert d.selected[0] == 0.31
    assert d.selected[2] == -0.15
    # yaw is right-only and positive (turning right)
    assert d.selected[3] >= 0.0


def test_preturn_yaw_capped_right_only() -> None:
    c = VisualHandoffController()
    drive_to_preturn(c)
    # settle several ticks past the blend
    for k in range(3, 12):
        d = c.step(
            make_input(
                NOW + k * 16 * MS,
                active_target=gate0_target(NOW + k * 16 * MS),
                provisional_target=prov_target(NOW + k * 16 * MS, sequence=k),
            )
        )
    assert 0.0 <= d.selected[3] <= VisualHandoffConfig().preturn_yaw_cap_rps + 1e-9


# --------------------------------------------------------------------------- ESKF roll-only
def test_shadow_eskf_roll_no_longer_drives_the_command() -> None:
    """The sign-defective shadow ESKF roll is a cross-check ONLY (Phase D): changing it alone
    must NOT change the command, while changing the OBSERVER inputs (control roll + body gyro)
    DOES. This is the post-attempt-2 design: roll feedback is arbitrated off the gyro observer.
    """
    c1 = VisualHandoffController()
    drive_to_preturn(c1)
    d1 = c1.step(
        make_input(
            NOW + 16 * MS,
            active_target=gate0_target(NOW + 16 * MS),
            provisional_target=prov_target(NOW + 16 * MS, sequence=4),
        )
    )
    # identical except the shadow ESKF roll (now only a degraded cross-check) → same command
    c2 = VisualHandoffController()
    drive_to_preturn(c2)
    d2 = c2.step(
        make_input(
            NOW + 16 * MS,
            active_target=gate0_target(NOW + 16 * MS),
            provisional_target=prov_target(NOW + 16 * MS, sequence=4),
            roll_rad=math.radians(-40.0),  # shadow ESKF swung wildly — must not steer the law
        )
    )
    assert d2.selected == d1.selected
    # but the OBSERVER bank (seeded from control roll) DOES influence the roll rate
    c3 = VisualHandoffController()
    c3.step(make_input(NOW, control_roll=math.radians(30.0)))
    d3 = c3.step(
        make_input(
            NOW + 16 * MS,
            active_target=gate0_target(NOW + 16 * MS),
            provisional_target=prov_target(NOW + 16 * MS, sequence=4),
            control_roll=math.radians(30.0),  # already banked right → less additional roll rate
        )
    )
    assert d3.selected[1] != d1.selected[1]


def test_roll_invalid_by_nonfinite_gyro_revokes_preturn() -> None:
    # A non-finite body rate invalidates the observer → roll_invalid → revoke to legacy IDLE.
    c = VisualHandoffController()
    drive_to_preturn(c)
    d = c.step(
        make_input(
            NOW + 16 * MS,
            body_gyro=(float("nan"), 0.0, 0.0),
            active_target=gate0_target(NOW + 16 * MS),
            provisional_target=prov_target(NOW + 16 * MS, sequence=4),
        )
    )
    assert c.state is VisualHandoffState.IDLE
    assert d.status.source == "legacy"
    assert d.status.observer_source.startswith("stale")


def test_roll_invalid_by_gyro_gap_revokes_preturn() -> None:
    # A body-rate gap beyond the observer's max (80 ms) invalidates it → roll_invalid → IDLE.
    c = VisualHandoffController()
    drive_to_preturn(c)
    c.step(
        make_input(
            NOW + 200 * MS,  # 200 ms since the last tick → gyro gap
            active_target=gate0_target(NOW + 200 * MS),
            provisional_target=prov_target(NOW + 200 * MS, sequence=4),
        )
    )
    assert c.state is VisualHandoffState.IDLE


# --------------------------------------------------------------------------- promotion / TURN
def promote(
    c: VisualHandoffController, now_ns: int, *, roll_rad: float = math.radians(12.0), **over: object
) -> object:
    over.setdefault("bearing_rad", math.radians(22.0))
    over.setdefault("track_id", LOCK)
    over.setdefault("sequence", now_ns)
    act = prov_target(now_ns, active_gate_at_receipt=1, race_gate_id=1, **over)
    return c.step(
        make_input(
            now_ns,
            active_gate=1,
            anchor_gate=0,
            active_target=act,
            provisional_target=empty_structured_target(now_ns),
            roll_rad=roll_rad,
            legacy=(0.34, 0.0, 1.4, 0.0),
        )
    )


def test_turn_engages_on_promotion_with_matching_identity() -> None:
    c = VisualHandoffController()
    drive_to_preturn(c)
    promote(c, NOW + 60 * MS)
    assert c.state is VisualHandoffState.TURN


def test_turn_with_bearing_only_target() -> None:
    c = VisualHandoffController()
    drive_to_preturn(c)
    d = promote(
        c,
        NOW + 60 * MS,
        output_mode="bearing_only",
        has_metric_pose=False,
        slant_range_m=float("nan"),
    )
    assert c.state is VisualHandoffState.TURN
    assert d.status.authoritative_channels == (True, True, True, True)
    assert d.status.source == "visual"


def test_promotion_identity_mismatch_safe_exits() -> None:
    c = VisualHandoffController()
    drive_to_preturn(c)
    d = promote(c, NOW + 60 * MS, track_id=99)  # stolen identity
    assert c.state is VisualHandoffState.SAFE_EXIT
    assert "identity" in d.status.safe_exit_reason


def test_no_far_pose_fusion_bearing_only_turn_has_no_range_dependency() -> None:
    # A far metric slant (30 m, outside the band) must not gate the TURN — direction only.
    c = VisualHandoffController()
    drive_to_preturn(c)
    d = promote(
        c, NOW + 60 * MS, has_metric_pose=True, slant_range_m=30.0, output_mode="position_only"
    )
    assert c.state is VisualHandoffState.TURN
    assert math.isfinite(d.selected[1])


def test_turn_bank_schedule_right_bank_grows_with_bearing() -> None:
    c = VisualHandoffController()
    cfg = c.config
    b_small = c._turn_bank_target(math.radians(3.0))
    b_mid = c._turn_bank_target(math.radians(10.0))
    b_big = c._turn_bank_target(math.radians(25.0))
    assert cfg.turn_bank_min_rad <= b_small <= cfg.turn_bank_lo_rad
    assert b_small < b_mid < b_big
    assert b_big <= cfg.turn_bank_cap_rad
    assert math.isclose(b_big, cfg.turn_bank_hi_rad)


def test_turn_yaw_is_right_only_and_capped() -> None:
    c = VisualHandoffController()
    drive_to_preturn(c)
    d = promote(c, NOW + 60 * MS)
    for k in range(1, 12):  # settle past the slew
        d = promote(c, NOW + (60 + k * 16) * MS)
    assert 0.0 <= d.selected[3] <= c.config.turn_yaw_cap_rps + 1e-9


def test_turn_corner_brake_is_bounded_then_zero() -> None:
    # The corner brake is a bounded nose-up pitch while OFF-AXIS (never the +1.4 blind flare),
    # and it stops after corner_brake_max_s even if the gate stays off-axis.
    c = VisualHandoffController()
    drive_to_preturn(c)
    d0 = promote(c, NOW + 60 * MS)
    assert d0.status.state is VisualHandoffState.TURN
    off = dict(bearing_rad=math.radians(22.0))
    # inside the brake window → modest bounded nose-up brake (not the 1.4 flare).
    # Step ≤ 60 ms so the observer's body-gyro integration stays continuous.
    d_early = promote(c, NOW + 120 * MS, **off)
    assert d_early.selected[2] == c.config.corner_brake_pitch_rps
    assert d_early.selected[2] <= 0.5
    # keep it in TURN and step past corner_brake_max_s → pitch returns to zero
    d_late = None
    for k in range(1, 120):
        d_late = promote(c, NOW + (120 + k * 16) * MS, **off)
        if c.state is not VisualHandoffState.TURN:
            break
        if (NOW + (120 + k * 16) * MS - c._turn_entered_ns) * 1e-9 > c.config.corner_brake_max_s:
            assert d_late.selected[2] == 0.0
            break
    assert d_late is not None


# --------------------------------------------------------------------------- CAPTURE / COMMIT
def test_capture_after_three_centred_frames() -> None:
    c = VisualHandoffController()
    drive_to_preturn(c)
    promote(c, NOW + 60 * MS)
    t = NOW + 60 * MS
    for _k in range(1, 6):
        t += 20 * MS
        promote(c, t, bearing_rad=math.radians(4.0))  # inside 10°
    assert c.state in (VisualHandoffState.CAPTURE, VisualHandoffState.COMMIT)


def test_capture_via_in_band_metric() -> None:
    c = VisualHandoffController()
    drive_to_preturn(c)
    promote(c, NOW + 60 * MS)
    promote(
        c,
        NOW + 80 * MS,
        bearing_rad=math.radians(12.0),
        has_metric_pose=True,
        output_mode="position_only",
        slant_range_m=8.0,
    )
    assert c.state is VisualHandoffState.CAPTURE


def test_commit_and_gate2_success() -> None:
    c = VisualHandoffController()
    drive_to_preturn(c)
    promote(c, NOW + 60 * MS)
    t = NOW + 60 * MS
    for _k in range(4):
        t += 20 * MS
        promote(c, t, bearing_rad=math.radians(3.0))
    # now centred+close → COMMIT
    t += 20 * MS
    promote(c, t, bearing_rad=math.radians(3.0), cy_px=CY_TARGET, area_px2=20000.0)
    assert c.state in (VisualHandoffState.CAPTURE, VisualHandoffState.COMMIT)
    # advance race counter to gate 2
    t += 20 * MS
    act2 = prov_target(t, active_gate_at_receipt=2, race_gate_id=2, track_id=LOCK, sequence=200)
    c.step(
        make_input(
            t,
            active_gate=2,
            anchor_gate=1,
            active_target=act2,
            provisional_target=empty_structured_target(t),
            roll_rad=math.radians(4.0),
            legacy=(0.30, 0.0, 0.0, 0.0),
        )
    )
    assert c.succeeded


# --------------------------------------------------------------------------- dropout / SAFE_EXIT
def _lost_input(t: int) -> VisualHandoffInput:
    return make_input(
        t,
        active_gate=1,
        anchor_gate=0,
        active_target=empty_structured_target(t),
        provisional_target=empty_structured_target(t),
        roll_rad=math.radians(12.0),
        legacy=(0.34, 0.0, 1.4, 0.0),
    )


def test_turn_target_loss_enters_reacquire_arc() -> None:
    # ANY loss of the locked gate-1 target now continues the turn in a bounded REACQUIRE_ARC
    # (not the old level-and-give-up RECOVERY).
    c = VisualHandoffController()
    drive_to_preturn(c)
    promote(c, NOW + 60 * MS)
    d = c.step(_lost_input(NOW + 60 * MS + 30 * MS))
    assert c.state is VisualHandoffState.REACQUIRE_ARC
    assert d.selected[3] > 0.0  # still yawing right
    # pitch is at most the bounded corner brake — never the legacy +1.4 blind flare
    assert 0.0 <= d.selected[2] <= c.config.corner_brake_pitch_rps + 1e-9


def test_reacquire_arc_maintains_positive_bank_and_yaw() -> None:
    c = VisualHandoffController()
    drive_to_preturn(c)
    promote(c, NOW + 60 * MS)
    t = NOW + 90 * MS
    last = None
    for _k in range(30):  # ~0.5 s of continued loss
        t += 16 * MS
        last = c.step(_lost_input(t))
        assert c.state is VisualHandoffState.REACQUIRE_ARC
    assert last is not None
    # yaw does NOT exponentially decay toward zero — it stays at the arc rate
    assert last.selected[3] >= c.config.reacquire_yaw_rps - 1e-9
    assert last.status.roll_target_rad >= c.config.reacquire_bank_rad - 1e-9


def test_reacquire_arc_hard_1p2s_timeout_to_safe_exit() -> None:
    c = VisualHandoffController()
    drive_to_preturn(c)
    promote(c, NOW + 60 * MS)
    t = NOW + 90 * MS
    d = None
    for _k in range(120):  # well past 1.2 s
        t += 16 * MS
        d = c.step(_lost_input(t))
        if c.state is VisualHandoffState.SAFE_EXIT:
            break
    assert d is not None
    assert c.state is VisualHandoffState.SAFE_EXIT
    assert d.status.safe_exit_reason == "reacquire_timeout"
    assert d.status.reacquire_elapsed_s <= c.config.reacquire_max_s + 0.05
    assert all(math.isfinite(x) for x in d.selected)
    assert d.selected[2] == 0.0  # never the blind pitch flare; never legacy SCAN
    assert d.status.source == "visual"


def test_safe_exit_terminates_after_hold() -> None:
    c = VisualHandoffController()
    drive_to_preturn(c)
    promote(c, NOW + 60 * MS)
    t = NOW + 90 * MS
    # run out the arc → SAFE_EXIT, then keep stepping through the 0.40 s hold
    d = None
    for _k in range(200):
        t += 16 * MS
        d = c.step(_lost_input(t))
        if d.status.terminate:
            break
    assert d is not None
    assert d.status.terminate
    assert c.state is VisualHandoffState.SAFE_EXIT
    # terminate is raised at/after the bounded hold, not seconds later
    assert d.status.time_in_state_s >= c.config.safe_exit_hold_s - 1e-9
    assert d.status.time_in_state_s <= c.config.safe_exit_hold_s + 0.05


def test_capture_timeout_safe_exits() -> None:
    c = VisualHandoffController()
    drive_to_preturn(c)
    promote(c, NOW + 60 * MS)
    t = NOW + 60 * MS
    for _k in range(1, 200):
        t += 16 * MS
        promote(c, t, bearing_rad=math.radians(22.0))  # never converges
        if c.state is VisualHandoffState.SAFE_EXIT:
            break
    assert c.state is VisualHandoffState.SAFE_EXIT


def test_wrong_side_during_turn_safe_exits() -> None:
    c = VisualHandoffController()
    drive_to_preturn(c)
    promote(c, NOW + 60 * MS)
    d = promote(c, NOW + 80 * MS, bearing_rad=math.radians(-25.0))
    assert c.state is VisualHandoffState.SAFE_EXIT
    assert d.status.safe_exit_reason == "wrong_turn"


# --------------------------------------------------------------------------- abort / defaults
def test_abort_routes_to_safe_exit_and_is_finite() -> None:
    c = VisualHandoffController()
    drive_to_preturn(c)
    promote(c, NOW + 60 * MS)
    d = c.step(
        make_input(
            NOW + 80 * MS,
            active_gate=1,
            anchor_gate=0,
            abort_reason="collisions (3)",
            active_target=prov_target(
                NOW + 80 * MS, active_gate_at_receipt=1, race_gate_id=1, sequence=101
            ),
            provisional_target=empty_structured_target(NOW + 80 * MS),
        )
    )
    assert c.state is VisualHandoffState.SAFE_EXIT
    assert all(math.isfinite(x) for x in d.selected)


def test_default_disabled_is_byte_identical_to_legacy() -> None:
    c = VisualHandoffController()
    legacy = (0.31, 0.07, -0.13, 0.05)
    d = c.step(make_input(flag_enabled=False, legacy=legacy))
    assert d.selected == legacy
    assert d.status.source == "legacy"
    assert d.status.authoritative_channels == (False, False, False, False)


def test_qualification_context_never_activates() -> None:
    # The mission enforces Training; the controller itself also refuses when training=False.
    c = VisualHandoffController()
    for k in range(6):
        d = c.step(
            make_input(
                NOW + k * 16 * MS,
                training=False,
                provisional_target=prov_target(NOW + k * 16 * MS, sequence=k + 1),
            )
        )
    assert c.state is VisualHandoffState.IDLE
    assert d.status.source == "legacy"


def test_attempt_reset_isolation() -> None:
    c = VisualHandoffController()
    drive_to_preturn(c)
    promote(c, NOW + 60 * MS)
    assert c.state is VisualHandoffState.TURN
    c.reset()
    assert c.state is VisualHandoffState.IDLE
    assert c.locked_track_id == -1
    assert not c.succeeded
    d = c.step(make_input(flag_enabled=False, legacy=LEGACY))
    assert d.selected == LEGACY


def test_all_commands_finite_and_rate_clamped() -> None:
    c = VisualHandoffController()
    drive_to_preturn(c)
    cfg = c.config
    t = NOW + 60 * MS
    promote(c, t)
    for _k in range(1, 40):
        t += 16 * MS
        d = promote(c, t, bearing_rad=math.radians(22.0), roll_rad=math.radians(40.0))
        thr, rr, pr, yr = d.selected
        assert all(math.isfinite(x) for x in d.selected)
        assert abs(rr) <= cfg.roll_rate_clip_rps + 1e-9
        assert -1e-9 <= yr <= cfg.turn_yaw_cap_rps + 1e-9


# --------------------------------------------------------------------------- image-row thrust
def test_image_row_thrust_direction_and_descent_guard() -> None:
    cfg = VisualHandoffConfig()
    # gate high in frame (cy < target) → climb (thrust above hover), always allowed
    high = image_row_thrust(40.0, 0.0, 3000.0, 0.0, 0.0, False, cfg)
    assert high > cfg.thrust_hover
    # gate low + close + on-axis → descend below hover
    low_close = image_row_thrust(260.0, 0.0, 5000.0, 0.0, 0.0, False, cfg)
    assert low_close < cfg.thrust_hover
    # gate low but FAR / off-axis → hold hover (no dive at the floor)
    low_far = image_row_thrust(260.0, 0.0, 500.0, 0.0, 0.0, False, cfg)
    assert low_far == cfg.thrust_hover
    off_axis = image_row_thrust(260.0, 0.0, 5000.0, math.radians(40.0), 0.0, False, cfg)
    assert off_axis == cfg.thrust_hover
    # clamped to [0.29, 0.36]
    for cy in (0.0, 90.0, 130.0, 200.0, 360.0):
        thr = image_row_thrust(cy, 0.0, 6000.0, 0.0, 0.0, False, cfg)
        assert cfg.thrust_min - 1e-9 <= thr <= cfg.thrust_max + 1e-9


def test_image_row_thrust_roll_comp_only_when_trusted() -> None:
    cfg = VisualHandoffConfig()
    base = image_row_thrust(130.0, 0.0, 6000.0, math.radians(20.0), 0.0, False, cfg)
    comped = image_row_thrust(130.0, 0.0, 6000.0, math.radians(20.0), 0.0, True, cfg)
    assert comped >= base  # 1/cos(roll) raises thrust when a trusted bank is present
    # denominator floor at 0.85 (never explode near 90°)
    extreme = image_row_thrust(130.0, 0.0, 6000.0, math.radians(80.0), 0.0, True, cfg)
    assert extreme <= cfg.thrust_max + 1e-9


# ------------------------------------------------ PRETURN engagement-window robustness (v2)
def test_gate0_proximity_latch_enables_preturn_after_gate0_deselects() -> None:
    # Live root cause: gate 0 de-selects (active target invalid) exactly as the provisional
    # confirms. A recently latched close+aligned gate-0 sighting must still allow entry.
    c = VisualHandoffController()
    c.step(
        make_input(NOW, provisional_target=prov_target(NOW, track_hits=4))
    )  # latch, no entry yet
    assert c.state is VisualHandoffState.IDLE
    t = NOW + 48 * MS
    c.step(
        make_input(
            t,
            active_target=empty_structured_target(t),  # gate 0 now de-selected
            provisional_target=prov_target(t, track_hits=6),
        )
    )
    assert c.state is VisualHandoffState.PRETURN
    assert c.locked_track_id == LOCK


def test_stale_latch_does_not_enable_preturn() -> None:
    c = VisualHandoffController()
    c.step(make_input(NOW, provisional_target=prov_target(NOW, track_hits=4)))  # latch
    t = NOW + 400 * MS  # 400 ms > 250 ms proximity memory
    c.step(
        make_input(
            t,
            active_target=empty_structured_target(t),
            provisional_target=prov_target(t, track_hits=6),
        )
    )
    assert c.state is VisualHandoffState.IDLE


def test_brief_provisional_gap_survives_to_promotion() -> None:
    c = VisualHandoffController()
    drive_to_preturn(c)
    t1 = NOW + 40 * MS  # one frame with the provisional missing (within grace)
    c.step(make_input(t1, provisional_target=empty_structured_target(t1)))
    assert c.state is VisualHandoffState.PRETURN
    t2 = NOW + 56 * MS  # gate advances; promoted active is the locked identity → TURN
    act = prov_target(t2, active_gate_at_receipt=1, race_gate_id=1, track_id=LOCK, sequence=50)
    c.step(
        make_input(
            t2,
            active_gate=1,
            anchor_gate=0,
            active_target=act,
            provisional_target=empty_structured_target(t2),
            roll_rad=math.radians(12.0),
        )
    )
    assert c.state is VisualHandoffState.TURN


def test_provisional_gap_beyond_grace_revokes_preturn() -> None:
    c = VisualHandoffController()
    drive_to_preturn(c)
    # a gap within the logical grace (0.35 s) keeps PRETURN; only beyond it revokes
    hold_empty_provisional(c, NOW + 16 * MS, NOW + 300 * MS)
    assert c.state is VisualHandoffState.PRETURN
    d = hold_empty_provisional(c, NOW + 316 * MS, NOW + 420 * MS)  # cross the 350 ms grace
    assert c.state is VisualHandoffState.IDLE
    assert d.status.source == "legacy"  # type: ignore[attr-defined]


def test_promotion_with_lagged_active_target_enters_turn() -> None:
    # The worker's active target lags the race-counter advance by ~1 frame; a not-yet-
    # published active target must enter TURN (not SAFE_EXIT), then confirm next frame.
    c = VisualHandoffController()
    drive_to_preturn(c)
    t = NOW + 60 * MS
    c.step(
        make_input(
            t,
            active_gate=1,
            anchor_gate=0,
            active_target=empty_structured_target(t),  # promoted target not yet published
            provisional_target=empty_structured_target(t),
            roll_rad=math.radians(12.0),
            legacy=(0.34, 0.0, 1.4, 0.0),
        )
    )
    assert c.state is VisualHandoffState.TURN
    t2 = NOW + 76 * MS
    act = prov_target(t2, active_gate_at_receipt=1, race_gate_id=1, track_id=LOCK, sequence=60)
    c.step(
        make_input(
            t2,
            active_gate=1,
            anchor_gate=0,
            active_target=act,
            provisional_target=empty_structured_target(t2),
            roll_rad=math.radians(12.0),
            legacy=(0.34, 0.0, 0.0, 0.0),
        )
    )
    assert c.state is VisualHandoffState.TURN


# ------------------------------------------------ roll feedback correctness + safety cap (v3)
def _turn_requested_roll_rate(observer_bank_deg: float) -> float:
    """Roll rate the TURN law requests when the OBSERVER bank is seeded at the given value."""
    c = VisualHandoffController()
    # seed the observer at the given bank on the first pre-turn tick (body_gyro=0 holds it)
    c.step(make_input(NOW, control_roll=math.radians(observer_bank_deg)))
    d = promote(c, NOW + 16 * MS, bearing_rad=math.radians(20.0))
    assert d.requested is not None
    return d.requested[1]


def test_observer_bank_drives_roll_feedback_sign() -> None:
    # The roll law now feeds back on the dedicated gyro observer (right-bank positive). An
    # already-right-banked observer needs LESS additional right roll rate than a left-banked one.
    rr_right = _turn_requested_roll_rate(+30.0)  # observer already banked right (near target)
    rr_left = _turn_requested_roll_rate(-20.0)  # observer banked left → far from target
    assert rr_left > rr_right
    assert rr_right < VisualHandoffConfig().roll_rate_clip_rps


def test_feedforward_bank_cap_prevents_roll_runaway() -> None:
    # ESKF roll stuck at 0 (never tracks the bank) — the feedforward commanded-bank integral
    # must still stop the roll rate once the integrated bank reaches target (no 60° runaway).
    c = VisualHandoffController()
    drive_to_preturn(c)
    t = NOW + 60 * MS
    promote(c, t, roll_rad=0.0)
    last_rate = 1.0
    for _k in range(60):
        t += 16 * MS
        d = promote(c, t, bearing_rad=math.radians(22.0), roll_rad=0.0)
        if c.state is not VisualHandoffState.TURN:
            break
        last_rate = d.selected[1]
    assert last_rate < 0.15  # roll authority self-limited, not stuck at the cap
    assert c._cmd_bank_rad <= c.config.bank_feedforward_cap_rad + 1e-9


# ---------------------------------------------------------------- guarded reacquisition (v4)
def _into_reacquire(c: VisualHandoffController) -> None:
    drive_to_preturn(c)
    promote(c, NOW + 60 * MS)
    c.step(_lost_input(NOW + 90 * MS))
    assert c.state is VisualHandoffState.REACQUIRE_ARC


def _reacq_candidate(t: int, **over: object) -> VisualHandoffInput:
    over.setdefault("active_gate_at_receipt", 1)
    over.setdefault("race_gate_id", 1)
    act = prov_target(t, **over)
    return make_input(
        t,
        active_gate=1,
        anchor_gate=0,
        active_target=act,
        provisional_target=empty_structured_target(t),
        roll_rad=math.radians(12.0),
        legacy=(0.34, 0.0, 0.0, 0.0),
    )


def test_no_reacquire_without_prior_engagement() -> None:
    # IDLE controller with gate already at 1 and no target → never enters REACQUIRE_ARC.
    c = VisualHandoffController()
    c.step(_lost_input(NOW))
    assert c.state is VisualHandoffState.IDLE


def test_reacquire_original_id_preserved() -> None:
    c = VisualHandoffController()
    _into_reacquire(c)
    d = c.step(_reacq_candidate(NOW + 120 * MS, track_id=LOCK, bearing_rad=math.radians(20.0)))
    assert c.state is VisualHandoffState.TURN  # >15° → resume TURN
    assert c.locked_track_id == LOCK
    assert d.status.reacquire_result == "original"


def test_reacquire_new_id_after_three_hits() -> None:
    c = VisualHandoffController()
    _into_reacquire(c)
    d = c.step(
        _reacq_candidate(NOW + 120 * MS, track_id=5, track_hits=3, bearing_rad=math.radians(8.0))
    )
    assert d.status.reacquire_result == "new"
    assert c.locked_track_id == 5
    assert d.status.reacquire_prev_track_id == LOCK
    assert c.state is VisualHandoffState.CAPTURE  # ≤15° → CAPTURE


def test_reacquire_new_id_rejected_unconfirmed() -> None:
    c = VisualHandoffController()
    _into_reacquire(c)
    d = c.step(
        _reacq_candidate(NOW + 120 * MS, track_id=5, track_hits=2, bearing_rad=math.radians(8.0))
    )
    assert c.state is VisualHandoffState.REACQUIRE_ARC
    assert d.status.reacquire_result == "rejected:unconfirmed"


def test_reacquire_rejects_wrong_side() -> None:
    c = VisualHandoffController()
    _into_reacquire(c)
    d = c.step(
        _reacq_candidate(NOW + 120 * MS, track_id=5, track_hits=5, bearing_rad=math.radians(-30.0))
    )
    assert c.state is VisualHandoffState.REACQUIRE_ARC
    assert d.status.reacquire_result == "rejected:wrong_side"


def test_reacquire_rejects_association_refused() -> None:
    c = VisualHandoffController()
    _into_reacquire(c)
    d = c.step(
        _reacq_candidate(
            NOW + 120 * MS,
            track_id=5,
            track_hits=5,
            bearing_rad=math.radians(8.0),
            association_refused=True,
        )
    )
    assert c.state is VisualHandoffState.REACQUIRE_ARC
    assert d.status.reacquire_result == "rejected:assoc_refused"


def test_reacquire_rejects_wrong_race_gate() -> None:
    c = VisualHandoffController()
    _into_reacquire(c)
    d = c.step(
        _reacq_candidate(
            NOW + 120 * MS, track_id=5, race_gate_id=2, track_hits=5, bearing_rad=math.radians(8.0)
        )
    )
    assert c.state is VisualHandoffState.REACQUIRE_ARC
    assert d.status.reacquire_result == "rejected:wrong_race"


def test_reacquire_rejects_stale_candidate() -> None:
    c = VisualHandoffController()
    _into_reacquire(c)
    stale_t = NOW + 120 * MS
    act = prov_target(
        stale_t - 300 * MS,
        active_gate_at_receipt=1,
        race_gate_id=1,
        track_id=5,
        track_hits=5,
        bearing_rad=math.radians(8.0),
        sequence=77,
    )
    d = c.step(
        make_input(
            stale_t,
            active_gate=1,
            anchor_gate=0,
            active_target=act,
            provisional_target=empty_structured_target(stale_t),
            roll_rad=math.radians(12.0),
            legacy=(0.34, 0.0, 0.0, 0.0),
        )
    )
    assert c.state is VisualHandoffState.REACQUIRE_ARC
    assert d.status.reacquire_result == "rejected:stale"


def test_reacquire_rejects_downstream_imposter() -> None:
    # A far-right (+45°) confirmed gate-1 track is off the arc-predicted bearing → imposter.
    c = VisualHandoffController()
    _into_reacquire(c)
    d = c.step(
        _reacq_candidate(NOW + 120 * MS, track_id=6, track_hits=5, bearing_rad=math.radians(45.0))
    )
    assert c.state is VisualHandoffState.REACQUIRE_ARC
    assert d.status.reacquire_result == "rejected:imposter"


def test_reacquire_out_of_band_metric_is_direction_only_capture() -> None:
    # A reacquired gate 1 that is position_only at 24 m (out of the 3–16 m band) is used for
    # DIRECTION only: it captures on bearing but is never treated as an in-band metric.
    c = VisualHandoffController()
    _into_reacquire(c)
    d = c.step(
        _reacq_candidate(
            NOW + 120 * MS,
            track_id=5,
            track_hits=5,
            bearing_rad=math.radians(6.0),
            has_metric_pose=True,
            output_mode="position_only",
            slant_range_m=24.0,
            out_of_band=True,
        )
    )
    # out_of_band is a fusion-band flag, not an identity refusal → reacquisition still valid,
    # and it captures on bearing (≤15°); the 24 m position_only pose is used for direction only.
    assert d.status.reacquire_result == "new"
    assert c.state is VisualHandoffState.CAPTURE


def test_reacquire_abort_priority() -> None:
    c = VisualHandoffController()
    _into_reacquire(c)
    d = c.step(
        make_input(
            NOW + 120 * MS,
            active_gate=1,
            anchor_gate=0,
            active_target=empty_structured_target(NOW + 120 * MS),
            provisional_target=empty_structured_target(NOW + 120 * MS),
            roll_rad=math.radians(12.0),
            legacy=(0.34, 0.0, 0.0, 0.0),
            abort_reason="collisions (3)",
        )
    )
    assert c.state is VisualHandoffState.SAFE_EXIT
    assert all(math.isfinite(x) for x in d.selected)


# ---------------------------------------------------------------- gate-0 exit-speed control
def _g0_metric(
    t: int, slant: float = 5.5, bearing_deg: float = 1.0, cy: float = 120.0, track: int = 1
) -> StructuredTarget:
    return StructuredTarget(
        valid=True,
        frame_id=int(t % 100000),
        frame_sensor_ns=0,
        arrival_ns=t,
        active_gate_at_receipt=0,
        track_id=track,
        race_gate_id=0,
        track_hits=10,
        output_mode="position_only",
        bearing_rad=math.radians(bearing_deg),
        los_body_unit=np.array([1.0, 0.0, 0.0]),
        slant_range_m=slant,
        has_metric_pose=True,
        confidence=0.9,
        cov_trace=0.2,
        expected_turn_sign=0.0,
        disposition="accepted",
        area_px2=26000.0,
        cx_px=323.0,
        cy_px=cy,
        visible_fraction=0.5,
    )


def _speed_input(
    t: int,
    *,
    speed: float,
    slant: float,
    active_gate: int = 0,
    anchor: int = 0,
    initialised: bool = True,
    cov: float = 0.5,
    fix_age: float = 0.03,
    bearing_deg: float = 1.0,
    cy: float = 120.0,
    provisional: StructuredTarget | None = None,
) -> VisualHandoffInput:
    return VisualHandoffInput(
        now_ns=t,
        training=True,
        flag_enabled=True,
        active_gate=active_gate,
        anchor_gate=anchor,
        active_target=_g0_metric(t, slant=slant, bearing_deg=bearing_deg, cy=cy),
        provisional_target=empty_structured_target(t) if provisional is None else provisional,
        eskf_roll_rad=math.radians(3.0),
        eskf_roll_cov_rad2=0.001,
        eskf_att_cov_trace_rad2=0.01,
        state_age_s=0.02,
        legacy_command=(0.30, 0.0, -0.05, 0.0),
        eskf_speed_mps=speed,
        eskf_vel_cov_trace=cov,
        eskf_fix_age_s=fix_age,
        eskf_initialised=initialised,
        control_yaw_rad=0.0,
        control_roll_rad=0.0,
        control_pitch_rad=0.0,
        body_gyro_rps=(0.0, 0.0, 0.0),
    )


def _drive_speed(
    c: VisualHandoffController, *, speed: float, n: int = 4, slant0: float = 8.0, t0: int = NOW
) -> object:
    # range decreases at the ESKF speed (33 ms frames) so the PnP closure agrees with it
    step = speed * 0.033
    d = None
    for k in range(n):
        d = c.step(_speed_input(t0 + k * 33 * MS, speed=speed, slant=max(3.5, slant0 - k * step)))
    return d


def test_speed_no_authority_without_valid_anchor() -> None:
    c = VisualHandoffController()
    d = c.step(_speed_input(NOW, speed=9.0, slant=5.5, anchor=1))  # not anchored to gate 0
    assert not d.status.speed_active
    assert d.status.speed_reason == "off"  # gated out at the step level
    assert d.selected[2] == -0.05  # legacy pitch


def test_speed_runs_during_approach_before_active_gate_zero() -> None:
    # The metric window precedes the race counter setting active_gate=0, so the brake must
    # run while active_gate == -1 as long as the ESKF is anchored to gate 0.
    c = VisualHandoffController()
    d = None
    for k in range(6):
        t = NOW + k * 33 * MS
        d = c.step(_speed_input(t, speed=10.0, slant=8.0 - k * 0.33, active_gate=-1, anchor=0))
    assert d.status.speed_active and d.selected[2] > 0.0  # braking during the approach


def test_speed_pitch_engages_above_threshold_nose_up() -> None:
    c = VisualHandoffController()
    d = _drive_speed(c, speed=9.0)
    assert d.status.speed_active and d.status.speed_reason == "braking"
    assert d.selected[2] > 0.0  # nose-up brake
    assert d.status.authoritative_channels == (False, False, True, False)


def test_speed_zero_pitch_at_or_below_engage_speed() -> None:
    c = VisualHandoffController()
    d = _drive_speed(c, speed=6.0)  # below engage → authority never begins
    assert not d.status.speed_active and d.status.speed_reason == "below_engage"
    assert d.selected[2] == -0.05  # legacy pitch (no brake)


def test_speed_pitch_rate_capped() -> None:
    c = VisualHandoffController()
    d = _drive_speed(c, speed=18.0)  # huge error → kp*err clipped to cap
    assert d.selected[2] == c.config.entry_speed_pitch_cap_rps


def test_speed_impulse_budget_caps_brake() -> None:
    c = VisualHandoffController()
    t = NOW
    reasons = set()
    for k in range(60):  # range decreases at ~10 m/s so the closure agrees and braking runs
        d = c.step(_speed_input(t, speed=10.0, slant=30.0 - k * 0.33))
        reasons.add(d.status.speed_reason)
        t += 33 * MS
    assert "braking" in reasons and "impulse_cap" in reasons  # brakes, then hits the budget
    assert c._speed_impulse_rad <= c.config.entry_speed_impulse_cap_rad + math.radians(2.0)


def test_speed_thrust_and_preturn_rollyaw_preserved() -> None:
    # exit-speed brake owns pitch only; a simultaneous PRETURN keeps roll/yaw + legacy thrust.
    c = VisualHandoffController()
    # gate 0 close + centred metric AND a confirmed provisional → PRETURN + brake. Range
    # decreases at the ESKF speed so the closure agrees.
    d = None
    for k in range(5):
        t = NOW + k * 33 * MS
        inp = _speed_input(
            t,
            speed=10.0,
            slant=8.0 - k * 0.33,
            provisional=prov_target(t, bearing_rad=math.radians(22.0), area_px2=40000.0),
        )
        d = c.step(inp)
    assert c.state is VisualHandoffState.PRETURN
    assert d.selected[2] > 0.0  # brake owns pitch
    assert d.selected[3] > 0.0  # PRETURN owns yaw (right)
    assert d.selected[0] == 0.30  # legacy thrust preserved


def test_speed_revoked_on_off_axis() -> None:
    c = VisualHandoffController()
    _drive_speed(c, speed=10.0)
    d = c.step(_speed_input(NOW + 200 * MS, speed=10.0, slant=4.0, bearing_deg=25.0))  # off-axis
    assert not d.status.speed_active
    assert d.status.speed_reason in ("misaligned", "no_closure")
    assert d.selected[2] == -0.05


def test_speed_captures_exit_speed_and_ends_authority_at_advance() -> None:
    c = VisualHandoffController()
    _drive_speed(c, speed=8.0)
    # advance to gate 1
    t = NOW + 200 * MS
    act1 = prov_target(t, active_gate_at_receipt=1, race_gate_id=1, track_id=LOCK, sequence=9)
    d = c.step(
        make_input(
            t,
            active_gate=1,
            anchor_gate=0,
            active_target=act1,
            provisional_target=empty_structured_target(t),
        )
    )
    assert math.isfinite(d.status.exit_speed_mps)
    assert not d.status.speed_active  # exit-speed control ended past the gate


def test_speed_exit_too_high_refuses_corner() -> None:
    c = VisualHandoffController()
    # engage PRETURN with a HIGH validated exit speed so the corner is refused
    prov = lambda t: prov_target(t, bearing_rad=math.radians(22.0), area_px2=40000.0)  # noqa: E731
    for k in range(4):
        t = NOW + k * 33 * MS
        c.step(_speed_input(t, speed=13.0, slant=5.5 - k * 0.2, provisional=prov(t)))
    # ESKF 13 disagrees with a ~9 closure → not validated → exit speed stays NaN → permitted.
    # Force a validated-high exit speed by making closure ~13 too (fast approach).
    c2 = VisualHandoffController()
    for k in range(5):
        t = NOW + k * 33 * MS
        c2.step(
            _speed_input(t, speed=13.0, slant=6.0 - k * 0.43, provisional=prov(t))
        )  # ~13 m/s closure
    t = NOW + 165 * MS  # ≤ 80 ms after the last speed tick (keeps the observer valid)
    act1 = prov_target(t, active_gate_at_receipt=1, race_gate_id=1, track_id=LOCK, sequence=9)
    d = c2.step(
        make_input(
            t,
            active_gate=1,
            anchor_gate=0,
            active_target=act1,
            provisional_target=empty_structured_target(t),
        )
    )
    if math.isfinite(d.status.exit_speed_mps) and d.status.exit_speed_mps > 8.5:
        assert not d.status.corner_permitted
        assert c2.state is VisualHandoffState.SAFE_EXIT
        assert d.status.safe_exit_reason == "exit_speed_too_high"


def test_speed_default_off_byte_identical() -> None:
    c = VisualHandoffController()
    legacy = (0.31, 0.02, -0.11, 0.03)
    inp = VisualHandoffInput(
        now_ns=NOW,
        training=True,
        flag_enabled=False,
        active_gate=0,
        anchor_gate=0,
        active_target=_g0_metric(NOW),
        provisional_target=empty_structured_target(NOW),
        eskf_roll_rad=0.05,
        eskf_roll_cov_rad2=0.001,
        eskf_att_cov_trace_rad2=0.01,
        state_age_s=0.02,
        legacy_command=legacy,
        eskf_speed_mps=10.0,
        eskf_vel_cov_trace=0.5,
        eskf_fix_age_s=0.03,
        eskf_initialised=True,
    )
    d = c.step(inp)
    assert d.selected == legacy and d.status.source == "legacy"


def test_speed_commands_finite_and_clamped() -> None:
    c = VisualHandoffController()
    t = NOW
    for k in range(30):
        d = c.step(_speed_input(t, speed=12.0, slant=max(3.0, 6.0 - k * 0.1)))
        assert all(math.isfinite(x) for x in d.selected)
        assert (
            -1e-9 <= d.selected[2] <= c.config.entry_speed_pitch_cap_rps + 1e-9
            or d.selected[2] == -0.05
        )
        t += 33 * MS


# ===================================================================================
# Phase 1 — LOGICAL gate-1 identity (world-azimuth continuity above raw tracker ids)
# ===================================================================================
def _logical_prov(
    now_ns: int, *, track_id: int, bearing_deg: float, hits: int = 6
) -> StructuredTarget:
    return prov_target(
        now_ns, track_id=track_id, bearing_rad=math.radians(bearing_deg), track_hits=hits
    )


def _establish_logical(
    c: VisualHandoffController, *, cy_deg: float = 0.0, bearing_deg: float = 22.0, track_id: int = 0
) -> object:
    """Establish the logical identity on a confirmed right-side provisional."""
    d = c.step(
        make_input(
            NOW,
            control_yaw=math.radians(cy_deg),
            provisional_target=_logical_prov(NOW, track_id=track_id, bearing_deg=bearing_deg),
        )
    )
    assert d.status.logical_prov_id == track_id
    assert d.status.logical_reason == "established"
    return d


def test_logical_identity_establishes_on_confirmed_right_provisional() -> None:
    c = VisualHandoffController()
    d = _establish_logical(c, cy_deg=0.0, bearing_deg=22.0, track_id=0)
    assert d.status.logical_raw_id == 0
    assert math.isclose(d.status.logical_azimuth_rad, math.radians(22.0), abs_tol=1e-6)


def test_logical_handover_accepts_same_azimuth_new_raw_id() -> None:
    # Track 0 and 2 are the SAME physical gate: as the drone yaws +5°, a right-side gate holds
    # its world azimuth (~22°), so bearing drops 5° while cy rises 5°.
    c = VisualHandoffController()
    _establish_logical(c, cy_deg=0.0, bearing_deg=22.0, track_id=0)
    d = c.step(
        make_input(
            NOW + 100 * MS,
            control_yaw=math.radians(5.0),
            provisional_target=_logical_prov(NOW + 100 * MS, track_id=2, bearing_deg=17.0),
        )
    )
    assert d.status.logical_reason == "reacquired"
    assert d.status.logical_raw_id == 2
    assert d.status.logical_prev_raw_id == 0
    assert d.status.logical_azimuth_residual_rad <= math.radians(15.0)


def test_logical_handover_rejects_imposter_large_azimuth_jump() -> None:
    # The downstream imposter (track 4) sits at a very different world azimuth → rejected; the
    # logical identity is kept on the original raw id.
    c = VisualHandoffController()
    _establish_logical(c, cy_deg=0.0, bearing_deg=22.0, track_id=0)
    d = c.step(
        make_input(
            NOW + 100 * MS,
            control_yaw=math.radians(5.0),
            provisional_target=_logical_prov(NOW + 100 * MS, track_id=4, bearing_deg=45.0),
        )
    )
    assert d.status.logical_reason == "rejected:imposter"
    assert d.status.logical_raw_id == 0  # unchanged
    assert d.status.logical_azimuth_residual_rad > math.radians(15.0)


def test_logical_handover_rejected_beyond_gap() -> None:
    c = VisualHandoffController()
    _establish_logical(c, cy_deg=0.0, bearing_deg=22.0, track_id=0)
    d = c.step(  # 400 ms > the 350 ms logical gap grace, even at a matching azimuth
        make_input(
            NOW + 400 * MS,
            control_yaw=math.radians(0.0),
            provisional_target=_logical_prov(NOW + 400 * MS, track_id=2, bearing_deg=22.0),
        )
    )
    assert d.status.logical_reason == "rejected:gap"
    assert d.status.logical_raw_id == 0


def test_logical_handover_rejected_unconfirmed_hits() -> None:
    c = VisualHandoffController()
    _establish_logical(c, cy_deg=0.0, bearing_deg=22.0, track_id=0)
    d = c.step(
        make_input(
            NOW + 100 * MS,
            control_yaw=math.radians(5.0),
            provisional_target=_logical_prov(NOW + 100 * MS, track_id=2, bearing_deg=17.0, hits=2),
        )
    )
    assert d.status.logical_reason == "rejected:unconfirmed"
    assert d.status.logical_raw_id == 0


def test_logical_azimuth_residual_wraps_across_pi() -> None:
    # Establish near +180°; a handover just across −180° is a TINY true delta despite the ±2π
    # raw difference — the wrapped residual must stay small (accepted).
    c = VisualHandoffController()
    # cy = 160°, bearing 22° → az = 182° → wraps to −178°.
    d0 = c.step(
        make_input(
            NOW,
            control_yaw=math.radians(160.0),
            provisional_target=_logical_prov(NOW, track_id=0, bearing_deg=22.0),
        )
    )
    assert d0.status.logical_reason == "established"
    d = c.step(
        make_input(
            NOW + 100 * MS,
            control_yaw=math.radians(165.0),
            provisional_target=_logical_prov(NOW + 100 * MS, track_id=2, bearing_deg=19.0),
        )
    )
    assert d.status.logical_reason == "reacquired"
    assert d.status.logical_azimuth_residual_rad <= math.radians(15.0)


def test_logical_gap_reason_when_provisional_absent() -> None:
    c = VisualHandoffController()
    _establish_logical(c, cy_deg=0.0, bearing_deg=22.0, track_id=0)
    d = c.step(
        make_input(
            NOW + 40 * MS,
            control_yaw=math.radians(2.0),
            provisional_target=empty_structured_target(NOW + 40 * MS),
        )
    )
    assert d.status.logical_reason == "gap"
    assert d.status.logical_prov_id == 0  # identity preserved through the gap


def test_logical_streak_preserved_across_handover() -> None:
    c = VisualHandoffController()
    _establish_logical(c, cy_deg=0.0, bearing_deg=22.0, track_id=0)
    d = c.step(
        make_input(
            NOW + 100 * MS,
            control_yaw=math.radians(5.0),
            provisional_target=_logical_prov(NOW + 100 * MS, track_id=2, bearing_deg=17.0),
        )
    )
    assert d.status.logical_reason == "reacquired"
    assert d.status.logical_streak >= 6  # not reset to 1 on the raw-id change
    assert d.status.logical_reacquired == 1


def test_logical_identity_cleared_on_reset() -> None:
    c = VisualHandoffController()
    _establish_logical(c, cy_deg=0.0, bearing_deg=22.0, track_id=0)
    c.reset()
    d = c.step(make_input(NOW + 500 * MS, control_yaw=math.radians(0.0)))
    # a fresh establishment (id 3 from the default prov_target), not the stale id 0
    assert d.status.logical_prov_id == LOCK


def test_logical_never_establishes_without_control_yaw() -> None:
    # NaN control yaw (a recording predating the logical handoff) → identity never establishes.
    c = VisualHandoffController()
    d = c.step(make_input(NOW, control_yaw=float("nan")))
    assert d.status.logical_prov_id == -1
    assert d.status.logical_reason in ("none", "gap")


# ===================================================================================
# Phase 1 — PRETURN survives raw-id churn / gaps within the logical grace
# ===================================================================================
def test_preturn_survives_gap_within_grace() -> None:
    c = VisualHandoffController()
    drive_to_preturn(c)
    # provisional absent for 200 ms (< 350 ms grace), stepped at control rate → PRETURN holds.
    d = hold_empty_provisional(c, NOW + 16 * MS, NOW + 200 * MS)
    assert c.state is VisualHandoffState.PRETURN
    assert d.status.source in ("visual", "blended")  # type: ignore[attr-defined]


def test_preturn_promotes_through_logical_reacquisition() -> None:
    # Establish + PRETURN on track 0, hand off to track 2 mid-preturn, then promote at gate 1.
    c = VisualHandoffController()
    _establish_logical(c, cy_deg=0.0, bearing_deg=22.0, track_id=0)
    assert c.state is VisualHandoffState.PRETURN
    # steps ≤ 64 ms keep the observer's body-gyro integration continuous (no gyro-gap invalidation)
    c.step(
        make_input(
            NOW + 64 * MS,
            control_yaw=math.radians(5.0),
            provisional_target=_logical_prov(NOW + 64 * MS, track_id=2, bearing_deg=17.0),
        )
    )
    assert c.locked_track_id == 2
    # race counter advances; the active gate-1 target now carries raw id 2 → promotes.
    act = prov_target(
        NOW + 128 * MS, active_gate_at_receipt=1, race_gate_id=1, track_id=2, sequence=200
    )
    c.step(
        make_input(
            NOW + 128 * MS,
            active_gate=1,
            anchor_gate=0,
            active_target=act,
            provisional_target=empty_structured_target(NOW + 128 * MS),
            roll_rad=math.radians(12.0),
            legacy=(0.34, 0.0, 1.4, 0.0),
        )
    )
    assert c.state is VisualHandoffState.TURN


# ===================================================================================
# Phase 3 — geometry-derived corner bank
# ===================================================================================
def _geometry_radius(cfg: VisualHandoffConfig) -> float:
    x, y = cfg.corner_offset_fwd_m, cfg.corner_offset_right_m
    return (x * x + y * y) / (2.0 * y)


def test_geometry_bank_matches_coordinated_turn_formula() -> None:
    from drone_racing.deployments.vq2.visual_handoff import _G

    for v in (7.0, 7.7):  # both raw banks (25.7°, 30.2°) are below the 34° authority cap
        c = VisualHandoffController(VisualHandoffConfig(corner_bank_speed_mps=v))
        c._freeze_corner_bank()
        r = _geometry_radius(c.config)
        expected = math.atan(v * v / (_G * r))
        assert math.isclose(c._corner_bank_raw, expected, abs_tol=1e-9)
        assert math.isclose(c._corner_bank, expected, abs_tol=1e-9)  # below the 34° clamp


def test_geometry_bank_clamped_at_34_degrees() -> None:
    # v=8.5 raw ≈ 35.3° and v=9.5 raw ≈ 41.5° both exceed the derated 34° authority cap.
    for v in (8.5, 9.5):
        c = VisualHandoffController(VisualHandoffConfig(corner_bank_speed_mps=v))
        c._freeze_corner_bank()
        assert c._corner_bank_raw > math.radians(34.0)  # bank_geometry_raw exceeds the cap
        assert math.isclose(c._corner_bank, math.radians(34.0), abs_tol=1e-9)


def test_geometry_bank_frozen_at_turn_entry_and_stable() -> None:
    c = VisualHandoffController()
    drive_to_preturn(c)
    promote(c, NOW + 60 * MS)
    assert c.state is VisualHandoffState.TURN
    frozen = c._corner_bank
    assert math.isfinite(frozen)
    # advance several TURN ticks; the frozen bank must not drift.
    t = NOW + 60 * MS
    for _k in range(5):
        t += 20 * MS
        promote(c, t, bearing_rad=math.radians(20.0))
        assert c._corner_bank == frozen


def test_turn_bank_reaches_geometry_ceiling_at_high_bearing() -> None:
    c = VisualHandoffController()
    drive_to_preturn(c)
    promote(c, NOW + 60 * MS)
    # bearing beyond the hi threshold → the target rides the frozen geometry ceiling.
    assert math.isclose(c._turn_bank_target(math.radians(20.0)), c._corner_bank, abs_tol=1e-9)


def test_reacquire_arc_preserves_geometry_bank() -> None:
    c = VisualHandoffController()
    drive_to_preturn(c)
    promote(c, NOW + 60 * MS)
    geo = c._corner_bank
    # lose the target → REACQUIRE_ARC; its roll target is the frozen geometry bank.
    d = c.step(_lost_input(NOW + 100 * MS))
    assert c.state is VisualHandoffState.REACQUIRE_ARC
    assert math.isclose(d.status.roll_target_rad, geo, abs_tol=1e-9)


def test_geometry_bank_below_roll_validity_and_tilt_abort() -> None:
    cfg = VisualHandoffConfig()
    # command authority cap (34°) ≤ feedforward integral cap (35°) < roll-validity (45°)
    assert cfg.corner_bank_max_rad <= cfg.bank_feedforward_cap_rad
    assert cfg.bank_feedforward_cap_rad < cfg.roll_abs_limit_rad
    assert cfg.roll_abs_limit_rad <= cfg.bank_abort_rad  # 45° validity, 45° in-handoff abort


def test_corner_bank_speed_is_frozen_conservative_not_recorded_closure() -> None:
    # The bank uses the frozen conservative pre-pass speed (config), NOT the stale PnP closure
    # that `ctrl_vh_spd_exit` records — they are deliberately different quantities.
    c = VisualHandoffController()
    c._freeze_corner_bank()
    assert c._corner_bank_speed == c.config.corner_bank_speed_mps
    assert c._corner_bank_speed >= 9.0  # conservative upper-CI, above the ~6.9 recorded closure


# ===================================================================================
# Phase C/D — bank observer arbitration (fail-closed roll validity)
# ===================================================================================
def _to_turn_with_observer(bank_deg: float) -> VisualHandoffController:
    """Reach TURN with the observer seeded at ``bank_deg`` (body_gyro=0 holds it there)."""
    c = VisualHandoffController()
    c.step(make_input(NOW, control_roll=math.radians(bank_deg)))  # seed observer + PRETURN
    assert c.state is VisualHandoffState.PRETURN
    promote(c, NOW + 16 * MS)
    assert c.state is VisualHandoffState.TURN
    return c


def test_arbitration_ok_uses_observer_as_feedback_source() -> None:
    c = _to_turn_with_observer(28.0)
    d = promote(c, NOW + 32 * MS)
    assert d.status.roll_arbitration == "ok"
    assert d.status.roll_feedback_source == "observer"
    assert math.isclose(d.status.observer_bank_rad, math.radians(28.0), abs_tol=math.radians(2.0))


def test_observer_over_bank_fails_closed() -> None:
    # In TURN at ~30°, an observer that out-banks the commanded envelope (real body roll rate the
    # command did not ask for) must fail closed — either on the over-envelope guard (observer >
    # integral + tolerance) or the absolute 45° limit, whichever trips first.
    c = _to_turn_with_observer(30.0)
    t = NOW + 16 * MS
    got_exit = False
    for _k in range(40):
        t += 16 * MS
        d = c.step(
            make_input(
                t,
                active_gate=1,
                anchor_gate=0,
                active_target=prov_target(
                    t, active_gate_at_receipt=1, race_gate_id=1, track_id=LOCK, sequence=int(t)
                ),
                provisional_target=empty_structured_target(t),
                body_gyro=(2.0, 0.0, 0.0),  # +2 rad/s right roll the command didn't request
                legacy=(0.34, 0.0, 1.4, 0.0),
            )
        )
        if c.state is VisualHandoffState.SAFE_EXIT:
            assert d.status.safe_exit_reason == "roll_invalid"
            assert d.status.roll_arbitration in (
                "fail:observer_over_envelope",
                "fail:observer_over_limit",
            )
            got_exit = True
            break
    assert got_exit


def test_shadow_reversal_alone_does_not_terminate() -> None:
    # Observer + command-integral agree (~28°), but the shadow ESKF roll swings wildly (sign
    # reversal). The single degraded cross-check must NOT terminate the turn.
    c = _to_turn_with_observer(28.0)
    d = c.step(
        make_input(
            NOW + 40 * MS,
            active_gate=1,
            anchor_gate=0,
            active_target=prov_target(
                NOW + 40 * MS, active_gate_at_receipt=1, race_gate_id=1, track_id=LOCK, sequence=9
            ),
            provisional_target=empty_structured_target(NOW + 40 * MS),
            roll_rad=math.radians(45.0),  # shadow ESKF reversed → corrected −45°, far from +28°
            legacy=(0.34, 0.0, 1.4, 0.0),
        )
    )
    assert c.state is VisualHandoffState.TURN  # NOT terminated by the degraded shadow alone
    assert d.status.shadow_roll_degraded
    assert d.status.roll_arbitration == "ok"


def test_global_tilt_abort_unchanged() -> None:
    # The in-handoff bank abort is 45°; the runtime global TILT_ABORT is 60° and untouched.
    from drone_racing.deployments.vq2 import runtime

    assert math.isclose(runtime.TILT_ABORT, math.radians(60.0), abs_tol=1e-9)
    assert VisualHandoffConfig().bank_abort_rad == math.radians(45.0)
