"""Deterministic tests for the gate-relative, velocity-aware shadow policy.

STRICT SHADOW MODE: these lock the fail-closed contract, the ESKF relative-position
sign, the right-turn convention, and the nonzero-closure racing objective. Nothing here
grants command authority; the policy only ever produces hypothetical desired actual
FRD rates + thrust through the single canonical conversion boundary.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest

from drone_racing.deployments.vq2.attitude import quat_to_euler, quat_to_matrix
from drone_racing.deployments.vq2.eskf import ESKF
from drone_racing.deployments.vq2.follower import FollowerConfig, StateQualityLimits
from drone_racing.deployments.vq2.gate_relative import (
    GateApproachConfig,
    GateApproachLimits,
    GateApproachStatus,
    GateRelativeObservation,
    GateRelativePolicy,
    StructuredTarget,
    rotation_body_to_ned,
)

# --------------------------------------------------------------------------- helpers


def make_config(**overrides: object) -> GateApproachConfig:
    conversion = FollowerConfig(
        quality=StateQualityLimits(
            max_age_s=0.5, max_position_cov_trace_m2=50.0, max_velocity_cov_trace_m2ps2=25.0
        )
    )
    return GateApproachConfig(conversion=conversion, limits=GateApproachLimits(), **overrides)


def make_obs(**overrides: object) -> GateRelativeObservation:
    to_gate = np.array([8.0, 3.0, 0.0])  # gate 8 m fwd (N), 3 m right (E), level
    base: dict[str, object] = {
        "arrival_time_ns": 1_000,
        "active_race_gate": 1,
        "anchor_gate": 1,
        "initialised": True,
        "position_drone_rel_gate_ned_m": -to_gate,  # p = drone - gate
        "velocity_ned_mps": np.zeros(3),
        "euler_rpy_rad": np.zeros(3),
        "cov_diag": np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.01, 0.01, 0.01]),
        "fix_age_s": 0.05,
        "target_mode": "metric",
        "target_bearing_rad": math.radians(20.0),  # gate to the right
        "target_slant_m": float(np.linalg.norm(to_gate)),
        "target_age_s": 0.05,
    }
    base.update(overrides)
    return GateRelativeObservation(**base)  # type: ignore[arg-type]


def run(obs: GateRelativeObservation, **cfg: object):
    policy = GateRelativePolicy(make_config(**cfg))
    return policy, policy.update(obs, now_ns=obs.arrival_time_ns)


# --------------------------------------------------------------------------- sign


def test_rotation_matches_quaternion_convention() -> None:
    q = np.array([0.8, 0.1, -0.2, 0.55])
    q = q / np.linalg.norm(q)
    roll, pitch, yaw = quat_to_euler(q)
    assert np.allclose(rotation_body_to_ned(roll, pitch, yaw), quat_to_matrix(q), atol=1e-9)


def test_eskf_relative_position_sign_is_drone_minus_gate() -> None:
    """Regression: reanchor sets p = -R@z (= drone - gate), so to-gate = -p."""
    kf = ESKF()  # default q = [1,0,0,0] -> R = I
    gate_body = np.array([10.0, 4.0, -1.0])  # gate ahead/right/up in body
    kf.reanchor(gate_body)
    assert np.allclose(kf.p, -gate_body)  # p = -R@z, R=I
    obs = make_obs(
        position_drone_rel_gate_ned_m=kf.p,
        target_slant_m=float(np.linalg.norm(gate_body)),
        target_bearing_rad=0.0,
    )
    _, result = run(obs)
    assert result.eligible
    # policy's to-gate must equal -p and point at the gate the ESKF measured
    assert np.allclose(result.to_gate_ned_m, -np.asarray(kf.p))
    assert np.allclose(result.to_gate_ned_m, gate_body)


# --------------------------------------------------------------------------- eligible


def test_fresh_confirmed_metric_target_produces_finite_setpoint() -> None:
    _, result = run(make_obs())
    assert result.status is GateApproachStatus.OK
    sp = result.setpoint
    assert sp is not None
    assert np.all(np.isfinite(sp.body_rates_rps))
    assert math.isfinite(sp.collective_thrust)
    assert result.closure_speed_mps is not None and result.closure_speed_mps > 0.0


def test_no_map_ready_requirement() -> None:
    # Building/using the policy touches no CourseMap and never calls require_flight_ready.
    _, result = run(make_obs())
    assert result.eligible


# ---------------------------------------------------------------------- fail closed


@pytest.mark.parametrize(
    ("overrides", "expected"),
    [
        ({"initialised": False}, GateApproachStatus.NOT_INITIALISED),
        ({"anchor_gate": 2}, GateApproachStatus.ANCHOR_MISMATCH),
        ({"active_race_gate": 2}, GateApproachStatus.ANCHOR_MISMATCH),
        ({"target_mode": "none"}, GateApproachStatus.NO_TARGET),
        ({"target_mode": "bearing"}, GateApproachStatus.BEARING_ONLY),
        ({"association_refused": True}, GateApproachStatus.ASSOCIATION_REFUSED),
        ({"out_of_band": True}, GateApproachStatus.OUT_OF_BAND),
        ({"target_slant_m": 25.0}, GateApproachStatus.OUT_OF_BAND),
        ({"fix_age_s": 2.0}, GateApproachStatus.STALE),
        ({"target_age_s": 2.0}, GateApproachStatus.STALE),
        (
            {"cov_diag": np.array([100.0, 100.0, 100.0, 1.0, 1.0, 1.0, 0.01, 0.01, 0.01])},
            GateApproachStatus.UNCERTAIN,
        ),
        (
            {"cov_diag": np.array([1.0, 1.0, 1.0, 400.0, 400.0, 4.0, 0.01, 0.01, 0.01])},
            GateApproachStatus.UNCERTAIN,
        ),
    ],
)
def test_fail_closed_paths(overrides: dict[str, object], expected: GateApproachStatus) -> None:
    _, result = run(make_obs(**overrides))
    assert result.status is expected
    assert result.setpoint is None


def test_wrong_turn_target_fails_closed() -> None:
    # Map says gate 1 is a RIGHT turn (sign +1); a confirmed target to the LEFT is refused.
    obs = make_obs(target_bearing_rad=math.radians(-25.0))
    _, result = run(obs, expected_turn_sign=1.0)
    assert result.status is GateApproachStatus.WRONG_TURN
    assert result.setpoint is None


def test_excessive_tilt_fails_closed() -> None:
    obs = make_obs(euler_rpy_rad=np.array([0.0, math.radians(70.0), 0.0]))
    _, result = run(obs)
    assert result.status is GateApproachStatus.EXCESSIVE_TILT


def test_nonfinite_state_fails_closed() -> None:
    obs = make_obs(velocity_ned_mps=np.array([np.nan, 0.0, 0.0]))
    _, result = run(obs)
    assert result.status is GateApproachStatus.INVALID_STATE


# ------------------------------------------------------------------ right-turn sign


def test_right_side_gate_commands_right_turn() -> None:
    """Gate to the right (east, drone facing north) -> heading right, yaw-rate right."""
    to_gate = np.array([6.0, 5.0, 0.0])  # +east = right
    obs = make_obs(
        position_drone_rel_gate_ned_m=-to_gate,
        target_slant_m=float(np.linalg.norm(to_gate)),
        target_bearing_rad=math.radians(30.0),
    )
    _, result = run(obs)
    assert result.eligible
    assert result.desired_heading_rad is not None and result.desired_heading_rad > 0.0  # right
    # FRD body-z is down; a right (nose-east) turn from yaw 0 is a POSITIVE body yaw rate.
    assert result.setpoint is not None and result.setpoint.body_rates_rps[2] > 0.0
    # accelerate toward the gate: east (right) component of requested accel is positive.
    assert result.requested_acceleration_ned_mps2[1] > 0.0


def test_left_side_gate_commands_left_turn() -> None:
    to_gate = np.array([6.0, -5.0, 0.0])  # -east = left
    obs = make_obs(
        position_drone_rel_gate_ned_m=-to_gate,
        target_slant_m=float(np.linalg.norm(to_gate)),
        target_bearing_rad=math.radians(-30.0),
    )
    _, result = run(obs)
    assert result.eligible
    assert result.desired_heading_rad is not None and result.desired_heading_rad < 0.0
    assert result.setpoint is not None and result.setpoint.body_rates_rps[2] < 0.0


# ------------------------------------------------------------- closure objective


def test_positive_closure_objective_outside_handoff() -> None:
    obs = make_obs()  # distance ~8.5 m, well outside handoff
    _, result = run(obs)
    assert result.eligible
    assert result.closure_speed_mps is not None and result.closure_speed_mps > 0.0
    # velocity reference points toward the gate: requested accel has positive LOS component
    los = result.to_gate_ned_m / result.distance_m
    assert float(result.requested_acceleration_ned_mps2 @ los) > 0.0


def test_closure_speed_is_bounded_and_nonzero_far_away() -> None:
    far = np.array([15.0, 0.0, 0.0])
    obs = make_obs(
        position_drone_rel_gate_ned_m=-far,
        target_slant_m=15.0,
        target_bearing_rad=0.0,
    )
    _, result = run(obs)
    assert result.eligible
    cfg = make_config()
    assert cfg.closure_speed_min_mps <= result.closure_speed_mps <= cfg.closure_speed_max_mps


def test_does_not_zero_forward_for_large_but_visible_bearing() -> None:
    # A large (but in-FOV) bearing must NOT suppress closure (unlike the legacy law).
    obs = make_obs(target_bearing_rad=math.radians(45.0))
    _, result = run(obs)
    assert result.eligible
    assert result.closure_speed_mps is not None and result.closure_speed_mps > 0.0


# --------------------------------------------------------------- near-gate handoff


def test_near_gate_handoff_avoids_los_singularity() -> None:
    tiny = np.array([0.0, 0.0, 0.0])  # exactly at the gate: LOS undefined
    obs = make_obs(
        position_drone_rel_gate_ned_m=tiny,
        target_slant_m=3.5,
        target_bearing_rad=0.0,
    )
    _, result = run(obs)  # must not raise / divide by zero
    assert result.status is GateApproachStatus.NEAR_GATE_HANDOFF
    assert result.setpoint is None


def test_handoff_triggers_on_perception_slant() -> None:
    obs = make_obs(target_slant_m=2.5)  # inside handoff even if ESKF distance is larger
    _, result = run(obs)
    assert result.status is GateApproachStatus.NEAR_GATE_HANDOFF


# ------------------------------------------------------------------- constraints


def test_output_obeys_tilt_rate_and_thrust_limits() -> None:
    # A large velocity error drives a huge requested accel; the conversion must clamp.
    obs = make_obs(velocity_ned_mps=np.array([-12.0, -8.0, -3.0]))
    _, result = run(obs)
    assert result.eligible
    sp = result.setpoint
    assert sp is not None
    cfg = make_config()
    rate_limit = np.asarray(cfg.conversion.max_body_rate_rps)
    assert np.all(np.abs(sp.body_rates_rps) <= rate_limit + 1e-9)
    assert (
        cfg.conversion.min_thrust - 1e-9 <= sp.collective_thrust <= cfg.conversion.max_thrust + 1e-9
    )
    desired_down = sp.desired_rotation_body_to_ned[:, 2]
    tilt = math.acos(float(np.clip(desired_down[2], -1.0, 1.0)))
    assert tilt <= cfg.conversion.max_tilt_rad + 1e-6
    assert sp.saturation_flags  # something saturated under this aggressive demand


# ----------------------------------------------------------------- determinism


def test_deterministic_replay_identical_decisions() -> None:
    obs = make_obs()
    policy = GateRelativePolicy(make_config())
    a = policy.update(obs, now_ns=obs.arrival_time_ns)
    b = policy.update(obs, now_ns=obs.arrival_time_ns)
    assert a.status is b.status
    assert a.setpoint is not None and b.setpoint is not None
    assert np.array_equal(a.setpoint.body_rates_rps, b.setpoint.body_rates_rps)
    assert a.setpoint.collective_thrust == b.setpoint.collective_thrust
    assert np.array_equal(a.requested_acceleration_ned_mps2, b.requested_acceleration_ned_mps2)


def test_attempt_state_isolation() -> None:
    obs = make_obs()
    policy = GateRelativePolicy(make_config())
    first = policy.update(obs, now_ns=obs.arrival_time_ns)
    policy.reset()
    after = policy.update(obs, now_ns=obs.arrival_time_ns)
    assert first.status is after.status
    assert np.array_equal(
        first.setpoint.body_rates_rps,  # type: ignore[union-attr]
        after.setpoint.body_rates_rps,  # type: ignore[union-attr]
    )


def test_record_guidance_is_log_only() -> None:
    """mission._record_guidance only appends sg_*/st_* fields from immutable worker
    snapshots; it performs no guidance and touches nothing on the command path."""
    from drone_racing.deployments.vq2 import mission
    from drone_racing.deployments.vq2.gate_relative import (
        GateApproachResult,
        GateApproachStatus,
        GuidanceDecision,
        empty_structured_target,
    )

    decision = GuidanceDecision(
        decision_mono_ns=1_900,
        now_ns=1_900,
        active_gate=1,
        anchor_gate=0,
        track_id=7,
        target_arrival_ns=1_800,
        mode="transition",
        result=GateApproachResult(status=GateApproachStatus.ANCHOR_MISMATCH),
    )
    row: dict[str, object] = {"mono_ns": 2_000, "req": (0.3, 0, 0, 0)}
    mission._record_guidance(row, empty_structured_target(1_800), decision, 2_000)
    added = set(row) - {"mono_ns", "req"}
    assert added and all(k.startswith(("sg_", "st_")) for k in added)
    assert row["sg_status"] == "anchor_mismatch" and row["sg_track"] == 7
    assert row["sg_decage"] == pytest.approx((2_000 - 1_900) * 1e-9)


# ============================ TRANSITION_ACQUIRE (gate 0 -> 1 bridge) ===============

_NOW = 2_000_000_000


def make_target(**over: object) -> StructuredTarget:
    """A confirmed right-side gate-1 structured target (far/out-of-band by default)."""
    bearing = float(over.pop("bearing", math.radians(35.0)))  # type: ignore[arg-type]
    base: dict[str, object] = {
        "valid": True,
        "frame_id": 1,
        "frame_sensor_ns": 0,
        "arrival_ns": _NOW,
        "active_gate_at_receipt": 1,
        "track_id": 7,
        "race_gate_id": 1,
        "track_hits": 4,
        "output_mode": "position_only",
        "bearing_rad": bearing,
        "los_body_unit": np.array([math.cos(bearing), math.sin(bearing), 0.0]),
        "slant_range_m": 31.0,  # OUT of the 3-16 m fusion band on purpose
        "has_metric_pose": True,
        "confidence": 0.9,
        "cov_trace": 0.3,
        "expected_turn_sign": 1.0,
        "disposition": "assoc_refused",
        "association_refused": False,
        "wrong_turn": False,
        "out_of_band": True,
    }
    base.update(over)
    return StructuredTarget(**base)  # type: ignore[arg-type]


def step_transition(target: StructuredTarget, **over: object):
    policy = GateRelativePolicy(make_config())
    state: dict[str, object] = {
        "now_ns": _NOW,
        "active_gate": 1,
        "anchor_gate": 0,  # still anchored to gate 0 -> transition regime
        "initialised": True,
        "position_drone_rel_gate_ned_m": np.zeros(3),
        "velocity_ned_mps": np.array([12.0, 0.0, 0.0]),  # 12 m/s north (gate-0 pass speed)
        "euler_rpy_rad": np.zeros(3),
        "cov_diag": np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.01, 0.01, 0.01]),
        "fix_age_s": 0.05,
        "target": target,
        "transition_age_s": 0.3,
    }
    state.update(over)
    return policy.step(**state)  # type: ignore[arg-type]


def test_transition_acquire_fires_at_gate_advance() -> None:
    d = step_transition(make_target())
    assert d.mode == "transition"
    assert d.result.status is GateApproachStatus.TRANSITION_ACQUIRE
    assert d.result.setpoint is not None
    assert d.track_id == 7  # real structured track id reached guidance


def test_transition_commands_right_turn() -> None:
    d = step_transition(make_target())  # gate +35 deg right, flying north
    r = d.result
    assert r.desired_heading_rad is not None and r.desired_heading_rad > 0.0
    assert r.setpoint is not None and r.setpoint.body_rates_rps[2] > 0.0  # yaw right
    assert r.requested_acceleration_ned_mps2[1] > 0.0  # accelerate east (right)


def test_transition_preserves_useful_closure_not_braked_to_cap() -> None:
    r = step_transition(make_target()).result
    # existing along-LOS closure ~= 12*cos(35) ~= 9.8 m/s: preserved, NOT the 4 m/s cap
    assert r.closure_speed_mps is not None and r.closure_speed_mps > 8.0
    assert r.distance_m is None  # far-range PnP scale is NOT trusted


def test_transition_out_of_band_far_target_still_acts_on_direction() -> None:
    r = step_transition(make_target(slant_range_m=31.0, out_of_band=True)).result
    assert r.status is GateApproachStatus.TRANSITION_ACQUIRE  # direction-only, no fusion


def test_transition_never_exceeds_safety_speed() -> None:
    r = step_transition(make_target(), velocity_ned_mps=np.array([20.0, 0.0, 0.0])).result  # 20 m/s
    assert r.closure_speed_mps is not None and r.closure_speed_mps <= 12.0 + 1e-9


@pytest.mark.parametrize(
    ("t_over", "s_over", "expected"),
    [
        ({"wrong_turn": True}, {}, GateApproachStatus.WRONG_TURN),
        ({"bearing": math.radians(-30.0)}, {}, GateApproachStatus.WRONG_TURN),  # left = wrong
        ({"association_refused": True}, {}, GateApproachStatus.ASSOCIATION_REFUSED),
        ({"track_hits": 1}, {}, GateApproachStatus.NO_TARGET),  # not continuous
        ({"race_gate_id": 2}, {}, GateApproachStatus.NO_TARGET),  # wrong association
        ({"valid": False}, {}, GateApproachStatus.NO_TARGET),
        ({"bearing": math.radians(70.0)}, {}, GateApproachStatus.OUT_OF_BAND),  # out of FOV
        ({}, {"transition_age_s": 5.0}, GateApproachStatus.TRANSITION_TIMEOUT),
        ({"arrival_ns": _NOW - 700_000_000}, {}, GateApproachStatus.STALE),
        (
            {},
            {"cov_diag": np.array([1.0, 1.0, 1.0, 400.0, 400.0, 4.0, 0.01, 0.01, 0.01])},
            GateApproachStatus.UNCERTAIN,
        ),
        (
            {},
            {"euler_rpy_rad": np.array([0.0, math.radians(70.0), 0.0])},
            GateApproachStatus.EXCESSIVE_TILT,
        ),
    ],
)
def test_transition_fail_closed_paths(t_over, s_over, expected) -> None:  # noqa: ANN001
    # bearing is a helper kwarg; map it through make_target's popping logic
    d = step_transition(make_target(**t_over), **s_over)
    assert d.result.status is expected
    assert d.result.setpoint is None


def test_transition_switches_to_anchored_after_in_band_reanchor() -> None:
    # once the ESKF reanchors to gate 1 (anchor == active), the dispatcher uses the
    # anchored policy with the in-band metric target.
    tgt = make_target(slant_range_m=8.0, out_of_band=False)
    d = step_transition(
        tgt,
        anchor_gate=1,  # reanchored
        position_drone_rel_gate_ned_m=-np.array([6.0, 4.0, 0.0]),
        velocity_ned_mps=np.zeros(3),
    )
    assert d.mode == "anchored"
    assert d.result.status is GateApproachStatus.OK


def test_gate0_anchored_is_turn_unconstrained() -> None:
    # gate 0: expected_turn_sign 0 -> a left-bearing target is NOT a wrong-turn refusal.
    tgt = make_target(
        race_gate_id=0,
        expected_turn_sign=0.0,
        bearing=math.radians(-20.0),
        slant_range_m=8.0,
        out_of_band=False,
    )
    d = step_transition(
        tgt,
        active_gate=0,
        anchor_gate=0,
        position_drone_rel_gate_ned_m=-np.array([6.0, -2.0, 0.0]),
        velocity_ned_mps=np.zeros(3),
    )
    assert d.mode == "anchored"
    assert d.result.status is GateApproachStatus.OK  # not WRONG_TURN


def test_gate1_anchored_left_target_is_wrong_turn() -> None:
    # gate 1 turn guard is live (+1): an anchored gate-1 target to the LEFT is refused.
    tgt = make_target(
        bearing=math.radians(-25.0), slant_range_m=8.0, out_of_band=False, expected_turn_sign=1.0
    )
    d = step_transition(
        tgt,
        anchor_gate=1,
        position_drone_rel_gate_ned_m=-np.array([6.0, -3.0, 0.0]),
        velocity_ned_mps=np.zeros(3),
    )
    assert d.mode == "anchored"
    assert d.result.status is GateApproachStatus.WRONG_TURN


def test_transition_deterministic_replay() -> None:
    a = step_transition(make_target())
    b = step_transition(make_target())
    assert a.result.status is b.result.status
    assert np.array_equal(
        a.result.setpoint.body_rates_rps,  # type: ignore[union-attr]
        b.result.setpoint.body_rates_rps,  # type: ignore[union-attr]
    )


def test_guidance_decision_age_reports_staleness() -> None:
    d = step_transition(make_target())
    assert d.age_s(d.decision_mono_ns) == 0.0
    assert d.age_s(d.decision_mono_ns + 500_000_000) == pytest.approx(0.5)


def test_policy_never_applies_command_calibration() -> None:
    """Separation of concerns: the policy emits desired ACTUAL rates; calibration (the
    sim sign/scale inversion) lives only at the single command boundary, never here.

    Checked against actual import statements + code (not docstring prose)."""
    source = Path("src/drone_racing/deployments/vq2/gate_relative.py").read_text()
    code_lines = [
        line
        for line in source.splitlines()
        if line.strip().startswith(("import ", "from ")) or "RATE_GAIN" in line
    ]
    code = "\n".join(code_lines)
    assert "calibration" not in code  # no calibration import
    assert "RATE_GAIN" not in code  # never touches the sim rate inversion/scale
    assert "runtime" not in code and "mission" not in code  # no command-path coupling
