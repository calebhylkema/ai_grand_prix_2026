"""Offline tests for the fail-closed VQ2 course-reference follower."""

from __future__ import annotations

import math
from pathlib import Path

import casadi as ca
import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from drone_racing.deployments.vq2.course_map import (
    CourseMap,
    GateLeg,
    GateSpec,
    MapEvidence,
    MapFrame,
    MapStatus,
)
from drone_racing.deployments.vq2.follower import (
    AlignmentStatus,
    ConservativeFollower,
    FollowerConfig,
    FollowerStatus,
    FrameAlignment,
    MapAlignedState,
    MapFrameBridge,
    StateQualityLimits,
    VQ2EstimateSnapshot,
    acceleration_to_rate_thrust,
    evaluate_guidance_trace,
)
from drone_racing.deployments.vq2.reference import CourseReference, build_course_reference
from drone_racing.planning.dynamics import NU, NX, VQ2DynamicsConfig, rk4_vq2_step
from drone_racing.utils.types import PlannerConfig


def _course() -> CourseMap:
    evidence = MapEvidence("structured-v2", ("a", "b", "c"), 3, 60)
    return CourseMap(
        schema_version=1,
        course_id="guidance-test",
        status=MapStatus.VALIDATED,
        frame=MapFrame.GATE0_LOCAL_NED,
        units="si",
        drone_radius_m=0.14,
        tracking_margin_m=0.20,
        gates=(
            GateSpec(
                0,
                (0.0, 0.0, 0.0),
                (0.01, 0.01, 0.01),
                (1.5, 1.5),
                (1.0, 0.0, 0.0),
                evidence,
            ),
            GateSpec(
                1,
                (9.2, 6.4, -0.6),
                (0.01, 0.01, 0.01),
                (1.5, 1.5),
                (1.0, 0.0, 0.0),
                evidence,
            ),
        ),
        legs=(GateLeg(0, 1, 1, (3.0, 16.0), (0.4, 1.9)),),
    )


def _reference() -> CourseReference:
    config = PlannerConfig(
        max_velocity_mps=4.0,
        min_velocity_mps=0.8,
        max_lateral_accel_mps2=3.5,
        max_longitudinal_accel_mps2=2.0,
        max_braking_accel_mps2=3.0,
    )
    return build_course_reference(_course(), config, num_points=300)


def _quality() -> StateQualityLimits:
    return StateQualityLimits(
        max_age_s=0.10,
        max_position_cov_trace_m2=0.5,
        max_velocity_cov_trace_m2ps2=0.5,
    )


def _snapshot(
    anchor: int,
    position_anchor: np.ndarray,
    *,
    arrival_ns: int = 1_000_000_000,
    velocity: np.ndarray | None = None,
    quaternion: np.ndarray | None = None,
    covariance: np.ndarray | None = None,
) -> VQ2EstimateSnapshot:
    return VQ2EstimateSnapshot(
        arrival_time_ns=arrival_ns,
        anchor_gate_id=anchor,
        position_anchor_ned_m=position_anchor,
        velocity_ned_mps=np.zeros(3) if velocity is None else velocity,
        quaternion_wxyz=(np.array([1.0, 0.0, 0.0, 0.0]) if quaternion is None else quaternion),
        covariance=np.eye(15) * 0.01 if covariance is None else covariance,
    )


def _state(
    position: np.ndarray,
    *,
    timestamp_ns: int = 1_000_000_000,
    velocity: np.ndarray | None = None,
    rotation: np.ndarray | None = None,
    covariance: np.ndarray | None = None,
) -> MapAlignedState:
    return MapAlignedState(
        arrival_time_ns=timestamp_ns,
        anchor_gate_id=0,
        position_ned_m=position,
        velocity_ned_mps=np.zeros(3) if velocity is None else velocity,
        rotation_body_to_ned=np.eye(3) if rotation is None else rotation,
        covariance=np.eye(15) * 0.01 if covariance is None else covariance,
    )


def test_map_bridge_rotates_world_blocks_but_not_local_error_blocks() -> None:
    course = _course()
    bridge = MapFrameBridge(
        course,
        FrameAlignment(math.pi / 2.0, "reviewed-launch-heading", True),
    )
    covariance = np.eye(15) * 0.01
    covariance[0:3, 0:3] = np.diag([1.0, 2.0, 3.0])
    covariance[6:9, 6:9] = np.diag([4.0, 5.0, 6.0])
    result = bridge.update(_snapshot(0, np.array([-2.0, 0.0, 0.0]), covariance=covariance))
    assert result.status is AlignmentStatus.OK
    assert result.state is not None
    assert result.state.position_ned_m == pytest.approx([0.0, -2.0, 0.0])
    assert np.diag(result.state.covariance[0:3, 0:3]) == pytest.approx([2.0, 1.0, 3.0])
    assert np.diag(result.state.covariance[6:9, 6:9]) == pytest.approx([4.0, 5.0, 6.0])


def test_reanchor_preserves_global_position_and_rejects_large_jump() -> None:
    course = _course()
    bridge = MapFrameBridge(course, FrameAlignment(0.0, "reviewed", True))
    before = np.array([9.0, 6.2, -0.5])
    first = bridge.update(_snapshot(0, before, arrival_ns=1_000_000_000))
    second = bridge.update(_snapshot(1, before - course.gate(1).center, arrival_ns=1_010_000_000))
    assert first.status is AlignmentStatus.OK
    assert second.status is AlignmentStatus.OK
    assert second.reanchor_jump_m == pytest.approx(0.0, abs=1e-12)

    bridge.reset()
    bridge.update(_snapshot(0, before, arrival_ns=2_000_000_000))
    jumped = bridge.update(_snapshot(1, np.array([2.0, 0.0, 0.0]), arrival_ns=2_010_000_000))
    assert jumped.status is AlignmentStatus.REANCHOR_JUMP
    assert jumped.state is None
    assert jumped.reanchor_jump_m is not None and jumped.reanchor_jump_m > 0.75


def test_bridge_fails_closed_on_alignment_anchor_and_quaternion_errors() -> None:
    course = _course()
    unaligned = MapFrameBridge(course, FrameAlignment(0.0, "candidate", False))
    assert unaligned.update(_snapshot(0, np.zeros(3))).status is AlignmentStatus.UNALIGNED

    bridge = MapFrameBridge(course, FrameAlignment(0.0, "reviewed", True))
    assert (
        bridge.update(_snapshot(0, np.zeros(3)), active_gate_id=1).status
        is AlignmentStatus.ANCHOR_MISMATCH
    )
    bad_quaternion = _snapshot(0, np.zeros(3), quaternion=np.array([2.0, 0.0, 0.0, 0.0]))
    assert bridge.update(bad_quaternion).status is AlignmentStatus.INVALID_STATE


@pytest.mark.parametrize(
    ("acceleration", "axis", "sign"),
    [
        (np.array([2.0, 0.0, 0.0]), 1, -1.0),  # forward -> nose-down pitch
        (np.array([-2.0, 0.0, 0.0]), 1, 1.0),  # brake -> nose-up pitch
        (np.array([0.0, 2.0, 0.0]), 0, 1.0),  # right -> right-wing-down roll
    ],
)
def test_acceleration_conversion_has_vq2_frd_rate_signs(
    acceleration: np.ndarray, axis: int, sign: float
) -> None:
    config = FollowerConfig(_quality())
    setpoint = acceleration_to_rate_thrust(acceleration, 0.0, np.eye(3), config)
    assert math.copysign(1.0, setpoint.body_rates_rps[axis]) == sign
    assert np.allclose(
        setpoint.desired_rotation_body_to_ned.T @ setpoint.desired_rotation_body_to_ned,
        np.eye(3),
        atol=1e-10,
    )
    assert np.linalg.det(setpoint.desired_rotation_body_to_ned) == pytest.approx(1.0)


def test_hover_climb_and_descent_collective_are_correct() -> None:
    config = FollowerConfig(_quality())
    hover = acceleration_to_rate_thrust(np.zeros(3), 0.0, np.eye(3), config)
    climb = acceleration_to_rate_thrust(np.array([0.0, 0.0, -1.0]), 0.0, np.eye(3), config)
    descent = acceleration_to_rate_thrust(np.array([0.0, 0.0, 1.0]), 0.0, np.eye(3), config)
    assert hover.collective_thrust == pytest.approx(0.30)
    assert climb.collective_thrust > hover.collective_thrust
    assert descent.collective_thrust < hover.collective_thrust
    assert hover.body_rates_rps == pytest.approx(np.zeros(3), abs=1e-12)


def test_acceleration_conversion_enforces_all_command_limits_and_near_pi_is_finite() -> None:
    config = FollowerConfig(_quality(), max_tilt_rad=0.15)
    upside_yaw = Rotation.from_euler("z", math.pi).as_matrix()
    setpoint = acceleration_to_rate_thrust(
        np.array([100.0, -100.0, -100.0]),
        0.0,
        upside_yaw,
        config,
    )
    assert np.linalg.norm(setpoint.requested_acceleration_ned_mps2[:2]) <= 3.5 + 1e-12
    assert abs(setpoint.requested_acceleration_ned_mps2[2]) <= 2.0
    tilt = math.acos(setpoint.desired_rotation_body_to_ned[2, 2])
    assert tilt <= 0.15 + 1e-12
    assert np.all(np.abs(setpoint.body_rates_rps) <= np.array([2.0, 2.0, 1.0]) + 1e-12)
    assert abs(setpoint.body_rates_rps[2]) == pytest.approx(1.0)
    assert 0.25 <= setpoint.collective_thrust <= 0.45
    assert np.all(np.isfinite(setpoint.body_rates_rps))
    assert {"horizontal_acceleration", "vertical_acceleration", "tilt", "body_rate"} <= set(
        setpoint.saturation_flags
    )


def test_follower_fails_closed_for_stale_uncertain_invalid_and_completed_states() -> None:
    course = _course()
    reference = _reference()
    config = FollowerConfig(_quality())

    follower = ConservativeFollower(reference, course, config)
    start = reference.racing_line.positions[0]
    stale = follower.update(_state(start), now_ns=1_200_000_000)
    assert stale.status is FollowerStatus.STALE and stale.setpoint is None

    covariance = np.eye(15)
    uncertain = ConservativeFollower(reference, course, config).update(
        _state(start, covariance=covariance), now_ns=1_000_000_000
    )
    assert uncertain.status is FollowerStatus.UNCERTAIN and uncertain.setpoint is None

    tilted = Rotation.from_euler("x", 1.0).as_matrix()
    invalid = ConservativeFollower(reference, course, config).update(
        _state(start, rotation=tilted), now_ns=1_000_000_000
    )
    assert invalid.status is FollowerStatus.INVALID_STATE and invalid.setpoint is None

    completed = ConservativeFollower(reference, course, config).update(
        _state(reference.racing_line.positions[-1]), now_ns=1_000_000_000
    )
    assert completed.status is FollowerStatus.REFERENCE_COMPLETE
    assert completed.setpoint is None


def test_follower_output_is_shadow_setpoint_with_visibility_diagnostics() -> None:
    course = _course()
    reference = _reference()
    follower = ConservativeFollower(reference, course, FollowerConfig(_quality()))
    state = _state(reference.racing_line.positions[5])
    result = follower.update(state, now_ns=state.arrival_time_ns)
    assert result.status is FollowerStatus.OK
    assert result.setpoint is not None
    assert result.reference is not None
    assert result.reference.speed_mps >= np.linalg.norm(result.reference.velocity_ned_mps) - 1e-9
    assert result.gate_bearing_rad is not None
    assert result.gate_elevation_rad is not None
    assert result.gate_fov_margin_px is not None
    assert result.setpoint.collective_thrust <= 0.45


def test_rate_thrust_setpoint_drives_vq2_dynamics_in_expected_direction() -> None:
    config = FollowerConfig(_quality())
    dynamics = VQ2DynamicsConfig()
    x_symbol = ca.SX.sym("x", NX)
    u_symbol = ca.SX.sym("u", NU)
    step = ca.Function(
        "follower_vq2_step",
        [x_symbol, u_symbol],
        [rk4_vq2_step(x_symbol, u_symbol, 0.02, dynamics)],
    )
    state = np.zeros(NX)
    for _ in range(60):
        rotation = Rotation.from_euler("xyz", state[6:9]).as_matrix()
        command = acceleration_to_rate_thrust(np.array([2.0, 1.0, 0.0]), 0.0, rotation, config)
        control = np.concatenate(([command.collective_thrust], command.body_rates_rps))
        state = np.asarray(step(state, control)).reshape(-1)
    assert state[3] > 0.5  # north/forward velocity
    assert state[4] > 0.2  # east/right velocity
    assert np.all(np.isfinite(state))


def test_trace_evaluator_reports_monotonic_bounded_shadow_outputs() -> None:
    course = _course()
    reference = _reference()
    bridge = MapFrameBridge(course, FrameAlignment(0.0, "reviewed", True))
    follower = ConservativeFollower(reference, course, FollowerConfig(_quality()))
    indices = np.linspace(0, len(reference.racing_line.positions) - 2, 80, dtype=int)
    snapshots = [
        _snapshot(
            0,
            reference.racing_line.positions[index],
            arrival_ns=1_000_000_000 + sequence * 20_000_000,
            velocity=reference.racing_line.tangents[index]
            * min(reference.racing_line.speeds[index], 1.8),
        )
        for sequence, index in enumerate(indices)
    ]
    metrics = evaluate_guidance_trace(bridge, follower, snapshots)
    assert metrics.sample_count == len(snapshots)
    assert metrics.valid_setpoints == len(snapshots) - 1
    assert metrics.status_counts == {"ok": len(snapshots) - 1, "reference_complete": 1}
    assert metrics.backward_progress_jumps == 0
    assert metrics.progress_end > metrics.progress_start
    assert metrics.thrust_min >= 0.25
    assert metrics.thrust_max <= 0.45
    assert metrics.max_tilt_rad <= 0.30 + 1e-12
    assert metrics.max_body_rate_rps <= 2.0
    assert metrics.minimum_gate_clearance_m > 0.0
    assert metrics.latency_max_ms < 100.0
    repeated = evaluate_guidance_trace(bridge, follower, snapshots)
    assert repeated.valid_setpoints == metrics.valid_setpoints
    assert repeated.progress_start == pytest.approx(metrics.progress_start)
    assert repeated.progress_end == pytest.approx(metrics.progress_end)


def test_follower_module_has_no_active_control_dependencies() -> None:
    import drone_racing.deployments.vq2.follower as follower_module

    forbidden = ("mission", "runtime", "calibration", "DroneCommand", "RATE_GAIN")
    source = Path(follower_module.__file__).read_text(encoding="utf-8")
    import_lines = [line for line in source.splitlines() if line.startswith(("import ", "from "))]
    assert not any(token in line for token in forbidden for line in import_lines)
