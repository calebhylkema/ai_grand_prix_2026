"""Tests for drone_racing.estimation.gate_mapper.

Strategy: construct a synthetic scenario with a known drone state and a
known gate world pose, render the gate's corners into the image using
OpenCV's projection, feed the resulting detection into the mapper, and
verify the recovered gate pose matches ground truth.
"""

from __future__ import annotations

import cv2
import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from drone_racing.estimation.gate_mapper import (
    GateMapper,
    GateMapperConfig,
    estimate_gate_pose_world,
)
from drone_racing.utils.config import load_config
from drone_racing.utils.transforms import (
    body_to_camera_rotation,
    invert_transform,
    pose_to_matrix,
    quat_identity,
)
from drone_racing.utils.types import (
    CameraConfig,
    GateDetection,
    State,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def camera_config() -> CameraConfig:
    return load_config().camera


@pytest.fixture()
def gate_corners_local() -> np.ndarray:
    return load_config().gate.corners_3d_array


def _render_detection(
    gate_position_w: np.ndarray,
    gate_orientation_w: np.ndarray,
    drone_position_w: np.ndarray,
    drone_orientation_w: np.ndarray,
    corners_local: np.ndarray,
    camera: CameraConfig,
    *,
    timestamp_ns: int = 1000,
    gate_id: int = 0,
    confidence: float = 0.9,
    pixel_noise_std: float = 0.0,
    rng: np.random.Generator | None = None,
) -> GateDetection:
    """Render a synthetic GateDetection from a known drone/gate configuration.

    Computes T_CG = (T_WB @ T_BC)^-1 @ T_WG, then projects the local
    corners through OpenCV.
    """
    t_wg = pose_to_matrix(gate_position_w, gate_orientation_w)
    t_wb = pose_to_matrix(drone_position_w, drone_orientation_w)
    r_bc = body_to_camera_rotation(camera.tilt_deg).T  # R_BC = R_CB^T
    t_bc = np.eye(4, dtype=np.float64)
    t_bc[:3, :3] = r_bc
    t_wc = t_wb @ t_bc
    t_cg = invert_transform(t_wc) @ t_wg

    rvec, _ = cv2.Rodrigues(t_cg[:3, :3])
    tvec = t_cg[:3, 3].reshape(3, 1)
    img_pts, _ = cv2.projectPoints(
        corners_local, rvec, tvec, camera.intrinsic_matrix, camera.distortion_coeffs
    )
    corners_2d = img_pts.reshape(-1, 2)
    if pixel_noise_std > 0.0:
        gen = rng or np.random.default_rng(0)
        corners_2d = corners_2d + gen.normal(0.0, pixel_noise_std, corners_2d.shape)

    return GateDetection(
        timestamp_ns=timestamp_ns,
        corners_2d=corners_2d,
        confidence=confidence,
        gate_id=gate_id,
    )


def _gate_facing(drone_pos: np.ndarray, gate_pos: np.ndarray) -> np.ndarray:
    """Return a gate orientation quaternion whose +Z normal points toward
    *drone_pos* (so the gate is a vertical plane visible from the drone)."""
    # Gate-local +Z should point toward drone.  Build a rotation that sends
    # local +Z onto the world unit vector (drone - gate), then pick yaw so
    # the gate plane is upright (local Y stays roughly world +Z).
    direction = drone_pos - gate_pos
    direction = direction / (np.linalg.norm(direction) + 1e-12)
    # Start: local +Z → world direction via axis-angle from (+Z) to direction.
    z_axis = np.array([0.0, 0.0, 1.0])
    v = np.cross(z_axis, direction)
    c = float(np.dot(z_axis, direction))
    if np.linalg.norm(v) < 1e-9:
        if c > 0:
            r = Rotation.identity()
        else:
            r = Rotation.from_rotvec(np.pi * np.array([1.0, 0.0, 0.0]))
    else:
        angle = float(np.arccos(np.clip(c, -1.0, 1.0)))
        axis = v / np.linalg.norm(v)
        r = Rotation.from_rotvec(angle * axis)
    return r.as_quat()


def _drone_looking_at(drone_pos: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Return a body-to-world quaternion [x,y,z,w] for a drone at *drone_pos*
    whose body-X axis (FRD forward) points at *target*.

    Only yaw is set; roll and pitch are zero.
    """
    dx, dy = (target - drone_pos)[:2]
    yaw = float(np.arctan2(dy, dx))
    return Rotation.from_euler("ZYX", [yaw, 0.0, 0.0]).as_quat()


def _state_from(pos: np.ndarray, quat: np.ndarray, *, ts: int = 1000) -> State:
    s = State.identity(ts)
    s.position = pos.astype(np.float64)
    s.orientation = quat.astype(np.float64)
    return s


# ---------------------------------------------------------------------------
# Tests: pose recovery
# ---------------------------------------------------------------------------


class TestEstimateGatePoseWorld:
    """The pure PnP-chain helper recovers ground-truth gate poses."""

    def test_recovers_gate_at_origin(
        self, camera_config: CameraConfig, gate_corners_local: np.ndarray
    ) -> None:
        gate_pos = np.array([0.0, 0.0, 3.0])
        gate_ori = Rotation.from_euler("ZYX", [np.pi, 0.0, np.pi / 2]).as_quat()
        # Gate normal along +X world by rotating gate-local +Z onto +X.
        # Simpler: put the drone somewhere and aim it at the gate.
        drone_pos = np.array([-5.0, 0.0, 3.0])
        drone_ori = _drone_looking_at(drone_pos, gate_pos)

        det = _render_detection(
            gate_pos,
            gate_ori,
            drone_pos,
            drone_ori,
            gate_corners_local,
            camera_config,
        )
        state = _state_from(drone_pos, drone_ori)

        result = estimate_gate_pose_world(det, state, gate_corners_local, camera_config)
        assert result is not None
        pos_w, ori_w, corners_w, reproj = result
        np.testing.assert_allclose(pos_w, gate_pos, atol=1e-4)
        # Quaternion can flip sign
        dot = abs(float(np.dot(ori_w, gate_ori)))
        assert dot > 0.999
        assert reproj < 0.1

    def test_recovers_rotated_gate(
        self, camera_config: CameraConfig, gate_corners_local: np.ndarray
    ) -> None:
        gate_pos = np.array([10.0, 2.0, 3.5])
        gate_ori = Rotation.from_euler("ZYX", [np.pi / 2, 0.0, np.pi / 2]).as_quat()
        drone_pos = np.array([10.0, -3.0, 3.0])
        drone_ori = _drone_looking_at(drone_pos, gate_pos)

        det = _render_detection(
            gate_pos,
            gate_ori,
            drone_pos,
            drone_ori,
            gate_corners_local,
            camera_config,
        )
        state = _state_from(drone_pos, drone_ori)

        result = estimate_gate_pose_world(det, state, gate_corners_local, camera_config)
        assert result is not None
        pos_w, ori_w, _, _ = result
        np.testing.assert_allclose(pos_w, gate_pos, atol=1e-3)
        dot = abs(float(np.dot(ori_w, gate_ori)))
        assert dot > 0.999

    def test_returns_none_with_few_corners(
        self, camera_config: CameraConfig, gate_corners_local: np.ndarray
    ) -> None:
        det = GateDetection(
            timestamp_ns=0,
            corners_2d=np.array([[100, 200], [300, 200], [200, 300]], dtype=np.float64),
            confidence=0.5,
            gate_id=0,
        )
        state = _state_from(np.zeros(3), quat_identity())
        result = estimate_gate_pose_world(det, state, gate_corners_local, camera_config)
        assert result is None


# ---------------------------------------------------------------------------
# Tests: GateMapper aggregation
# ---------------------------------------------------------------------------


class TestGateMapper:
    def test_single_observation_not_promoted(
        self, camera_config: CameraConfig, gate_corners_local: np.ndarray
    ) -> None:
        """A single observation is insufficient (needs ``min_observations``)."""
        mapper = GateMapper(camera_config, gate_corners_local)
        gate_pos = np.array([0.0, 0.0, 3.0])
        drone_pos = np.array([-5.0, 0.0, 3.0])
        gate_ori = _gate_facing(drone_pos, gate_pos)
        drone_ori = _drone_looking_at(drone_pos, gate_pos)

        det = _render_detection(
            gate_pos,
            gate_ori,
            drone_pos,
            drone_ori,
            gate_corners_local,
            camera_config,
            gate_id=7,
        )
        state = _state_from(drone_pos, drone_ori)
        assert mapper.observe(det, state)
        assert mapper.observation_count(7) == 1
        assert mapper.waypoints() == []

    def test_promotes_after_min_observations(
        self, camera_config: CameraConfig, gate_corners_local: np.ndarray
    ) -> None:
        mapper = GateMapper(camera_config, gate_corners_local, GateMapperConfig(min_observations=3))
        gate_pos = np.array([8.0, 1.0, 3.0])
        gate_ori = Rotation.from_euler("ZYX", [0.3, 0.0, np.pi / 2]).as_quat()

        # Observe from three different vantage points
        for i, offset in enumerate([(-5.0, 0.0, 0.0), (-4.0, 1.0, 0.2), (-5.5, -0.5, -0.3)]):
            drone_pos = gate_pos + np.array([-5.0, 0.0, 0.0]) + np.array(offset) * 0.5
            drone_ori = _drone_looking_at(drone_pos, gate_pos)
            det = _render_detection(
                gate_pos,
                gate_ori,
                drone_pos,
                drone_ori,
                gate_corners_local,
                camera_config,
                gate_id=3,
                timestamp_ns=1000 + i,
            )
            state = _state_from(drone_pos, drone_ori)
            assert mapper.observe(det, state)

        waypoints = mapper.waypoints()
        assert len(waypoints) == 1
        wp = waypoints[0]
        assert wp.gate_id == 3
        assert wp.order == 0
        np.testing.assert_allclose(wp.position_w, gate_pos, atol=1e-3)
        dot = abs(float(np.dot(wp.orientation_w, gate_ori)))
        assert dot > 0.999
        # Corners should be the gate-local model transformed into world frame
        assert wp.corners_3d_w.shape == gate_corners_local.shape

    def test_rejects_missing_gate_id(
        self, camera_config: CameraConfig, gate_corners_local: np.ndarray
    ) -> None:
        mapper = GateMapper(camera_config, gate_corners_local)
        gate_pos = np.array([0.0, 0.0, 3.0])
        drone_pos = np.array([-5.0, 0.0, 3.0])
        drone_ori = _drone_looking_at(drone_pos, gate_pos)
        det = _render_detection(
            gate_pos,
            _gate_facing(drone_pos, gate_pos),
            drone_pos,
            drone_ori,
            gate_corners_local,
            camera_config,
            gate_id=0,
        )
        det.gate_id = None
        state = _state_from(drone_pos, drone_ori)
        assert not mapper.observe(det, state)
        assert mapper.waypoints() == []

    def test_rejects_outlier_observation(
        self, camera_config: CameraConfig, gate_corners_local: np.ndarray
    ) -> None:
        """An observation far from the running mean should be rejected."""
        mapper = GateMapper(
            camera_config,
            gate_corners_local,
            GateMapperConfig(min_observations=2, position_consistency_m=1.0),
        )
        gate_pos = np.array([6.0, 0.0, 3.0])
        drone_pos = np.array([0.0, 0.0, 3.0])
        gate_ori = _gate_facing(drone_pos, gate_pos)
        drone_ori = _drone_looking_at(drone_pos, gate_pos)

        # Establish a run of good observations
        for i in range(2):
            det = _render_detection(
                gate_pos,
                gate_ori,
                drone_pos + np.array([0.0, i * 0.1, 0.0]),
                drone_ori,
                gate_corners_local,
                camera_config,
                gate_id=1,
                timestamp_ns=1000 + i,
            )
            state = _state_from(drone_pos + np.array([0.0, i * 0.1, 0.0]), drone_ori)
            assert mapper.observe(det, state)

        # Now inject an observation whose rendered geometry corresponds to a
        # gate 5m away from the real one — should be rejected.
        shifted_gate = gate_pos + np.array([0.0, 5.0, 0.0])
        bad_det = _render_detection(
            shifted_gate,
            gate_ori,
            drone_pos,
            drone_ori,
            gate_corners_local,
            camera_config,
            gate_id=1,
            timestamp_ns=2000,
        )
        state = _state_from(drone_pos, drone_ori)
        assert not mapper.observe(bad_det, state)

    def test_multiple_gates_ordered(
        self, camera_config: CameraConfig, gate_corners_local: np.ndarray
    ) -> None:
        mapper = GateMapper(
            camera_config,
            gate_corners_local,
            GateMapperConfig(min_observations=1),
        )
        gates = {
            10: np.array([5.0, 0.0, 3.0]),
            20: np.array([5.0, 5.0, 3.0]),
        }
        drone_pos = np.array([0.0, 0.0, 3.0])

        for gid, gpos in gates.items():
            drone_ori = _drone_looking_at(drone_pos, gpos)
            det = _render_detection(
                gpos,
                _gate_facing(drone_pos, gpos),
                drone_pos,
                drone_ori,
                gate_corners_local,
                camera_config,
                gate_id=gid,
            )
            state = _state_from(drone_pos, drone_ori)
            assert mapper.observe(det, state)

        waypoints = mapper.waypoints()
        assert {wp.gate_id for wp in waypoints} == {10, 20}
        # Orders are 0, 1 in first-seen order
        assert sorted(wp.order for wp in waypoints) == [0, 1]

    def test_reset_clears_state(
        self, camera_config: CameraConfig, gate_corners_local: np.ndarray
    ) -> None:
        mapper = GateMapper(camera_config, gate_corners_local)
        gate_pos = np.array([0.0, 0.0, 3.0])
        drone_pos = np.array([-5.0, 0.0, 3.0])
        drone_ori = _drone_looking_at(drone_pos, gate_pos)
        det = _render_detection(
            gate_pos,
            _gate_facing(drone_pos, gate_pos),
            drone_pos,
            drone_ori,
            gate_corners_local,
            camera_config,
            gate_id=5,
        )
        state = _state_from(drone_pos, drone_ori)
        mapper.observe(det, state)
        assert mapper.observation_count(5) == 1
        mapper.reset()
        assert mapper.observation_count(5) == 0
        assert mapper.tracked_gate_ids == []
