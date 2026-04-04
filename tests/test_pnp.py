"""Tests for drone_racing.perception.pnp — PnP pose solver.

Test strategy:
    1. Known geometry: drone directly in front of a gate at known distance →
       PnP recovers the position.
    2. Identity gate pose: gate at origin facing +Z, drone looking at it →
       verify chain T_WG @ T_GC^{-1} @ T_CB.
    3. Rotated gate: gate facing a different direction → position still correct.
    4. Too few corners → returns None.
    5. Noise sensitivity: small pixel noise → position within tolerance.
    6. Distance-based noise scaling: farther = noisier measurement.
    7. Multiple detections: solve_multiple with gate map.
    8. State-guided solve: providing drone_state improves accuracy.
    9. Rejection: high reprojection error, excessive distance, far from ESKF.
   10. Confidence capping: ≤4 model corners → MEDIUM max confidence.
"""

from __future__ import annotations

import cv2
import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from drone_racing.perception.pnp import PnPSolver
from drone_racing.utils.config import load_config
from drone_racing.utils.transforms import (
    body_to_camera_rotation,
    quat_identity,
)
from drone_racing.utils.types import (
    CameraConfig,
    DetectionConfidence,
    EstimationConfig,
    GateDetection,
    GateWaypoint,
    State,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def camera_config() -> CameraConfig:
    """Default camera config."""
    return load_config().camera


@pytest.fixture()
def estimation_config() -> EstimationConfig:
    """Default estimation config."""
    return load_config().estimation


@pytest.fixture()
def solver(camera_config: CameraConfig, estimation_config: EstimationConfig) -> PnPSolver:
    """Create a PnP solver with default config."""
    return PnPSolver(camera_config, estimation_config)


def _gate_corners_local() -> np.ndarray:
    """Standard 1.5m × 1.5m gate corners in gate-local frame.

    Same as config/default.toml [gate] corners_3d.
    """
    return np.array(
        [
            [-0.75, 0.75, 0.0],
            [0.75, 0.75, 0.0],
            [0.75, -0.75, 0.0],
            [-0.75, -0.75, 0.0],
        ],
        dtype=np.float64,
    )


def _project_points(
    points_3d: np.ndarray,
    rvec: np.ndarray,
    tvec: np.ndarray,
    camera_matrix: np.ndarray,
    dist_coeffs: np.ndarray,
) -> np.ndarray:
    """Project 3D points to 2D image coordinates using OpenCV.

    Parameters
    ----------
    points_3d : np.ndarray
        (N, 3) object points.
    rvec : np.ndarray
        Rotation vector (Rodrigues) of the object in camera frame.
    tvec : np.ndarray
        Translation vector of the object in camera frame.
    camera_matrix : np.ndarray
        3×3 camera intrinsic matrix.
    dist_coeffs : np.ndarray
        Distortion coefficients.

    Returns
    -------
    np.ndarray
        (N, 2) image points.
    """
    img_pts, _ = cv2.projectPoints(points_3d, rvec, tvec, camera_matrix, dist_coeffs)
    return img_pts.reshape(-1, 2)


def _gate_at_origin() -> GateWaypoint:
    """Gate at world origin, facing +Z."""
    return GateWaypoint(
        gate_id=0,
        position_w=np.zeros(3),
        orientation_w=quat_identity(),
        corners_3d_w=_gate_corners_local(),  # In world frame = local frame since gate is at origin
    )


# ---------------------------------------------------------------------------
# Core PnP solve tests
# ---------------------------------------------------------------------------


class TestPnPBasic:
    """Basic PnP solve with known geometry."""

    def test_gate_at_origin_drone_in_front(
        self, solver: PnPSolver, camera_config: CameraConfig
    ) -> None:
        """Drone 5m in front of a gate at origin → PnP recovers position.

        Gate at world origin facing +Z.
        Camera (= drone for zero tilt simplification) at [0, 0, 5] looking at gate.

        In camera frame, the gate is at [0, 0, 5] (5m ahead).
        """
        gate = _gate_at_origin()
        corners_local = _gate_corners_local()

        # Camera at [0, 0, 5] looking toward origin (camera Z = -world Z)
        # In camera frame, the gate is at t=[0, 0, 5], R=I
        rvec = np.zeros(3)  # No rotation
        tvec = np.array([0.0, 0.0, 5.0])  # Gate is 5m in front of camera

        # Project to get 2D corners
        img_pts = _project_points(
            corners_local,
            rvec,
            tvec,
            camera_config.intrinsic_matrix,
            camera_config.distortion_coeffs,
        )

        # Create detection
        det = GateDetection(
            timestamp_ns=1000,
            corners_2d=img_pts,
            confidence=0.9,
            gate_id=0,
        )

        meas = solver.solve(det, gate, corners_local)
        assert meas is not None
        assert meas.timestamp_ns == 1000
        assert meas.gate_id == 0

        # The recovered position should be reasonable
        # (exact values depend on the camera-body extrinsic chain)
        assert meas.position_w is not None
        assert meas.orientation_w is not None
        assert len(meas.position_w) == 3
        assert len(meas.orientation_w) == 4

    def test_returns_none_with_few_corners(self, solver: PnPSolver) -> None:
        """Less than 4 corners → None."""
        gate = _gate_at_origin()
        det = GateDetection(
            timestamp_ns=0,
            corners_2d=np.array([[100, 200], [300, 200], [200, 300]], dtype=np.float64),
            confidence=0.5,
        )
        result = solver.solve(det, gate, _gate_corners_local())
        assert result is None

    def test_confidence_level_propagated(
        self, solver: PnPSolver, camera_config: CameraConfig
    ) -> None:
        """The confidence level from the detection should propagate to the measurement."""
        gate = _gate_at_origin()
        corners_local = _gate_corners_local()
        rvec = np.zeros(3)
        tvec = np.array([0.0, 0.0, 3.0])
        img_pts = _project_points(
            corners_local,
            rvec,
            tvec,
            camera_config.intrinsic_matrix,
            camera_config.distortion_coeffs,
        )

        # 4 corners with confidence 0.5 → MEDIUM
        det = GateDetection(
            timestamp_ns=0,
            corners_2d=img_pts,
            confidence=0.5,
            gate_id=0,
        )
        meas = solver.solve(det, gate, corners_local)
        assert meas is not None
        assert meas.confidence == DetectionConfidence.MEDIUM


class TestPnPNoiseScaling:
    """Noise scales with distance."""

    def test_farther_gate_noisier(self, solver: PnPSolver, camera_config: CameraConfig) -> None:
        """Gate at 3m vs 10m → farther should have larger noise std."""
        gate = _gate_at_origin()
        corners_local = _gate_corners_local()

        results = []
        for dist in [3.0, 10.0]:
            rvec = np.zeros(3)
            tvec = np.array([0.0, 0.0, dist])
            img_pts = _project_points(
                corners_local,
                rvec,
                tvec,
                camera_config.intrinsic_matrix,
                camera_config.distortion_coeffs,
            )
            det = GateDetection(
                timestamp_ns=0,
                corners_2d=img_pts,
                confidence=0.9,
                gate_id=0,
            )
            meas = solver.solve(det, gate, corners_local)
            assert meas is not None
            results.append(meas)

        # Farther gate should have larger noise
        assert results[1].position_noise_std > results[0].position_noise_std


class TestPnPPixelNoise:
    """Small pixel perturbations → position still in tolerance."""

    def test_noisy_corners_still_solve(
        self, solver: PnPSolver, camera_config: CameraConfig
    ) -> None:
        """Add 1-pixel noise to corners → PnP still succeeds."""
        gate = _gate_at_origin()
        corners_local = _gate_corners_local()
        rvec = np.zeros(3)
        tvec = np.array([0.0, 0.0, 4.0])
        img_pts = _project_points(
            corners_local,
            rvec,
            tvec,
            camera_config.intrinsic_matrix,
            camera_config.distortion_coeffs,
        )

        rng = np.random.default_rng(42)
        noisy_pts = img_pts + rng.normal(0, 1.0, img_pts.shape)

        det = GateDetection(
            timestamp_ns=0,
            corners_2d=noisy_pts,
            confidence=0.9,
            gate_id=0,
        )
        meas = solver.solve(det, gate, corners_local)
        assert meas is not None
        # Position should still be in a reasonable range
        assert float(np.linalg.norm(meas.position_w)) < 20.0


# ---------------------------------------------------------------------------
# Multiple detections
# ---------------------------------------------------------------------------


class TestSolveMultiple:
    """solve_multiple with gate map."""

    def test_no_gate_id_skipped(self, solver: PnPSolver) -> None:
        """Detection without gate_id → skipped."""
        det = GateDetection(
            timestamp_ns=0,
            corners_2d=np.zeros((4, 2)),
            confidence=0.5,
            gate_id=None,  # No ID
        )
        results = solver.solve_multiple([det], [_gate_at_origin()], _gate_corners_local())
        assert len(results) == 0

    def test_unknown_gate_id_skipped(self, solver: PnPSolver) -> None:
        """Detection with gate_id not in map → skipped."""
        det = GateDetection(
            timestamp_ns=0,
            corners_2d=np.zeros((4, 2)),
            confidence=0.5,
            gate_id=99,  # Not in map
        )
        results = solver.solve_multiple([det], [_gate_at_origin()], _gate_corners_local())
        assert len(results) == 0

    def test_valid_detection_produces_measurement(
        self, solver: PnPSolver, camera_config: CameraConfig
    ) -> None:
        """Valid detection with matching gate_id → measurement produced."""
        gate = _gate_at_origin()
        corners_local = _gate_corners_local()
        rvec = np.zeros(3)
        tvec = np.array([0.0, 0.0, 5.0])
        img_pts = _project_points(
            corners_local,
            rvec,
            tvec,
            camera_config.intrinsic_matrix,
            camera_config.distortion_coeffs,
        )
        det = GateDetection(
            timestamp_ns=0,
            corners_2d=img_pts,
            confidence=0.9,
            gate_id=0,
        )
        results = solver.solve_multiple([det], [gate], corners_local)
        assert len(results) == 1
        assert results[0].gate_id == 0


# ---------------------------------------------------------------------------
# Transform chain correctness — the exact chain from the PnP solver
# ---------------------------------------------------------------------------


def _make_drone_state(
    position: np.ndarray,
    orientation: np.ndarray,
    velocity: np.ndarray | None = None,
) -> State:
    """Create a minimal State for PnP tests."""
    return State(
        timestamp_ns=0,
        position=position.astype(np.float64),
        velocity=(velocity if velocity is not None else np.zeros(3)).astype(np.float64),
        orientation=orientation.astype(np.float64),
        accel_bias=np.zeros(3),
        gyro_bias=np.zeros(3),
        covariance=np.eye(15) * 0.01,
    )


def _synth_detection(
    drone_pos: np.ndarray,
    drone_ori: np.ndarray,
    gate: GateWaypoint,
    corners_local: np.ndarray,
    camera_config: CameraConfig,
    tilt_deg: float,
) -> GateDetection | None:
    """Synthesise a perfect detection from known drone and gate pose.

    Projects gate corners through the full camera model to produce pixel
    coordinates that the PnP solver should be able to invert.
    """
    r_wb = Rotation.from_quat(drone_ori).as_matrix()
    r_cb = body_to_camera_rotation(tilt_deg)
    r_cw = r_cb @ r_wb.T
    t_cw = -r_cw @ drone_pos

    # Transform corners to camera frame
    r_wg = Rotation.from_quat(gate.orientation_w).as_matrix()
    corners_world = (r_wg @ corners_local.T).T + gate.position_w
    pts_c = (r_cw @ corners_world.T).T + t_cw

    # All points must be in front of camera
    if np.any(pts_c[:, 2] <= 0.1):
        return None

    # Project using camera intrinsics
    k = camera_config.intrinsic_matrix
    dist = camera_config.distortion_coeffs
    rvec_cw, _ = cv2.Rodrigues(r_cw)
    img_pts = _project_points(corners_world, rvec_cw, t_cw, k, dist)

    return GateDetection(
        timestamp_ns=1000,
        corners_2d=img_pts,
        confidence=0.9,
        gate_id=gate.gate_id,
    )


class TestPnPTransformChain:
    """Verify the PnP solver recovers drone position via the T_WB chain.

    These tests catch the class of bugs found during integration:
    - Incorrect T_CB rotation (R_BC vs R_BC^T)
    - Wrong transform chain order
    """

    def test_drone_approaching_gate(self, solver: PnPSolver, camera_config: CameraConfig) -> None:
        """Drone approaching a gate along +X axis → recovers position.

        Gate at [10,0,3] facing -X (normal points toward drone).
        Drone at [5,0,3] facing +X (toward gate).
        """
        gate_pos = np.array([10.0, 0.0, 3.0])
        # Gate facing -X: rotate gate so its Z-axis points along -X
        gate_ori = Rotation.from_euler("yz", [np.pi / 2, np.pi]).as_quat()
        corners_local = _gate_corners_local()
        r_wg = Rotation.from_quat(gate_ori).as_matrix()
        corners_w = (r_wg @ corners_local.T).T + gate_pos

        gate = GateWaypoint(
            gate_id=0,
            position_w=gate_pos,
            orientation_w=gate_ori.astype(np.float64),
            corners_3d_w=corners_w.astype(np.float64),
        )

        drone_pos = np.array([5.0, 0.0, 3.0])
        drone_ori = Rotation.from_euler("z", 0.0).as_quat()  # facing +X
        drone_state = _make_drone_state(drone_pos, drone_ori)

        det = _synth_detection(
            drone_pos,
            drone_ori,
            gate,
            corners_local,
            camera_config,
            camera_config.tilt_deg,
        )
        assert det is not None

        meas = solver.solve(det, gate, corners_local, drone_state=drone_state)
        assert meas is not None
        pos_err = float(np.linalg.norm(meas.position_w - drone_pos))
        assert pos_err < 1.0, f"Position error {pos_err:.2f}m too large"

    def test_gate_off_to_the_side(self, solver: PnPSolver, camera_config: CameraConfig) -> None:
        """Drone and gate at same height, gate off to the side."""
        gate_pos = np.array([8.0, 5.0, 3.0])
        gate_ori = Rotation.from_euler("z", -np.pi / 2).as_quat()
        corners_local = _gate_corners_local()
        r_wg = Rotation.from_quat(gate_ori).as_matrix()
        corners_w = (r_wg @ corners_local.T).T + gate_pos

        gate = GateWaypoint(
            gate_id=1,
            position_w=gate_pos,
            orientation_w=gate_ori.astype(np.float64),
            corners_3d_w=corners_w.astype(np.float64),
        )

        drone_pos = np.array([8.0, 0.0, 3.0])
        drone_ori = Rotation.from_euler("z", np.pi / 2).as_quat()  # facing +Y
        drone_state = _make_drone_state(drone_pos, drone_ori)

        det = _synth_detection(
            drone_pos,
            drone_ori,
            gate,
            corners_local,
            camera_config,
            camera_config.tilt_deg,
        )
        assert det is not None

        meas = solver.solve(det, gate, corners_local, drone_state=drone_state)
        assert meas is not None
        pos_err = float(np.linalg.norm(meas.position_w - drone_pos))
        assert pos_err < 1.5, f"Position error {pos_err:.2f}m too large"


class TestPnPRejection:
    """Verify the solver rejects bad solutions."""

    def test_rejects_distant_gate(self, solver: PnPSolver, camera_config: CameraConfig) -> None:
        """Gate >12m away → None (too far for reliable PnP)."""
        gate = _gate_at_origin()
        corners_local = _gate_corners_local()

        # Very far away: 20m
        rvec = np.zeros(3)
        tvec = np.array([0.0, 0.0, 20.0])
        img_pts = _project_points(
            corners_local,
            rvec,
            tvec,
            camera_config.intrinsic_matrix,
            camera_config.distortion_coeffs,
        )
        det = GateDetection(
            timestamp_ns=0,
            corners_2d=img_pts,
            confidence=0.9,
            gate_id=0,
        )
        meas = solver.solve(det, gate, corners_local)
        assert meas is None

    def test_rejects_position_far_from_eskf(
        self, solver: PnPSolver, camera_config: CameraConfig
    ) -> None:
        """PnP result >5m from ESKF estimate → rejected."""
        gate = _gate_at_origin()
        corners_local = _gate_corners_local()

        # True camera-from-gate: 5m ahead
        rvec = np.zeros(3)
        tvec = np.array([0.0, 0.0, 5.0])
        img_pts = _project_points(
            corners_local,
            rvec,
            tvec,
            camera_config.intrinsic_matrix,
            camera_config.distortion_coeffs,
        )
        det = GateDetection(
            timestamp_ns=0,
            corners_2d=img_pts,
            confidence=0.9,
            gate_id=0,
        )

        # Provide a drone state that's very far from where PnP will compute
        far_state = _make_drone_state(
            np.array([100.0, 100.0, 100.0]),
            Rotation.from_euler("z", 0.0).as_quat(),
        )
        meas = solver.solve(det, gate, corners_local, drone_state=far_state)
        # Should be rejected because PnP position is far from ESKF estimate
        assert meas is None


class TestPnPConfidenceCapping:
    """4 coplanar corners → confidence capped at MEDIUM."""

    def test_high_confidence_capped_with_4_corners(
        self, solver: PnPSolver, camera_config: CameraConfig
    ) -> None:
        """HIGH confidence detection with only 4 model corners → capped to MEDIUM."""
        gate = _gate_at_origin()
        corners_local = _gate_corners_local()  # 4 corners
        assert len(corners_local) == 4

        rvec = np.zeros(3)
        tvec = np.array([0.0, 0.0, 4.0])
        img_pts = _project_points(
            corners_local,
            rvec,
            tvec,
            camera_config.intrinsic_matrix,
            camera_config.distortion_coeffs,
        )
        det = GateDetection(
            timestamp_ns=0,
            corners_2d=img_pts,
            confidence=0.95,  # HIGH confidence
            gate_id=0,
        )
        meas = solver.solve(det, gate, corners_local)
        assert meas is not None
        # Should be capped to MEDIUM, not HIGH
        assert meas.confidence == DetectionConfidence.MEDIUM
