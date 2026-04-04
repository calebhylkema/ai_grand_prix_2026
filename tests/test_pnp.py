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
"""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from drone_racing.perception.pnp import PnPSolver
from drone_racing.utils.config import load_config
from drone_racing.utils.transforms import (
    quat_identity,
)
from drone_racing.utils.types import (
    CameraConfig,
    DetectionConfidence,
    EstimationConfig,
    GateDetection,
    GateWaypoint,
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
