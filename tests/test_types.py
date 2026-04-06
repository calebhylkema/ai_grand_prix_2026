"""Tests for drone_racing.utils.types — data classes and config models."""

from __future__ import annotations

import numpy as np
from numpy.testing import assert_allclose

from drone_racing.utils.types import (
    CameraConfig,
    DetectionConfidence,
    EstimationConfig,
    GateConfig,
    GateDetection,
    State,
)


class TestState:
    """State dataclass."""

    def test_identity_defaults(self) -> None:
        s = State.identity(timestamp_ns=1000)
        assert s.timestamp_ns == 1000
        assert_allclose(s.position, [0, 0, 0])
        assert_allclose(s.velocity, [0, 0, 0])
        assert_allclose(s.orientation, [0, 0, 0, 1])
        assert_allclose(s.accel_bias, [0, 0, 0])
        assert_allclose(s.gyro_bias, [0, 0, 0])
        assert s.covariance.shape == (15, 15)

    def test_identity_covariance_is_symmetric(self) -> None:
        s = State.identity()
        assert_allclose(s.covariance, s.covariance.T)


class TestGateDetection:
    """GateDetection confidence mapping."""

    def test_high_confidence(self) -> None:
        det = GateDetection(
            timestamp_ns=0,
            corners_2d=np.zeros((8, 2)),
            confidence=0.9,
        )
        assert det.confidence_level == DetectionConfidence.HIGH

    def test_medium_confidence(self) -> None:
        det = GateDetection(
            timestamp_ns=0,
            corners_2d=np.zeros((4, 2)),
            confidence=0.5,
        )
        assert det.confidence_level == DetectionConfidence.MEDIUM

    def test_low_confidence_few_corners(self) -> None:
        det = GateDetection(
            timestamp_ns=0,
            corners_2d=np.zeros((1, 2)),
            confidence=0.9,
        )
        assert det.confidence_level == DetectionConfidence.LOW

    def test_low_confidence_low_score(self) -> None:
        det = GateDetection(
            timestamp_ns=0,
            corners_2d=np.zeros((4, 2)),
            confidence=0.2,
        )
        assert det.confidence_level == DetectionConfidence.LOW


class TestCameraConfig:
    """CameraConfig property methods."""

    def test_intrinsic_matrix_shape(self) -> None:
        cfg = CameraConfig()
        k = cfg.intrinsic_matrix
        assert k.shape == (3, 3)
        assert_allclose(k[0, 0], 600.0)
        assert_allclose(k[1, 1], 600.0)
        assert_allclose(k[0, 2], 640.0)
        assert_allclose(k[1, 2], 360.0)

    def test_distortion_coeffs(self) -> None:
        cfg = CameraConfig()
        d = cfg.distortion_coeffs
        assert d.shape == (5,)
        assert_allclose(d, 0.0)


class TestGateConfig:
    """GateConfig corner array."""

    def test_corners_3d_array_shape(self) -> None:
        cfg = GateConfig()
        corners = cfg.corners_3d_array
        assert corners.shape == (8, 3)

    def test_corners_centred_at_origin(self) -> None:
        cfg = GateConfig()
        centroid = cfg.corners_3d_array.mean(axis=0)
        assert_allclose(centroid, [0, 0, 0], atol=1e-14)


class TestEstimationConfig:
    """EstimationConfig default values match config/default.toml."""

    def test_defaults(self) -> None:
        cfg = EstimationConfig()
        assert cfg.imu_rate_hz == 200.0
        assert cfg.gyro_noise_std == 0.01
        assert cfg.accel_noise_std == 0.1
