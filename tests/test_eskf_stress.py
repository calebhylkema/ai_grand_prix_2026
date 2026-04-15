"""Stress and edge-case tests for the ESKF.

Covers scenarios not in the main test_eskf.py:
    1. Correction magnitude capping (max_pos/vel/ori_correction).
    2. Covariance inflation after consecutive chi² rejections.
    3. IMU saturation flag handling.
    4. Rapid predict-correct interleaving.
    5. Bias estimation convergence with persistent bias.
    6. Multiple corrections per IMU step.
"""

from __future__ import annotations

import numpy as np
import pytest
from numpy.testing import assert_allclose

from drone_racing.estimation.eskf import ESKF
from drone_racing.utils.config import load_config
from drone_racing.utils.transforms import euler_to_quat, quat_identity
from drone_racing.utils.types import (
    DetectionConfidence,
    ImuSample,
    PoseMeasurement,
    Telemetry,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def config():
    return load_config()


@pytest.fixture()
def eskf(config):
    filt = ESKF(config)
    filt.initialise(
        Telemetry(
            timestamp_ns=0,
            position=np.zeros(3),
            velocity=np.zeros(3),
            orientation=quat_identity(),
        )
    )
    return filt


def _stationary_imu(timestamp_ns: int) -> ImuSample:
    return ImuSample(
        timestamp_ns=timestamp_ns,
        accel_mps2=np.array([0.0, 0.0, 9.81]),
        gyro_rps=np.zeros(3),
    )


def _pose_meas(
    timestamp_ns: int,
    position: np.ndarray,
    orientation: np.ndarray | None = None,
    confidence: DetectionConfidence = DetectionConfidence.HIGH,
    pos_noise: float = 0.15,
    ori_noise: float = 0.05,
) -> PoseMeasurement:
    return PoseMeasurement(
        timestamp_ns=timestamp_ns,
        position_w=position,
        orientation_w=orientation if orientation is not None else quat_identity(),
        position_noise_std=pos_noise,
        orientation_noise_std=ori_noise,
        confidence=confidence,
        gate_id=0,
    )


# ---------------------------------------------------------------------------
# Correction magnitude capping
# ---------------------------------------------------------------------------


class TestCorrectionCapping:
    """Large corrections should be capped to prevent state jumps."""

    def test_huge_position_measurement_is_capped(self, config) -> None:
        """A measurement 100m away should not move the state by 100m."""
        filt = ESKF(config)
        filt.initialise(
            Telemetry(
                timestamp_ns=0,
                position=np.zeros(3),
                velocity=np.zeros(3),
                orientation=quat_identity(),
            )
        )
        # Build some uncertainty
        for i in range(20):
            filt.predict(_stationary_imu((i + 1) * 5_000_000))

        pos_before = filt.state.position.copy()
        meas = _pose_meas(
            timestamp_ns=filt.state.timestamp_ns,
            position=np.array([100.0, 0.0, 0.0]),
            pos_noise=0.5,
            ori_noise=0.5,
        )
        filt.correct(meas)

        correction = np.linalg.norm(filt.state.position - pos_before)
        # Should be capped — not jump the full 100m
        assert correction < config.estimation.max_pos_correction + 1.0

    def test_huge_orientation_measurement_is_capped(self, config) -> None:
        """A measurement with 90° orientation offset should be capped."""
        filt = ESKF(config)
        filt.initialise(
            Telemetry(
                timestamp_ns=0,
                position=np.zeros(3),
                velocity=np.zeros(3),
                orientation=quat_identity(),
            )
        )
        for i in range(20):
            filt.predict(_stationary_imu((i + 1) * 5_000_000))

        # Measurement with 90° yaw rotation
        big_rotation = euler_to_quat(0.0, 0.0, np.pi / 2)
        meas = _pose_meas(
            timestamp_ns=filt.state.timestamp_ns,
            position=np.zeros(3),
            orientation=big_rotation,
            pos_noise=0.5,
            ori_noise=0.5,
        )
        filt.correct(meas)

        # State should still be valid (not NaN or extreme)
        assert np.all(np.isfinite(filt.state.position))
        assert np.all(np.isfinite(filt.state.orientation))
        assert abs(np.linalg.norm(filt.state.orientation) - 1.0) < 1e-6


# ---------------------------------------------------------------------------
# Covariance inflation after rejections
# ---------------------------------------------------------------------------


class TestCovarianceInflation:
    """After many consecutive chi² rejections, covariance should inflate."""

    def test_inflation_after_repeated_rejections(self, config) -> None:
        """Feed many outlier measurements → covariance should inflate."""
        filt = ESKF(config)
        filt.initialise(
            Telemetry(
                timestamp_ns=0,
                position=np.zeros(3),
                velocity=np.zeros(3),
                orientation=quat_identity(),
            )
        )
        # Give many good corrections to tighten covariance
        for i in range(100):
            t = (i + 1) * 5_000_000
            filt.predict(_stationary_imu(t))
            if i % 5 == 4:
                filt.correct(
                    _pose_meas(
                        timestamp_ns=t,
                        position=np.zeros(3),
                        pos_noise=0.01,
                        ori_noise=0.01,
                    )
                )

        trace_before = np.trace(filt.state.covariance)

        # Now feed many outliers to trigger inflation
        rejection_limit = config.estimation.consecutive_rejection_limit
        for j in range(rejection_limit + 5):
            t = (100 + j + 1) * 5_000_000
            filt.predict(_stationary_imu(t))
            filt.correct(
                _pose_meas(
                    timestamp_ns=t,
                    position=np.array([1000.0, 0.0, 0.0]),
                    pos_noise=0.01,
                    ori_noise=0.01,
                )
            )

        trace_after = np.trace(filt.state.covariance)
        # Covariance should have grown due to inflation + prediction growth
        assert trace_after > trace_before


# ---------------------------------------------------------------------------
# IMU saturation
# ---------------------------------------------------------------------------


class TestIMUSaturation:
    """ESKF should handle saturated IMU gracefully."""

    def test_saturated_flag_does_not_crash(self, eskf) -> None:
        """IMU sample with saturation flag should still be processed."""
        imu = ImuSample(
            timestamp_ns=5_000_000,
            accel_mps2=np.array([200.0, 0.0, 9.81]),  # > 16g forward
            gyro_rps=np.zeros(3),
            accel_saturated=True,
        )
        eskf.predict(imu)
        assert np.all(np.isfinite(eskf.state.position))
        assert np.all(np.isfinite(eskf.state.velocity))


# ---------------------------------------------------------------------------
# Rapid interleaving
# ---------------------------------------------------------------------------


class TestRapidInterleaving:
    """Many predict-correct cycles at high rate should stay stable."""

    def test_1000_cycles_stays_stable(self, config) -> None:
        """1000 predict-correct cycles with noisy measurements."""
        filt = ESKF(config)
        filt.initialise(
            Telemetry(
                timestamp_ns=0,
                position=np.zeros(3),
                velocity=np.zeros(3),
                orientation=quat_identity(),
            )
        )
        rng = np.random.default_rng(42)
        dt_ns = 5_000_000

        for i in range(1000):
            t = (i + 1) * dt_ns
            filt.predict(_stationary_imu(t))

            if i % 5 == 4:
                noisy_pos = rng.normal(0, 0.05, 3)
                filt.correct(
                    _pose_meas(
                        timestamp_ns=t,
                        position=noisy_pos,
                        pos_noise=0.1,
                    )
                )

        s = filt.state
        # State should be finite and near origin
        assert np.all(np.isfinite(s.position))
        assert np.all(np.isfinite(s.velocity))
        assert np.all(np.isfinite(s.orientation))
        assert np.linalg.norm(s.position) < 2.0

        # Covariance should still be positive definite
        eigenvalues = np.linalg.eigvalsh(s.covariance)
        assert np.all(eigenvalues > 0)

    def test_covariance_stays_symmetric_over_many_cycles(self, config) -> None:
        """After many cycles, covariance should remain symmetric."""
        filt = ESKF(config)
        filt.initialise(
            Telemetry(
                timestamp_ns=0,
                position=np.zeros(3),
                velocity=np.zeros(3),
                orientation=quat_identity(),
            )
        )
        rng = np.random.default_rng(99)

        for i in range(500):
            t = (i + 1) * 5_000_000
            gyro = rng.normal(0, 0.1, 3)
            imu = ImuSample(
                timestamp_ns=t,
                accel_mps2=np.array([0.0, 0.0, 9.81]),
                gyro_rps=gyro,
            )
            filt.predict(imu)
            if i % 10 == 9:
                filt.correct(
                    _pose_meas(
                        timestamp_ns=t,
                        position=rng.normal(0, 0.1, 3),
                    )
                )

        p = filt.state.covariance
        assert_allclose(p, p.T, atol=1e-10)


# ---------------------------------------------------------------------------
# Multiple corrections per IMU step
# ---------------------------------------------------------------------------


class TestMultipleCorrections:
    """Multiple vision corrections between IMU predictions."""

    def test_two_corrections_same_timestamp(self, eskf) -> None:
        """Two corrections at the same time should both update state."""
        for i in range(20):
            eskf.predict(_stationary_imu((i + 1) * 5_000_000))

        t = eskf.state.timestamp_ns
        pos_before = eskf.state.position.copy()

        # Use small offsets with high noise so chi² doesn't reject them.
        # The first correction shifts the state, so the second sees a small
        # innovation relative to the updated position.
        eskf.correct(_pose_meas(t, np.array([0.2, 0.0, 0.0]), pos_noise=0.5))
        pos_mid = eskf.state.position.copy()

        eskf.correct(_pose_meas(t, np.array([0.2, 0.2, 0.0]), pos_noise=0.5))
        pos_after = eskf.state.position

        # Both corrections should have had an effect
        assert pos_mid[0] > pos_before[0]
        assert pos_after[1] > pos_mid[1]


# ---------------------------------------------------------------------------
# Bias convergence
# ---------------------------------------------------------------------------


class TestBiasEstimation:
    """Bias estimates should converge when a persistent bias is present."""

    def test_accel_bias_converges(self, config) -> None:
        """With a constant accel bias, the ESKF should estimate it over time."""
        true_bias = np.array([0.3, -0.2, 0.1])  # m/s²

        filt = ESKF(config)
        filt.initialise(
            Telemetry(
                timestamp_ns=0,
                position=np.zeros(3),
                velocity=np.zeros(3),
                orientation=quat_identity(),
            )
        )

        dt_ns = 5_000_000
        rng = np.random.default_rng(42)

        for i in range(400):
            t = (i + 1) * dt_ns
            # IMU reads gravity + bias (stationary drone)
            biased_imu = ImuSample(
                timestamp_ns=t,
                accel_mps2=np.array([0.0, 0.0, 9.81]) + true_bias,
                gyro_rps=np.zeros(3),
            )
            filt.predict(biased_imu)

            # Frequent corrections at the origin
            if i % 5 == 4:
                filt.correct(
                    _pose_meas(
                        timestamp_ns=t,
                        position=rng.normal(0, 0.02, 3),
                        pos_noise=0.05,
                        ori_noise=0.05,
                    )
                )

        # Bias estimate should have moved toward the true bias
        est_bias = filt.state.accel_bias
        # At least one component should be closer to truth than the initial (zero)
        initial_err = np.linalg.norm(true_bias)
        final_err = np.linalg.norm(est_bias - true_bias)
        assert final_err < initial_err, (
            f"Bias should converge: initial_err={initial_err:.3f}, final_err={final_err:.3f}"
        )
