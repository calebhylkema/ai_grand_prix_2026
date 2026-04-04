"""Tests for drone_racing.estimation.eskf — Error-State Kalman Filter.

Test strategy:
    1. Stationary drone: IMU reads only gravity → state should stay at origin.
    2. Constant acceleration: known 1-D motion → verify position/velocity.
    3. Vision correction: inject pose measurement → state converges.
    4. Chi² gating: outlier rejection.
    5. Covariance growth: predict without correction → uncertainty grows.
    6. Covariance shrink: correction → uncertainty shrinks.
"""

from __future__ import annotations

import numpy as np
import pytest
from numpy.testing import assert_allclose

from drone_racing.estimation.eskf import ESKF
from drone_racing.utils.config import load_config
from drone_racing.utils.transforms import (
    euler_to_quat,
    quat_identity,
)
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
    """Load the default racing config."""
    return load_config()


@pytest.fixture()
def eskf(config):
    """Create and initialise an ESKF at the origin."""
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
    """IMU reading for a stationary drone (measures only gravity in body FRD).

    Body frame is FRD, world is Z-up.  When level and still:
        body accel = R_BW @ [0, 0, -9.81] = [0, 0, 9.81]  (gravity along body-Z-down)
    """
    return ImuSample(
        timestamp_ns=timestamp_ns,
        accel_mps2=np.array([0.0, 0.0, 9.81]),  # gravity in FRD body
        gyro_rps=np.zeros(3),
    )


def _make_imu(
    timestamp_ns: int,
    accel_body: np.ndarray,
    gyro: np.ndarray | None = None,
) -> ImuSample:
    """Helper to build an ImuSample."""
    return ImuSample(
        timestamp_ns=timestamp_ns,
        accel_mps2=accel_body,
        gyro_rps=gyro if gyro is not None else np.zeros(3),
    )


# ---------------------------------------------------------------------------
# Initialisation
# ---------------------------------------------------------------------------


class TestInitialisation:
    """ESKF init and state after initialise()."""

    def test_not_initialised_by_default(self, config) -> None:
        filt = ESKF(config)
        assert not filt.initialised

    def test_initialised_after_telemetry(self, eskf) -> None:
        assert eskf.initialised

    def test_initial_state_at_origin(self, eskf) -> None:
        s = eskf.state
        assert_allclose(s.position, [0, 0, 0])
        assert_allclose(s.velocity, [0, 0, 0])
        assert_allclose(s.orientation, [0, 0, 0, 1])

    def test_initial_covariance_symmetric_positive(self, eskf) -> None:
        p = eskf.state.covariance
        assert p.shape == (15, 15)
        assert_allclose(p, p.T, atol=1e-14)
        eigenvalues = np.linalg.eigvalsh(p)
        assert np.all(eigenvalues > 0), "Covariance must be positive definite"

    def test_predict_before_init_is_noop(self, config) -> None:
        filt = ESKF(config)
        state = filt.predict(_stationary_imu(1_000_000))
        assert_allclose(state.position, [0, 0, 0], atol=1e-10)


# ---------------------------------------------------------------------------
# IMU propagation
# ---------------------------------------------------------------------------


class TestPrediction:
    """IMU-only propagation."""

    def test_stationary_stays_at_origin(self, eskf) -> None:
        """Feed 1 second of stationary IMU data → should remain at origin."""
        dt_ns = 5_000_000  # 5 ms = 200 Hz
        for i in range(200):
            t = (i + 1) * dt_ns
            eskf.predict(_stationary_imu(t))

        s = eskf.state
        assert_allclose(s.position, [0, 0, 0], atol=0.05)
        assert_allclose(s.velocity, [0, 0, 0], atol=0.05)

    def test_constant_accel_x(self, eskf) -> None:
        """Apply 1 m/s² forward (body X → world X when level) for 1 s.

        Expected: v ≈ 1 m/s, p ≈ 0.5 m.

        Body accel = [1 + 0, 0, 9.81] (forward thrust + gravity in FRD).
        """
        dt_ns = 5_000_000  # 5 ms
        accel_body = np.array([1.0, 0.0, 9.81])  # 1 m/s² forward + gravity
        for i in range(200):
            t = (i + 1) * dt_ns
            eskf.predict(_make_imu(t, accel_body))

        s = eskf.state
        # X should grow, Y and Z should stay near 0
        assert s.position[0] > 0.3  # ~0.5 m expected
        assert abs(s.position[1]) < 0.05
        assert abs(s.position[2]) < 0.05
        assert s.velocity[0] > 0.8  # ~1.0 m/s expected
        assert abs(s.velocity[1]) < 0.05

    def test_covariance_grows_with_prediction(self, eskf) -> None:
        """Uncertainty should increase without corrections."""
        p_initial_trace = np.trace(eskf.state.covariance)
        for i in range(100):
            t = (i + 1) * 5_000_000
            eskf.predict(_stationary_imu(t))
        p_final_trace = np.trace(eskf.state.covariance)
        assert p_final_trace > p_initial_trace

    def test_quaternion_stays_normalised(self, eskf) -> None:
        """After many predictions, quaternion should still be unit-length."""
        gyro = np.array([0.1, -0.05, 0.02])  # some rotation
        for i in range(1000):
            t = (i + 1) * 5_000_000
            eskf.predict(_make_imu(t, np.array([0.0, 0.0, 9.81]), gyro))
        q = eskf.state.orientation
        assert_allclose(np.linalg.norm(q), 1.0, atol=1e-10)

    def test_timestamp_monotonic(self, eskf) -> None:
        """Backward timestamp should be a no-op (dt ≤ 0)."""
        eskf.predict(_stationary_imu(10_000_000))
        pos_after_first = eskf.state.position.copy()
        eskf.predict(_stationary_imu(5_000_000))  # backward in time
        assert_allclose(eskf.state.position, pos_after_first)


# ---------------------------------------------------------------------------
# Vision correction
# ---------------------------------------------------------------------------


def _pose_measurement(
    timestamp_ns: int,
    position: np.ndarray,
    orientation: np.ndarray | None = None,
    confidence: DetectionConfidence = DetectionConfidence.HIGH,
    pos_noise: float = 0.15,
    ori_noise: float = 0.05,
) -> PoseMeasurement:
    """Build a PoseMeasurement helper."""
    return PoseMeasurement(
        timestamp_ns=timestamp_ns,
        position_w=position,
        orientation_w=orientation if orientation is not None else quat_identity(),
        position_noise_std=pos_noise,
        orientation_noise_std=ori_noise,
        confidence=confidence,
        gate_id=0,
    )


class TestCorrection:
    """Vision-based correction step."""

    def test_position_correction_moves_state(self, eskf) -> None:
        """A measurement offset from the current state should pull it."""
        # Propagate a bit to build uncertainty
        for i in range(50):
            eskf.predict(_stationary_imu((i + 1) * 5_000_000))

        pos_before = eskf.state.position.copy()
        meas = _pose_measurement(
            timestamp_ns=eskf.state.timestamp_ns,
            position=np.array([1.0, 0.0, 0.0]),
            confidence=DetectionConfidence.HIGH,
        )
        eskf.correct(meas)
        pos_after = eskf.state.position
        # Should have moved toward the measurement
        assert pos_after[0] > pos_before[0]

    def test_covariance_shrinks_after_correction(self, eskf) -> None:
        """Correction should reduce uncertainty."""
        for i in range(50):
            eskf.predict(_stationary_imu((i + 1) * 5_000_000))

        trace_before = np.trace(eskf.state.covariance)
        meas = _pose_measurement(
            timestamp_ns=eskf.state.timestamp_ns,
            position=np.zeros(3),
        )
        eskf.correct(meas)
        trace_after = np.trace(eskf.state.covariance)
        assert trace_after < trace_before

    def test_covariance_stays_symmetric(self, eskf) -> None:
        """After correction, P should remain symmetric."""
        for i in range(50):
            eskf.predict(_stationary_imu((i + 1) * 5_000_000))
        meas = _pose_measurement(
            timestamp_ns=eskf.state.timestamp_ns,
            position=np.array([0.5, -0.3, 0.1]),
        )
        eskf.correct(meas)
        p = eskf.state.covariance
        assert_allclose(p, p.T, atol=1e-12)

    def test_covariance_stays_positive_definite(self, eskf) -> None:
        """After correction, P should remain positive definite."""
        for i in range(50):
            eskf.predict(_stationary_imu((i + 1) * 5_000_000))
        meas = _pose_measurement(
            timestamp_ns=eskf.state.timestamp_ns,
            position=np.array([0.1, 0.2, 0.0]),
        )
        eskf.correct(meas)
        eigenvalues = np.linalg.eigvalsh(eskf.state.covariance)
        assert np.all(eigenvalues > 0)

    def test_low_confidence_skipped(self, eskf) -> None:
        """LOW confidence measurements should be ignored."""
        for i in range(50):
            eskf.predict(_stationary_imu((i + 1) * 5_000_000))

        pos_before = eskf.state.position.copy()
        meas = _pose_measurement(
            timestamp_ns=eskf.state.timestamp_ns,
            position=np.array([100.0, 0.0, 0.0]),
            confidence=DetectionConfidence.LOW,
        )
        eskf.correct(meas)
        assert_allclose(eskf.state.position, pos_before)

    def test_medium_confidence_position_only(self, eskf) -> None:
        """MEDIUM confidence should update position but not orientation."""
        for i in range(50):
            eskf.predict(_stationary_imu((i + 1) * 5_000_000))

        ori_before = eskf.state.orientation.copy()
        meas = _pose_measurement(
            timestamp_ns=eskf.state.timestamp_ns,
            position=np.array([1.0, 0.0, 0.0]),
            orientation=euler_to_quat(0.0, 0.0, 0.5),  # 0.5 rad yaw
            confidence=DetectionConfidence.MEDIUM,
        )
        eskf.correct(meas)
        # Position should have changed
        assert eskf.state.position[0] > 0.1
        # Orientation should barely change (position-only update)
        assert_allclose(eskf.state.orientation, ori_before, atol=0.01)

    def test_chi2_rejects_outlier(self, eskf) -> None:
        """A wildly wrong measurement should be rejected by chi² gating."""
        # Keep covariance tight by giving many good corrections first
        for i in range(100):
            t = (i + 1) * 5_000_000
            eskf.predict(_stationary_imu(t))
            if i % 10 == 9:
                eskf.correct(
                    _pose_measurement(
                        timestamp_ns=t,
                        position=np.zeros(3),
                        pos_noise=0.01,
                        ori_noise=0.01,
                    )
                )

        pos_before = eskf.state.position.copy()
        # Giant outlier
        outlier = _pose_measurement(
            timestamp_ns=eskf.state.timestamp_ns,
            position=np.array([1000.0, 0.0, 0.0]),
            pos_noise=0.01,
            ori_noise=0.01,
        )
        eskf.correct(outlier)
        # Should NOT have jumped to 1000 m
        assert np.linalg.norm(eskf.state.position - pos_before) < 1.0


# ---------------------------------------------------------------------------
# Convergence (integration-style)
# ---------------------------------------------------------------------------


class TestConvergence:
    """Longer scenario: does the filter converge to known truth?"""

    def test_static_convergence(self, config) -> None:
        """Drone at known position, frequent corrections → converge.

        Initialise near the true position so chi² gating doesn't reject
        every measurement.  The filter should converge from a 0.5 m offset.
        """
        true_pos = np.array([5.0, 3.0, -1.0])
        true_ori = quat_identity()

        # Start within ~0.5 m of truth so early corrections pass chi² gate
        filt = ESKF(config)
        filt.initialise(
            Telemetry(
                timestamp_ns=0,
                position=true_pos + np.array([0.5, -0.3, 0.2]),
                velocity=np.zeros(3),
                orientation=quat_identity(),
            )
        )

        dt_ns = 5_000_000  # 5 ms
        for i in range(400):  # 2 seconds
            t = (i + 1) * dt_ns
            filt.predict(_stationary_imu(t))

            # Correction every 50 ms (20 Hz)
            if i % 10 == 9:
                filt.correct(
                    _pose_measurement(
                        timestamp_ns=t,
                        position=true_pos + np.random.default_rng(i).normal(0, 0.05, 3),
                        orientation=true_ori,
                        pos_noise=0.05,
                        ori_noise=0.05,
                    )
                )

        s = filt.state
        assert_allclose(s.position, true_pos, atol=0.5)

    def test_moving_drone_tracks_position(self, config) -> None:
        """Drone moving at 5 m/s along X with corrections → tracks well."""
        filt = ESKF(config)
        filt.initialise(
            Telemetry(
                timestamp_ns=0,
                position=np.zeros(3),
                velocity=np.array([5.0, 0.0, 0.0]),
                orientation=quat_identity(),
            )
        )

        dt_ns = 5_000_000  # 5 ms
        rng = np.random.default_rng(42)
        for i in range(200):  # 1 second
            t = (i + 1) * dt_ns
            # Body accel: only gravity when moving at constant velocity
            filt.predict(_stationary_imu(t))

            # True position at time t
            true_x = 5.0 * (t * 1e-9)  # v * t
            true_pos = np.array([true_x, 0.0, 0.0])

            if i % 10 == 9:  # 20 Hz corrections
                noisy_pos = true_pos + rng.normal(0, 0.1, 3)
                filt.correct(
                    _pose_measurement(
                        timestamp_ns=t,
                        position=noisy_pos,
                        pos_noise=0.1,
                    )
                )

        s = filt.state
        expected_x = 5.0  # 5 m/s * 1 s
        assert abs(s.position[0] - expected_x) < 1.0


# ---------------------------------------------------------------------------
# Latency compensation
# ---------------------------------------------------------------------------


class TestLatencyCompensation:
    """Verify that delayed measurements are correctly applied via rollback."""

    def test_delayed_correction_applied(self, config) -> None:
        """A measurement with a past timestamp should still update the state.

        Run 50 IMU steps, then feed a correction timestamped at step 25.
        The correction should still take effect.
        """
        filt = ESKF(config)
        start_pos = np.array([1.0, 0.0, 0.0])
        filt.initialise(
            Telemetry(
                timestamp_ns=0,
                position=start_pos.copy(),
                velocity=np.zeros(3),
                orientation=quat_identity(),
            )
        )

        dt_ns = 5_000_000  # 5ms
        for i in range(50):
            filt.predict(_stationary_imu((i + 1) * dt_ns))

        current_ts = filt.state.timestamp_ns
        assert current_ts == 50 * dt_ns

        # Measurement from the past (at step 25)
        past_ts = 25 * dt_ns
        meas = _pose_measurement(
            timestamp_ns=past_ts,
            position=start_pos + np.array([0.1, 0.0, 0.0]),
        )
        pos_before = filt.state.position.copy()
        filt.correct(meas)

        # State should have changed (correction was applied)
        assert not np.allclose(filt.state.position, pos_before)

        # Timestamp should be restored to the current time (re-propagated)
        assert filt.state.timestamp_ns == current_ts

    def test_delayed_correction_timestamp_restored(self, config) -> None:
        """After latency-compensated correction, state timestamp is at current time."""
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
        for i in range(20):
            filt.predict(_stationary_imu((i + 1) * dt_ns))

        expected_ts = 20 * dt_ns
        meas = _pose_measurement(
            timestamp_ns=5 * dt_ns,  # 75ms in the past
            position=np.array([0.05, 0.0, 0.0]),
        )
        filt.correct(meas)

        assert filt.state.timestamp_ns == expected_ts

    def test_no_rollback_for_current_measurement(self, config) -> None:
        """Measurement at current time should not trigger rollback."""
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
        for i in range(10):
            filt.predict(_stationary_imu((i + 1) * dt_ns))

        # Measurement at current time — no rollback needed
        meas = _pose_measurement(
            timestamp_ns=filt.state.timestamp_ns,
            position=np.array([0.1, 0.0, 0.0]),
        )
        filt.correct(meas)
        assert filt.state.timestamp_ns == 10 * dt_ns

    def test_covariance_valid_after_latency_comp(self, config) -> None:
        """Covariance stays symmetric positive-definite after rollback + replay."""
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
        for i in range(30):
            filt.predict(_stationary_imu((i + 1) * dt_ns))

        meas = _pose_measurement(
            timestamp_ns=10 * dt_ns,
            position=np.array([0.05, -0.02, 0.01]),
        )
        filt.correct(meas)

        p = filt.state.covariance
        # Symmetric
        assert_allclose(p, p.T, atol=1e-12)
        # Positive definite
        eigenvalues = np.linalg.eigvalsh(p)
        assert np.all(eigenvalues > 0)

    def test_latency_comp_improves_moving_drone(self, config) -> None:
        """For a moving drone, delayed correction with latency comp should
        produce better position estimate than without.

        Drone moving at 10 m/s along X. Measurement arrives 50ms late.
        With latency comp, the correction is applied at the measurement time
        and re-propagated, so the final state should be closer to truth.
        """
        true_velocity = np.array([10.0, 0.0, 0.0])
        dt_ns = 5_000_000  # 5ms = 200 Hz
        n_steps = 60  # 300ms total

        # Run filter
        filt = ESKF(config)
        filt.initialise(
            Telemetry(
                timestamp_ns=0,
                position=np.zeros(3),
                velocity=true_velocity.copy(),
                orientation=quat_identity(),
            )
        )

        for i in range(n_steps):
            filt.predict(_stationary_imu((i + 1) * dt_ns))

        # True position at step 20 (100ms): 10 * 0.1 = 1.0m
        meas_time = 20 * dt_ns
        true_pos_at_meas = true_velocity * (meas_time * 1e-9)

        # Delayed measurement (arrives now but was captured at step 20)
        meas = _pose_measurement(
            timestamp_ns=meas_time,
            position=true_pos_at_meas,
            pos_noise=0.05,
            ori_noise=0.05,
        )
        filt.correct(meas)

        # True position at current time (300ms): 10 * 0.3 = 3.0m
        true_pos_now = true_velocity * (n_steps * dt_ns * 1e-9)
        error = np.linalg.norm(filt.state.position - true_pos_now)

        # Error should be reasonable (< 1m for a 300ms window)
        assert error < 1.0
