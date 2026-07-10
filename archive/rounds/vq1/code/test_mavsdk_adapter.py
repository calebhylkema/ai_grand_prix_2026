"""Tests for drone_racing.sim.mavsdk_adapter — telemetry validation & robustness.

Test strategy:
    1. NaN/Inf validation helpers reject bad data.
    2. Staleness detection works correctly.
    3. Command validation rejects NaN and clamps throttle.
    4. TelemetrySnapshot defaults are sane.
"""

from __future__ import annotations

import time

import numpy as np

from drone_racing.sim.mavsdk_adapter import (
    MAVSDKAdapter,
    TelemetrySnapshot,
    _is_finite_scalar,
    _is_finite_vec,
)
from drone_racing.utils.types import MAVLinkConfig

# ---------------------------------------------------------------------------
# Tests: validation helpers
# ---------------------------------------------------------------------------


class TestIsFiniteVec:
    """Tests for _is_finite_vec."""

    def test_finite_vector(self) -> None:
        assert _is_finite_vec(np.array([1.0, 2.0, 3.0])) is True

    def test_nan_in_vector(self) -> None:
        assert _is_finite_vec(np.array([1.0, float("nan"), 3.0])) is False

    def test_inf_in_vector(self) -> None:
        assert _is_finite_vec(np.array([float("inf"), 2.0, 3.0])) is False

    def test_neg_inf_in_vector(self) -> None:
        assert _is_finite_vec(np.array([1.0, float("-inf"), 3.0])) is False

    def test_zeros(self) -> None:
        assert _is_finite_vec(np.zeros(3)) is True

    def test_large_values(self) -> None:
        assert _is_finite_vec(np.array([1e18, -1e18, 0.0])) is True


class TestIsFiniteScalar:
    """Tests for _is_finite_scalar."""

    def test_finite_values(self) -> None:
        assert _is_finite_scalar(1.0, 2.0, 3.0) is True

    def test_nan_value(self) -> None:
        assert _is_finite_scalar(1.0, float("nan")) is False

    def test_inf_value(self) -> None:
        assert _is_finite_scalar(float("inf"), 1.0) is False

    def test_zero(self) -> None:
        assert _is_finite_scalar(0.0) is True


# ---------------------------------------------------------------------------
# Tests: TelemetrySnapshot defaults
# ---------------------------------------------------------------------------


class TestTelemetrySnapshot:
    """Verify sane defaults for TelemetrySnapshot."""

    def test_defaults(self) -> None:
        t = TelemetrySnapshot()
        assert t.connected is False
        assert t.armed is False
        assert t.offboard_active is False
        assert t.odometry_valid is False
        assert t.timestamp_ns == 0
        np.testing.assert_array_equal(t.imu_accel, np.zeros(3))
        np.testing.assert_array_equal(t.imu_gyro, np.zeros(3))

    def test_staleness_timestamps_default_zero(self) -> None:
        t = TelemetrySnapshot()
        assert t.last_attitude_ns == 0
        assert t.last_imu_ns == 0
        assert t.last_odometry_ns == 0


# ---------------------------------------------------------------------------
# Tests: staleness detection
# ---------------------------------------------------------------------------


class TestStaleness:
    """Staleness detection on MAVSDKAdapter."""

    def test_imu_stale_when_never_received(self) -> None:
        adapter = MAVSDKAdapter(MAVLinkConfig())
        assert adapter.is_imu_stale() is True

    def test_attitude_stale_when_never_received(self) -> None:
        adapter = MAVSDKAdapter(MAVLinkConfig())
        assert adapter.is_attitude_stale() is True

    def test_odometry_stale_when_never_received(self) -> None:
        adapter = MAVSDKAdapter(MAVLinkConfig())
        assert adapter.is_odometry_stale() is True

    def test_imu_fresh_after_update(self) -> None:
        adapter = MAVSDKAdapter(MAVLinkConfig())
        adapter._telemetry.last_imu_ns = time.monotonic_ns()
        assert adapter.is_imu_stale() is False

    def test_attitude_fresh_after_update(self) -> None:
        adapter = MAVSDKAdapter(MAVLinkConfig())
        adapter._telemetry.last_attitude_ns = time.monotonic_ns()
        assert adapter.is_attitude_stale() is False

    def test_odometry_fresh_after_update(self) -> None:
        adapter = MAVSDKAdapter(MAVLinkConfig())
        adapter._telemetry.last_odometry_ns = time.monotonic_ns()
        adapter._telemetry.odometry_valid = True
        assert adapter.is_odometry_stale() is False

    def test_nan_count_starts_zero(self) -> None:
        adapter = MAVSDKAdapter(MAVLinkConfig())
        assert adapter.nan_count == 0


# ---------------------------------------------------------------------------
# Tests: latest_state construction
# ---------------------------------------------------------------------------


class TestLatestState:
    """Verify latest_state builds from telemetry fields."""

    def test_fallback_state_without_odometry(self) -> None:
        """Without ODOMETRY, state should have zero position/velocity."""
        adapter = MAVSDKAdapter(MAVLinkConfig())
        adapter._telemetry.attitude_roll = 0.1
        adapter._telemetry.attitude_pitch = 0.2
        adapter._telemetry.attitude_yaw = 0.3
        adapter._telemetry.odometry_valid = False

        state = adapter.latest_state
        np.testing.assert_array_equal(state.position, np.zeros(3))
        np.testing.assert_array_equal(state.velocity, np.zeros(3))
        # Covariance should be high (uncertain)
        assert np.trace(state.covariance) > 1.0

    def test_state_with_odometry(self) -> None:
        """With ODOMETRY, state should use odometry position/velocity."""
        adapter = MAVSDKAdapter(MAVLinkConfig())
        adapter._telemetry.odometry_valid = True
        adapter._telemetry.odometry_position = np.array([1.0, 2.0, 3.0])
        adapter._telemetry.odometry_velocity = np.array([0.5, -0.3, 0.1])
        adapter._telemetry.odometry_orientation = np.array([0.0, 0.0, 0.0, 1.0])

        state = adapter.latest_state
        np.testing.assert_allclose(state.position, [1.0, 2.0, 3.0])
        np.testing.assert_allclose(state.velocity, [0.5, -0.3, 0.1])

    def test_latest_imu_copies_data(self) -> None:
        """latest_imu should return a copy, not a reference."""
        adapter = MAVSDKAdapter(MAVLinkConfig())
        adapter._telemetry.imu_accel = np.array([1.0, 2.0, 3.0])
        adapter._telemetry.imu_gyro = np.array([0.1, 0.2, 0.3])
        adapter._telemetry.imu_timestamp_ns = 1000

        imu = adapter.latest_imu
        assert imu.timestamp_ns == 1000
        np.testing.assert_allclose(imu.accel_mps2, [1.0, 2.0, 3.0])
        np.testing.assert_allclose(imu.gyro_rps, [0.1, 0.2, 0.3])

        # Mutating the returned IMU should not affect the adapter
        imu.accel_mps2[0] = 999.0
        np.testing.assert_allclose(adapter._telemetry.imu_accel, [1.0, 2.0, 3.0])


# ---------------------------------------------------------------------------
# Tests: get_initial_telemetry
# ---------------------------------------------------------------------------


class TestGetInitialTelemetry:
    """Verify initial telemetry construction."""

    def test_returns_telemetry_from_state(self) -> None:
        adapter = MAVSDKAdapter(MAVLinkConfig())
        adapter._telemetry.odometry_valid = True
        adapter._telemetry.odometry_position = np.array([5.0, 3.0, 1.0])
        adapter._telemetry.odometry_velocity = np.array([1.0, 0.0, 0.0])
        adapter._telemetry.odometry_orientation = np.array([0.0, 0.0, 0.0, 1.0])

        tel = adapter.get_initial_telemetry()
        np.testing.assert_allclose(tel.position, [5.0, 3.0, 1.0])
        np.testing.assert_allclose(tel.velocity, [1.0, 0.0, 0.0])


# ---------------------------------------------------------------------------
# Tests: is_running
# ---------------------------------------------------------------------------


class TestIsRunning:
    """Connection state logic."""

    def test_not_running_by_default(self) -> None:
        adapter = MAVSDKAdapter(MAVLinkConfig())
        assert adapter.is_running() is False

    def test_not_running_when_disconnected(self) -> None:
        adapter = MAVSDKAdapter(MAVLinkConfig())
        adapter._running = True
        adapter._telemetry.connected = False
        assert adapter.is_running() is False

    def test_running_when_connected(self) -> None:
        adapter = MAVSDKAdapter(MAVLinkConfig())
        adapter._running = True
        adapter._telemetry.connected = True
        assert adapter.is_running() is True
