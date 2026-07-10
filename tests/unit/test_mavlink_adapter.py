"""Tests for the pymavlink-based sim adapter — no live sim required.

Covers the same surfaces as the older MAVSDKAdapter tests (validation
helpers, snapshot defaults, staleness detection, state construction,
command validation) and adds coverage for the sim-specific message
dispatch (RACE_STATUS, TRACK_INFO, COLLISION).
"""

from __future__ import annotations

import struct
import time
from types import SimpleNamespace

import numpy as np
import pytest

from drone_racing.sim.mavlink_adapter import (
    MAVLinkAdapter,
    TelemetrySnapshot,
    _is_finite_scalar,
    _is_finite_vec,
)
from drone_racing.utils.types import MAVLinkConfig


# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------
class TestIsFiniteVec:
    def test_finite(self) -> None:
        assert _is_finite_vec(np.array([1.0, 2.0, 3.0])) is True

    def test_nan(self) -> None:
        assert _is_finite_vec(np.array([1.0, float("nan"), 3.0])) is False

    def test_inf(self) -> None:
        assert _is_finite_vec(np.array([float("inf"), 0.0, 0.0])) is False


class TestIsFiniteScalar:
    def test_finite(self) -> None:
        assert _is_finite_scalar(1.0, 2.0, 3.0) is True

    def test_nan(self) -> None:
        assert _is_finite_scalar(1.0, float("nan")) is False


# ---------------------------------------------------------------------------
# TelemetrySnapshot defaults
# ---------------------------------------------------------------------------
class TestTelemetrySnapshot:
    def test_defaults(self) -> None:
        t = TelemetrySnapshot()
        assert t.connected is False
        assert t.armed is False
        assert t.offboard_active is False
        assert t.odometry_valid is False
        np.testing.assert_array_equal(t.imu_accel, np.zeros(3))
        np.testing.assert_array_equal(t.imu_gyro, np.zeros(3))


# ---------------------------------------------------------------------------
# Staleness
# ---------------------------------------------------------------------------
class TestStaleness:
    def test_imu_stale_when_never_received(self) -> None:
        adapter = MAVLinkAdapter(MAVLinkConfig())
        assert adapter.is_imu_stale() is True

    def test_attitude_stale_when_never_received(self) -> None:
        adapter = MAVLinkAdapter(MAVLinkConfig())
        assert adapter.is_attitude_stale() is True

    def test_odometry_stale_when_never_received(self) -> None:
        adapter = MAVLinkAdapter(MAVLinkConfig())
        assert adapter.is_odometry_stale() is True

    def test_imu_fresh_after_update(self) -> None:
        adapter = MAVLinkAdapter(MAVLinkConfig())
        adapter._telemetry.last_imu_ns = time.monotonic_ns()
        assert adapter.is_imu_stale() is False

    def test_odometry_fresh_after_update(self) -> None:
        adapter = MAVLinkAdapter(MAVLinkConfig())
        adapter._telemetry.last_odometry_ns = time.monotonic_ns()
        adapter._telemetry.odometry_valid = True
        assert adapter.is_odometry_stale() is False


# ---------------------------------------------------------------------------
# State construction
# ---------------------------------------------------------------------------
class TestLatestState:
    def test_fallback_state_without_odometry(self) -> None:
        adapter = MAVLinkAdapter(MAVLinkConfig())
        adapter._telemetry.attitude_roll = 0.1
        adapter._telemetry.attitude_pitch = 0.2
        adapter._telemetry.attitude_yaw = 0.3

        state = adapter.latest_state
        np.testing.assert_array_equal(state.position, np.zeros(3))
        np.testing.assert_array_equal(state.velocity, np.zeros(3))
        assert np.trace(state.covariance) > 1.0

    def test_state_with_odometry(self) -> None:
        adapter = MAVLinkAdapter(MAVLinkConfig())
        adapter._telemetry.odometry_valid = True
        adapter._telemetry.odometry_position = np.array([1.0, 2.0, 3.0])
        adapter._telemetry.odometry_velocity = np.array([0.5, -0.3, 0.1])
        adapter._telemetry.odometry_orientation = np.array([0.0, 0.0, 0.0, 1.0])

        state = adapter.latest_state
        np.testing.assert_allclose(state.position, [1.0, 2.0, 3.0])
        np.testing.assert_allclose(state.velocity, [0.5, -0.3, 0.1])

    def test_latest_imu_returns_copy(self) -> None:
        adapter = MAVLinkAdapter(MAVLinkConfig())
        adapter._telemetry.imu_accel = np.array([1.0, 2.0, 3.0])
        adapter._telemetry.imu_gyro = np.array([0.1, 0.2, 0.3])
        adapter._telemetry.imu_timestamp_ns = 1000

        imu = adapter.latest_imu
        imu.accel_mps2[0] = 999.0
        np.testing.assert_allclose(adapter._telemetry.imu_accel, [1.0, 2.0, 3.0])


class TestIsRunning:
    def test_not_running_by_default(self) -> None:
        adapter = MAVLinkAdapter(MAVLinkConfig())
        assert adapter.is_running() is False

    def test_running_when_connected(self) -> None:
        adapter = MAVLinkAdapter(MAVLinkConfig())
        adapter._running = True
        adapter._telemetry.connected = True
        assert adapter.is_running() is True


# ---------------------------------------------------------------------------
# Message dispatch — call private handlers with mock messages
# ---------------------------------------------------------------------------
class TestDispatch:
    """Verify that incoming-message handlers update telemetry correctly."""

    def test_attitude_update(self) -> None:
        adapter = MAVLinkAdapter(MAVLinkConfig())
        msg = SimpleNamespace(
            roll=0.1,
            pitch=-0.2,
            yaw=0.3,
            rollspeed=0.4,
            pitchspeed=-0.5,
            yawspeed=0.6,
        )
        adapter._on_attitude(msg)

        t = adapter.telemetry
        assert t.attitude_roll == pytest.approx(0.1)
        assert t.attitude_pitch == pytest.approx(-0.2)
        assert t.attitude_yaw == pytest.approx(0.3)
        assert t.attitude_yawspeed == pytest.approx(0.6)
        assert t.last_attitude_ns > 0

    def test_attitude_rejects_nan(self) -> None:
        adapter = MAVLinkAdapter(MAVLinkConfig())
        msg = SimpleNamespace(
            roll=float("nan"),
            pitch=0.0,
            yaw=0.0,
            rollspeed=0.0,
            pitchspeed=0.0,
            yawspeed=0.0,
        )
        adapter._on_attitude(msg)
        assert adapter.nan_count == 1
        assert adapter.telemetry.last_attitude_ns == 0

    def test_highres_imu_update(self) -> None:
        adapter = MAVLinkAdapter(MAVLinkConfig())
        msg = SimpleNamespace(
            xacc=0.0,
            yacc=0.0,
            zacc=-9.81,
            xgyro=0.01,
            ygyro=-0.02,
            zgyro=0.03,
        )
        adapter._on_highres_imu(msg)
        t = adapter.telemetry
        np.testing.assert_allclose(t.imu_accel, [0.0, 0.0, -9.81])
        np.testing.assert_allclose(t.imu_gyro, [0.01, -0.02, 0.03])
        assert t.last_imu_ns > 0

    def test_odometry_converts_ned_to_enu(self) -> None:
        adapter = MAVLinkAdapter(MAVLinkConfig())
        # NED (north=1, east=2, down=3) should become ENU (east=2, north=1, up=-3)
        msg = SimpleNamespace(
            x=1.0,
            y=2.0,
            z=3.0,
            vx=0.5,
            vy=-0.5,
            vz=0.1,
            q=[1.0, 0.0, 0.0, 0.0],  # scalar-first identity
            reset_counter=0,
        )
        adapter._on_odometry(msg)
        t = adapter.telemetry
        assert t.odometry_valid is True
        np.testing.assert_allclose(t.odometry_position, [2.0, 1.0, -3.0])
        np.testing.assert_allclose(t.odometry_velocity, [-0.5, 0.5, -0.1])

    def test_local_position_skipped_when_odometry_valid(self) -> None:
        adapter = MAVLinkAdapter(MAVLinkConfig())
        adapter._telemetry.odometry_valid = True
        adapter._telemetry.odometry_position = np.array([10.0, 10.0, 10.0])
        msg = SimpleNamespace(x=1.0, y=2.0, z=3.0, vx=0.0, vy=0.0, vz=0.0)
        adapter._on_local_position_ned(msg)
        # Position must not get overwritten by the worse stream.
        np.testing.assert_allclose(adapter.telemetry.odometry_position, [10.0, 10.0, 10.0])

    def test_local_position_used_when_no_odometry(self) -> None:
        adapter = MAVLinkAdapter(MAVLinkConfig())
        msg = SimpleNamespace(x=1.0, y=2.0, z=3.0, vx=0.5, vy=-0.5, vz=0.1)
        adapter._on_local_position_ned(msg)
        t = adapter.telemetry
        assert t.odometry_valid is True
        np.testing.assert_allclose(t.odometry_position, [2.0, 1.0, -3.0])

    def test_encapsulated_race_status(self) -> None:
        adapter = MAVLinkAdapter(MAVLinkConfig())
        payload = struct.pack("<BQqqIq", 1, 12345, -1, -1, 4, 0)
        msg = SimpleNamespace(data=payload, seqnr=0)
        adapter._on_encapsulated_data(msg)
        rs = adapter.race_status
        assert rs is not None
        assert rs.active_gate_index == 4
        assert rs.race_started is False

    def test_encapsulated_track_info_full_transfer(self) -> None:
        adapter = MAVLinkAdapter(MAVLinkConfig())
        # Track payload: 2 gates, each (gate_id, p, q, w, h)
        track_payload = struct.pack("<H", 2)
        for gid in (0, 1):
            track_payload += struct.pack(
                "<Hfffffffff",
                gid,
                float(gid),
                0.0,
                -2.0,
                1.0,
                0.0,
                0.0,
                0.0,
                1.5,
                1.5,
            )

        # 1) Sim broadcasts handshake
        handshake = SimpleNamespace(width=12, packets=1)
        adapter._on_data_transmission_handshake(handshake)

        # 2) Sim sends chunked payload (one chunk here for simplicity)
        chunk_payload = struct.pack("<BH", 2, 12) + track_payload
        chunk_msg = SimpleNamespace(data=chunk_payload, seqnr=0)
        adapter._on_encapsulated_data(chunk_msg)

        assert len(adapter.gate_map) == 2
        assert adapter.gate_map[0].gate_id == 0
        assert adapter.gate_map[1].gate_id == 1
        assert adapter.gate_map[1].width_m == pytest.approx(1.5)

    def test_collision_event_recorded(self) -> None:
        adapter = MAVLinkAdapter(MAVLinkConfig())
        msg = SimpleNamespace(id=1001, threat_level=2, horizontal_minimum_delta=4.2)
        adapter._on_collision(msg)
        events = adapter.collisions
        assert len(events) == 1
        assert events[0]["collision_id"] == 1001.0
        assert events[0]["threat_level"] == 2.0
        assert events[0]["impulse"] == pytest.approx(4.2)


# ---------------------------------------------------------------------------
# Gate map conversion
# ---------------------------------------------------------------------------
class TestGateWaypoints:
    def test_empty_gate_map_returns_empty(self) -> None:
        adapter = MAVLinkAdapter(MAVLinkConfig())
        assert adapter.gate_waypoints() == []

    def test_gate_map_converts_ned_to_enu(self) -> None:
        adapter = MAVLinkAdapter(MAVLinkConfig())
        # Inject a single gate at NED (north=10, east=5, down=-2) with identity orientation.
        track_payload = struct.pack("<H", 1) + struct.pack(
            "<Hfffffffff",
            0,
            10.0,
            5.0,
            -2.0,
            1.0,
            0.0,
            0.0,
            0.0,
            1.5,
            1.5,
        )
        handshake = SimpleNamespace(width=1, packets=1)
        adapter._on_data_transmission_handshake(handshake)
        chunk_payload = struct.pack("<BH", 2, 1) + track_payload
        adapter._on_encapsulated_data(SimpleNamespace(data=chunk_payload, seqnr=0))

        waypoints = adapter.gate_waypoints()
        assert len(waypoints) == 1
        # NED (10, 5, -2) → ENU (5, 10, 2)
        np.testing.assert_allclose(waypoints[0].position_w, [5.0, 10.0, 2.0])
        assert waypoints[0].corners_3d_w.shape == (4, 3)
        assert waypoints[0].order == 0
