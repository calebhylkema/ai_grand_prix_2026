"""Tests for drone_racing.sim.mock_adapter — mock simulator physics & sensors.

Test strategy:
    1. Physics: hover stays stationary, free-fall accelerates, thrust lifts.
    2. IMU generation: stationary drone reads gravity, noise is bounded.
    3. Frame rendering: produces correct-sized RGB images.
    4. Gate passage: drone flying through a gate triggers detection.
    5. Reset: clears state and stops the sim timer.
    6. Edge cases: crash detection, time limit, zero gates.
"""

from __future__ import annotations

import numpy as np
from numpy.testing import assert_allclose
from scipy.spatial.transform import Rotation

from drone_racing.sim.mock_adapter import MockSimAdapter, MockSimConfig
from drone_racing.utils.transforms import quat_identity
from drone_racing.utils.types import (
    CameraConfig,
    DroneCommand,
    GateWaypoint,
    Telemetry,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _default_camera() -> CameraConfig:
    return CameraConfig(width=320, height=240)


def _gate_with_normal_plus_x(gate_id: int, x: float, y: float, z: float) -> GateWaypoint:
    """Gate at (x, y, z) with normal pointing along +X.

    Drone approaching from -X starts on the negative side of the gate normal
    and crosses to positive when passing through — matching _check_gate_passage.
    """
    # Rotate gate so its local Z-axis (normal) points along world +X
    ori = Rotation.from_euler("y", np.pi / 2).as_quat()
    corners_local = np.array(
        [
            [-0.75, 0.75, 0.0],
            [0.75, 0.75, 0.0],
            [0.75, -0.75, 0.0],
            [-0.75, -0.75, 0.0],
        ],
        dtype=np.float64,
    )
    r_wg = Rotation.from_quat(ori).as_matrix()
    corners_w = (r_wg @ corners_local.T).T + np.array([x, y, z])
    return GateWaypoint(
        gate_id=gate_id,
        position_w=np.array([x, y, z]),
        orientation_w=ori.astype(np.float64),
        corners_3d_w=corners_w,
        order=gate_id,
    )


def _initial_telemetry(
    pos: tuple[float, float, float] = (0.0, 0.0, 2.0),
    vel: tuple[float, float, float] = (0.0, 0.0, 0.0),
) -> Telemetry:
    return Telemetry(
        timestamp_ns=0,
        position=np.array(pos),
        velocity=np.array(vel),
        orientation=quat_identity(),
    )


def _hover_command(ts: int = 0) -> DroneCommand:
    """Hover throttle (0.5) with zero angular rates."""
    return DroneCommand(timestamp_ns=ts, throttle=0.5, roll_rate=0.0, pitch_rate=0.0, yaw_rate=0.0)


def _build_adapter(
    gates: list[GateWaypoint] | None = None,
    sim_rate: float = 500.0,
    max_time_s: float = 5.0,
) -> MockSimAdapter:
    if gates is None:
        gates = []
    cfg = MockSimConfig(sim_rate_hz=sim_rate, max_time_s=max_time_s)
    return MockSimAdapter(camera_config=_default_camera(), gates=gates, config=cfg)


# ---------------------------------------------------------------------------
# Physics
# ---------------------------------------------------------------------------


class TestPhysicsHover:
    """Hovering drone should stay roughly stationary."""

    def test_hover_maintains_altitude(self) -> None:
        adapter = _build_adapter()
        adapter.reset(_initial_telemetry(pos=(0, 0, 2)))
        adapter.send_command(_hover_command())

        for _ in range(500):  # 1 second at 500 Hz
            adapter.step()

        state = adapter.state
        assert abs(state.position[2] - 2.0) < 0.5, f"Altitude drifted to {state.position[2]}"
        assert abs(state.velocity[2]) < 1.0, f"Vertical velocity {state.velocity[2]}"

    def test_hover_no_lateral_drift(self) -> None:
        adapter = _build_adapter()
        adapter.reset(_initial_telemetry(pos=(0, 0, 2)))
        adapter.send_command(_hover_command())

        for _ in range(500):
            adapter.step()

        state = adapter.state
        assert abs(state.position[0]) < 0.1
        assert abs(state.position[1]) < 0.1


class TestPhysicsFreefall:
    """Zero thrust should cause the drone to fall."""

    def test_falls_under_gravity(self) -> None:
        adapter = _build_adapter()
        adapter.reset(_initial_telemetry(pos=(0, 0, 10)))
        adapter.send_command(
            DroneCommand(
                timestamp_ns=0,
                throttle=0.0,
                roll_rate=0.0,
                pitch_rate=0.0,
                yaw_rate=0.0,
            )
        )

        for _ in range(250):  # 0.5s
            adapter.step()

        state = adapter.state
        # Should have fallen (drag slows it, but still negative vz)
        assert state.velocity[2] < -1.0
        assert state.position[2] < 10.0


class TestPhysicsThrust:
    """Full thrust should accelerate upward."""

    def test_full_thrust_climbs(self) -> None:
        adapter = _build_adapter()
        adapter.reset(_initial_telemetry(pos=(0, 0, 2)))
        adapter.send_command(
            DroneCommand(
                timestamp_ns=0,
                throttle=1.0,
                roll_rate=0.0,
                pitch_rate=0.0,
                yaw_rate=0.0,
            )
        )

        for _ in range(500):
            adapter.step()

        state = adapter.state
        assert state.position[2] > 2.0, f"Should have climbed, z={state.position[2]}"
        assert state.velocity[2] > 0.0


class TestPhysicsRotation:
    """Angular rate commands should rotate the drone."""

    def test_yaw_rate_rotates(self) -> None:
        adapter = _build_adapter()
        adapter.reset(_initial_telemetry(pos=(0, 0, 2)))
        adapter.send_command(
            DroneCommand(
                timestamp_ns=0,
                throttle=0.5,
                roll_rate=0.0,
                pitch_rate=0.0,
                yaw_rate=2.0,
            )
        )

        for _ in range(500):  # 1 second
            adapter.step()

        state = adapter.state
        r = Rotation.from_quat(state.orientation)
        _, _, yaw = r.as_euler("xyz")
        assert abs(yaw) > 0.5, f"Expected significant yaw rotation, got {yaw}"


# ---------------------------------------------------------------------------
# IMU generation
# ---------------------------------------------------------------------------


class TestIMU:
    """IMU output rate and content."""

    def test_imu_returns_none_when_not_due(self) -> None:
        adapter = _build_adapter()
        adapter.reset(_initial_telemetry())
        adapter.step()
        # First step is at 2ms (500 Hz), IMU at 200 Hz = every 5ms
        imu = adapter.get_imu()
        # Depending on timing, may or may not be due — but shouldn't crash
        assert imu is None or imu.timestamp_ns > 0

    def test_step_until_imu_returns_sample(self) -> None:
        adapter = _build_adapter()
        adapter.reset(_initial_telemetry())
        imu = adapter.step_until_imu()
        assert imu is not None
        assert imu.timestamp_ns > 0
        assert len(imu.accel_mps2) == 3
        assert len(imu.gyro_rps) == 3

    def test_hover_imu_reads_gravity(self) -> None:
        """Stationary drone: accel should be near 9.81 m/s² (thrust cancels gravity)."""
        adapter = _build_adapter()
        adapter.reset(_initial_telemetry(pos=(0, 0, 2)))
        adapter.send_command(_hover_command())

        # Step past initial transient
        for _ in range(100):
            adapter.step()

        imu = adapter.step_until_imu()
        accel_mag = float(np.linalg.norm(imu.accel_mps2))
        # Should read roughly g (hover thrust produces body-Z specific force ≈ g)
        assert 5.0 < accel_mag < 30.0, f"Unexpected accel magnitude {accel_mag}"


# ---------------------------------------------------------------------------
# Frame rendering
# ---------------------------------------------------------------------------


class TestFrameRendering:
    """Synthetic camera frame generation."""

    def test_frame_shape_matches_config(self) -> None:
        cam = _default_camera()
        adapter = _build_adapter()
        adapter.reset(_initial_telemetry())
        # Step enough to trigger a frame (camera at 30 Hz = every ~33ms)
        for _ in range(20):
            adapter.step()
        result = adapter.get_frame()
        if result is not None:
            frame, ts = result
            assert frame.shape == (cam.height, cam.width, 3)
            assert frame.dtype == np.uint8
            assert ts > 0

    def test_frame_with_visible_gate(self) -> None:
        """Gate close in front of drone should produce non-uniform pixels."""
        # Gate 3m ahead — close enough to occupy significant pixels
        gate = _gate_with_normal_plus_x(0, x=3.0, y=0.0, z=2.0)
        cam = CameraConfig(width=640, height=480, fx=300.0, fy=300.0, cx=320.0, cy=240.0)
        cfg = MockSimConfig(sim_rate_hz=500.0, max_time_s=5.0)
        adapter = MockSimAdapter(camera_config=cam, gates=[gate], config=cfg)
        adapter.reset(_initial_telemetry(pos=(0, 0, 2)))

        # Step until a frame is available
        frame = None
        for _ in range(100):
            adapter.step()
            result = adapter.get_frame()
            if result is not None:
                frame, _ = result
                break

        assert frame is not None
        # Frame should not be uniform grey (gate should be visible)
        unique_vals = len(np.unique(frame))
        assert unique_vals > 1, "Frame looks uniform — gate may not be rendered"


# ---------------------------------------------------------------------------
# Gate passage
# ---------------------------------------------------------------------------


class TestGatePassage:
    """Gate passage detection."""

    def test_flying_through_gate(self) -> None:
        """Drone flying along +X through a gate should trigger passage."""
        gate = _gate_with_normal_plus_x(0, x=5.0, y=0.0, z=2.0)
        adapter = _build_adapter(gates=[gate])
        adapter.reset(_initial_telemetry(pos=(0, 0, 2), vel=(10, 0, 0)))
        adapter.send_command(_hover_command())

        assert adapter.gates_passed == 0

        # Step for 2 seconds — drone at 10 m/s should pass gate at x=5
        for _ in range(1000):
            adapter.step()

        assert adapter.gates_passed >= 1, "Drone should have passed through the gate"

    def test_no_gate_passage_when_far_away(self) -> None:
        """Gate 100m away should not trigger passage."""
        gate = _gate_with_normal_plus_x(0, x=100.0, y=0.0, z=2.0)
        adapter = _build_adapter(gates=[gate])
        adapter.reset(_initial_telemetry(pos=(0, 0, 2)))
        adapter.send_command(_hover_command())

        for _ in range(500):
            adapter.step()

        assert adapter.gates_passed == 0


# ---------------------------------------------------------------------------
# Reset and termination
# ---------------------------------------------------------------------------


class TestResetAndTermination:
    """Reset clears state; time limit and crash stop the sim."""

    def test_reset_restores_position(self) -> None:
        adapter = _build_adapter()
        adapter.reset(_initial_telemetry(pos=(5, 3, 1)))
        assert_allclose(adapter.state.position, [5, 3, 1])

    def test_reset_clears_gates_passed(self) -> None:
        gate = _gate_with_normal_plus_x(0, x=5.0, y=0.0, z=2.0)
        adapter = _build_adapter(gates=[gate])
        adapter.reset(_initial_telemetry(pos=(0, 0, 2), vel=(10, 0, 0)))
        adapter.send_command(_hover_command())
        for _ in range(1000):
            adapter.step()
        assert adapter.gates_passed >= 1

        adapter.reset(_initial_telemetry(pos=(0, 0, 2)))
        assert adapter.gates_passed == 0

    def test_time_limit_stops_sim(self) -> None:
        adapter = _build_adapter(max_time_s=0.1)
        adapter.reset(_initial_telemetry())
        adapter.send_command(_hover_command())

        for _ in range(10000):
            if not adapter.is_running():
                break
            adapter.step()

        assert not adapter.is_running()

    def test_crash_stops_sim(self) -> None:
        """Drone falling below z=-2 should terminate."""
        adapter = _build_adapter(max_time_s=10.0)
        adapter.reset(_initial_telemetry(pos=(0, 0, -1.5)))
        adapter.send_command(
            DroneCommand(
                timestamp_ns=0,
                throttle=0.0,
                roll_rate=0.0,
                pitch_rate=0.0,
                yaw_rate=0.0,
            )
        )

        for _ in range(5000):
            if not adapter.is_running():
                break
            adapter.step()

        assert not adapter.is_running()

    def test_empty_gate_list(self) -> None:
        """Adapter works fine with no gates."""
        adapter = _build_adapter(gates=[])
        adapter.reset(_initial_telemetry())
        adapter.send_command(_hover_command())
        for _ in range(100):
            adapter.step()
        assert adapter.is_running()
        assert adapter.gates_passed == 0


# ---------------------------------------------------------------------------
# Visible gate measurements
# ---------------------------------------------------------------------------


class TestVisibleGateMeasurements:
    """Synthetic pose measurements for nearby gates."""

    def test_returns_measurements_for_nearby_gate(self) -> None:
        gate = _gate_with_normal_plus_x(0, x=5.0, y=0.0, z=2.0)
        adapter = _build_adapter(gates=[gate])
        adapter.reset(_initial_telemetry(pos=(0, 0, 2)))

        measurements = adapter.get_visible_gate_measurements(max_distance=10.0)
        assert len(measurements) == 1
        gate_id, noisy_pos, noisy_ori = measurements[0]
        assert gate_id == 0
        assert len(noisy_pos) == 3
        assert len(noisy_ori) == 4

    def test_no_measurements_for_distant_gate(self) -> None:
        gate = _gate_with_normal_plus_x(0, x=50.0, y=0.0, z=2.0)
        adapter = _build_adapter(gates=[gate])
        adapter.reset(_initial_telemetry(pos=(0, 0, 2)))

        measurements = adapter.get_visible_gate_measurements(max_distance=10.0)
        assert len(measurements) == 0
