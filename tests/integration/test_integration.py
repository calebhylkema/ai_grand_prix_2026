"""End-to-end integration test: image → detect → PnP → ESKF → planner → PID → command.

Validates that all modules in the autonomy pipeline connect correctly and produce
reasonable outputs when wired together with synthetic data.

Test scenario:
    - 3 gates forming a triangular course in the XY plane at Z=0
    - Drone starts at a known position, facing the first gate
    - A synthetic image with a coloured rectangle simulates a gate detection
    - The full pipeline runs: detect → PnP → ESKF correct → plan → PID → command
"""

from __future__ import annotations

import numpy as np
import pytest

from drone_racing.control.pid import PIDController, TrajectorySetpoint
from drone_racing.estimation.eskf import ESKF
from drone_racing.perception.classical import ClassicalDetector, ClassicalDetectorConfig
from drone_racing.perception.pnp import PnPSolver
from drone_racing.planning.spline import build_racing_line
from drone_racing.utils.config import load_config
from drone_racing.utils.transforms import quat_identity, quat_to_rotation_matrix
from drone_racing.utils.types import (
    DroneCommand,
    GateWaypoint,
    ImuSample,
    State,
    Telemetry,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def config() -> load_config:
    """Load default racing config."""
    return load_config()


@pytest.fixture()
def gate_corners_local(config: load_config) -> np.ndarray:
    """Gate corner positions in gate-local frame (from config)."""
    return config.gate.corners_3d_array


@pytest.fixture()
def triangular_course() -> list[GateWaypoint]:
    """Three gates forming a triangle in the XY plane.

    Gate 0: at (10, 0, 0) facing -X (drone approaches from +X side)
    Gate 1: at (0, 10, 0) facing -Y
    Gate 2: at (0, -10, 0) facing +Y
    """
    gates = [
        GateWaypoint(
            gate_id=0,
            position_w=np.array([10.0, 0.0, 0.0]),
            orientation_w=quat_identity(),
            corners_3d_w=_gate_corners_world(np.array([10.0, 0.0, 0.0]), quat_identity()),
        ),
        GateWaypoint(
            gate_id=1,
            position_w=np.array([0.0, 10.0, 0.0]),
            orientation_w=quat_identity(),
            corners_3d_w=_gate_corners_world(np.array([0.0, 10.0, 0.0]), quat_identity()),
        ),
        GateWaypoint(
            gate_id=2,
            position_w=np.array([0.0, -10.0, 0.0]),
            orientation_w=quat_identity(),
            corners_3d_w=_gate_corners_world(np.array([0.0, -10.0, 0.0]), quat_identity()),
        ),
    ]
    return gates


def _gate_corners_local() -> np.ndarray:
    """Standard 1.5m × 1.5m gate corners in gate-local frame."""
    return np.array(
        [
            [-0.75, 0.75, 0.0],
            [0.75, 0.75, 0.0],
            [0.75, -0.75, 0.0],
            [-0.75, -0.75, 0.0],
        ],
        dtype=np.float64,
    )


def _gate_corners_world(position: np.ndarray, orientation: np.ndarray) -> np.ndarray:
    """Transform local gate corners to world frame."""
    r = quat_to_rotation_matrix(orientation)
    local = _gate_corners_local()
    return (r @ local.T).T + position


def _make_grey_image(width: int = 1280, height: int = 720) -> np.ndarray:
    """Create a uniform grey RGB image."""
    return np.full((height, width, 3), 128, dtype=np.uint8)


def _draw_coloured_rect(
    image: np.ndarray,
    center: tuple[int, int],
    size: tuple[int, int],
    colour_rgb: tuple[int, int, int] = (255, 0, 0),
) -> np.ndarray:
    """Draw a filled coloured rectangle on an image."""
    cx, cy = center
    w, h = size
    x1 = max(0, cx - w // 2)
    y1 = max(0, cy - h // 2)
    x2 = min(image.shape[1], cx + w // 2)
    y2 = min(image.shape[0], cy + h // 2)
    image[y1:y2, x1:x2] = colour_rgb
    return image


def _detector_config() -> ClassicalDetectorConfig:
    """Config tuned for bright red rectangles on grey backgrounds."""
    return ClassicalDetectorConfig(
        hsv_lower=(0, 100, 100),
        hsv_upper=(10, 255, 255),
        morph_kernel_size=3,
        min_contour_area=200.0,
        max_contour_area=500_000.0,
        min_confidence=0.1,
    )


# ---------------------------------------------------------------------------
# Test: Full pipeline end-to-end
# ---------------------------------------------------------------------------


class TestFullPipeline:
    """Wire the complete pipeline with synthetic data."""

    def test_detect_to_command(
        self,
        config: load_config,
        gate_corners_local: np.ndarray,
        triangular_course: list[GateWaypoint],
    ) -> None:
        """Full pipeline: image → detect → PnP → ESKF → plan → PID → command.

        This verifies that all modules can be wired together and produce
        a valid DroneCommand at the end.
        """
        # --- Setup ---
        # Drone starts at [15, 0, 0] looking toward gate 0 at [10, 0, 0]
        drone_pos = np.array([15.0, 0.0, 0.0])
        drone_vel = np.zeros(3)
        drone_ori = quat_identity()
        t0_ns = 1_000_000_000  # 1 second

        # 1. Initialise ESKF with known starting state
        eskf = ESKF(config)
        telemetry = Telemetry(
            timestamp_ns=t0_ns,
            position=drone_pos.copy(),
            velocity=drone_vel.copy(),
            orientation=drone_ori.copy(),
        )
        eskf.initialise(telemetry)
        assert eskf.initialised

        # 2. Run a few IMU prediction steps (stationary, gravity-only)
        #    Accelerometer reads [0, 0, 9.81] in body frame (compensating gravity)
        for i in range(1, 11):
            imu = ImuSample(
                timestamp_ns=t0_ns + i * 5_000_000,  # 5ms steps = 200 Hz
                accel_mps2=np.array([0.0, 0.0, 9.81]),
                gyro_rps=np.zeros(3),
            )
            state = eskf.predict(imu)

        # State should still be close to starting position
        assert float(np.linalg.norm(state.position - drone_pos)) < 0.5

        # 3. Detect a gate in a synthetic image
        cam_config = config.camera
        img = _make_grey_image(cam_config.width, cam_config.height)
        # Draw a red rectangle near image center (simulating gate 0 in view)
        img = _draw_coloured_rect(img, (640, 360), (120, 120))

        detector = ClassicalDetector(_detector_config())
        detections = detector.detect(img, timestamp_ns=t0_ns + 50_000_000)
        assert len(detections) >= 1, "Detector should find the red rectangle"

        det = detections[0]
        assert det.corners_2d.shape[1] == 2
        assert det.confidence > 0.0

        # 4. Solve PnP (assign gate_id so PnP can match)
        det.gate_id = 0
        pnp_solver = PnPSolver(cam_config, config.estimation)
        measurement = pnp_solver.solve(det, triangular_course[0], gate_corners_local)
        assert measurement is not None, "PnP should succeed with 4 corners"
        assert measurement.position_w is not None
        assert len(measurement.position_w) == 3

        # 5. ESKF correction with the PnP measurement
        state_before = eskf.state
        cov_trace_before = float(np.trace(state_before.covariance))

        state_after = eskf.correct(measurement)
        cov_trace_after = float(np.trace(state_after.covariance))

        # Covariance should shrink (or stay same if gated) — at least not explode
        assert cov_trace_after < cov_trace_before * 2.0

        # 6. Build a racing line from the course
        gate_positions = np.array([g.position_w for g in triangular_course])
        gate_orientations = np.array([g.orientation_w for g in triangular_course])

        racing_line = build_racing_line(
            gate_positions, gate_orientations, config.planner, num_points=200, closed=True
        )
        assert racing_line.total_length > 0.0
        assert len(racing_line.positions) == 200

        # 7. Query the racing line for a setpoint
        current_state = eskf.state
        progress = racing_line.closest_progress(current_state.position)
        # Look slightly ahead
        lookahead = min(progress + 0.05, 1.0)
        target_pos, target_tangent, target_speed = racing_line.at_progress(lookahead)

        assert len(target_pos) == 3
        assert len(target_tangent) == 3
        assert target_speed > 0.0

        # 8. Create trajectory setpoint and run PID
        # Compute desired yaw from tangent direction
        desired_yaw = float(np.arctan2(target_tangent[1], target_tangent[0]))

        setpoint = TrajectorySetpoint(
            position=target_pos,
            velocity=target_tangent * target_speed,
            yaw=desired_yaw,
        )

        pid = PIDController(config.control)
        command = pid.update(current_state, setpoint)

        # 9. Verify the output is a valid DroneCommand
        assert isinstance(command, DroneCommand)
        assert 0.0 <= command.throttle <= 1.0
        assert abs(command.roll_rate) <= config.control.max_roll_rate
        assert abs(command.pitch_rate) <= config.control.max_pitch_rate
        assert abs(command.yaw_rate) <= config.control.max_yaw_rate


# ---------------------------------------------------------------------------
# Test: Module interface contracts
# ---------------------------------------------------------------------------


class TestModuleContracts:
    """Verify that module outputs match downstream module input expectations."""

    def test_pnp_output_matches_eskf_input(self, config: load_config) -> None:
        """PnPSolver.solve() output is directly consumable by ESKF.correct()."""
        import cv2

        cam_config = config.camera
        gate = GateWaypoint(
            gate_id=0,
            position_w=np.zeros(3),
            orientation_w=quat_identity(),
            corners_3d_w=_gate_corners_local(),
        )
        corners_local = _gate_corners_local()

        # Project gate corners to get 2D points
        rvec = np.zeros(3)
        tvec = np.array([0.0, 0.0, 5.0])
        img_pts, _ = cv2.projectPoints(
            corners_local, rvec, tvec, cam_config.intrinsic_matrix, cam_config.distortion_coeffs
        )

        from drone_racing.utils.types import GateDetection

        det = GateDetection(
            timestamp_ns=1000,
            corners_2d=img_pts.reshape(-1, 2),
            confidence=0.9,
            gate_id=0,
        )

        solver = PnPSolver(cam_config, config.estimation)
        meas = solver.solve(det, gate, corners_local)
        assert meas is not None

        # Now feed into ESKF — should not raise
        eskf = ESKF(config)
        eskf.initialise(
            Telemetry(
                timestamp_ns=0,
                position=np.zeros(3),
                velocity=np.zeros(3),
                orientation=quat_identity(),
            )
        )
        # Run one IMU step to advance timestamp
        eskf.predict(
            ImuSample(
                timestamp_ns=500,
                accel_mps2=np.array([0.0, 0.0, 9.81]),
                gyro_rps=np.zeros(3),
            )
        )
        state = eskf.correct(meas)
        assert isinstance(state, State)

    def test_racing_line_to_pid_setpoint(self, config: load_config) -> None:
        """RacingLine.at_progress() output can construct a TrajectorySetpoint for PID."""
        # Build a simple 3-gate racing line
        positions = np.array(
            [[0.0, 0.0, 0.0], [10.0, 0.0, 0.0], [10.0, 10.0, 0.0]], dtype=np.float64
        )
        orientations = np.array(
            [quat_identity(), quat_identity(), quat_identity()], dtype=np.float64
        )
        racing_line = build_racing_line(
            positions, orientations, config.planner, num_points=100, closed=False
        )

        pos, tangent, speed = racing_line.at_progress(0.5)

        # Build a setpoint
        yaw = float(np.arctan2(tangent[1], tangent[0]))
        setpoint = TrajectorySetpoint(
            position=pos,
            velocity=tangent * speed,
            yaw=yaw,
        )

        # Feed to PID with a hover state
        state = State.identity(timestamp_ns=1_000_000)
        state.position = np.array([5.0, 0.0, 0.0])

        pid = PIDController(config.control)
        cmd = pid.update(state, setpoint)
        assert isinstance(cmd, DroneCommand)
        assert 0.0 <= cmd.throttle <= 1.0

    def test_eskf_state_usable_by_pid(self, config: load_config) -> None:
        """ESKF output State is directly usable by PIDController.update()."""
        eskf = ESKF(config)
        eskf.initialise(
            Telemetry(
                timestamp_ns=0,
                position=np.array([1.0, 2.0, 3.0]),
                velocity=np.zeros(3),
                orientation=quat_identity(),
            )
        )
        # Predict a few steps
        for i in range(1, 6):
            eskf.predict(
                ImuSample(
                    timestamp_ns=i * 5_000_000,
                    accel_mps2=np.array([0.0, 0.0, 9.81]),
                    gyro_rps=np.zeros(3),
                )
            )

        state = eskf.state
        setpoint = TrajectorySetpoint(
            position=np.array([2.0, 2.0, 3.0]),
        )

        pid = PIDController(config.control)
        cmd = pid.update(state, setpoint)
        assert isinstance(cmd, DroneCommand)


# ---------------------------------------------------------------------------
# Test: Multi-step pipeline simulation
# ---------------------------------------------------------------------------


class TestMultiStepSimulation:
    """Simulate several frames of the pipeline running in sequence."""

    def test_multiple_imu_then_vision_correction(self, config: load_config) -> None:
        """Run 50 IMU predictions, then a vision correction, then more predictions.

        Verifies the pipeline doesn't accumulate errors or crash over time.
        """
        eskf = ESKF(config)
        start_pos = np.array([5.0, 0.0, 0.0])
        eskf.initialise(
            Telemetry(
                timestamp_ns=0,
                position=start_pos.copy(),
                velocity=np.zeros(3),
                orientation=quat_identity(),
            )
        )

        # Phase 1: 50 IMU predictions (250 ms at 200 Hz)
        for i in range(1, 51):
            imu = ImuSample(
                timestamp_ns=i * 5_000_000,
                accel_mps2=np.array([0.0, 0.0, 9.81]),
                gyro_rps=np.zeros(3),
            )
            state = eskf.predict(imu)

        # Position should still be close (stationary + gravity-compensated IMU)
        assert float(np.linalg.norm(state.position - start_pos)) < 1.0

        # Covariance should have grown
        cov_before_correction = float(np.trace(state.covariance))

        # Phase 2: Vision correction (direct PoseMeasurement, skip perception)
        from drone_racing.utils.types import DetectionConfidence, PoseMeasurement

        meas = PoseMeasurement(
            timestamp_ns=51 * 5_000_000,
            position_w=start_pos + np.array([0.05, -0.02, 0.01]),  # Slight offset
            orientation_w=quat_identity(),
            position_noise_std=0.15,
            orientation_noise_std=0.05,
            confidence=DetectionConfidence.HIGH,
            gate_id=0,
        )
        state = eskf.correct(meas)

        # Covariance should shrink after correction
        cov_after_correction = float(np.trace(state.covariance))
        assert cov_after_correction < cov_before_correction

        # Phase 3: 50 more IMU predictions
        for i in range(52, 102):
            imu = ImuSample(
                timestamp_ns=i * 5_000_000,
                accel_mps2=np.array([0.0, 0.0, 9.81]),
                gyro_rps=np.zeros(3),
            )
            state = eskf.predict(imu)

        # Still close to starting position (stationary drone)
        assert float(np.linalg.norm(state.position - start_pos)) < 1.5

    def test_repeated_detect_pnp_correct_cycle(
        self, config: load_config, gate_corners_local: np.ndarray
    ) -> None:
        """Run the detect → PnP → correct cycle 3 times to verify stability."""
        cam_config = config.camera
        eskf = ESKF(config)
        eskf.initialise(
            Telemetry(
                timestamp_ns=0,
                position=np.zeros(3),
                velocity=np.zeros(3),
                orientation=quat_identity(),
            )
        )

        gate = GateWaypoint(
            gate_id=0,
            position_w=np.array([0.0, 0.0, 5.0]),
            orientation_w=quat_identity(),
            corners_3d_w=_gate_corners_world(np.array([0.0, 0.0, 5.0]), quat_identity()),
        )

        detector = ClassicalDetector(_detector_config())
        pnp_solver = PnPSolver(cam_config, config.estimation)

        for cycle in range(3):
            base_ns = cycle * 100_000_000  # 100ms per cycle

            # IMU predictions
            for i in range(1, 11):
                eskf.predict(
                    ImuSample(
                        timestamp_ns=base_ns + i * 5_000_000,
                        accel_mps2=np.array([0.0, 0.0, 9.81]),
                        gyro_rps=np.zeros(3),
                    )
                )

            # Detect
            img = _make_grey_image(cam_config.width, cam_config.height)
            img = _draw_coloured_rect(img, (640, 360), (100, 100))
            dets = detector.detect(img, timestamp_ns=base_ns + 50_000_000)

            if len(dets) >= 1:
                dets[0].gate_id = 0
                meas = pnp_solver.solve(dets[0], gate, gate_corners_local)
                if meas is not None:
                    eskf.correct(meas)

        # ESKF should still have a valid state (no NaN, no explosion)
        final_state = eskf.state
        assert not np.any(np.isnan(final_state.position))
        assert not np.any(np.isnan(final_state.velocity))
        assert not np.any(np.isnan(final_state.orientation))
        assert np.all(np.isfinite(final_state.covariance))

    def test_full_loop_planning_and_control(self, config: load_config) -> None:
        """Full loop: ESKF state → racing line query → PID command, repeated."""
        eskf = ESKF(config)
        drone_start = np.array([0.0, 0.0, 0.0])
        eskf.initialise(
            Telemetry(
                timestamp_ns=0,
                position=drone_start.copy(),
                velocity=np.zeros(3),
                orientation=quat_identity(),
            )
        )

        # Build racing line through 3 gates
        gate_positions = np.array(
            [[5.0, 0.0, 0.0], [5.0, 5.0, 0.0], [0.0, 5.0, 0.0]], dtype=np.float64
        )
        gate_orientations = np.array(
            [quat_identity(), quat_identity(), quat_identity()], dtype=np.float64
        )
        racing_line = build_racing_line(
            gate_positions, gate_orientations, config.planner, num_points=200, closed=True
        )

        pid = PIDController(config.control)
        commands: list[DroneCommand] = []

        for step in range(20):
            t_ns = (step + 1) * 20_000_000  # 20ms = 50 Hz

            # IMU predict
            eskf.predict(
                ImuSample(
                    timestamp_ns=t_ns,
                    accel_mps2=np.array([0.0, 0.0, 9.81]),
                    gyro_rps=np.zeros(3),
                )
            )

            state = eskf.state

            # Plan: find closest progress and look ahead
            progress = racing_line.closest_progress(state.position)
            lookahead = min(progress + 0.02, 1.0)
            target_pos, target_tangent, target_speed = racing_line.at_progress(lookahead)

            yaw = float(np.arctan2(target_tangent[1], target_tangent[0]))
            setpoint = TrajectorySetpoint(
                position=target_pos,
                velocity=target_tangent * target_speed,
                yaw=yaw,
            )

            # Control
            cmd = pid.update(state, setpoint)
            commands.append(cmd)

        # All commands should be valid
        assert len(commands) == 20
        for cmd in commands:
            assert 0.0 <= cmd.throttle <= 1.0
            assert np.isfinite(cmd.roll_rate)
            assert np.isfinite(cmd.pitch_rate)
            assert np.isfinite(cmd.yaw_rate)
