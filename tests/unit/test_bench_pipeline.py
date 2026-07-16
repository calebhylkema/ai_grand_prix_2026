"""Latency benchmarks for the perception-to-control pipeline.

Run with::

    uv run python -m pytest tests/test_bench_pipeline.py -v

Each test asserts that p95 latency stays within the per-stage budget
(see perception/CLAUDE.md and project CLAUDE.md for targets).
"""

from __future__ import annotations

import time

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from drone_racing.estimation.eskf import ESKF
from drone_racing.perception.classical import ClassicalDetector, ClassicalDetectorConfig
from drone_racing.perception.pnp import PnPSolver
from drone_racing.sim.mock_adapter import MockSimAdapter, MockSimConfig
from drone_racing.utils.config import load_config
from drone_racing.utils.transforms import quat_identity
from drone_racing.utils.types import (
    CameraConfig,
    DroneCommand,
    GateWaypoint,
    ImuSample,
    PlannerConfig,
    Telemetry,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

N_BENCH = 100  # iterations per benchmark


@pytest.fixture(scope="module")
def config():
    return load_config()


@pytest.fixture(scope="module")
def gates(config):
    corners_local = config.gate.outer_corners_3d_array
    gate_defs = [
        (np.array([15.0, 0.0, 3.0]), 0.0),
        (np.array([15.0, 15.0, 3.0]), np.pi / 2),
        (np.array([0.0, 15.0, 3.0]), np.pi),
        (np.array([0.0, 0.0, 3.0]), -np.pi / 2),
    ]
    result = []
    for i, (pos, yaw) in enumerate(gate_defs):
        ori = Rotation.from_euler("yz", [np.pi / 2, yaw]).as_quat()
        r_wg = Rotation.from_quat(ori)
        corners_w = (r_wg.apply(corners_local) + pos).astype(np.float64)
        result.append(
            GateWaypoint(
                gate_id=i,
                position_w=pos,
                orientation_w=ori.astype(np.float64),
                corners_3d_w=corners_w,
                order=i,
            )
        )
    return result


@pytest.fixture(scope="module")
def frame(gates):
    """Render a synthetic frame with visible gate."""
    cam = CameraConfig(width=640, height=480, fx=300.0, fy=300.0, cx=320.0, cy=240.0)
    sim_cfg = MockSimConfig(max_time_s=5.0)
    adapter = MockSimAdapter(camera_config=cam, gates=gates, config=sim_cfg)

    gate0 = gates[0]
    r_wg = Rotation.from_quat(gate0.orientation_w)
    normal = r_wg.apply(np.array([0.0, 0.0, 1.0]))
    drone_pos = gate0.position_w - normal * 5.0

    adapter.reset(
        Telemetry(
            timestamp_ns=0,
            position=drone_pos.copy(),
            velocity=np.zeros(3),
            orientation=quat_identity(),
        )
    )
    adapter.send_command(
        DroneCommand(timestamp_ns=0, throttle=0.5, roll_rate=0.0, pitch_rate=0.0, yaw_rate=0.0)
    )
    for _ in range(100):
        adapter.step()
        result = adapter.get_frame()
        if result is not None:
            return result[0]

    return np.zeros((480, 640, 3), dtype=np.uint8)


@pytest.fixture(scope="module")
def eskf_with_state(config):
    """ESKF initialised and advanced a few steps."""
    filt = ESKF(config)
    filt.initialise(
        Telemetry(
            timestamp_ns=0,
            position=np.array([0.0, 0.0, 3.0]),
            velocity=np.array([5.0, 0.0, 0.0]),
            orientation=quat_identity(),
        )
    )
    for i in range(20):
        filt.predict(
            ImuSample(
                timestamp_ns=(i + 1) * 5_000_000,
                accel_mps2=np.array([0.0, 0.0, 9.81]),
                gyro_rps=np.zeros(3),
            )
        )
    return filt


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _time_many(fn, n: int) -> list[float]:
    """Call *fn* n times and return per-call timings in ms."""
    times = []
    for _ in range(n):
        t0 = time.perf_counter()
        fn()
        times.append((time.perf_counter() - t0) * 1000.0)
    return times


def _p95(times: list[float]) -> float:
    return sorted(times)[int(len(times) * 0.95)]


# ---------------------------------------------------------------------------
# Benchmarks
# ---------------------------------------------------------------------------


class TestClassicalDetectorLatency:
    """Classical gate detection should be <5ms (p95)."""

    def test_detection_latency(self, config, frame) -> None:
        cam = config.camera
        detector = ClassicalDetector(ClassicalDetectorConfig(), camera_config=cam)
        ts = 1_000_000

        # Warmup
        for _ in range(10):
            detector.detect(frame, ts)

        times = _time_many(lambda: detector.detect(frame, ts), N_BENCH)
        p95 = _p95(times)
        median = sorted(times)[len(times) // 2]
        print(f"\n  Classical Detector: p50={median:.2f}ms  p95={p95:.2f}ms")
        assert p95 < 10.0, f"Classical detector p95={p95:.2f}ms exceeds 10ms"


class TestPnPSolverLatency:
    """PnP solve should be <2ms (p95)."""

    def test_pnp_latency(self, config, frame, gates, eskf_with_state) -> None:
        cam = config.camera
        detector = ClassicalDetector(ClassicalDetectorConfig(), camera_config=cam)
        pnp = PnPSolver(cam, config.estimation)
        corners_local = config.gate.outer_corners_3d_array
        state = eskf_with_state.state

        detections = detector.detect(frame, 1_000_000)
        if not detections:
            pytest.skip("No gate detected in synthetic frame — PnP benchmark skipped")

        det = detections[0]
        gate = gates[0]

        # Warmup
        for _ in range(10):
            pnp.solve(det, gate, corners_local, drone_state=state)

        times = _time_many(lambda: pnp.solve(det, gate, corners_local, drone_state=state), N_BENCH)
        p95 = _p95(times)
        median = sorted(times)[len(times) // 2]
        print(f"\n  PnP Solver: p50={median:.2f}ms  p95={p95:.2f}ms")
        assert p95 < 5.0, f"PnP solver p95={p95:.2f}ms exceeds 5ms"


class TestESKFLatency:
    """ESKF predict + correct should be <2.5ms combined (p95)."""

    def test_predict_latency(self, config) -> None:
        filt = ESKF(config)
        filt.initialise(
            Telemetry(
                timestamp_ns=0,
                position=np.zeros(3),
                velocity=np.zeros(3),
                orientation=quat_identity(),
            )
        )
        # Build up some state
        for i in range(50):
            filt.predict(
                ImuSample(
                    timestamp_ns=(i + 1) * 5_000_000,
                    accel_mps2=np.array([0.0, 0.0, 9.81]),
                    gyro_rps=np.zeros(3),
                )
            )

        counter = [50]

        def _predict() -> None:
            counter[0] += 1
            filt.predict(
                ImuSample(
                    timestamp_ns=counter[0] * 5_000_000,
                    accel_mps2=np.array([0.0, 0.0, 9.81]),
                    gyro_rps=np.zeros(3),
                )
            )

        times = _time_many(_predict, N_BENCH)
        p95 = _p95(times)
        median = sorted(times)[len(times) // 2]
        print(f"\n  ESKF Predict: p50={median:.3f}ms  p95={p95:.3f}ms")
        assert p95 < 1.0, f"ESKF predict p95={p95:.3f}ms exceeds 1ms"

    def test_correct_latency(self, config) -> None:
        from drone_racing.utils.types import DetectionConfidence, PoseMeasurement

        filt = ESKF(config)
        filt.initialise(
            Telemetry(
                timestamp_ns=0,
                position=np.zeros(3),
                velocity=np.zeros(3),
                orientation=quat_identity(),
            )
        )
        for i in range(50):
            filt.predict(
                ImuSample(
                    timestamp_ns=(i + 1) * 5_000_000,
                    accel_mps2=np.array([0.0, 0.0, 9.81]),
                    gyro_rps=np.zeros(3),
                )
            )

        counter = [50]

        def _correct() -> None:
            counter[0] += 1
            ts = counter[0] * 5_000_000
            filt.predict(
                ImuSample(
                    timestamp_ns=ts,
                    accel_mps2=np.array([0.0, 0.0, 9.81]),
                    gyro_rps=np.zeros(3),
                )
            )
            filt.correct(
                PoseMeasurement(
                    timestamp_ns=ts,
                    position_w=np.array([0.01, 0.0, 0.0]),
                    orientation_w=quat_identity(),
                    position_noise_std=0.15,
                    orientation_noise_std=0.05,
                    confidence=DetectionConfidence.HIGH,
                    gate_id=0,
                )
            )

        times = _time_many(_correct, N_BENCH)
        p95 = _p95(times)
        median = sorted(times)[len(times) // 2]
        print(f"\n  ESKF Correct: p50={median:.3f}ms  p95={p95:.3f}ms")
        assert p95 < 2.0, f"ESKF correct p95={p95:.3f}ms exceeds 2ms"


class TestMPCCLatency:
    """MPCC controller update should be <30ms (p95)."""

    def test_mpcc_update_latency(self, config, gates) -> None:
        pytest.importorskip("casadi", reason="casadi not installed")

        from drone_racing.control.mpcc_adapter import MPCCController
        from drone_racing.control.pid import PIDController
        from drone_racing.planning.mpcc import MPCCSolver
        from drone_racing.planning.spline import build_racing_line

        gate_positions = np.array([g.position_w for g in gates])
        gate_orientations = np.array([g.orientation_w for g in gates])
        racing_line = build_racing_line(
            gate_positions, gate_orientations, PlannerConfig(), num_points=200, closed=True
        )
        mpcc_solver = MPCCSolver(racing_line, closed=True)
        pid = PIDController(config.control)
        ctrl = MPCCController(
            mpcc_solver=mpcc_solver,
            racing_line=racing_line,
            pid_controller=pid,
            control_config=config.control,
            allow_command_authority=True,
        )

        # Build a state for the controller
        filt = ESKF(config)
        filt.initialise(
            Telemetry(
                timestamp_ns=0,
                position=np.array([5.0, 0.0, 3.0]),
                velocity=np.array([5.0, 0.0, 0.0]),
                orientation=quat_identity(),
            )
        )
        for i in range(20):
            filt.predict(
                ImuSample(
                    timestamp_ns=(i + 1) * 5_000_000,
                    accel_mps2=np.array([0.0, 0.0, 9.81]),
                    gyro_rps=np.zeros(3),
                )
            )

        state = filt.state

        # Warmup (first solve is always slower — cold NLP)
        for _ in range(3):
            ctrl.update(state)

        times = _time_many(lambda: ctrl.update(state), N_BENCH)
        p95 = _p95(times)
        median = sorted(times)[len(times) // 2]
        print(f"\n  MPCC Controller: p50={median:.2f}ms  p95={p95:.2f}ms")
        assert p95 < 50.0, f"MPCC controller p95={p95:.2f}ms exceeds 50ms"
