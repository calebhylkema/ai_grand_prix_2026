#!/usr/bin/env python3
"""Benchmark the full perception-to-control pipeline.

Measures per-stage and end-to-end latency through the pipeline::

    Frame → ClassicalDetector → PnP Solver → ESKF Correct → MPCC Controller → DroneCommand

Target: <40ms total pipeline latency (see CLAUDE.md).

Usage::

    uv run python scripts/bench_pipeline.py
    uv run python scripts/bench_pipeline.py --iterations 500 --warmup 50

The benchmark uses the MockSimAdapter to generate realistic synthetic frames
with visible gates, then runs each pipeline stage and reports timing statistics.
"""

from __future__ import annotations

import argparse
import logging
import statistics
import sys
import time
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

# Ensure project root is on path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from drone_racing.control.mpcc_adapter import MPCCController
from drone_racing.control.pid import PIDController
from drone_racing.estimation.eskf import ESKF
from drone_racing.perception.classical import ClassicalDetector, ClassicalDetectorConfig
from drone_racing.perception.pnp import PnPSolver
from drone_racing.planning.spline import build_racing_line
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

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Setup helpers
# ---------------------------------------------------------------------------

_LATENCY_TARGET_MS = 40.0


def _build_gate_circuit() -> tuple[list[GateWaypoint], np.ndarray]:
    """Build a 4-gate square circuit and return gates + local corners."""
    config = load_config()
    corners_local = config.gate.outer_corners_3d_array

    gate_defs: list[tuple[np.ndarray, float]] = [
        (np.array([15.0, 0.0, 3.0]), 0.0),
        (np.array([15.0, 15.0, 3.0]), np.pi / 2),
        (np.array([0.0, 15.0, 3.0]), np.pi),
        (np.array([0.0, 0.0, 3.0]), -np.pi / 2),
    ]

    gates: list[GateWaypoint] = []
    for i, (pos, yaw) in enumerate(gate_defs):
        orientation = Rotation.from_euler("yz", [np.pi / 2, yaw]).as_quat()
        r_wg = Rotation.from_quat(orientation)
        corners_world = (r_wg.apply(corners_local) + pos).astype(np.float64)
        gates.append(
            GateWaypoint(
                gate_id=i,
                position_w=pos,
                orientation_w=orientation.astype(np.float64),
                corners_3d_w=corners_world,
                order=i,
            )
        )
    return gates, corners_local


def _build_pipeline(
    gates: list[GateWaypoint],
) -> tuple[ClassicalDetector, PnPSolver, ESKF, MPCCController | None]:
    """Construct all pipeline stages.

    Returns
    -------
    tuple
        (detector, pnp_solver, eskf, mpcc_controller_or_none).
        MPCC controller is None if CasADi is not installed.
    """
    config = load_config()
    cam = config.camera

    detector = ClassicalDetector(ClassicalDetectorConfig(), camera_config=cam)
    pnp_solver = PnPSolver(cam, config.estimation)

    eskf = ESKF(config)
    eskf.initialise(
        Telemetry(
            timestamp_ns=0,
            position=np.array([0.0, 0.0, 3.0]),
            velocity=np.array([5.0, 0.0, 0.0]),
            orientation=quat_identity(),
        )
    )

    # Try to build MPCC controller (requires casadi)
    mpcc_ctrl: MPCCController | None = None
    try:
        from drone_racing.planning.mpcc import MPCCSolver

        gate_positions = np.array([g.position_w for g in gates])
        gate_orientations = np.array([g.orientation_w for g in gates])
        racing_line = build_racing_line(
            gate_positions, gate_orientations, PlannerConfig(), num_points=200, closed=True
        )
        mpcc_solver = MPCCSolver(racing_line, closed=True)
        pid = PIDController(config.control)
        mpcc_ctrl = MPCCController(
            mpcc_solver=mpcc_solver,
            racing_line=racing_line,
            pid_controller=pid,
            control_config=config.control,
        )
    except ImportError:
        logger.warning("CasADi not installed — MPCC benchmark will be skipped")

    return detector, pnp_solver, eskf, mpcc_ctrl


def _generate_frame_with_gate(
    gates: list[GateWaypoint],
    cam: CameraConfig,
    drone_pos: np.ndarray,
) -> np.ndarray:
    """Use MockSimAdapter to render a frame with visible gates.

    Parameters
    ----------
    gates : list[GateWaypoint]
        Gate circuit.
    cam : CameraConfig
        Camera intrinsics.
    drone_pos : np.ndarray
        Drone position (near a gate for visibility).

    Returns
    -------
    np.ndarray
        (H, W, 3) uint8 RGB frame.
    """
    sim_cfg = MockSimConfig(max_time_s=5.0)
    adapter = MockSimAdapter(camera_config=cam, gates=gates, config=sim_cfg)
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

    # Step until a frame is available
    for _ in range(100):
        adapter.step()
        result = adapter.get_frame()
        if result is not None:
            return result[0]

    # Fallback: synthetic image
    return np.random.default_rng(42).integers(0, 255, (cam.height, cam.width, 3), dtype=np.uint8)


# ---------------------------------------------------------------------------
# Benchmark runner
# ---------------------------------------------------------------------------


def run_benchmark(n_iterations: int, n_warmup: int) -> None:
    """Run the pipeline benchmark and print results.

    Parameters
    ----------
    n_iterations : int
        Number of timed iterations.
    n_warmup : int
        Number of warmup iterations (not timed).
    """
    config = load_config()
    cam = config.camera
    gates, corners_local = _build_gate_circuit()
    detector, pnp_solver, eskf, mpcc_ctrl = _build_pipeline(gates)

    # Position the drone 5m behind gate 0 (facing it along +X)
    gate0 = gates[0]
    r_wg = Rotation.from_quat(gate0.orientation_w)
    gate_normal = r_wg.apply(np.array([0.0, 0.0, 1.0]))
    drone_pos = gate0.position_w - gate_normal * 5.0

    print("Generating test frame from MockSim...")
    frame = _generate_frame_with_gate(gates, cam, drone_pos)
    print(f"Frame shape: {frame.shape}, dtype: {frame.dtype}")

    # Pre-build a state for PnP and MPCC (simulate a few IMU steps)
    for i in range(20):
        eskf.predict(
            ImuSample(
                timestamp_ns=(i + 1) * 5_000_000,
                accel_mps2=np.array([0.0, 0.0, 9.81]),
                gyro_rps=np.zeros(3),
            )
        )
    state = eskf.state

    # Storage for timing results
    detect_times: list[float] = []
    pnp_times: list[float] = []
    eskf_predict_times: list[float] = []
    eskf_correct_times: list[float] = []
    mpcc_times: list[float] = []
    total_times: list[float] = []

    total_iters = n_warmup + n_iterations
    print(f"\nRunning {n_warmup} warmup + {n_iterations} timed iterations...\n")

    for i in range(total_iters):
        is_warmup = i < n_warmup
        timestamp_ns = (i + 100) * 5_000_000

        t_total_start = time.perf_counter()

        # --- Stage 1: Gate detection ---
        t0 = time.perf_counter()
        detections = detector.detect(frame, timestamp_ns)
        t_detect = (time.perf_counter() - t0) * 1000.0

        # --- Stage 2: PnP pose solve ---
        t0 = time.perf_counter()
        measurement = None
        if detections:
            measurement = pnp_solver.solve(detections[0], gate0, corners_local, drone_state=state)
        t_pnp = (time.perf_counter() - t0) * 1000.0

        # --- Stage 3a: ESKF predict (IMU) ---
        t0 = time.perf_counter()
        imu = ImuSample(
            timestamp_ns=timestamp_ns,
            accel_mps2=np.array([0.0, 0.0, 9.81]),
            gyro_rps=np.zeros(3),
        )
        state = eskf.predict(imu)
        t_eskf_predict = (time.perf_counter() - t0) * 1000.0

        # --- Stage 3b: ESKF correct (vision) ---
        t0 = time.perf_counter()
        if measurement is not None:
            state = eskf.correct(measurement)
        t_eskf_correct = (time.perf_counter() - t0) * 1000.0

        # --- Stage 4: MPCC controller ---
        t0 = time.perf_counter()
        if mpcc_ctrl is not None:
            _cmd = mpcc_ctrl.update(state)
        t_mpcc = (time.perf_counter() - t0) * 1000.0

        t_total = (time.perf_counter() - t_total_start) * 1000.0

        if not is_warmup:
            detect_times.append(t_detect)
            pnp_times.append(t_pnp)
            eskf_predict_times.append(t_eskf_predict)
            eskf_correct_times.append(t_eskf_correct)
            mpcc_times.append(t_mpcc)
            total_times.append(t_total)

    # --- Report ---
    def _stats(label: str, times: list[float], target_ms: float | None = None) -> None:
        if not times:
            print(f"  {label:30s}  (no data)")
            return
        mn = min(times)
        p50 = statistics.median(times)
        p95 = sorted(times)[int(len(times) * 0.95)]
        p99 = sorted(times)[int(len(times) * 0.99)]
        mx = max(times)
        status = ""
        if target_ms is not None:
            status = " OK" if p95 < target_ms else " SLOW"
        print(
            f"  {label:30s}  min={mn:7.2f}  p50={p50:7.2f}  p95={p95:7.2f}  p99={p99:7.2f}  max={mx:7.2f} ms{status}"
        )

    print("=" * 100)
    print(f"Pipeline Benchmark Results ({n_iterations} iterations)")
    print("=" * 100)
    _stats("Classical Detector", detect_times, target_ms=5.0)
    _stats("PnP Solver", pnp_times, target_ms=2.0)
    _stats("ESKF Predict (IMU)", eskf_predict_times, target_ms=0.5)
    _stats("ESKF Correct (Vision)", eskf_correct_times, target_ms=2.0)
    if mpcc_ctrl is not None:
        _stats("MPCC Controller", mpcc_times, target_ms=30.0)
    else:
        print("  MPCC Controller                 (skipped — casadi not installed)")
    print("-" * 100)
    _stats("TOTAL PIPELINE", total_times, target_ms=_LATENCY_TARGET_MS)
    print("=" * 100)

    print(f"\nDetection rate: {len(detections)} gate(s) detected per frame")

    # Pass/fail summary
    p95_total = sorted(total_times)[int(len(total_times) * 0.95)]
    if p95_total < _LATENCY_TARGET_MS:
        print(f"\nPASS: p95 total = {p95_total:.2f} ms < {_LATENCY_TARGET_MS} ms target")
    else:
        print(f"\nWARN: p95 total = {p95_total:.2f} ms >= {_LATENCY_TARGET_MS} ms target")


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Benchmark perception-to-control pipeline latency",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--iterations", type=int, default=200, help="Timed iterations")
    parser.add_argument("--warmup", type=int, default=20, help="Warmup iterations")
    args = parser.parse_args()

    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")
    run_benchmark(args.iterations, args.warmup)


if __name__ == "__main__":
    main()
