#!/usr/bin/env python3
"""Run the full autonomy pipeline in the mock simulator.

Pipeline::

    Mock Camera → Classical Detect → Tracker → PnP → ESKF → Racing Line → Controller → Command → Physics

Controller options:
    - ``pid`` (default): PID cascade with lookahead on the racing line
    - ``mpcc``: MPCC trajectory optimiser with PID fallback (safety ladder)

Usage::

    uv run python scripts/run_sim.py
    uv run python scripts/run_sim.py --controller mpcc --max-time 30
    uv run python scripts/run_sim.py --controller pid --debug
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

# Ensure project root is on path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from drone_racing.control.pid import PIDController, TrajectorySetpoint
from drone_racing.control.rate_limiter import CommandRateLimiter
from drone_racing.estimation.eskf import ESKF
from drone_racing.logging.flight_logger import FlightLogger
from drone_racing.perception.classical import ClassicalDetector
from drone_racing.perception.neural import NeuralDetector, NeuralDetectorConfig
from drone_racing.perception.pnp import PnPSolver
from drone_racing.perception.tracker import GateTracker
from drone_racing.planning.spline import build_racing_line
from drone_racing.sim.mock_adapter import MockSimAdapter, MockSimConfig
from drone_racing.utils.config import RacingConfig, load_config
from drone_racing.utils.types import (
    DetectionConfidence,
    GateWaypoint,
    PoseMeasurement,
    Telemetry,
)


def build_gate_circuit(config: RacingConfig) -> list[GateWaypoint]:
    """Create a simple 4-gate square circuit for testing.

    Gates are arranged in a square pattern at a constant height,
    each facing inward so the drone flies through them in order.

    Returns
    -------
    list[GateWaypoint]
        Four gates forming a ~40m square circuit at 3m altitude.
    """
    gate_cfg = config.gate
    corners_local = gate_cfg.outer_corners_3d_array

    # Gate positions: square circuit, Z=3m (above ground)
    # Drone starts near gate 0 and flies counter-clockwise
    gate_defs: list[tuple[np.ndarray, float]] = [
        (np.array([15.0, 0.0, 3.0]), 0.0),  # Gate 0: drone approaches along +X
        (np.array([15.0, 15.0, 3.0]), np.pi / 2),  # Gate 1: drone approaches along +Y
        (np.array([0.0, 15.0, 3.0]), np.pi),  # Gate 2: drone approaches along -X
        (np.array([0.0, 0.0, 3.0]), -np.pi / 2),  # Gate 3: drone approaches along -Y
    ]

    gates: list[GateWaypoint] = []
    for i, (pos, yaw) in enumerate(gate_defs):
        # Gate orientation: gate normal (local Z) must point along the approach
        # direction.  Pitch +90° makes local Z → world +X, then yaw rotates
        # to the desired approach heading.
        orientation = Rotation.from_euler("yz", [np.pi / 2, yaw]).as_quat()  # [x,y,z,w]

        # Transform local corners to world frame
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

    return gates


def run_pipeline(
    config: RacingConfig,
    max_time: float = 60.0,
    debug: bool = False,
    log_path: str | None = None,
    controller_type: str = "pid",
    use_perception: bool = True,
    detector_type: str = "classical",
    rl_checkpoint: str | None = None,
) -> dict[str, object]:
    """Run the full autonomy pipeline in the mock simulator.

    Parameters
    ----------
    config : RacingConfig
        Full stack configuration.
    max_time : float
        Maximum simulation time in seconds.
    debug : bool
        If True, enable verbose logging.
    log_path : str | None
        Path to save the flight log. None = auto-generate.
    controller_type : str
        Controller to use: ``"pid"``, ``"mpcc"``, or ``"rl"``.
    use_perception : bool
        If True (default), use detector + PnP pipeline.
        If False, use ground-truth-based synthetic measurements.
    detector_type : str
        Detector backend: ``"classical"`` (4-corner) or ``"neural"`` (8-corner ONNX).
    rl_checkpoint : str | None
        Path to SB3 model ``.zip`` (required when ``controller_type="rl"``).

    Returns
    -------
    dict[str, object]
        Run statistics: gates_passed, laps, elapsed time, etc.
    """
    if debug:
        logging.getLogger("drone_racing").setLevel(logging.DEBUG)

    # --- Build gate circuit ---
    gates = build_gate_circuit(config)
    logger = logging.getLogger(__name__)
    logger.info("Gate circuit: %d gates", len(gates))

    # --- Build racing line ---
    gate_positions = np.array([g.position_w for g in gates])
    gate_orientations = np.array([g.orientation_w for g in gates])
    racing_line = build_racing_line(
        gate_positions, gate_orientations, config.planner, num_points=200, closed=True
    )
    logger.info("Racing line: %.1fm total length", racing_line.total_length)

    # --- Initialise modules ---
    use_neural = detector_type in ("neural", "hybrid") and use_perception
    use_hybrid = detector_type == "hybrid" and use_perception
    sim = MockSimAdapter(
        camera_config=config.camera,
        gates=gates,
        config=MockSimConfig(max_time_s=max_time),
    )
    # Tune estimation for mock sim: no real IMU bias, PnP is reasonably accurate
    from drone_racing.utils.types import EstimationConfig

    # Neural detector keypoints are less precise than classical corners,
    # so PnP measurements need higher noise to avoid corrupting the ESKF.
    if use_neural and not use_hybrid:
        pnp_pos_noise = 8.0  # neural keypoints → very noisy PnP (rough hint only)
        pnp_ori_noise = 3.0
        chi2_threshold = 100.0  # accept most measurements — noise model handles weighting
    else:
        # Classical and hybrid both use classical-quality corners
        pnp_pos_noise = 0.5
        pnp_ori_noise = 0.5
        chi2_threshold = 20.0
    mock_est = EstimationConfig(
        imu_rate_hz=config.estimation.imu_rate_hz,
        gyro_noise_std=config.estimation.gyro_noise_std,
        accel_noise_std=config.estimation.accel_noise_std,
        gyro_bias_std=1e-6,  # mock sim has no IMU bias — suppress random walk
        accel_bias_std=1e-6,
        pnp_position_noise_std=pnp_pos_noise,
        pnp_orientation_noise_std=pnp_ori_noise,
        chi2_gate_threshold=chi2_threshold,
    )
    from drone_racing.utils.config import RacingConfig

    mock_config = RacingConfig(
        estimation=mock_est,
        camera=config.camera,
        gate=config.gate,
        control=config.control,
        planner=config.planner,
    )
    eskf = ESKF(mock_config)

    # Perception modules (only needed when use_perception=True)
    hybrid_detector = None
    if use_hybrid:
        from drone_racing.perception.hybrid import HybridDetector, HybridDetectorConfig

        # Hybrid: neural bboxes + classical corner refinement → precise PnP
        gate_corners_3d_local = config.gate.corners_3d_array
        hybrid_cfg = HybridDetectorConfig(
            neural=NeuralDetectorConfig(
                model_path="runs/pose/runs/pose/train/weights/best.onnx",
                conf_threshold=0.15,
                num_keypoints=8,
                providers=["CPUExecutionProvider"],
            ),
        )
        hybrid_detector = HybridDetector(hybrid_cfg, camera_config=config.camera)
        hybrid_detector.load()
        logger.info("Detector: hybrid (neural bbox + classical corners)")
    elif use_neural:
        # Neural detector outputs 8 geometrically-ordered corners
        # (TL,TR,BR,BL outer + inner), matching GateConfig.corners_3d.
        gate_corners_3d_local = config.gate.corners_3d_array
        neural_cfg = NeuralDetectorConfig(
            model_path="runs/pose/runs/pose/train/weights/best.onnx",
            conf_threshold=0.15,
            num_keypoints=8,
            providers=["CPUExecutionProvider"],
        )
        neural_detector = NeuralDetector(neural_cfg)
        neural_detector.load()
        logger.info("Detector: neural (8-corner ONNX)")
    else:
        # Classical detector's polygon approximation is too noisy for
        # 8-corner PnP — use outer 4 corners only.
        gate_corners_3d_local = config.gate.outer_corners_3d_array
        logger.info("Detector: classical (4-corner)")

    classical_detector = ClassicalDetector(camera_config=config.camera)
    tracker = GateTracker()
    pnp = PnPSolver(config.camera, mock_est, max_reprojection_px=50.0)

    # Override control gains for mock sim (gentler than real hardware)
    from drone_racing.utils.types import ControlConfig

    mock_control = ControlConfig(
        position_kp=[0.8, 0.8, 1.5],
        position_ki=[0.0, 0.0, 0.02],
        position_kd=[0.5, 0.5, 0.8],
        velocity_kp=[1.5, 1.5, 2.0],
        attitude_kp=[4.0, 4.0, 2.0],
        rate_kp=[6.0, 6.0, 3.0],
        max_thrust=config.control.max_thrust,
        max_roll_rate=4.0,
        max_pitch_rate=4.0,
        max_yaw_rate=3.0,
        max_tilt_angle=0.8,  # ~45 degrees max tilt
    )
    pid_controller = PIDController(mock_control)

    # Build MPCC or RL controller if requested
    mpcc_controller = None
    rl_controller = None
    if controller_type == "mpcc":
        from drone_racing.control.mpcc_adapter import MPCCController
        from drone_racing.planning.mpcc import MPCCConfig, MPCCSolver

        mpcc_cfg = MPCCConfig()
        mpcc_solver = MPCCSolver(racing_line, mpcc_cfg, closed=True)
        mpcc_controller = MPCCController(
            mpcc_solver=mpcc_solver,
            racing_line=racing_line,
            pid_controller=pid_controller,
            control_config=mock_control,
            attitude_tau=mpcc_cfg.dynamics.attitude_tau,
        )
        logger.info("Controller: MPCC with PID fallback")
    elif controller_type == "rl":
        from drone_racing.control.rl_controller import RLController

        rl_controller = RLController.from_checkpoint(
            rl_checkpoint,
            racing_line=racing_line,
            control_config=mock_control,
        )
        logger.info("Controller: RL policy from %s", rl_checkpoint)
    else:
        logger.info("Controller: PID cascade")

    rate_limiter = CommandRateLimiter()
    flight_log = FlightLogger()

    # --- Initial conditions ---
    # Start between gate 0 and gate 1, heading towards gate 1
    start_pos_rl, start_tan, _ = racing_line.at_progress(0.08)
    start_pos = start_pos_rl.copy()
    start_yaw = float(np.arctan2(start_tan[1], start_tan[0]))
    start_vel = start_tan * 1.0  # gentle forward velocity along path
    start_ori = Rotation.from_euler("z", start_yaw).as_quat().astype(np.float64)
    start_telemetry = Telemetry(
        timestamp_ns=0,
        position=start_pos,
        velocity=start_vel,
        orientation=start_ori,
    )

    sim.reset(start_telemetry)
    eskf.initialise(start_telemetry)
    pid_controller.reset()
    start_progress = racing_line.closest_progress(start_pos)
    if mpcc_controller is not None:
        mpcc_controller.reset(progress=start_progress)
    if rl_controller is not None:
        rl_controller.reset(progress=start_progress)

    flight_log.set_metadata("start_position", start_pos)
    flight_log.set_metadata("num_gates", len(gates))

    # --- Main loop ---
    wall_start = time.perf_counter()
    imu_count = 0
    frame_count = 0
    correction_count = 0
    pipeline_times: list[float] = []
    prev_gates_passed = 0
    bootstrap_logged = False

    logger.info("Starting simulation loop (max %.1fs)...", max_time)

    while sim.is_running():
        # Step physics until next IMU sample
        imu = sim.step_until_imu()
        if not sim.is_running():
            break

        imu_count += 1

        # --- IMU prediction ---
        eskf.predict(imu)

        # --- Vision pipeline (at camera rate) ---
        frame_result = sim.get_frame()
        if frame_result is not None:
            t_pipe_start = time.perf_counter()
            frame, frame_ts = frame_result
            frame_count += 1

            if use_perception:
                # Full perception pipeline: Detect → Track → PnP → ESKF
                if use_hybrid and hybrid_detector is not None:
                    detections = hybrid_detector.detect(frame, frame_ts)
                elif use_neural:
                    detections = neural_detector.detect(frame, frame_ts)
                else:
                    detections = classical_detector.detect(frame, frame_ts)
                tracked = tracker.update(detections)

                # Assign gate map IDs using current state estimate
                gate_projections: dict[int, np.ndarray] = {}
                est = eskf.state
                r_wb = Rotation.from_quat(est.orientation).as_matrix()
                r_cb = np.linalg.inv(
                    Rotation.from_euler("y", -np.deg2rad(config.camera.tilt_deg)).as_matrix()
                    @ np.array([[0, 0, 1], [1, 0, 0], [0, 1, 0]], dtype=np.float64)
                )
                r_cw = r_cb @ r_wb.T
                t_cw = -r_cw @ est.position
                k = config.camera.intrinsic_matrix

                for gate in gates:
                    pt_c = r_cw @ gate.position_w + t_cw
                    if pt_c[2] > 0.5:
                        px = k[0, 0] * pt_c[0] / pt_c[2] + k[0, 2]
                        py = k[1, 1] * pt_c[1] / pt_c[2] + k[1, 2]
                        gate_projections[gate.gate_id] = np.array([px, py])

                tracker.assign_gate_map_ids(gate_projections)

                # PnP solve (tries all gates if ID unknown/wrong)
                measurements = pnp.solve_multiple(
                    tracked, gates, gate_corners_3d_local, drone_state=eskf.state
                )
            else:
                # Synthetic measurements from ground truth (for testing
                # estimation & control in isolation from perception)
                measurements = []
                visible = sim.get_visible_gate_measurements(
                    position_noise_std=0.15,
                    orientation_noise_std=0.05,
                    max_distance=25.0,
                )
                for gate_id, noisy_pos, noisy_ori in visible:
                    measurements.append(
                        PoseMeasurement(
                            timestamp_ns=frame_ts,
                            position_w=noisy_pos,
                            orientation_w=noisy_ori,
                            position_noise_std=mock_est.pnp_position_noise_std,
                            orientation_noise_std=mock_est.pnp_orientation_noise_std,
                            confidence=DetectionConfidence.HIGH,
                            gate_id=gate_id,
                        )
                    )

            for meas in measurements:
                eskf.correct(meas)
                correction_count += 1

            t_pipe_end = time.perf_counter()
            pipeline_times.append((t_pipe_end - t_pipe_start) * 1000.0)

        # --- Planning + Control ---
        current_state = eskf.state

        # Controller selection:
        # - MPCC + synthetic measurements: works perfectly, use for MPCC dev
        # - PID + real perception: works perfectly, use for perception dev
        # - MPCC + classical perception: degrades to PID — 4-corner PnP
        #   produces biased measurements under aggressive maneuvers.
        # - MPCC + neural perception: 8-corner PnP should be reliable
        #   enough for MPCC — test this hypothesis.
        mpcc_active = mpcc_controller is not None
        if mpcc_active and use_perception and not use_neural:
            # Classical 4-corner PnP is unreliable under MPCC's aggressive
            # maneuvers — fall back to PID.  Neural 8-corner PnP should
            # fix this once keypoint accuracy improves (needs real sim data).
            mpcc_active = False
            if not bootstrap_logged:
                bootstrap_logged = True
                logger.info(
                    "MPCC + classical perception: using PID (4-corner PnP "
                    "unreliable under aggressive maneuvers; use --detector "
                    "neural or --no-perception for MPCC)"
                )

        if rl_controller is not None:
            cmd = rl_controller.update(current_state)
        elif mpcc_active:
            cmd = mpcc_controller.update(current_state)
        else:
            # PID cascade with lookahead
            progress = racing_line.closest_progress(current_state.position)
            lookahead = (progress + 0.015) % 1.0
            target_pos, target_tangent, target_speed = racing_line.at_progress(lookahead)

            pos_error = target_pos - current_state.position
            err_dist = float(np.linalg.norm(pos_error))
            max_err = 3.0

            if err_dist > max_err:
                target_pos = current_state.position + pos_error * (max_err / err_dist)

            desired_yaw = float(np.arctan2(target_tangent[1], target_tangent[0]))
            ff_velocity = target_tangent * min(target_speed, 8.0) * 0.4

            setpoint = TrajectorySetpoint(
                position=target_pos,
                velocity=ff_velocity,
                yaw=desired_yaw,
            )
            cmd = pid_controller.update(current_state, setpoint)

        cmd = rate_limiter.limit(cmd)
        sim.send_command(cmd)

        # --- Logging ---
        flight_log.log_state(current_state)
        flight_log.log_command(cmd)
        flight_log.log_ground_truth(sim.state)

        # Periodic diagnostics (every 2s of sim time)
        if imu_count % 400 == 0:
            gt = sim.state
            pos_err = float(np.linalg.norm(current_state.position - gt.position))
            cov_trace = float(np.trace(current_state.covariance))
            logger.info(
                "DIAG t=%.1fs | pos_err=%.2fm | cov_tr=%.1f | corrections=%d | "
                "est_z=%.2f gt_z=%.2f | est_pos=%s gt_pos=%s",
                current_state.timestamp_ns * 1e-9,
                pos_err,
                cov_trace,
                correction_count,
                current_state.position[2],
                gt.position[2],
                np.array2string(current_state.position, precision=2),
                np.array2string(gt.position, precision=2),
            )

        # Gate passage events
        if sim.gates_passed > prev_gates_passed:
            gate_id = (sim.gates_passed - 1) % len(gates)
            flight_log.log_gate_passage(current_state.timestamp_ns, gate_id, sim.gates_passed)
            prev_gates_passed = sim.gates_passed

    # --- Results ---
    wall_elapsed = time.perf_counter() - wall_start
    sim_elapsed = eskf.state.timestamp_ns * 1e-9

    stats: dict[str, object] = {
        "controller": controller_type,
        "detector": detector_type,
        "gates_passed": sim.gates_passed,
        "laps_completed": sim.laps_completed,
        "sim_time_s": sim_elapsed,
        "wall_time_s": wall_elapsed,
        "realtime_factor": sim_elapsed / max(wall_elapsed, 1e-6),
        "imu_samples": imu_count,
        "camera_frames": frame_count,
        "mean_pipeline_ms": float(np.mean(pipeline_times)) if pipeline_times else 0.0,
        "p95_pipeline_ms": float(np.percentile(pipeline_times, 95)) if pipeline_times else 0.0,
        "max_pipeline_ms": float(np.max(pipeline_times)) if pipeline_times else 0.0,
    }

    # Save flight log
    if log_path is None:
        Path("logs").mkdir(exist_ok=True)
        log_path = f"logs/flight_{int(time.time())}.npz"
    flight_log.save(log_path)

    # Print summary
    print("\n" + "=" * 60)
    print("  SIMULATION COMPLETE")
    print("=" * 60)
    print(f"  Controller:       {controller_type.upper()}")
    print(f"  Detector:         {detector_type if use_perception else 'synthetic'}")
    print(f"  Gates passed:     {sim.gates_passed}")
    print(f"  Laps completed:   {sim.laps_completed}")
    print(f"  Sim time:         {sim_elapsed:.1f}s")
    print(f"  Wall time:        {wall_elapsed:.1f}s")
    print(f"  Realtime factor:  {stats['realtime_factor']:.1f}x")
    print(f"  IMU samples:      {imu_count}")
    print(f"  Camera frames:    {frame_count}")
    print(f"  Corrections:      {correction_count}")
    print(
        f"  Pipeline latency: {stats['mean_pipeline_ms']:.1f}ms avg, "
        f"{stats['p95_pipeline_ms']:.1f}ms p95, "
        f"{stats['max_pipeline_ms']:.1f}ms max"
    )
    print(f"  Flight log:       {log_path}")
    print("=" * 60)

    return stats


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="Run AI Grand Prix simulation")
    parser.add_argument("--config", type=str, default=None, help="Path to TOML config file")
    parser.add_argument(
        "--max-time", type=float, default=60.0, help="Max simulation time (seconds)"
    )
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")
    parser.add_argument("--log-path", type=str, default=None, help="Flight log output path")
    parser.add_argument(
        "--controller",
        type=str,
        default="pid",
        choices=["pid", "mpcc", "rl"],
        help="Controller type: pid (default), mpcc, or rl",
    )
    parser.add_argument(
        "--rl-checkpoint",
        type=str,
        default=None,
        help="Path to SB3 model .zip (required when --controller rl)",
    )
    parser.add_argument(
        "--no-perception",
        action="store_true",
        help="Use synthetic measurements instead of real perception pipeline",
    )
    parser.add_argument(
        "--detector",
        type=str,
        default="classical",
        choices=["classical", "neural", "hybrid"],
        help="Detector backend: classical, neural, or hybrid (neural bbox + classical corners)",
    )
    args = parser.parse_args()

    if args.controller == "rl" and args.rl_checkpoint is None:
        parser.error("--rl-checkpoint is required when --controller rl")

    # Setup logging
    log_level = logging.DEBUG if args.debug else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )

    config = load_config(args.config)
    run_pipeline(
        config,
        max_time=args.max_time,
        debug=args.debug,
        log_path=args.log_path,
        controller_type=args.controller,
        use_perception=not args.no_perception,
        detector_type=args.detector,
        rl_checkpoint=args.rl_checkpoint,
    )


if __name__ == "__main__":
    main()
