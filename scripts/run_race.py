#!/usr/bin/env python3
"""Run the autonomy pipeline against the real AI Grand Prix simulator.

Connects to the simulator via MAVSDK over MAVLink v2 / UDP, receives
telemetry and vision, runs the perception→estimation→planning→control
pipeline, and sends commands back.

This is the competition entry point.  For offline testing with the mock
simulator, use ``scripts/run_sim.py`` instead.

Usage::

    uv run python scripts/run_race.py
    uv run python scripts/run_race.py --address udp://:14540
    uv run python scripts/run_race.py --debug

Requirements:
    - AI Grand Prix simulator running (Windows, downloaded package)
    - Active internet connection (anti-cheat)
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from drone_racing.control.pid import PIDController, TrajectorySetpoint
from drone_racing.control.rate_limiter import CommandRateLimiter
from drone_racing.estimation.eskf import ESKF
from drone_racing.estimation.gate_mapper import GateMapper, GateMapperConfig
from drone_racing.logging.flight_logger import FlightLogger
from drone_racing.logging.frame_recorder import FrameRecorder, FrameRecorderConfig
from drone_racing.perception.classical import ClassicalDetector
from drone_racing.perception.tracker import GateTracker
from drone_racing.planning.mpcc import MPCCSolver
from drone_racing.planning.spline import RacingLine, build_racing_line
from drone_racing.sim.mavsdk_adapter import MAVSDKAdapter
from drone_racing.utils.config import load_config
from drone_racing.utils.types import DroneCommand, GateDetection, State

logger = logging.getLogger(__name__)

# VQ1 max run time (from spec)
MAX_RUN_TIME_S = 8 * 60  # 8 minutes

# Minimum number of consolidated gate waypoints required before we attempt to
# build a racing line and hand off from PID hover to MPCC tracking.  A closed
# spline requires at least 2 distinct gates; more gives a stabler line.
MIN_MAPPED_GATES_FOR_MPCC = 3


def _extract_euler(state: State) -> tuple[float, float, float]:
    """Return (roll, pitch, yaw) in radians from the body quaternion."""
    from scipy.spatial.transform import Rotation

    yaw, pitch, roll = Rotation.from_quat(state.orientation).as_euler("ZYX")
    return float(roll), float(pitch), float(yaw)


def _mpcc_to_command(
    mpcc_input: np.ndarray,
    state: State,
    attitude_kp: tuple[float, float, float],
    timestamp_ns: int,
) -> DroneCommand:
    """Convert an MPCC input [t_cmd, phi_cmd, theta_cmd, psi_dot_cmd] into a
    rate-mode :class:`DroneCommand` by closing an attitude P-loop.

    The MPCC model outputs attitude setpoints + yaw rate; the flight
    controller expects body rates, so we drive roll/pitch as ``kp * error``.
    """
    t_cmd = float(mpcc_input[0])
    phi_cmd = float(mpcc_input[1])
    theta_cmd = float(mpcc_input[2])
    psi_dot_cmd = float(mpcc_input[3])

    roll, pitch, _ = _extract_euler(state)
    return DroneCommand(
        timestamp_ns=timestamp_ns,
        throttle=float(np.clip(t_cmd, 0.0, 1.0)),
        roll_rate=attitude_kp[0] * (phi_cmd - roll),
        pitch_rate=attitude_kp[1] * (theta_cmd - pitch),
        yaw_rate=psi_dot_cmd,
    )


def _build_racing_line_from_waypoints(
    waypoints: list, planner_config, *, closed: bool = True
) -> RacingLine:
    """Build a racing line from :class:`GateWaypoint` list."""
    positions = np.stack([wp.position_w for wp in waypoints])
    orientations = np.stack([wp.orientation_w for wp in waypoints])
    return build_racing_line(positions, orientations, planner_config, num_points=200, closed=closed)


def _maybe_get_camera_frame(adapter: MAVSDKAdapter) -> tuple[np.ndarray, int] | None:
    """Fetch the latest (frame, timestamp_ns) from the adapter if available.

    The camera stream is not yet implemented in the adapter; this probes for a
    ``latest_frame`` attribute so the pipeline works the moment vision lands
    without further changes here.
    """
    frame = getattr(adapter, "latest_frame", None)
    ts = getattr(adapter, "latest_frame_timestamp_ns", None)
    if frame is None or ts is None:
        return None
    return frame, int(ts)


async def run_race(
    config_path: str | None = None,
    address: str | None = None,
    debug: bool = False,
    record_dir: str | None = None,
    record_stride: int = 1,
) -> dict[str, object]:
    """Run the full autonomy pipeline against the real simulator.

    Parameters
    ----------
    config_path : str | None
        Path to TOML config file.
    address : str | None
        Override MAVSDK connection address.
    debug : bool
        Enable verbose logging.

    Returns
    -------
    dict[str, object]
        Run statistics.
    """
    if debug:
        logging.getLogger("drone_racing").setLevel(logging.DEBUG)

    config = load_config(config_path)

    # Override address if provided
    if address is not None:
        config.mavlink.system_address = address

    # --- Connect to simulator ---
    adapter = MAVSDKAdapter(config.mavlink)
    await adapter.connect()

    # Wait a moment for initial telemetry to arrive
    logger.info("Waiting for initial telemetry...")
    await asyncio.sleep(1.0)

    # --- Initialise pipeline modules ---
    eskf = ESKF(config)
    pid_controller = PIDController(config.control)
    rate_limiter = CommandRateLimiter()
    flight_log = FlightLogger()

    # Perception + mapping
    detector = ClassicalDetector(camera_config=config.camera)
    tracker = GateTracker()
    mapper = GateMapper(
        camera_config=config.camera,
        gate_corners_3d_local=config.gate.corners_3d_array,
        config=GateMapperConfig(),
    )
    attitude_kp = (
        config.control.attitude_kp[0],
        config.control.attitude_kp[1],
        config.control.attitude_kp[2],
    )

    # Planning state (built lazily once enough gates are mapped)
    mpcc: MPCCSolver | None = None
    racing_line: RacingLine | None = None
    mpcc_failures = 0

    # Optional frame + telemetry recorder for offline training-data capture.
    recorder: FrameRecorder | None = None
    if record_dir is not None:
        recorder = FrameRecorder(
            output_dir=record_dir,
            camera_config=config.camera,
            config=FrameRecorderConfig(stride=max(1, int(record_stride))),
        )
        logger.info("Frame recording enabled → %s (stride=%d)", record_dir, record_stride)

    # De-dupe guard so we never record the same frame twice.
    last_recorded_ts: int | None = None

    # Initialise ESKF from initial telemetry
    initial_telemetry = adapter.get_initial_telemetry()
    eskf.initialise(initial_telemetry)
    pid_controller.reset()

    logger.info(
        "Pipeline initialised. Initial position: %s",
        initial_telemetry.position,
    )

    # --- Start offboard control ---
    await adapter.start_offboard()

    # --- Main control loop ---
    command_period = 1.0 / config.mavlink.command_rate_hz
    start_time = time.monotonic()
    loop_count = 0

    logger.info(
        "Starting control loop at %.0f Hz (max %ds)...",
        config.mavlink.command_rate_hz,
        MAX_RUN_TIME_S,
    )

    try:
        while adapter.is_running():
            loop_start = time.monotonic()
            elapsed = loop_start - start_time

            # Check max run time
            if elapsed >= MAX_RUN_TIME_S:
                logger.warning("Max run time reached (%.0fs). Stopping.", elapsed)
                break

            loop_count += 1

            # --- Get latest telemetry ---
            imu = adapter.latest_imu
            eskf.predict(imu)

            # --- Vision pipeline ---
            # The MAVSDK adapter does not yet expose camera frames; _maybe_get_
            # camera_frame returns None until the vision stream lands.  When it
            # does, detector → tracker → gate_mapper runs automatically.
            frame_info = _maybe_get_camera_frame(adapter)
            tracked: list[GateDetection] = []
            if frame_info is not None:
                frame, frame_ts_ns = frame_info
                raw_detections = detector.detect(frame, frame_ts_ns)
                tracked = tracker.update(raw_detections)
                mapper.observe_many(tracked, eskf.state)
                if recorder is not None and frame_ts_ns != last_recorded_ts:
                    recorder.record(frame, frame_ts_ns, eskf.state, tracked)
                    last_recorded_ts = frame_ts_ns

            # --- Planning + Control ---
            current_state = eskf.state

            # Build / rebuild the racing line once we have enough consolidated
            # gate waypoints.  Rebuilding each frame is cheap relative to MPCC,
            # but we only do it when the waypoint count changes to keep MPCC's
            # cached interpolants stable.
            waypoints = mapper.waypoints()
            if len(waypoints) >= MIN_MAPPED_GATES_FOR_MPCC and mpcc is None:
                try:
                    racing_line = _build_racing_line_from_waypoints(
                        waypoints, config.planner, closed=True
                    )
                    mpcc = MPCCSolver.from_planner_config(racing_line, config.planner, closed=True)
                    logger.info(
                        "Racing line built from %d gates — switching to MPCC",
                        len(waypoints),
                    )
                except Exception:  # noqa: BLE001
                    logger.exception("Failed to build racing line — staying on PID hover")
                    mpcc = None
                    racing_line = None

            cmd: DroneCommand | None = None
            if mpcc is not None and racing_line is not None:
                try:
                    # MPCC state is 9D: [px, py, pz, vx, vy, vz, phi, theta, psi]
                    roll, pitch, yaw = _extract_euler(current_state)
                    mpcc_state = np.concatenate(
                        [
                            current_state.position,
                            current_state.velocity,
                            np.array([roll, pitch, yaw]),
                        ]
                    )
                    progress = racing_line.closest_progress(current_state.position)
                    sol = mpcc.solve(mpcc_state, progress)
                    if sol.success and sol.inputs.shape[0] > 0:
                        cmd = _mpcc_to_command(
                            sol.inputs[0],
                            current_state,
                            attitude_kp,
                            timestamp_ns=current_state.timestamp_ns,
                        )
                        mpcc_failures = 0
                    else:
                        mpcc_failures += 1
                except Exception:  # noqa: BLE001
                    logger.exception("MPCC solve raised — falling back to PID this tick")
                    mpcc_failures += 1

                if mpcc_failures >= 10:
                    logger.warning(
                        "MPCC failed %d ticks in a row — reverting to PID hover",
                        mpcc_failures,
                    )
                    mpcc = None
                    racing_line = None

            if cmd is None:
                # PID hover baseline (and fallback while gates are being mapped).
                setpoint = TrajectorySetpoint(
                    position=current_state.position,
                    yaw=0.0,
                )
                cmd = pid_controller.update(current_state, setpoint)

            cmd = rate_limiter.limit(cmd)

            # --- Send command ---
            await adapter.send_command(cmd)

            # --- Logging ---
            flight_log.log_state(current_state)
            flight_log.log_command(cmd)

            # Periodic diagnostics (every 5s)
            if loop_count % int(5 * config.mavlink.command_rate_hz) == 0:
                t = adapter.telemetry
                mode = "MPCC" if mpcc is not None else "HOVER"
                logger.info(
                    "DIAG t=%.1fs | mode=%s | mapped_gates=%d | odom_valid=%s | "
                    "pos=%s | att=(%.1f, %.1f, %.1f)°",
                    elapsed,
                    mode,
                    len(mapper.waypoints()),
                    t.odometry_valid,
                    np.array2string(t.odometry_position, precision=2),
                    np.degrees(t.attitude_roll),
                    np.degrees(t.attitude_pitch),
                    np.degrees(t.attitude_yaw),
                )

            # --- Rate control ---
            loop_elapsed = time.monotonic() - loop_start
            sleep_time = command_period - loop_elapsed
            if sleep_time > 0:
                await asyncio.sleep(sleep_time)

    except KeyboardInterrupt:
        logger.info("Interrupted by user.")
    finally:
        await adapter.stop()
        if recorder is not None:
            recorder.close()

    # --- Results ---
    elapsed = time.monotonic() - start_time
    stats: dict[str, object] = {
        "elapsed_s": elapsed,
        "loop_count": loop_count,
        "avg_loop_hz": loop_count / max(elapsed, 1e-6),
    }

    # Save flight log
    Path("logs").mkdir(exist_ok=True)
    log_path = f"logs/race_{int(time.time())}.npz"
    flight_log.save(log_path)

    logger.info(
        "Race complete. %.1fs elapsed, %d loops (%.0f Hz avg). Log: %s",
        elapsed,
        loop_count,
        stats["avg_loop_hz"],
        log_path,
    )

    return stats


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Run AI Grand Prix autonomy against real simulator"
    )
    parser.add_argument("--config", type=str, default=None, help="Path to TOML config file")
    parser.add_argument(
        "--address",
        type=str,
        default=None,
        help="MAVSDK connection address (e.g. udp://:14540)",
    )
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")
    parser.add_argument(
        "--record",
        type=str,
        default=None,
        metavar="DIR",
        help="Record camera frames + telemetry to DIR for offline training-data capture.",
    )
    parser.add_argument(
        "--record-stride",
        type=int,
        default=1,
        help="When recording, keep every Nth frame (default 1 = every frame).",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.debug else logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )

    asyncio.run(
        run_race(
            config_path=args.config,
            address=args.address,
            debug=args.debug,
            record_dir=args.record,
            record_stride=args.record_stride,
        )
    )


if __name__ == "__main__":
    main()
