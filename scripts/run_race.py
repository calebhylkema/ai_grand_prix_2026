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
from drone_racing.logging.flight_logger import FlightLogger
from drone_racing.sim.mavsdk_adapter import MAVSDKAdapter
from drone_racing.utils.config import load_config

logger = logging.getLogger(__name__)

# VQ1 max run time (from spec)
MAX_RUN_TIME_S = 8 * 60  # 8 minutes


async def run_race(
    config_path: str | None = None,
    address: str | None = None,
    debug: bool = False,
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
            # TODO: Vision stream capture not yet implemented.
            # Waiting on separate vision spec from organizers.
            # For now, if ODOMETRY is available, use it as a position correction.
            if adapter.telemetry.odometry_valid:
                # ODOMETRY can supplement/replace ESKF for VQ1
                # This is a temporary shortcut until vision pipeline is connected
                pass

            # --- Planning + Control ---
            current_state = eskf.state

            # TODO: Replace with racing line planner once gate positions are known.
            # For VQ1 initial testing, use a simple hover or waypoint-based approach.
            # The racing line needs gate positions which we'll discover from vision.

            # Simple hover command as baseline (keeps drone stable)
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
                logger.info(
                    "DIAG t=%.1fs | odom_valid=%s | pos=%s | att=(%.1f, %.1f, %.1f)°",
                    elapsed,
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
        )
    )


if __name__ == "__main__":
    main()
