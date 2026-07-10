#!/usr/bin/env python3
"""Authoritatively identify body-rate sign conventions for the AI-GP sim.

Sends a small known body-rate command on one axis at a time and prints how
the corresponding attitude angle changes.  The combination tells us:

- Does positive ``body_roll_rate`` make ``ATTITUDE.roll`` go up or down?
- Does positive ``body_pitch_rate`` make ``ATTITUDE.pitch`` go up or down?
- Does positive ``body_yaw_rate`` make ``ATTITUDE.yaw`` go up or down?

Each axis is probed in two race attempts (one for positive rate, one for
negative).  Between attempts the script sends Backspace to the sim window
to trigger a race restart, so the user only has to click Race at the very
beginning.

Run while a race is *already live* or with the sim at the VQ1 Race screen.
The script blocks until ``RaceStatus.is_live`` flips True before each phase.
"""

from __future__ import annotations

import asyncio
import logging
import math
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import keyboard  # noqa: E402

from drone_racing.sim.mavlink_adapter import MAVLinkAdapter  # noqa: E402
from drone_racing.utils.config import load_config  # noqa: E402
from drone_racing.utils.types import DroneCommand  # noqa: E402

logger = logging.getLogger(__name__)


HOVER_THROTTLE: float = 0.30
PROBE_RATE: float = 0.5  # rad/s — small enough not to instantly tumble
SEND_HZ: float = 50.0
PROBE_DURATION_S: float = 2.5
RESTART_KEY: str = "backspace"
RESTART_WAIT_S: float = 4.0
WAIT_FOR_LIVE_S: float = 60.0


async def _wait_for_race_live(adapter: MAVLinkAdapter, timeout_s: float) -> bool:
    deadline = time.monotonic() + timeout_s
    last_log = 0.0
    while time.monotonic() < deadline:
        rs = adapter.race_status
        if rs is not None and rs.is_live:
            return True
        now = time.monotonic()
        if now - last_log >= 2.0:
            last_log = now
            print(
                f"  waiting for race-live ... {('rs=' + str(rs) if rs else 'no RACE_STATUS yet')}"
            )
        await asyncio.sleep(0.1)
    return False


async def _send_rate_phase(
    adapter: MAVLinkAdapter,
    *,
    label: str,
    roll_rate: float,
    pitch_rate: float,
    yaw_rate: float,
    duration_s: float,
) -> None:
    print(f"\n--- {label}  rates=(r={roll_rate:+.2f}, p={pitch_rate:+.2f}, y={yaw_rate:+.2f}) ---")
    print("    t_phase  attitude_deg (roll, pitch, yaw)")

    # Snapshot the initial attitude as a baseline.
    t0 = adapter.telemetry
    r0 = math.degrees(t0.attitude_roll)
    p0 = math.degrees(t0.attitude_pitch)
    y0 = math.degrees(t0.attitude_yaw)
    print(f"    baseline ({r0:+.1f}, {p0:+.1f}, {y0:+.1f})")

    send_period = 1.0 / SEND_HZ
    start = time.monotonic()
    next_log = start
    while time.monotonic() - start < duration_s:
        cmd = DroneCommand(
            timestamp_ns=adapter.telemetry.imu_timestamp_ns,
            throttle=HOVER_THROTTLE,
            roll_rate=float(roll_rate),
            pitch_rate=float(pitch_rate),
            yaw_rate=float(yaw_rate),
        )
        await adapter.send_command(cmd)
        now = time.monotonic()
        if now >= next_log:
            next_log = now + 0.25
            t = adapter.telemetry
            r = math.degrees(t.attitude_roll)
            p = math.degrees(t.attitude_pitch)
            y = math.degrees(t.attitude_yaw)
            print(
                f"    {now - start:5.2f}s  ({r:+6.1f}, {p:+6.1f}, {y:+6.1f})  "
                f"delta=({r - r0:+5.1f}, {p - p0:+5.1f}, {y - y0:+5.1f})"
            )
        await asyncio.sleep(max(0.0, send_period - (time.monotonic() - now)))


async def _restart_race() -> None:
    print(f"\n[restart] sending {RESTART_KEY} keystroke to sim window ...")
    try:
        keyboard.send(RESTART_KEY)
    except Exception:
        logger.exception("Failed to send restart keystroke")
        return
    print(f"  waiting {RESTART_WAIT_S:.1f}s for restart ...")
    await asyncio.sleep(RESTART_WAIT_S)


PROBE_PLAN: list[tuple[str, dict[str, float]]] = [
    ("ROLL +", {"roll": +PROBE_RATE, "pitch": 0.0, "yaw": 0.0}),
    ("ROLL -", {"roll": -PROBE_RATE, "pitch": 0.0, "yaw": 0.0}),
    ("PITCH +", {"roll": 0.0, "pitch": +PROBE_RATE, "yaw": 0.0}),
    ("PITCH -", {"roll": 0.0, "pitch": -PROBE_RATE, "yaw": 0.0}),
    ("YAW +", {"roll": 0.0, "pitch": 0.0, "yaw": +PROBE_RATE}),
    ("YAW -", {"roll": 0.0, "pitch": 0.0, "yaw": -PROBE_RATE}),
]


async def probe() -> None:
    cfg = load_config()
    adapter = MAVLinkAdapter(cfg.mavlink)

    print("Opening MAVLink endpoint ...")
    await adapter.connect()
    await adapter.start_offboard()

    print(
        "\nWaiting for race-live for the FIRST probe.  Click Race in the sim "
        "if it isn't already running.  Subsequent probes will use the "
        f"{RESTART_KEY!r} key to restart automatically."
    )

    try:
        for i, (label, rates) in enumerate(PROBE_PLAN):
            if i > 0:
                await _restart_race()
            if not await _wait_for_race_live(adapter, WAIT_FOR_LIVE_S):
                print(f"Race never went live for {label} — aborting.")
                break
            # Brief settle so we don't sample mid-spawn-transient.
            await asyncio.sleep(0.5)
            await _send_rate_phase(
                adapter,
                label=label,
                roll_rate=rates["roll"],
                pitch_rate=rates["pitch"],
                yaw_rate=rates["yaw"],
                duration_s=PROBE_DURATION_S,
            )
    finally:
        print("\nProbe complete — stopping adapter.")
        await adapter.stop()

    print(
        "\n=== INTERPRETATION ===\n"
        "For each axis: if positive rate caused the corresponding attitude\n"
        "angle to INCREASE, the sim uses aerospace convention (rate sign\n"
        "matches angle sign).  If positive rate caused the angle to DECREASE,\n"
        "the sim uses right-hand-rule convention (opposite sign).\n"
        "Use the delta column in the per-phase output to read off the answer."
    )


def main() -> None:
    logging.basicConfig(level=logging.WARNING, format="%(name)s %(levelname)s %(message)s")
    asyncio.run(probe())


if __name__ == "__main__":
    main()
