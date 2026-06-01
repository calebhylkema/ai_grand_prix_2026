#!/usr/bin/env python3
"""Diagnostic probe to characterise sim response to velocity-target commands.

Runs a fixed sequence of small known velocity commands against the live sim
and prints ODOMETRY response so we can answer:

1. Does ``SET_POSITION_TARGET_LOCAL_NED`` in velocity mode actually drive the
   sim drone?  (Yesterday's smoke 007 tested position mode and the drone
   cruised at constant velocity instead of converging — strong suggestion the
   sim only honours velocity mode.  The reference example also uses velocity
   mode exclusively.)
2. Are the units metres / m/s as expected, or scaled (cm)?
3. Does the sim hold altitude when commanded ``vd=0``, or does it fall?

Sequence (each phase ~4 s, plus 1 s idle between):
- Phase A: climb at 1 m/s NED  (vd = -1.0)
- Phase B: 1 m/s south, hold altitude  (vn = -1.0)
- Phase C: 1 m/s west, hold altitude   (ve = -1.0)
- Phase D: hold (all zero)

All commands are small (1 m/s) so the drone stays near the start pad even
if our frame interpretation is wrong by a factor of a few.

Run while a race is live (Race button already clicked, countdown finished).
The probe will wait up to 120 s for ``RaceStatus.is_live`` to flip True.
"""

from __future__ import annotations

import asyncio
import logging
import sys
import time
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np  # noqa: E402

from drone_racing.sim.mavlink_adapter import MAVLinkAdapter  # noqa: E402
from drone_racing.utils.config import load_config  # noqa: E402
from drone_racing.utils.types import DroneCommand  # noqa: E402

logger = logging.getLogger(__name__)


SEND_HZ: float = 50.0
"""Velocity command resend rate while a phase is active.
Matches our main control loop rate; yesterday's probe at 20 Hz saw no response."""

ATTITUDE_WAKEUP_S: float = 2.0
"""Stream idle SET_ATTITUDE_TARGET commands for this long before the velocity
probe. Hypothesis: the sim's autopilot stays in a held / pre-flight state
until it has seen at least one stream of rate-target commands, after which it
honours subsequent SET_POSITION_TARGET_LOCAL_NED messages."""

LOG_PERIOD_S: float = 0.25
"""How often to print a telemetry sample."""

WAIT_FOR_LIVE_S: float = 120.0
"""Max wall-clock seconds to wait for ``RaceStatus.is_live`` before giving up."""

PROBE_SEQUENCE: list[dict[str, Any]] = [
    {"name": "A: climb 1 m/s (vd=-1.0)", "vn": 0.0, "ve": 0.0, "vd": -1.0, "duration_s": 4.0},
    {"name": "idle 1", "vn": 0.0, "ve": 0.0, "vd": 0.0, "duration_s": 1.0},
    {"name": "B: south 1 m/s (vn=-1.0)", "vn": -1.0, "ve": 0.0, "vd": 0.0, "duration_s": 4.0},
    {"name": "idle 2", "vn": 0.0, "ve": 0.0, "vd": 0.0, "duration_s": 1.0},
    {"name": "C: west 1 m/s (ve=-1.0)", "vn": 0.0, "ve": -1.0, "vd": 0.0, "duration_s": 4.0},
    {"name": "D: hold (all zero)", "vn": 0.0, "ve": 0.0, "vd": 0.0, "duration_s": 4.0},
]


def _fmt_vec(v: np.ndarray) -> str:
    """Format a 3-vector with consistent column widths."""
    return f"[{v[0]:>+7.2f} {v[1]:>+7.2f} {v[2]:>+7.2f}]"


async def _wait_for_race_live(adapter: MAVLinkAdapter, timeout_s: float) -> bool:
    """Block until ``adapter.race_status.is_live`` flips True."""
    deadline = time.monotonic() + timeout_s
    last_log = 0.0
    while time.monotonic() < deadline:
        rs = adapter.race_status
        if rs is not None and rs.is_live:
            return True
        now = time.monotonic()
        if now - last_log >= 2.0:
            last_log = now
            state = (
                f"start_ms={rs.race_start_boot_time_ms} sim_ms={rs.sim_boot_time_ms}"
                if rs is not None
                else "no RACE_STATUS yet"
            )
            print(f"  waiting for race-live ... {state}")
        await asyncio.sleep(0.1)
    return False


async def _run_phase(
    adapter: MAVLinkAdapter,
    *,
    name: str,
    vn: float,
    ve: float,
    vd: float,
    duration_s: float,
    pos_origin_enu: np.ndarray,
) -> None:
    """Stream a fixed velocity command for ``duration_s`` and log response."""
    print(f"\n--- {name} ---")
    print(
        "    t_phase  cmd_NED(vn ve vd)   "
        "pos_enu - origin            vel_enu                  att (r p y)°"
    )
    send_period = 1.0 / SEND_HZ
    start = time.monotonic()
    next_log = start
    while time.monotonic() - start < duration_s:
        await adapter.send_velocity_command(vn, ve, vd)
        now = time.monotonic()
        if now >= next_log:
            next_log = now + LOG_PERIOD_S
            t = adapter.telemetry
            dpos = t.odometry_position - pos_origin_enu
            print(
                f"    {now - start:6.2f}s  "
                f"[{vn:>+5.2f} {ve:>+5.2f} {vd:>+5.2f}]  "
                f"{_fmt_vec(dpos)}  {_fmt_vec(t.odometry_velocity)}  "
                f"({np.degrees(t.attitude_roll):>+6.1f} "
                f"{np.degrees(t.attitude_pitch):>+6.1f} "
                f"{np.degrees(t.attitude_yaw):>+6.1f})"
            )
        # Pace the loop without busy-waiting; remainder of send_period.
        await asyncio.sleep(max(0.0, send_period - (time.monotonic() - now)))


async def probe() -> None:
    cfg = load_config()
    adapter = MAVLinkAdapter(cfg.mavlink)

    print("Opening MAVLink endpoint ...")
    await adapter.connect()
    await adapter.start_offboard()

    print("Waiting for race-live (click Race in the sim if not already) ...")
    if not await _wait_for_race_live(adapter, WAIT_FOR_LIVE_S):
        print("Race never went live — aborting probe.")
        await adapter.stop()
        return

    rs_live = adapter.race_status
    if rs_live is not None:
        print(
            f"Race is live. RACE_STATUS: start={rs_live.race_start_boot_time_ms}ms "
            f"sim={rs_live.sim_boot_time_ms}ms gate={rs_live.active_gate_index} "
            f"finish={rs_live.race_finish_time_ns}"
        )
    # Give telemetry a moment to settle after the green light.
    await asyncio.sleep(0.5)

    # Wake-up phase: stream idle SET_ATTITUDE_TARGET so the autopilot engages
    # in rate-control mode before we test position-target.  Hypothesis from
    # smoke 007 (drone moved) vs. earlier probe (drone froze): the difference
    # is whether ATTITUDE_TARGET was streamed first.
    print(f"\n--- W: idle ATTITUDE_TARGET wake-up ({ATTITUDE_WAKEUP_S:.1f}s) ---")
    print("    streaming throttle=0, rates=0 via SET_ATTITUDE_TARGET")
    send_period = 1.0 / SEND_HZ
    wake_start = time.monotonic()
    while time.monotonic() - wake_start < ATTITUDE_WAKEUP_S:
        await adapter.send_command(
            DroneCommand(
                timestamp_ns=adapter.telemetry.imu_timestamp_ns,
                throttle=0.0,
                roll_rate=0.0,
                pitch_rate=0.0,
                yaw_rate=0.0,
            )
        )
        await asyncio.sleep(send_period)

    pos_origin = adapter.telemetry.odometry_position.copy()
    print(f"  origin pos_enu = {_fmt_vec(pos_origin)}")
    print(f"  origin vel_enu = {_fmt_vec(adapter.telemetry.odometry_velocity)}")
    print(
        f"  origin att (deg): roll={np.degrees(adapter.telemetry.attitude_roll):.1f}  "
        f"pitch={np.degrees(adapter.telemetry.attitude_pitch):.1f}  "
        f"yaw={np.degrees(adapter.telemetry.attitude_yaw):.1f}"
    )

    try:
        for phase in PROBE_SEQUENCE:
            await _run_phase(
                adapter,
                name=phase["name"],
                vn=phase["vn"],
                ve=phase["ve"],
                vd=phase["vd"],
                duration_s=phase["duration_s"],
                pos_origin_enu=pos_origin,
            )
    finally:
        print("\nProbe complete — stopping adapter.")
        await adapter.stop()


def main() -> None:
    logging.basicConfig(level=logging.WARNING, format="%(name)s %(levelname)s %(message)s")
    asyncio.run(probe())


if __name__ == "__main__":
    main()
