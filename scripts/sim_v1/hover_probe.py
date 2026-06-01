#!/usr/bin/env python3
"""Empirical hover-throttle probe for the AI-GP simulator.

Yesterday and today's experiments established two things:
1. ``SET_POSITION_TARGET_LOCAL_NED`` (position OR velocity mode) is not honoured
   by the sim's autopilot — the drone either ignores it entirely or auto-takes-off
   with ballistic uncontrolled motion.
2. ``SET_ATTITUDE_TARGET`` does drive the drone (yesterday's PID smoke tumbled it).

Before we re-tune the PID we need an empirical anchor for the throttle that
produces sustained hover.  The mock-sim PID code assumes ``_HOVER_THROTTLE =
0.5`` but that is unverified for the real sim — could easily be 0.3 or 0.7
given how the YBC reference sample sends 0.6 in their attitude-rate example.

Sequence (each phase 2 s, throttle sweep with rates=0):
- Phase 1: throttle 0.40
- Phase 2: throttle 0.50
- Phase 3: throttle 0.60
- Phase 4: throttle 0.70
- Phase 5: throttle 0.00 (brake / recovery)

At each phase we report the *world-frame vertical velocity* (ENU z).  The
phase whose final vertical velocity is closest to zero (or whose vertical
velocity is steady, neither growing nor shrinking) is the hover-throttle
neighbourhood.  Drone will drift laterally because the start pose is pitched
17.8° nose-down and we are sending zero rates — that's expected and orthogonal
to the hover measurement.

We stream commands from the instant race-live trips True so the sim doesn't
auto-take-off uncontrolled (observed 2026-06-02).
"""

from __future__ import annotations

import asyncio
import logging
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np  # noqa: E402

from drone_racing.sim.mavlink_adapter import MAVLinkAdapter  # noqa: E402
from drone_racing.utils.config import load_config  # noqa: E402
from drone_racing.utils.types import DroneCommand  # noqa: E402

logger = logging.getLogger(__name__)


SEND_HZ: float = 50.0
"""Attitude-target resend rate (matches our main control loop)."""

LOG_PERIOD_S: float = 0.25
"""How often to print a telemetry sample."""

WAIT_FOR_LIVE_S: float = 120.0
"""Max wall-clock seconds to wait for ``RaceStatus.is_live``."""

PROBE_SEQUENCE: list[dict[str, float]] = [
    {"name": "T=0.40", "throttle": 0.40, "duration_s": 2.0},
    {"name": "T=0.50", "throttle": 0.50, "duration_s": 2.0},
    {"name": "T=0.60", "throttle": 0.60, "duration_s": 2.0},
    {"name": "T=0.70", "throttle": 0.70, "duration_s": 2.0},
    {"name": "T=0.00 brake", "throttle": 0.00, "duration_s": 2.0},
]


def _fmt_vec(v: np.ndarray) -> str:
    return f"[{v[0]:>+7.2f} {v[1]:>+7.2f} {v[2]:>+7.2f}]"


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
    throttle: float,
    duration_s: float,
    pos_origin_enu: np.ndarray,
) -> tuple[float, float]:
    """Stream a fixed ATTITUDE_TARGET command and return final (vz_enu, |v_horiz|)."""
    print(f"\n--- {name}  throttle={throttle:.2f}, rates=0 ---")
    print("    t_phase  pos_enu - origin           vel_enu                  att (r p y)°")
    send_period = 1.0 / SEND_HZ
    start = time.monotonic()
    next_log = start
    last_vel = np.zeros(3)
    while time.monotonic() - start < duration_s:
        cmd = DroneCommand(
            timestamp_ns=adapter.telemetry.imu_timestamp_ns,
            throttle=float(throttle),
            roll_rate=0.0,
            pitch_rate=0.0,
            yaw_rate=0.0,
        )
        await adapter.send_command(cmd)
        now = time.monotonic()
        if now >= next_log:
            next_log = now + LOG_PERIOD_S
            t = adapter.telemetry
            dpos = t.odometry_position - pos_origin_enu
            last_vel = t.odometry_velocity.copy()
            print(
                f"    {now - start:6.2f}s  "
                f"{_fmt_vec(dpos)}  {_fmt_vec(t.odometry_velocity)}  "
                f"({np.degrees(t.attitude_roll):>+6.1f} "
                f"{np.degrees(t.attitude_pitch):>+6.1f} "
                f"{np.degrees(t.attitude_yaw):>+6.1f})"
            )
        await asyncio.sleep(max(0.0, send_period - (time.monotonic() - now)))
    horiz = float(np.linalg.norm(last_vel[:2]))
    vz = float(last_vel[2])
    return vz, horiz


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
            f"Race is live. start={rs_live.race_start_boot_time_ms}ms "
            f"sim={rs_live.sim_boot_time_ms}ms gate={rs_live.active_gate_index}"
        )

    # Capture origin before doing anything.
    pos_origin = adapter.telemetry.odometry_position.copy()
    print(f"\norigin pos_enu = {_fmt_vec(pos_origin)}")
    print(f"origin vel_enu = {_fmt_vec(adapter.telemetry.odometry_velocity)}")
    print(
        f"origin att (deg): roll={np.degrees(adapter.telemetry.attitude_roll):.1f}  "
        f"pitch={np.degrees(adapter.telemetry.attitude_pitch):.1f}  "
        f"yaw={np.degrees(adapter.telemetry.attitude_yaw):.1f}"
    )

    summary: list[tuple[str, float, float, float]] = []
    try:
        for phase in PROBE_SEQUENCE:
            vz, horiz = await _run_phase(
                adapter,
                name=phase["name"],
                throttle=phase["throttle"],
                duration_s=phase["duration_s"],
                pos_origin_enu=pos_origin,
            )
            summary.append((phase["name"], phase["throttle"], vz, horiz))
    finally:
        print("\nProbe complete — stopping adapter.")
        await adapter.stop()

    # Summary table — pick whichever throttle has |vz_enu| smallest while drone
    # is airborne (i.e. drone hasn't crashed back to z=0).
    print("\n=== final-sample vertical / horizontal speeds per throttle ===")
    print("  phase            throttle  vz_enu(m/s)  |v_horiz|(m/s)")
    for name, thr, vz, horiz in summary:
        print(f"  {name:14s}  {thr:>6.2f}     {vz:>+8.2f}      {horiz:>8.2f}")
    print(
        "\nThe phase whose vz_enu is closest to zero (with the drone still "
        "airborne) sits near hover-throttle.  Use that as the starting value "
        "for the PID's _HOVER_THROTTLE constant."
    )


def main() -> None:
    logging.basicConfig(level=logging.WARNING, format="%(name)s %(levelname)s %(message)s")
    asyncio.run(probe())


if __name__ == "__main__":
    main()
