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
import math
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
from drone_racing.sim.mavlink_adapter import MAVLinkAdapter
from drone_racing.utils.config import load_config
from drone_racing.utils.types import DroneCommand, GateDetection, State

logger = logging.getLogger(__name__)

# Measured hover throttle for the AI-GP sim (2026-06-02 via hover_probe.py).
# At pitch=17.8° nose-down, throttle 0.40 produced +2.5 m/s² vertical accel
# which back-solved to a level-hover throttle of ~0.30.  Smoke 015 showed
# the drone climbing ~5 m/s at 0.30 once stabilised level, so the level
# hover throttle is actually closer to 0.27.
HOVER_THROTTLE: float = 0.275

# SMOKE 024 (2026-06-02b): altitude-hold on throttle.  At fixed 0.275 the
# drone climbed ~2 m/s and orbited gate 0 from ~100 m above it, so it could
# never thread the gate.  Close a proportional loop on (target_up − drone_up)
# around the HOVER_THROTTLE baseline, clamped so a large altitude error can
# only command a gentle descent rather than a free-fall.
# SMOKE 050 (2026-06-02c): the course DESCENDS steeply (gate alts 0→−5→−14→
# −25 m) and gates are only 2.72 m tall (±1.36 m tolerance).  At the old gains
# the ±2 m altitude error overshot the hoop and the drone clipped the top bar
# at gate 2.  Tightened the gain and dropped the throttle floor so it can
# descend fast enough to stay inside the hoop on the steep legs.
# SMOKE 057 (2026-06-02c): P-only altitude was underdamped — on the steep
# gate-to-gate descents it stooped hard then overshot back up, oscillating and
# missing the next gate.  Add a derivative term on vertical velocity (from the
# trustworthy finite-diff Z) to damp the descent and kill the overshoot.
HOVER_KP_THROTTLE: float = 0.02  # throttle units per metre of altitude error
HOVER_KD_THROTTLE: float = 0.05  # throttle units per (m/s) of vertical velocity
HOVER_THROTTLE_MIN: float = 0.12
HOVER_THROTTLE_MAX: float = 0.40

# P-controller gain (rad/s per rad of attitude error) for the rate-mode
# hover controller.  axis_sign_probe.py (2026-06-02) showed the sim
# amplifies our commanded rate by ~2.4× (probably a built-in "rate gain"
# in the Acro mode rate loop), so 1.0 here behaves like 2.4 in the sim.
HOVER_KP_ATTITUDE: float = 1.0

# Pitch target (rad) for the hover controller.  Smoke 017 (2026-06-02)
# showed the drone obediently settles at the commanded pitch but moved in
# the *opposite* direction to what we wanted because of the unusual
# convention: positive ATTITUDE.pitch in this sim is nose-DOWN (confirmed
# by IMU physics: at att.pitch=+17.8° the IMU reads accel=(-3.0, 0, -9.34)
# which decomposes as -g·sin(17.8°) in body_x, characteristic of nose-down).
# For forward thrust toward the active gate we want a small POSITIVE target.
# Cruise pitch.  Speed sweep (smoke 065–069): 5°→41.5 s, 7°→33 s, 9°→30 s,
# 11°→28 s full course.  Settled on 9° — strong speed with comfortable
# threading margin for the 2.72 m gates; 11° rode the edge (push later under
# supervision for VQ2 fastest-time).
HOVER_PITCH_TARGET_RAD: float = +math.radians(9.0)
"""Tilt-angle clamp for the quaternion hover controller.  Beyond ~15° the
small-angle ``tilt ≈ atan2(a, g)`` linearisation breaks down and the drone
can run away laterally before the position term can pull it back."""

# SMOKE 029 (2026-06-02c): distance taper to thread (not graze) the gate.
# Smokes 026–028 showed odometry_velocity has no stable world-frame sign (a
# clean closing approach read +v in one run and −v in another — likely a
# body-frame quantity), so a velocity-based speed regulator is unreliable.
# Instead taper the forward pitch by gate distance — only gate_dist_m is
# trustworthy.  Far ⇒ full forward; within HOVER_SLOW_RADIUS the drone creeps
# in slowly so its lateral miss shrinks below the gate radius and it threads,
# instead of sailing ~3 m past the hoop and oscillating through the region.
# SMOKE 067 (2026-06-02c): the course is near-straight, so braking to 30% at
# every gate was over-slowing (racing principle: carry momentum, only slow for
# tight turns).  Slow later (10 m) and less (0.55 floor) to carry speed through.
HOVER_SLOW_RADIUS: float = 10.0  # m; begin slowing the approach inside this
HOVER_PITCH_MIN_FRAC: float = 0.55  # floor on the taper so it keeps speed up

# SMOKE 038 (2026-06-02c): lead-pursuit / proportional-navigation correction.
# Smoke 037 proved pure pursuit (point nose at gate, drive forward) has a
# stable circular orbit and never converges — it just shrinks with speed.  To
# converge we steer the nose to CANCEL cross-track velocity: bias the heading
# toward the side the drone is drifting so forward thrust kills the drift.
# Uses finite-difference world velocity (position is trustworthy; odometry
# velocity is frame-broken).  Sign of HOVER_KP_LEAD verified empirically.
HOVER_KP_LEAD: float = 0.3  # rad of heading bias per (m/s) of cross-track vel
HOVER_LEAD_MAX_RAD: float = math.radians(45.0)  # clamp on the bias

# SMOKE 059 (2026-06-02c): roll-based cross-track control.  Analysis of the
# alternating left/right gate clips showed lateral motion was 100% yaw-driven
# (roll held at 0), which lags and oscillates ±10 m.  Use ROLL to translate
# sideways toward the gate's lateral (East) line directly — a PD on the East
# offset + East velocity — and back off the yaw-lead so the two don't fight.
# Course runs ~straight in North so body-Y ≈ East; the world→body roll SIGN is
# unverified in this non-standard sim, so it's a flag to flip empirically.
HOVER_KP_ROLL: float = 0.05  # rad of roll per metre of lateral (East) offset
HOVER_KD_ROLL: float = 0.08  # rad of roll per (m/s) of lateral velocity
HOVER_ROLL_CLAMP_RAD: float = math.radians(12.0)  # bound on roll target
# Smoke 059 with +1.0 DIVERGED (East ran to +14 m, wrong way) — classic
# wrong-sign positive feedback, so the world→body roll sign is flipped here.
HOVER_ROLL_SIGN: float = -1.0

# SMOKE 054 (2026-06-02c): freeze the heading once very close to the gate.
# At close range the bearing to the gate whips around as the drone moves
# laterally, so the yaw target chases it and the drone spins in place instead
# of crossing the plane (smoke 053 at gate 2).  Once inside HOVER_COMMIT_RADIUS
# we lock the (lead-corrected) heading captured on entry and fly straight
# through; release only beyond HOVER_COMMIT_RELEASE so a wobble can't unlock.
# Pitch/taper are left alone (no full-forward punch — that overshot in 035).
HOVER_COMMIT_RADIUS: float = 10.0
HOVER_COMMIT_RELEASE: float = 12.0

# SMOKE 071 (2026-06-02c): EMA low-pass on the commanded rates to smooth the
# flight.  The stacked corrections (lead yaw + roll cross-track + commit/taper
# gating) are computed from instantaneous error with no smoothing, so the rate
# commands jump tick-to-tick and the flight is jolty.  Blend each new command
# with the previous one: cmd = a·new + (1−a)·prev.  Lower a = smoother but more
# lag; 0.35 is a balance that stays responsive enough to thread the gates.
HOVER_CMD_SMOOTH_ALPHA: float = 0.35
# EMA factor for the finite-diff velocity (noisy dP/dt) before it feeds the
# lead / roll / altitude-D terms.  Lower = smoother velocity, more lag.
HOVER_VEL_SMOOTH_ALPHA: float = 0.3

# Gravitational acceleration used for tilt-from-acceleration conversion.
_G: float = 9.81

# VQ1 max run time (from spec)
MAX_RUN_TIME_S = 8 * 60  # 8 minutes

# SMOKE 030 (2026-06-02c): abort a run early once the drone strays this far
# from its race-start position.  When the drone falls off the start platform
# before the race goes live, the captured reference is corrupt and the drone
# chases a phantom-distant gate clean out of the arena (observed flying 200–
# 580 m before the sim froze the run).  A real VQ1 course leg is nowhere near
# this far, so beyond MAX_EXCURSION_M we're certainly off-course — cut the run
# and (with --auto-restart) Backspace straight into the next attempt.
# NOTE: the VQ1 course is ~159 m long (gate 5 at North −159, ~161 m from the
# origin start), so 150 was cutting off the final stretch near gate 5.  Raised
# well past the course length; genuine runaways (300–580 m) and in-course
# stalls are still caught here or by the no-progress / stuck failsafes.
MAX_EXCURSION_M = 250.0

# SMOKE 057 (2026-06-02c): stationary-stuck failsafe.  Besides flying off
# (excursion), the drone also gets *stuck* — telemetry freezes (crash/DNF) or
# it hovers in one spot — and we'd otherwise wait out the whole run.  If it
# stays within STUCK_RADIUS of an anchor for STUCK_TIMEOUT_S while racing,
# reset and try again.  (A normally-progressing drone covers >STUCK_RADIUS
# between gates, so this only fires on genuine stalls/freezes, not on a
# wide orbit — that's a controller-tuning issue, handled separately.)
STUCK_RADIUS_M = 2.5
STUCK_TIMEOUT_S = 20.0  # only a true freeze/hover stays this still this long

# SMOKE 060 (2026-06-02c): no-progress failsafe.  The drone can also LOOP a
# gate — moving (so not "stuck") but circling without ever passing.  If no new
# gate is passed within NO_PROGRESS_TIMEOUT_S of the last pass (or race start),
# reset and retry instead of orbiting forever.  Set well above a legit slow
# thread so it only fires on genuine loops, never on a good-but-slow run.
NO_PROGRESS_TIMEOUT_S = 60.0
# Rough starts (bad takeoff) loop on gate 0.  A good start passes gate 0 in
# ~8 s, so use a much shorter timeout while still on the FIRST gate to recover
# from bad starts fast (saves ~40 s/attempt) without risking mid-course gates.
FIRST_GATE_TIMEOUT_S = 22.0
# Bad-takeoff detector: the sim's rough auto-takeoff launches the drone well
# above the start height (to +25 m) within seconds, while a good start stays
# near gate-0 altitude (~0).  If it climbs past this above the start while still
# on gate 0, reset immediately instead of waiting out the first-gate timeout.
BAD_TAKEOFF_CLIMB_M = 8.0

# SMOKE 043 (2026-06-02c): a race counts as "live" for the whole 8 min after
# its countdown, so if the sim is left running we connect and find a race that
# started long ago — the drone has been drifting/fell off the platform with no
# control, and handing off to it flies straight out of bounds.  Only hand
# control to a FRESH race: one whose age (sim_boot − race_start) is small.
# Otherwise keep Backspacing until a fresh countdown produces one.
STALE_RACE_MS = 8000  # a live race older than this at first sight ⇒ stale
STALE_RESTART_PERIOD_S = 3.0  # re-send Backspace at most this often

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


def _maybe_get_camera_frame(adapter: MAVLinkAdapter) -> tuple[np.ndarray, int] | None:
    """Fetch the latest (frame, timestamp_ns) from the adapter if available."""
    frame = adapter.latest_frame
    ts = adapter.latest_frame_timestamp_ns
    if frame is None or ts is None:
        return None
    return frame, int(ts)


async def run_race(
    config_path: str | None = None,
    address: str | None = None,
    debug: bool = False,
    record_dir: str | None = None,
    record_stride: int = 1,
    controller_type: str = "mpcc",
    rl_checkpoint: str | None = None,
    auto_restart: bool = False,
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
    record_dir : str | None
        Directory for frame recording.
    record_stride : int
        Keep every Nth frame when recording.
    controller_type : str
        Controller to use: ``"mpcc"`` (default), ``"pid"``, or ``"rl"``.
    rl_checkpoint : str | None
        Path to SB3 model ``.zip`` (required when ``controller_type="rl"``).

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
    adapter = MAVLinkAdapter(config.mavlink)
    await adapter.connect()

    # Start telemetry subscriptions BEFORE waiting/initialising so the snapshot
    # carries real message timestamps when ESKF.initialise() reads it.  Without
    # this ordering, telemetry.timestamp_ns is 0 and the first ESKF predict()
    # treats the monotonic-clock IMU timestamp as a multi-million-second dt
    # and the state explodes to ~10^12 m.
    await adapter.start_offboard()

    # With --auto-restart, send Backspace once at startup so the user can
    # leave the sim focused indefinitely and we'll trigger a fresh race for
    # each smoke run without them clicking anything.  Race-end auto-restart
    # also still fires per the main-loop hook below.
    if auto_restart:
        try:
            import keyboard

            keyboard.send("backspace")
            logger.info("auto-restart: sent Backspace at startup (sim window must be focused)")
        except Exception:
            logger.exception("auto-restart: failed to send startup Backspace")
        # Give the sim time to consume the keystroke and start a fresh
        # countdown before we begin probing race state.
        await asyncio.sleep(3.0)

    # Allow MAVLink + vision streams to populate the snapshot.
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

    # RL controller (built lazily once racing line exists)
    rl_controller = None
    use_rl = controller_type == "rl"
    use_mpcc = controller_type == "mpcc"

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

    # Initialise ESKF from initial telemetry (now carrying real timestamps).
    initial_telemetry = adapter.get_initial_telemetry()
    eskf.initialise(initial_telemetry)
    pid_controller.reset()

    logger.info(
        "Pipeline initialised. Initial position: %s",
        initial_telemetry.position,
    )

    # Track whether we've seen race_started flip True yet.  The sim shows a
    # "throttle down please" banner and refuses to start the race while we are
    # sending non-zero thrust, so we must stream pure-idle commands during the
    # countdown and only switch to the real controller once RACE_STATUS says
    # the race is live.
    race_was_started: bool = False

    # Hover target captured the moment the race goes live.  Using a *fixed*
    # target is essential — if we instead reused ``current_state.position`` as
    # the PID setpoint each tick, the setpoint would drift with the drone and
    # PID would see zero error while the drone flies off ballistically.
    hover_setpoint_pos: np.ndarray | None = None
    hover_setpoint_yaw_ned: float | None = None
    """Yaw (in NED convention, radians) captured at race-live.  Used by the
    ``hover`` controller as the fallback when no active gate is known."""

    # Active-gate tracking for the steer-the-drift hover strategy.
    last_active_gate_index: int = -1

    # Latched True when the drone strays past MAX_EXCURSION_M; forces idle and
    # (with --auto-restart) a Backspace, then clears when the race goes
    # not-live so the next attempt re-arms cleanly.
    excursion_aborted: bool = False

    # Previous position/timestamp for finite-difference world velocity, plus
    # the EMA-smoothed velocity estimate (reduces command jitter).
    prev_pos: np.ndarray | None = None
    prev_ts_ns: int | None = None
    fd_vel_smooth: np.ndarray | None = None

    # Throttle for re-Backspacing when we've joined a stale (already-running)
    # race.  0.0 ⇒ the first stale detection may Backspace immediately.
    last_stale_restart_t: float = 0.0

    # One-shot guard for dumping the gate-map course layout.
    gate_map_dumped: bool = False

    # Frozen (lead-corrected) heading captured on entering the gate commit
    # radius; None when not committed.  See HOVER_COMMIT_RADIUS.
    commit_yaw_ned: float | None = None

    # Stationary-stuck failsafe anchor (position + monotonic time).
    stuck_anchor_pos: np.ndarray | None = None
    stuck_anchor_t: float = 0.0

    # No-progress failsafe: monotonic time of the last gate pass / race start.
    last_progress_t: float = 0.0

    # Previous hover command for EMA rate smoothing (reset each race).
    hover_cmd_prev: DroneCommand | None = None

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

            # ESKF is currently bypassed for control: against the AI-GP sim
            # ODOMETRY is ground-truth and there is no PnP loop to fuse, so
            # leaving the ESKF in charge of `current_state` would only add a
            # frame-conversion drift bug (see project_eskf_bypass memo).  We
            # still run predict() so the state is available for log inspection
            # and for the physical stage later, but the live pipeline reads the
            # adapter's ODOMETRY-derived state directly.
            current_state = adapter.latest_state

            # Finite-difference world velocity from position.  The sim's
            # odometry_velocity is in a non-standard rotated frame for the
            # horizontal axes (smoke 037 offline check: |v| matched dP/dt but
            # direction was uncorrelated), whereas position is ground-truth, so
            # we derive a trustworthy world-frame velocity ourselves.
            fd_world_vel: np.ndarray | None = None
            if prev_pos is not None and prev_ts_ns is not None:
                dt_fd = (current_state.timestamp_ns - prev_ts_ns) / 1e9
                if dt_fd > 1e-3:
                    raw_v = (current_state.position - prev_pos) / dt_fd
                    # EMA-smooth the differentiated velocity (smoke 071): raw
                    # dP/dt is noisy at ~100 Hz and feeds the yaw-lead, roll
                    # cross-track and altitude-D terms, so its jitter is a main
                    # source of jolty control.  Low-pass it at the source.
                    if fd_vel_smooth is None:
                        fd_vel_smooth = raw_v
                    else:
                        fd_vel_smooth = (
                            HOVER_VEL_SMOOTH_ALPHA * raw_v
                            + (1.0 - HOVER_VEL_SMOOTH_ALPHA) * fd_vel_smooth
                        )
                    fd_world_vel = fd_vel_smooth
            prev_pos = current_state.position.copy()
            prev_ts_ns = current_state.timestamp_ns

            # One-shot dump of the course layout so we can reason about gate
            # geometry (turns, altitude changes, hoop sizes, facing) offline.
            if not gate_map_dumped and adapter.gate_map:
                gate_map_dumped = True
                for g in adapter.gate_map:
                    gn, ge, gd = g.position_ned
                    logger.info(
                        "GATEMAP id=%d pos_ned=(%.1f,%.1f,%.1f) alt_up=%.1f "
                        "wxyz=(%.3f,%.3f,%.3f,%.3f) w=%.2f h=%.2f",
                        g.gate_id,
                        gn,
                        ge,
                        gd,
                        -gd,
                        *g.orientation_ned_wxyz,
                        g.width_m,
                        g.height_m,
                    )

            # --- Vision pipeline ---
            frame_info = _maybe_get_camera_frame(adapter)
            tracked: list[GateDetection] = []
            if frame_info is not None:
                frame, frame_ts_ns = frame_info
                raw_detections = detector.detect(frame, frame_ts_ns)
                tracked = tracker.update(raw_detections)
                mapper.observe_many(tracked, current_state)
                if recorder is not None and frame_ts_ns != last_recorded_ts:
                    recorder.record(frame, frame_ts_ns, current_state, tracked)
                    last_recorded_ts = frame_ts_ns

            # --- Planning + Control ---

            # Build / rebuild the racing line once we have enough consolidated
            # gate waypoints.  Rebuilding each frame is cheap relative to MPCC,
            # but we only do it when the waypoint count changes to keep MPCC's
            # cached interpolants stable.
            waypoints = mapper.waypoints()
            if len(waypoints) >= MIN_MAPPED_GATES_FOR_MPCC and racing_line is None:
                try:
                    racing_line = _build_racing_line_from_waypoints(
                        waypoints, config.planner, closed=True
                    )
                    if use_rl:
                        from drone_racing.control.rl_controller import RLController

                        rl_controller = RLController.from_checkpoint(
                            rl_checkpoint,  # type: ignore[arg-type]
                            racing_line=racing_line,
                            control_config=config.control,
                        )
                        logger.info(
                            "Racing line built from %d gates — switching to RL",
                            len(waypoints),
                        )
                    elif use_mpcc:
                        mpcc = MPCCSolver.from_planner_config(
                            racing_line, config.planner, closed=True
                        )
                        logger.info(
                            "Racing line built from %d gates — switching to MPCC",
                            len(waypoints),
                        )
                except Exception:  # noqa: BLE001
                    logger.exception("Failed to build racing line — staying on PID hover")
                    mpcc = None
                    racing_line = None
                    rl_controller = None

            cmd: DroneCommand | None = None
            if rl_controller is not None and racing_line is not None:
                cmd = rl_controller.update(current_state)
            elif mpcc is not None and racing_line is not None:
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
                # PID hover baseline: hold the fixed take-off position (set
                # the moment the race went live).  Pre-race we have nothing to
                # hold yet — fall back to the current position so the PID has
                # a valid setpoint to compute against (the command is about to
                # be overridden to idle anyway).
                target_pos = (
                    hover_setpoint_pos if hover_setpoint_pos is not None else current_state.position
                )
                setpoint = TrajectorySetpoint(
                    position=target_pos,
                    yaw=0.0,
                )
                cmd = pid_controller.update(current_state, setpoint)

            # Race-start gate: override every command to idle until the sim
            # reports the race has gone live (green light).  ``race_started``
            # flips True the moment the user clicks Race (countdown begins),
            # but applying throttle during the countdown DQs the run, so we
            # gate on ``is_live`` which only fires after the countdown ends.
            rs = adapter.race_status
            race_live = rs is not None and rs.is_live

            # Active-gate transition logging: when active_gate_index advances,
            # we've just passed a gate (the only success signal we have).
            if rs is not None and race_live and rs.active_gate_index != last_active_gate_index:
                if last_active_gate_index >= 0:
                    logger.info(
                        "GATE PASSED — now targeting gate %d (was %d), t=%.1fs",
                        rs.active_gate_index,
                        last_active_gate_index,
                        elapsed,
                    )
                last_active_gate_index = rs.active_gate_index
                last_progress_t = time.monotonic()  # made progress; reset timer

            # Re-arm the start-transition on race end so the next Restart
            # captures a fresh hover target and resets PID integrator state.
            if rs is not None and rs.race_finished:
                if race_was_started and auto_restart:
                    # Only fire the keystroke on the transition into finished,
                    # so we don't spam Backspace once per tick.
                    try:
                        import keyboard

                        keyboard.send("backspace")
                        logger.info("auto-restart: race finished, sent Backspace to sim")
                    except Exception:
                        logger.exception("auto-restart: failed to send Backspace")
                race_was_started = False
                hover_setpoint_pos = None
                hover_setpoint_yaw_ned = None

            # Far-excursion cutout: if the drone has strayed past
            # MAX_EXCURSION_M from its race-start position, the run is
            # off-course (usually a bad platform start) — abort early and
            # restart instead of waiting for the out-of-bounds freeze.
            if race_live and hover_setpoint_pos is not None and not excursion_aborted:
                excursion = float(np.linalg.norm(current_state.position - hover_setpoint_pos))
                if excursion > MAX_EXCURSION_M:
                    logger.warning(
                        "Excursion %.0f m > %.0f m from start — off course, aborting run early.",
                        excursion,
                        MAX_EXCURSION_M,
                    )
                    excursion_aborted = True
                    race_was_started = False
                    hover_setpoint_pos = None
                    hover_setpoint_yaw_ned = None
                    if auto_restart:
                        try:
                            import keyboard

                            keyboard.send("backspace")
                            logger.info("auto-restart: excursion abort, sent Backspace to sim")
                        except Exception:
                            logger.exception("auto-restart: failed to send Backspace")

            # Stationary-stuck + no-progress failsafes: reset if the drone
            # hasn't moved beyond STUCK_RADIUS_M for STUCK_TIMEOUT_S (frozen /
            # hovering) OR hasn't passed a gate within NO_PROGRESS_TIMEOUT_S
            # (looping a gate), so we don't wait out the whole run.
            if race_live and race_was_started and not excursion_aborted:
                now_s = time.monotonic()
                if stuck_anchor_pos is None or (
                    float(np.linalg.norm(current_state.position - stuck_anchor_pos))
                    > STUCK_RADIUS_M
                ):
                    stuck_anchor_pos = current_state.position.copy()
                    stuck_anchor_t = now_s
                stuck = now_s - stuck_anchor_t > STUCK_TIMEOUT_S
                # Shorter timeout while still on gate 0 (rough-start recovery).
                on_first_gate = rs is not None and rs.active_gate_index <= 0
                progress_timeout = FIRST_GATE_TIMEOUT_S if on_first_gate else NO_PROGRESS_TIMEOUT_S
                no_progress = now_s - last_progress_t > progress_timeout
                # Bad takeoff: climbed far above the start while still on gate 0.
                bad_takeoff = (
                    on_first_gate
                    and hover_setpoint_pos is not None
                    and float(current_state.position[2]) - float(hover_setpoint_pos[2])
                    > BAD_TAKEOFF_CLIMB_M
                )
                if stuck or no_progress or bad_takeoff:
                    logger.warning(
                        "%s — resetting run.",
                        "Stuck stationary"
                        if stuck
                        else "Bad takeoff (climbed away)"
                        if bad_takeoff
                        else f"No gate passed in {progress_timeout:.0f}s (looping)",
                    )
                    excursion_aborted = True
                    race_was_started = False
                    hover_setpoint_pos = None
                    hover_setpoint_yaw_ned = None
                    stuck_anchor_pos = None
                    if auto_restart:
                        try:
                            import keyboard

                            keyboard.send("backspace")
                            logger.info("auto-restart: failsafe reset, sent Backspace to sim")
                        except Exception:
                            logger.exception("auto-restart: failed to send Backspace")

            # Stale-race guard: a race reads "live" for its whole 8 min, so if
            # we connected to one that started long ago the drone has been
            # uncontrolled (drifted / fell off the platform).  Refuse to hand
            # off to it and Backspace until a FRESH race (small age) appears.
            joined_stale = False
            if race_live and not race_was_started and rs is not None:
                race_age_ms = rs.sim_boot_time_ms - rs.race_start_boot_time_ms
                if race_age_ms > STALE_RACE_MS:
                    joined_stale = True
                    now_m = time.monotonic()
                    if auto_restart and now_m - last_stale_restart_t > STALE_RESTART_PERIOD_S:
                        last_stale_restart_t = now_m
                        try:
                            import keyboard

                            keyboard.send("backspace")
                            logger.warning(
                                "Joined stale race (age %.0fs) — Backspacing for a fresh start.",
                                race_age_ms / 1000.0,
                            )
                        except Exception:
                            logger.exception("auto-restart: failed to send Backspace")

            if not race_live:
                excursion_aborted = False
                stuck_anchor_pos = None
                hover_cmd_prev = None  # don't carry smoothing across races
            if not race_live or excursion_aborted or joined_stale:
                cmd = DroneCommand(
                    timestamp_ns=cmd.timestamp_ns,
                    throttle=0.0,
                    roll_rate=0.0,
                    pitch_rate=0.0,
                    yaw_rate=0.0,
                )
            elif not race_was_started:
                # Transition pre-race → live: clear PID integrator state so we
                # don't kick from any wind-up that accumulated during idle, and
                # capture the take-off pose as a fixed hover target.
                pid_controller.reset()
                hover_setpoint_pos = current_state.position.copy()
                hover_setpoint_yaw_ned = float(adapter.telemetry.attitude_yaw)
                race_was_started = True
                last_progress_t = time.monotonic()  # start the no-progress clock
                logger.info(
                    "Race started — handing control to %s (active_gate=%d) "
                    "hover_target=%s yaw=%.1f°",
                    controller_type.upper(),
                    rs.active_gate_index,
                    np.array2string(hover_setpoint_pos, precision=2),
                    np.degrees(hover_setpoint_yaw_ned),
                )

            cmd = rate_limiter.limit(cmd)

            # --- Send command ---
            # Three send paths depending on controller_type:
            #   - position : SET_POSITION_TARGET_LOCAL_NED with active gate
            #                (currently broken — see project_session memo)
            #   - hover    : SET_ATTITUDE_TARGET in quaternion mode targeting
            #                "level at captured yaw" + measured hover throttle
            #   - pid/mpcc/rl : rate-mode SET_ATTITUDE_TARGET (existing path)
            # Pre-race for every mode we send the idle rate-mode command so
            # the throttle-down banner stays satisfied.
            alt_message_sent = False
            if race_live and not excursion_aborted and not joined_stale:
                if controller_type == "position":
                    rs = adapter.race_status
                    gates = adapter.gate_map
                    if rs is not None and gates:
                        gate_idx = rs.active_gate_index
                        if 0 <= gate_idx < len(gates):
                            gate = gates[gate_idx]
                            n, e, d = gate.position_ned
                            await adapter.send_position_command(
                                north=n, east=e, down=d, yaw_deg=0.0
                            )
                            alt_message_sent = True
                elif controller_type == "hover" and hover_setpoint_yaw_ned is not None:
                    # Rate-mode hover for Acro mode (smoke 013, 2026-06-02).
                    # Quaternion mode produced uncontrolled yaw spin (smokes
                    # 010/011/012); Acro is a body-rate flight mode and the
                    # sim ignores the quaternion field, so we close the
                    # attitude loop ourselves.  Yaw target points at the
                    # active gate so the steered forward thrust carries us
                    # toward it; pitch and roll targets are zero (level).
                    # SMOKE 023 (2026-06-02b): gate steering RE-ENABLED with the
                    # CORRECTED yaw sign.  Smokes 019–022 proved the sim reports
                    # ATTITUDE.yaw with the OPPOSITE sign to standard NED (yaw is
                    # +CCW / ENU-style, same non-standard family as
                    # positive-pitch=nose-down).  The drone's nose points at
                    # heading (cosθ, −sinθ), i.e. heading = −θ.  So to face a
                    # bearing atan2(de, dn) we must command target = atan2(−de,
                    # dn).  gate_dist_m logged below validates we're closing in.
                    target_yaw_ned = hover_setpoint_yaw_ned
                    gate_dist_m = float("nan")
                    drone_up = float(current_state.position[2])
                    # Default: hold the RACE-START altitude, not the current one
                    # (smoke 065).  Gate 0 is at the start height, and anchoring
                    # here means a rough-start climb (e.g. the sim's auto-takeoff
                    # kick) gets braked immediately instead of the target just
                    # following the drone up while the gate map loads.
                    target_up = (
                        float(hover_setpoint_pos[2]) if hover_setpoint_pos is not None else drone_up
                    )
                    # Closing speed toward the gate (m/s); 0 ⇒ full forward cmd.
                    v_close = 0.0
                    # Roll target for lateral (cross-track) correction; 0 ⇒ level.
                    roll_target_eff = 0.0
                    rs = adapter.race_status
                    gates = adapter.gate_map
                    if rs is not None and gates and 0 <= rs.active_gate_index < len(gates):
                        gate = gates[rs.active_gate_index]
                        gate_n, gate_e, gate_d = gate.position_ned
                        # Drone position in NED: ENU x→NED-east, ENU y→NED-north.
                        drone_n = float(current_state.position[1])
                        drone_e = float(current_state.position[0])
                        dn = gate_n - drone_n
                        de = gate_e - drone_e
                        gate_dist_m = math.hypot(dn, de)
                        # Gate altitude in ENU-up = −(NED down).
                        target_up = -gate_d
                        # Roll-based lateral correction toward the gate's East
                        # line: PD on East offset (de) + East velocity.  Course
                        # runs ~straight in North so body-Y ≈ East.  Sign is a
                        # flag (HOVER_ROLL_SIGN) pending empirical confirmation.
                        v_east = float(fd_world_vel[0]) if fd_world_vel is not None else 0.0
                        roll_target_eff = HOVER_ROLL_SIGN * float(
                            np.clip(
                                HOVER_KP_ROLL * de - HOVER_KD_ROLL * v_east,
                                -HOVER_ROLL_CLAMP_RAD,
                                HOVER_ROLL_CLAMP_RAD,
                            )
                        )
                        if gate_dist_m > 1e-3:
                            # Closing speed = horizontal velocity projected onto
                            # the unit vector toward the gate.  ENU velocity:
                            # [0]=East, [1]=North; (de,dn) are East/North diffs.
                            # Positive ⇒ approaching.  (Smoke 027's clean 193 m
                            # approach confirmed this sign; smoke 026's apparent
                            # inversion was the launch transient dominating a
                            # short 21 m approach, not a frame flip.)
                            vel = current_state.velocity
                            v_close = (float(vel[0]) * de + float(vel[1]) * dn) / gate_dist_m
                        if dn * dn + de * de > 1.0:
                            # Negated yaw sign — see comment above.
                            base_yaw = math.atan2(-de, dn)
                            # Lead-pursuit: bias the heading to cancel cross-
                            # track velocity so we converge instead of orbit.
                            # Cross-track unit (E,N) = +90° rotate of the
                            # gate-bearing unit = (−dn, de)/dist.
                            lead = 0.0
                            if fd_world_vel is not None:
                                v_cross = (
                                    float(fd_world_vel[0]) * (-dn) + float(fd_world_vel[1]) * de
                                ) / gate_dist_m
                                lead = float(
                                    np.clip(
                                        HOVER_KP_LEAD * v_cross,
                                        -HOVER_LEAD_MAX_RAD,
                                        HOVER_LEAD_MAX_RAD,
                                    )
                                )
                            target_yaw_ned = base_yaw - lead

                    # Close-range heading freeze (smoke 054): lock the lead-
                    # corrected heading on entry so we fly straight through the
                    # plane instead of chasing the whipping close-range bearing.
                    if not math.isnan(gate_dist_m) and (
                        gate_dist_m < HOVER_COMMIT_RADIUS
                        or (commit_yaw_ned is not None and gate_dist_m < HOVER_COMMIT_RELEASE)
                    ):
                        if commit_yaw_ned is None:
                            commit_yaw_ned = target_yaw_ned
                        target_yaw_ned = commit_yaw_ned
                    else:
                        commit_yaw_ned = None

                    # PD altitude hold: P pulls toward the gate altitude, D
                    # (on the reliable finite-diff vertical velocity) damps the
                    # descent so it doesn't overshoot.  v_up>0 = climbing.
                    v_up = float(fd_world_vel[2]) if fd_world_vel is not None else 0.0
                    hover_throttle = float(
                        np.clip(
                            HOVER_THROTTLE
                            + HOVER_KP_THROTTLE * (target_up - drone_up)
                            - HOVER_KD_THROTTLE * v_up,
                            HOVER_THROTTLE_MIN,
                            HOVER_THROTTLE_MAX,
                        )
                    )

                    # Read attitude in NED from the adapter (raw MAVLink
                    # ATTITUDE values, aerospace convention).
                    t_att = adapter.telemetry
                    roll_now = float(t_att.attitude_roll)
                    pitch_now = float(t_att.attitude_pitch)
                    yaw_now = float(t_att.attitude_yaw)

                    # Wrap the yaw error to (−π, π] so we always rotate the
                    # short way around.
                    yaw_err = (target_yaw_ned - yaw_now + math.pi) % (2.0 * math.pi) - math.pi

                    # SMOKE 025/029 (2026-06-02c): break the pure-pursuit orbit
                    # with two terms.
                    #  (1) Alignment gate max(0, cos(yaw_err)): forward thrust
                    #      only when pointed at the gate, ~0 while the bearing
                    #      swings, so the drone coasts and lets yaw catch up
                    #      instead of looping (smoke 025).
                    #  (2) Distance taper: scale forward pitch down inside
                    #      HOVER_SLOW_RADIUS so the drone creeps the last metres
                    #      and threads the hoop instead of sailing past it
                    #      (smoke 029).
                    align = max(0.0, math.cos(yaw_err))
                    dist_taper = (
                        1.0
                        if math.isnan(gate_dist_m)
                        else float(
                            np.clip(
                                gate_dist_m / HOVER_SLOW_RADIUS,
                                HOVER_PITCH_MIN_FRAC,
                                1.0,
                            )
                        )
                    )
                    pitch_target_eff = HOVER_PITCH_TARGET_RAD * align * dist_taper

                    # Override the cmd computed upstream (PID hover fallback);
                    # rate-mode send_command picks this up below.  Sign
                    # conventions confirmed by axis_sign_probe.py 2026-06-02:
                    #   roll: aerospace  (positive rate → angle increases)
                    #   pitch: RH-rule   (positive rate → angle DECREASES)
                    #   yaw:   RH-rule   (positive rate → angle DECREASES)
                    # To drive each angle toward target=0 (or yaw toward
                    # target_yaw_ned) we pick the sign that produces an
                    # angle-change-toward-target with these conventions.
                    # We do NOT re-apply ``rate_limiter`` here — the upstream
                    # limit() call already updated its stored previous-cmd
                    # to whatever PID produced on this tick, and on tick 1
                    # the rate-of-change clamp would suppress our override.
                    # Our controller already produces bounded rates
                    # (|rate| ≤ KP·π ≈ 3 rad/s, well under sim limits).
                    cmd = DroneCommand(
                        timestamp_ns=current_state.timestamp_ns,
                        throttle=hover_throttle,
                        roll_rate=-HOVER_KP_ATTITUDE * (roll_now - roll_target_eff),
                        pitch_rate=+HOVER_KP_ATTITUDE * (pitch_now - pitch_target_eff),
                        yaw_rate=-HOVER_KP_ATTITUDE * yaw_err,
                    )
                    # EMA low-pass to smooth jolt: blend with the previous cmd.
                    if hover_cmd_prev is not None:
                        a = HOVER_CMD_SMOOTH_ALPHA
                        b = 1.0 - a
                        cmd = DroneCommand(
                            timestamp_ns=cmd.timestamp_ns,
                            throttle=a * cmd.throttle + b * hover_cmd_prev.throttle,
                            roll_rate=a * cmd.roll_rate + b * hover_cmd_prev.roll_rate,
                            pitch_rate=a * cmd.pitch_rate + b * hover_cmd_prev.pitch_rate,
                            yaw_rate=a * cmd.yaw_rate + b * hover_cmd_prev.yaw_rate,
                        )
                    hover_cmd_prev = cmd
                    # Verbose hover-controller log every 1 s so we can verify
                    # the yaw-toward-gate logic is actually computing what we
                    # expect (smoke 015 saw yaw stuck at 52° while target was
                    # -142° toward gate 0 — need to confirm command is sent).
                    if loop_count % int(config.mavlink.command_rate_hz) == 0:
                        logger.info(
                            "HOVER cmd: throttle=%.3f roll_rate=%+.3f "
                            "pitch_rate=%+.3f yaw_rate=%+.3f | "
                            "target_yaw=%+.1f° yaw_now=%+.1f° yaw_err=%+.1f° "
                            "| gate_idx=%d gate_dist=%.1fm dz=%+.1fm vc=%+.1f gate_map_size=%d",
                            cmd.throttle,
                            cmd.roll_rate,
                            cmd.pitch_rate,
                            cmd.yaw_rate,
                            np.degrees(target_yaw_ned),
                            np.degrees(yaw_now),
                            np.degrees(yaw_err),
                            rs.active_gate_index if rs is not None else -1,
                            gate_dist_m,
                            target_up - drone_up,
                            v_close,
                            len(adapter.gate_map),
                        )

            if not alt_message_sent:
                await adapter.send_command(cmd)

            # --- Logging ---
            flight_log.log_state(current_state)
            flight_log.log_command(cmd)

            # Periodic diagnostics (every 5s)
            if loop_count % int(5 * config.mavlink.command_rate_hz) == 0:
                t = adapter.telemetry
                if not race_live:
                    mode = "IDLE"
                elif controller_type == "position":
                    mode = "POS"
                elif controller_type == "hover":
                    mode = "QHOVER"
                elif mpcc is not None:
                    mode = "MPCC"
                else:
                    mode = "HOVER"
                rs_for_log = adapter.race_status
                race_info = (
                    f"start={rs_for_log.race_start_boot_time_ms}ms "
                    f"now={rs_for_log.sim_boot_time_ms}ms "
                    f"gate={rs_for_log.active_gate_index}"
                    if rs_for_log is not None
                    else "no_status"
                )
                logger.info(
                    "DIAG t=%.1fs | mode=%s | race=%s | mapped_gates=%d | "
                    "odom_valid=%s | pos=%s | att=(%.1f, %.1f, %.1f)°",
                    elapsed,
                    mode,
                    race_info,
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
    parser.add_argument(
        "--controller",
        type=str,
        default="mpcc",
        choices=["pid", "mpcc", "rl", "position", "hover"],
        help=(
            "Controller type. "
            "'mpcc' (default) and 'pid'/'rl' are rate-mode (currently tumble "
            "the drone — see project_pid_real_sim_todo memo). "
            "'position' sends SET_POSITION_TARGET_LOCAL_NED at the active gate "
            "(sim ignores it — see session memo). "
            "'hover' streams SET_ATTITUDE_TARGET in quaternion mode targeting "
            "'level at take-off yaw' + measured hover throttle — uses the "
            "sim's internal stabiliser so we don't fight the FRD/FLU "
            "body-frame confusion in our PID."
        ),
    )
    parser.add_argument(
        "--rl-checkpoint",
        type=str,
        default=None,
        help="Path to SB3 model .zip (required when --controller rl)",
    )
    parser.add_argument(
        "--auto-restart",
        action="store_true",
        help=(
            "Send Backspace to the sim window when a race finishes (DQ, time-out, "
            "or completion) to trigger an automatic restart.  Use to iterate "
            "smoke runs without manually clicking between attempts.  Requires "
            "the sim window to be focused for the keystroke to register."
        ),
    )
    args = parser.parse_args()

    if args.controller == "rl" and args.rl_checkpoint is None:
        parser.error("--rl-checkpoint is required when --controller rl")

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
            controller_type=args.controller,
            rl_checkpoint=args.rl_checkpoint,
            auto_restart=args.auto_restart,
        )
    )


if __name__ == "__main__":
    main()
