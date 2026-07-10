# VQ1 Official Spec — Consolidated Reference

> Sources: VADR-TS-001 (2026-03-09) + Competition Email Update (April 2026)

## Timeline

| Phase | Window | Notes |
|-------|--------|-------|
| VQ1 | May launch, open through end of VQ2 | Focus on **completion**, not speed |
| VQ2 | June launch, open until mid-late July | Fastest valid time counts |
| Physical Qualifier | September 2026, California | Real drones, no audience |
| Grand Prix Final | November 2026, Ohio | Head-to-head, audience, distractions |

## Simulator

- **Downloadable package**, runs locally
- **Windows only** (Linux not working)
- Active internet required (anti-cheat)
- Can run **multiple instances in parallel**
- Physics update: **120 Hz**
- Rigid-body flight model: thrust, aerodynamic drag, gravity, collision
- **Deterministic**: same course geometry, physics, and environment for all teams

## Communication Protocol — MAVLink v2

- **Transport:** UDP
- **SDK:** MAVSDK-compatible interfaces
- **Command rate:** 50–120 Hz recommended
- **Heartbeat rate:** minimum 2 Hz

### MAVLink Messages

| Message | Direction | Purpose |
|---------|-----------|---------|
| `HEARTBEAT` | Sim → Client | Connection status |
| `ATTITUDE` | Sim → Client | Vehicle attitude |
| `HIGHRES_IMU` | Sim → Client | IMU data |
| `ODOMETRY` | Sim → Client | (listed, no detail yet) |
| `TIMESYNC` | Sim → Client | Timing |
| `SET_POSITION_TARGET_LOCAL_NED` | Client → Sim | Control |
| `SET_ATTITUDE_TARGET` | Client → Sim | Control |

### Telemetry Provided

- Vehicle attitude and orientation
- Linear velocities
- System status flags
- Simulator navigation reference data

### Vision Stream

- Forward-facing FPV camera
- Parameters **not yet specified** (separate spec coming)
- No depth, no GPS, no additional sensors

## Course — VQ1

- Start gate → intermediate gates → finish gate
- **< 10 gates**
- Same course for all teams (no changes during the round)
- Gates visually distinctive from environment, consistent style
- Start and finish gate may differ slightly from intermediate gates
- Non-gated objects may be present
- **Full 3D** environment with elevation changes
- Minimal distractions, clear gate visibility

## Rules

- **Unlimited attempts** within the qualification window
- **8 minutes max** per attempt
- Gates must be passed in **correct order**
- No human interaction during runs — instant DQ
- Code must be accessible for review if required

## Software Environment

- Python 3.14.2 confirmed working; other environments allowed
- C/Cython extensions permitted
- Any software stack, tools, or frameworks

## Control Architecture (from spec)

```
Vision + Telemetry → Perception → Planning → Control → Pilot Commands → Stabilized Controller
```

## Still Pending from Organizers

- Coordinate system details (NED vs ENU, body frame convention)
- MAVLink message field details
- Gate dimensions and color
- Vision stream parameters (resolution, FPS, format)
- Camera intrinsics (before physical stage)

---

## Gap Analysis vs Our Current Stack

### Must Build / Change

1. **MAVSDK Integration Layer** — We have no MAVLink code. Need UDP connection,
   heartbeat maintenance, telemetry parsing, and command sending via MAVSDK-Python.
   This is the **#1 priority**.

2. **Windows Compatibility** — Sim is Windows-only. Our dev environment needs to work
   on Windows. Check all path handling, async patterns, etc.

3. **Control Command Interface** — Two options: `SET_ATTITUDE_TARGET` (rate commands:
   throttle/roll/pitch/yaw) and `SET_POSITION_TARGET_LOCAL_NED` (position/velocity).
   Need to understand which to use and when. Our current control module assumes
   direct rate commands — `SET_ATTITUDE_TARGET` is likely the match.

4. **ODOMETRY Message** — Listed in spec with no details. If this provides position
   estimates, it could supplement or replace our ESKF for VQ1. **Must investigate
   when sim drops.**

5. **Vision Stream Adapter** — Need to receive FPV frames from the sim. Format TBD
   but likely requires a capture/decode layer feeding into our perception pipeline.

6. **Sim Adapter (`drone_racing/sim/`)** — Currently a stub. Needs to become the
   real bridge: connect via MAVSDK, parse telemetry into our internal state, send
   control commands, capture vision frames.

### Already Aligned

- Gate detection pipeline (YOLO keypoint + classical fallback) — good for VQ1's
  clear visibility, even better once we get gate color/dimensions
- ESKF estimation — may be supplemented by ODOMETRY but still needed for high-rate fusion
- Planning/control pipeline architecture matches their described flow
- Python 3.14.2 — we target 3.12+ already

### Can Defer

- RL policy training — VQ1 is about completion, not speed. PID/MPCC sufficient.
- Complex obstacle avoidance — VQ1 has minimal distractions
- Sim-to-real transfer — not relevant until Physical Qualifier
