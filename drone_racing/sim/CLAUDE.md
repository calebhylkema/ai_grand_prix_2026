# Simulator Adapter Module

## Purpose

Adapter layer between our autonomy stack and the AI Grand Prix simulator.
Isolates all simulator-specific code so the core pipeline remains simulator-agnostic.

## Interface (confirmed — VADR-TS-001)

The simulator communicates via **MAVLink v2 over UDP** using **MAVSDK-compatible interfaces**.

### Telemetry (Simulator → Client)
| MAVLink Message | Data | Our Mapping |
|----------------|------|-------------|
| `HEARTBEAT` | Connection status | Connection health check |
| `ATTITUDE` | Roll, pitch, yaw + rates | `TelemetrySnapshot.attitude_*` |
| `HIGHRES_IMU` | Accel + gyro (body frame) | `ImuSample` for ESKF prediction |
| `ODOMETRY` | Position, velocity, orientation | Position estimate (if available) |
| `TIMESYNC` | Clock synchronisation | Timestamp alignment |

### Control (Client → Simulator)
| MAVLink Message | Data | Our Mapping |
|----------------|------|-------------|
| `SET_ATTITUDE_TARGET` | Throttle + body rates | `DroneCommand` (primary) |
| `SET_POSITION_TARGET_LOCAL_NED` | Position/velocity in NED | Waypoint mode (secondary) |

### Timing
- Physics: 120 Hz
- Command rate: 50–120 Hz recommended
- Heartbeat: minimum 2 Hz

## Frame Conversions

MAVLink uses NED (North-East-Down). Our internal pipeline uses Z-up (ENU-like).
All conversions happen at the adapter boundary via `transforms.ned_to_enu()` and friends.

- Position/velocity: swap X↔Y, negate Z
- Quaternions: frame rotation conversion via `quat_ned_to_enu()`
- IMU data: stays in body frame (FRD) — no conversion needed

## Files

- `mock_adapter.py` — Self-contained mock sim for offline testing (6-DOF physics + synthetic camera + IMU)
- `mavsdk_adapter.py` — Real simulator connection via MAVSDK over UDP

## Pre-Sim Development

Until the real simulator launches (May 2026), use `MockSimAdapter` for testing.
When the sim drops:
1. Run `scripts/vq1/run_race.py` to connect
2. Validate telemetry stream contents (especially ODOMETRY fields)
3. Tune command rate and control gains for the real physics model
4. Capture vision stream once format is specified

## Still Pending from Organizers
- Vision stream parameters (resolution, FPS, format, delivery mechanism)
- ODOMETRY field details (position frame, reference point)
- Coordinate system clarification (NED confirmed for MAVLink, but sim world frame TBD)
