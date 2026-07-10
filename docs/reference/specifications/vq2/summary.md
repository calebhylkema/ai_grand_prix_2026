# VQ2 Official Spec — Consolidated Reference

> Sources: VADR-TS-003 (Issue 00.03, 2026-06-24) + two VQ2-launch emails + `PyAIPilotExample-v2`
> reference client. Full spec text extracted to `docs/specs/vq2_spec_text.txt`. VQ2 went live **2026-06-30**.
> Sim build: **v1.0.3379** (VQ1 was 3364).

## ⚠️ THE HEADLINE CHANGE — ground-truth telemetry is REMOVED

Starting in VQ2, the simulator **no longer publishes** these messages (spec §9.3; confirmed by
comments in the reference client `mavlink_rx.py`):

| Removed message | What we lost |
|-----------------|--------------|
| `ATTITUDE` | vehicle orientation |
| `LOCAL_POSITION_NED` | absolute position |
| `ODOMETRY` | position + velocity + orientation quaternion |
| `GATE_INFO` / `TRACK_INFO` | **the gate map — gate poses/orientations/dimensions are nulled** |

**Consequence:** the entire VQ1 controller (`scripts/run_race.py --controller hover`), which flew on
ground-truth ODOMETRY + the TRACK_INFO gate list, is **non-functional in VQ2**. We must now run our
own state estimation and find gates from vision. This is the monocular-camera + IMU scenario the
pre-sim `perception/ → pnp → eskf` stack was built for (originally scoped for the physical qualifier).

## What the sim STILL provides in VQ2

| Message | Direction | Use |
|---------|-----------|-----|
| `HEARTBEAT` | Sim → Client | connection |
| `HIGHRES_IMU` | Sim → Client | **accel + gyro (body FRD)** — our only inertial sensor |
| Camera stream | Sim → Client (UDP 5600) | **30 Hz, 640×360 JPEG** — our only exteroceptive sensor |
| `RACE_STATUS` (ENCAPSULATED_DATA type 1) | Sim → Client | `active_gate_index`, `last_gate_race_time`, timing — our only progress/feedback signal |
| `COLLISION` | Sim → Client | id 1001=gate, 1002=environment; threat 1–2; impulse magnitude |
| `ACTUATOR_OUTPUT_STATUS` | Sim → Client | 4 motor outputs |
| `TIMESYNC` | Sim → Client | clock sync |
| `SET_ATTITUDE_TARGET` | Client → Sim | control (rate or quaternion) |
| `SET_POSITION_TARGET_LOCAL_NED` | Client → Sim | control (position/velocity; **re-test if honored**) |

## Simulator

- Windows only, runs locally, internet required (anti-cheat). Physics **120 Hz**, rigid-body
  (thrust, drag, gravity, collision). Deterministic — identical course/physics for all teams.
- **Two selectable event blocks** (spec §9.2):
  - **Training** — free, unlimited, **not** scored. Develop and validate here.
  - **Qualification (VQ2)** — time-tracked, counts toward the leaderboard / physical-qualifier standing.
- **Code-integrity policy:** submitting a competitive run gives DCL the right to audit the codebase.
  Suspected sim-manipulation/cheating triggers a formal audit. Keep the stack clean.

## Coordinate frames & camera (spec §3.7–3.8) — NEW, PRECISE

- World: **NED**, `MAV_FRAME_LOCAL_NED`, origin = point on ground where the drone armed.
- Body: **FRD** (X fwd, Y right, Z down). **Body → IMU = identity.**
- **Camera is tilted 20° UPWARD** from the body axis (shares body origin). *(VQ1 config wrongly
  assumed ~15° forward — must fix.)* Frames are NED; rotate into your CV library's convention.
- **Camera intrinsics:** pinhole, no distortion. 640×360, `[cx,cy]=[320,180]`,
  **`[fx,fy]=[320,320]`**, **VFoV = 90°**. Stream 30 Hz.

## Dimensions (spec §3.6–3.7)

- **Gate inner opening (fly-through target): 1.5 m × 1.5 m**, depth 0.26 m.
- Gate outer frame: 2.7 m × 2.7 m, depth 0.26 m.
- Drone chassis: 280 × 280 × 160 mm.
- Threading tolerance ≈ **±0.75 m** half-width — tighter than the ±1.36 m assumed in VQ1.

## Timing / control (spec §4.4)

- Physics 120 Hz. **Command rate < 100 Hz** (VQ1 said 50–120). Heartbeat ≥ 2 Hz.
- Reference client uses CONTROL_HZ 250 in its loop and also demonstrates
  `set_actuator_control_target` (direct motor RPM) as a control option.

## Vision stream wire format (unchanged from VQ1)

- UDP port 5600, little-endian, 24-byte header `<IHHIIQ` =
  `frame_id(u32), chunk_id(u16), total_chunks(u16), jpeg_size(u32), payload_size(u32), sim_time_ns(u64)`,
  followed by a JPEG slice. Reassemble chunks by `frame_id`.

## Course — VQ2

- Start gate → sequential intermediate gates → finish gate, **< 20 gates**.
- Higher fidelity than VQ1: **3D-scanned complex environments, dynamic lighting, static scene
  objects, vertical & horizontal obstacles, boundary elements** (spec §3.1).
- Deterministic; gates visually distinctive but consistent within the round.

## Scoring (spec §9.4)

- Leaderboard ranks by **fastest valid completed run**. Unlimited attempts.
- **Team-centric:** a team's rank = its single best time, by any member.
- VQ2 objective (spec §9.1): **qualify for the September Physical Qualifier.** Completion is the
  gate; time is the ranking.

---

## Gap analysis vs our stack (what VQ2 forces)

### Must build / change
1. **IMU-only attitude estimation (AHRS)** — no more `ATTITUDE`. Accel→roll/pitch, gyro→rates,
   visual yaw aiding. Linchpin: control + PnP world-chaining both need attitude.
2. **Vision-driven navigation** — no gate map. Detect gates in-image → PnP for *relative* pose
   (`perception/pnp.py` core already yields `T_CG` without a world map).
3. **Config fixes** — `camera.tilt_deg` 15→**20** (and sign = up), `fx=fy` 300→**320**,
   confirm gate 1.5 m, `command_rate_hz` < 100.
4. **Robust detector for complex env** — VQ1 classical HSV likely insufficient under dynamic
   lighting / 3D-scanned clutter. Prepare to train YOLOv8-pose on captured VQ2 frames.
5. **Obstacle handling** — VQ2 adds vertical/horizontal obstacles between gates.

### Reusable as-is / with rewiring
- `perception/` (classical, neural, tracker), `perception/pnp.py`, `estimation/eskf.py`,
  `estimation/gate_mapper.py`, `planning/mpcc.py` + `spline.py` — all now **mainline**.
- VQ1 control cascade *logic* (yaw lead-pursuit + roll cross-track + altitude PD, failsafes) —
  salvageable if fed vision bearing + estimated attitude instead of ground truth.
- Command-side sign/throttle findings (hover ≈0.275, rate directions) are physical — **re-verify**
  on build 3379 without an attitude readout (use camera horizon + IMU accel).

### Validation strategy (no ground truth in VQ2)
The **old VQ1 sim (v1.0.3364)** still broadcasts ODOMETRY + ATTITUDE. Use it as a ground-truth rig
to validate the new IMU-AHRS and vision-PnP estimators offline before trusting them in VQ2.
