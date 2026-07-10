# Roadmap

**Mission:** fastest, most reliable autonomous drone racing stack for the Anduril
AI Grand Prix 2026. Current status lives in [status.md](status.md) — this file is
future work only.

## Phase 2 — VQ2 (LIVE, until mid–late July 2026)

Environment: < 20 gates, 3D-scanned scenes, dynamic lighting, obstacles. Sensors:
HIGHRES_IMU + 640×360@30 Hz camera + RACE_STATUS + COLLISION only. Fastest valid
time counts; objective = qualify for the Physical Qualifier.

- [ ] Resolve the camera-tilt sign question (see status.md — do first)
- [ ] Thread gate 1 (map-and-plan executor, ADR-0003)
- [ ] Extend the course map to all legs; per-corner constants table
- [ ] Full-course completion → bank a valid Qualification-mode time
- [ ] Speed pass: trajectory through turns, higher cruise, command rate
- [ ] Perception upgrade if clutter demands it (YOLOv8-pose on captured VQ2 frames)
- [ ] MPCC racing line on ESKF state (only after reliable completion)

Success criteria — minimum: qualify. Target: top 25% time. Stretch: top 10%.

## Speed roadmap (ordered; only after reliable completion)

1. **Calibrated control** — done in principle (the 2.5× inverted gain was the wall).
2. **Trajectory through turns:** arrive centered in position AND velocity
   (geometric guidance on ESKF state), not reactive point-and-servo.
3. **Racing line:** B-spline through gate centers + curvature speed profile; then
   MPCC (`drone_racing.planning.mpcc`, p95 22 ms warm-started) on the ESKF state.
4. **Command rate** 50 → ~100 Hz (spec cap < 100 Hz).
5. **Cruise speed:** raise pitch/speed until threading degrades; brake-into-turn.
6. **Perception latency:** classical CV ~5 ms; YOLO only if clutter demands it
   (ONNX, <17 ms). Never exceed ~40 ms end-to-end.
7. **RL policy** (PPO, G&CNet-style) — only if classical+MPCC comfortably qualifies.
   Reference: MonoRace won A2RL at 28 m/s with the same sensor suite.

## Phase 3 — Physical Qualifier (September 2026, Southern California)

Sim-to-real is the challenge. The VQ2 stack (monocular + IMU, ESKF, vision steering)
is exactly the physical-stage problem — everything built now carries over.

- [ ] One-click camera-intrinsics calibration script
- [ ] ESKF auto-tuning from logged innovation sequences
- [ ] Detector retraining on real frames (domain randomization ready in rl_env)
- [ ] Conservative mode toggle: wider tunnels, lower max speed
- [ ] Post-flight analysis dashboard for the 2-week on-site period
- [ ] Consolidate the two ESKF implementations (package 16-state has a known
      frame-conversion drift; the validated VQ2 15-state filter is
      deployment-local) — a deliberate, separately-verified merge

Minimum: complete the course, don't crash. Target: qualify for finals (top 10).

## Phase 4 — Championship Finals (November 2026, Columbus, Ohio)

Head-to-head; reliability wins. Maximize completion rate, then aggressiveness.

- [ ] Adaptive race strategy (aggressiveness by position)
- [ ] Config presets: conservative / balanced / aggressive
- [ ] Crash recovery: resume after survivable impact
- [ ] RL / learned-depth obstacle avoidance (stretch goal per ADR-0002)
