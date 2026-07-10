# Project Status — VQ2

> **This is the single authoritative status document.** Update it in place; do not
> create new session-handoff files. Last updated: **2026-07-09** (repo restructure day).

## Where we are

**Phase:** VQ2 (live since 2026-06-30, sim v1.0.3379, spec VADR-TS-003). Deadline
mid–late July 2026 (~20 days remaining as of 2026-07-09). Objective: bank a valid
full-course time in Qualification mode → qualify for the September physical round.

**Active runner:** `aigp-vq2` (`drone_racing.deployments.vq2.mission` — the
map-and-plan phased executor, per [ADR-0003](../decisions/0003-course-map-and-plan.md)).
The superseded reactive controller's proven primitives (Flight runtime, calibrated
command boundary, thrust helpers) were extracted into
`drone_racing.deployments.vq2.runtime`. The reactive controller itself stays
runnable at `scripts/vq2/fly.py` until a live gate-0 regression (baseline: 100%
thread @ ~2.2 s) revalidates the extraction; then it moves to
`archive/vq2_experiments/reactive_fly/`.

## What works (validated)

- **Gate 0 threads 100%** (t ≈ 2.1–2.3 s, every run).
- **Rate + thrust control** is the only sim-honored interface. Calibrated command
  boundary verified: sim rate commands are inverted and ~2.5× scaled —
  `actual_gyro ≈ GAIN · commanded`, `GAIN = (roll −2.55, pitch −2.55, yaw −2.35)`;
  fix is `cmd = desired / GAIN` at one boundary (`aigp-calibrate --test rates --calib`
  shows ~+0.98 on all axes). **Hover thrust ≈ 0.30.**
- **Post-gate-0 corner survival:** arrest-then-look brake (1.0 s blind flare at
  hover+0.04, then nose to +0.10 so the cyan racing line returns to view and the
  line-thrust servo owns height closed-loop). 4/4 corner survival; gate-1
  acquisitions at z 9.8–12.6 m.
- **Course map v0** (mined offline from trace archives, no odometry): pad→gate 0 =
  8.6 m dead ahead; **gate 1 = 9.2 m fwd / 6.4 m RIGHT of gate 0 (34.8°, 11.2 m
  horizontal), +0.6 m up**; post-crossing z 21–23 m sightings are gate-2 imposters;
  a return-leg gate pair at z 12–14 m ~19° LEFT of exit heading is a trap
  (direction-gated accept rejects it).
- Perception: glow-resistant ring-test gate detection; ridge-walk cyan-line aim;
  offline validation harness over 1000+ real frames.
- IMU frames at 30 Hz stable; JPEG decode fine; comms proven against
  PyAIPilotExample-v2 (which itself does nothing — wire-protocol demo only).

## What is broken (current blocker)

**Gate 1 has never been threaded.** All 8 batches on 2026-07-09 failed identically in
the THREAD phase with an **orbit signature**: after a clean acquisition, gate bearing
drifts +34→+42° while yawing right at MAX_YAW with bank clamped → "target lost" →
re-scan → collide with floor clutter. Different thrust profiles did not change it.
The problem is perception or planning, **not control**.

## 🚨 Critical open question — camera tilt sign

- **Spec (VADR-TS-003 §3.8):** "camera is tilted upwards by 20°."
- **Our code implements 20° DOWN** (`_TILT = −20°` in the VQ2 perception module),
  justified 2026-07-03 from live imagery (level hover put the horizon at y≈64 with
  floor filling the frame) — but that reading was never re-examined.
- If the code is wrong, **all PnP poses are geometrically inverted** and the THREAD
  "gate 1" may be the wrong object entirely. This could explain 10 days of stalled
  control tuning.

**Validation plan (do this first):**
1. Flip the tilt sign to +20° (up, per spec); run gate 0 — it should break or fix
   things consistently.
2. Overlay detections + PnP poses + bearing lines on trace frames; compare against
   the mined course map geometry.
3. Use the VQ1 sim (`.local/simulators/vq1/`, still broadcasts ODOMETRY/ATTITUDE)
   as a ground-truth rig: if PnP is correct there with DOWN, and VQ2 differs, the
   builds differ.

## Next technical milestones (ordered)

1. **Validate the camera tilt** (above) — before any more control tuning.
2. **Simplify if needed:** clone the proven gate-0 primitives (yaw servo on bearing,
   CY servo on image row, bank on bearing) for gate 1 with zero fancy state machines;
   prove basic completion before optimizing.
3. Extend the course map table to gates ≥ 2 (mine new traces per leg).
4. Full-course completion → bank a Qualification-mode time.
5. Then the speed roadmap — see [roadmap](roadmap.md).

## Standing findings (do NOT redo these experiments)

- **Open-loop/blind vertical is dead:** every blind climb/hover ended at the ceiling
  or the floor. The gate-0 crossing carries a real ~−2 m/s sink; only closed-loop
  visual height (gate row / cyan line) works.
- **IMU-integrated vz is never control-grade near launch** (launch pins it at a −3 m/s
  fiction). Diagnostic only — never servo thrust on it.
- **Yaw does not steer.** Yaw points the nose; only tilting redirects travel. Any
  steering design must bank.
- **Attitude-level corner shaping failed three times** (blind-arc variants): the
  blind second's real job is swinging the camera around the corner at MAX_YAW —
  slower entry must never mean a slower head-turn.
- **ESKF `update_gravity` must be ground-only:** in cruise (|a|≈1.05 g) it passes its
  own gate and erases real nose-down pitch (the "cruise lie").
- The SET_POSITION_TARGET / velocity-setpoint interfaces are **ignored by the sim**
  (tested; drone tumbles). Rate+thrust only.
- Gate-hopping is real: after gate 0 the tracker once latched 3 different gates in
  0.3 s. Map-gated data association (range band + turn direction) is mandatory.

## Evidence base

- Per-attempt trace archives (frame + state in filename) — see
  [diagnose a flight](../how-to/diagnose-a-flight.md). Newest traces answer questions
  faster than new experiments; **read traces before changing anything.**
- Run recordings and frame corpora live under `.local/` (datasets, runs, logs).

## Operating norms

- Sim in **Training** mode for all development (Qualification runs are code-audited).
- Commit only when asked. Lint before committing.
- Before a Qualification submission, verify the rules allow persisted per-course
  constants (the course map) — flagged in ADR-0003.
