# 0003 — Fly the course from a map, not from instantaneous detections

## Status

Accepted (2026-07-09). Supersedes the reactive-branch approach of the old `fly.py`
controller for course navigation. (2026-07-09 restructure note: the reactive
controller stays runnable at `scripts/vq2/fly.py` until a live gate-0 regression
revalidates the extraction, then moves to `archive/vq2_experiments/reactive_fly/`;
its proven low-level primitives — calibrated rates, Flight failsafes, CY vertical
servo — were extracted into `drone_racing.deployments.vq2.runtime`. The mission
executor described below is `drone_racing.deployments.vq2.mission`, run as
`aigp-vq2`; the course-map miner is `drone_racing.deployments.vq2.mapping`.)

## Context

Ten days of VQ2 sessions (configs A→AX in fly.py) produced a controller that threads
gate 0 with 100% reliability but has never threaded gate 1. Each failed batch was
answered with another reactive branch or constant tweak; progress stalled (user
escalation 2026-07-09: "we have made no progress in 10 so something needs to change …
feel free to scrap stuff").

Forensics on the AX batch found the structural cause, not a tuning cause:

- **Imposter-gate hopping.** After crossing gate 0 the tracker latched three
  DIFFERENT gates within 0.3 s (PnP z 27.2 → 21.2 → 17.1 m — an impossible 35 m/s
  "closure"). Nothing in the stack knows where gates actually are, so data
  association is by image continuity only, and the controller flew the corner chord
  toward second-leg gates, into the corner scaffold/jet.
- **Geometric blindness to gate 1.** The camera is 20° DOWN; frame row 0 is only
  ~9° above body-horizontal. Gate 1 is ~0.6 m above gate 0's center, and we cross
  gate 0 sinking — the real gate 1 is physically out of frame (top) exactly when the
  reactive stack needs it, while the 21–23 m imposters are in frame.
- A full-pipeline audit found the one architecture box absent from the active
  stack: a gate map feeding a planner. State-of-the-art monocular racing systems
  (MonoRace, Swift) all fly a pre-built track map; perception corrects the plan
  rather than generating it.

The course is FIXED per event and Training attempts are unlimited, so map data is
free: any saved trace frame with TWO PnP-valid gates yields their relative position
with no odometry (`d_cam = pose_j − pose_i`, rotate through CAM_TO_BODY and the
roll/pitch leveling — the vertical component and separation are yaw-invariant).

## Decision

1. **Offline course mapping** (`drone_racing.deployments.vq2.mapping`): mine all
   saved fly_trace archives for PnP-valid gate observations and co-visible pairs;
   write `.local/datasets/vq2/map/observations.csv`. Map v0 (3 independent co-visible pairs across
   6 runs): gate 1 is 9.2 m fwd / 6.4 m right of gate 0 (34.8° right, 11.2 m
   horizontal), +0.6 m up; post-crossing sightings at z 21–23 m are gate 2.
2. **Map-based mission executor** (`drone_racing.deployments.vq2.mission`,
   run as `aigp-vq2`): phased plan instead of
   per-frame re-decision — thread gate 0 (proven servo), then on the RACE_STATUS
   gate-pass event: BRAKE hard (deep flare; we exit the catapulted gate-0 pass hot)
   while CLIMBING ~2 m (so the 20°-down camera can see the elevated gate 1), SCAN
   right toward the mapped bearing, and **accept a detection as gate 1 only if its
   PnP range is inside the mapped band (3–16 m)** — map-gated data association that
   rejects the imposters by construction. Then THREAD with the standard CY servo.
3. fly.py is untouched; no safety/failsafe logic is weakened (fly2 reuses
   `Flight.abort_reason` verbatim).

## Consequences

- Course knowledge is a persisted artifact (CSV + the mission's config TOMLs).
  **Before a
  Competitive submission, verify the rules allow persisted per-course constants**
  (runs are code-auditable); Training-phase use is unambiguous.
- The map is expressed in gate-relative terms (range bands, turn direction, relative
  elevation), not absolute coordinates, so modest spawn/heading variation is
  tolerated; a course change invalidates it (re-mine from fresh traces — one slow
  run suffices).
- Per-corner constants (turn direction, accept band, climb) generalize to later
  gates as a table once legs beyond gate 1 are mined from new traces.
- The reactive fly.py branch stack stops growing; future tuning effort goes into
  the plan (map accuracy, brake/climb profile) where failures are attributable.
