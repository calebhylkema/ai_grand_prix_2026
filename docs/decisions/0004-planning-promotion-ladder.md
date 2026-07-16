# 0004 — Promote mapping and planning by evidence, not solver availability

## Status

Accepted (2026-07-13). Refines ADR-0003. The existing VQ2 mission remains the active
controller until the promotion gates below pass.

## Context

The VQ2 mission threads gate 0 but has not threaded gate 1. The repository contained
three disconnected concepts:

- a reactive `G0 → BRAKE → SCAN → THREAD` mission with gate knowledge embedded in
  TOML constants;
- an empty version-controlled course-map directory plus 59 raw legacy observations,
  of which only three frames contained co-visible pairs; and
- a pre-sim racing-line/MPCC stack tested on synthetic closed courses using a generic
  Z-up attitude-command model rather than VQ2's NED rate/thrust interface.

Passing unit tests or converging in IPOPT does not establish frame correctness,
map identity, dynamic feasibility, estimator fitness, or safe flight integration.

## Decision

1. Course maps are typed, versioned deployment artifacts. Structural validation is
   separate from flight readiness. Historical/candidate maps may be inspected offline
   but fail closed when flight readiness is required.
2. The first reference is an open, arc-length-parameterized path with explicit gate
   plane/direction/aperture checks and acceleration/braking limits.
3. The VQ2 planning model uses world NED, body FRD, and collective-thrust/body-rate
   inputs. Its parameters must be identified from synchronized state and command logs;
   defaults are not flight authority.
4. Course references and model predictions run offline, then shadow-only. A
   conservative geometric follower is promoted before MPCC.
5. MPCC command authority is disabled by default. Enabling it requires an explicit
   constructor decision and does not imply that the active VQ2 deployment is ready.
6. RL remains deferred until the classical stack completes the course reliably and
   provides real residual/noise data.

## Promotion gates

- A map needs corrected-pipeline evidence from at least three independent attempts,
  resolved gate traversal normals, reviewed uncertainty, and `validated` status.
- A reference must cross every gate in order, in the correct direction, inside the
  aperture after vehicle, tracking, and 3σ map margins.
- Shadow planning must have zero backward progress jumps, no non-finite outputs, no
  command-loop effect, and no stale output consumption.
- The conservative follower must complete gate 0→1 without collision five times in a
  row before another leg or MPCC receives authority.

## Consequences

- `vq2_historical_v0.toml` records current evidence honestly but cannot fly.
- The existing reactive mission remains available as the regression baseline.
- The legacy MPCC can still be benchmarked, but its latency number is not deployment
  validation and its adapter defaults to shadow-only/PID fallback.
- New simulator data collection belongs to the perception/ESKF validation workflow;
  planning consumes the resulting reviewed artifacts offline.

