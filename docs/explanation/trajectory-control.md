# Trajectory and control

## Flight-validated lessons

- Yaw points the camera; roll/pitch redirect velocity. A steep corner needs both.
- Rate commands are inverted and scaled by the simulator. Apply calibration exactly
  once at the command boundary.
- Lost or unreliable state means bounded brake/hover, not aggressive open-loop drift.
- VQ2 accepts collective thrust and body rates only. Position and velocity setpoints
  are ignored.
- World calculations use NED; the body uses FRD; positive pitch is nose-up/braking.

## What is active

The active VQ2 mission is still the proven phase controller:
`G0 → BRAKE → SCAN → THREAD`. It does not consume the reusable racing-line, geometric
guidance, PID, MPCC, or RL controllers. Its gate-1 association range is explicitly
camera depth (`accept_depth_m`), while the typed map/shadow contract uses slant range.

The deployment-local ESKF and new reference planner remain shadow/offline. Do not
describe them as active guidance until their promotion gates pass.

## Course reference

The offline reference generator now:

1. requires a versioned, flight-ready course map;
2. builds an open path through approach, gate-center, and exit constraints;
3. resamples by physical arc length;
4. checks ordered crossing direction and shrunken gate aperture; and
5. limits speed by curvature, forward acceleration, and braking distance.

Progress must be projected locally around the previous progress. A global nearest
point can jump to the wrong leg near a self-intersection.

## Dynamics

The old MPCC model is retained for pre-sim regression tests. It uses a generic Z-up
attitude-command abstraction and is not a VQ2 flight model.

`VQ2DynamicsConfig` defines the correct planning contract: world NED state and
`[collective thrust, body p, body q, body r]` input. Hover thrust, effective drag, and
response parameters must be fitted from synchronized flight state and commands and
validated on held-out runs.

## MPCC and RL

MPCC is disabled for command authority by default. Before promotion it still needs
the identified VQ2 model plus explicit spatial gate/track constraints and a solver
deadline. Synthetic p95 latency only measures compute cost.

RL is a later performance layer. It becomes useful after conventional guidance can
reliably complete the mapped course and real logs provide the perception/dynamics
residual distribution needed for training and transfer.

See [ADR-0004](../decisions/0004-planning-promotion-ladder.md) for the acceptance
ladder and [safety](../reference/safety.md) for command constraints.
