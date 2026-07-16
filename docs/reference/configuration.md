# Configuration

## Deployment configs

Active VQ2 configuration lives in `deployments/vq2/configs/`:

- **`training.toml`** — development runs (sim in Training mode)
- **`qualification.toml`** — competitive runs (sim in Qualification mode;
  code-audited — keep the stack clean)

The two files are complete and separate — no layering/inheritance, so what you read
is what flies. Numerical flight values (gains, clamps, thresholds) were carried over
verbatim from the flight-validated controllers during the 2026-07-09 restructure.

`run.permit_transition_authority` is a permission boundary, not an enable switch.
Training sets the permission, but transition authority remains off unless the operator
also passes `--experimental-transition-authority`. Qualification sets the permission
false and rejects that CLI flag. The experiment affects only the short gate-0→gate-1
`TRANSITION_ACQUIRE` handoff; gate 0 and anchored guidance remain on the legacy controller.
As of 2026-07-13, the first bounded trial failed its target-continuity acceptance gate,
so the flag must not be used again until `docs/project/status.md` records a new review.

The VQ2 deployment intentionally **hardcodes the official camera intrinsics**
(fx=fy=320, 640×360) — they are spec constants, not tunables.

### Resolution policy (repo-bound, not wheel-packaged)

`drone_racing.deployments.vq2.config.load(name)` resolves named configs against
the **workspace root**, not the installed package:

1. `AIGP_WORKSPACE` env var, if set
2. else upward search from cwd for a `pyproject.toml` / `.local` marker
3. `<root>/deployments/vq2/configs/<name>.toml`

This is deliberate: deployment configs are version-controlled flight data, and
qualification runs are code-audited — the reviewed repo tree must be the single
source of what flies. Packaging configs into the wheel would create a second,
shadowable copy. Running an installed `aigp-vq2` outside the repo therefore
requires either `AIGP_WORKSPACE` pointing at a checkout or an explicit
`--config path/to/file.toml`.

## Package config

`src/drone_racing/config/default.toml` (packaged with the wheel; loaded by
`drone_racing.utils.config.load_config`) configures the pre-sim-lineage package
pipeline (perception/planning/control modules). It is not read by the VQ2 mission.

## Workspace paths

Filesystem locations are centralized in a workspace-paths object
(`drone_racing.utils.workspace`). Generated/local output defaults under `.local/`:

| Path | Contents |
|---|---|
| `.local/simulators/{vq1,vq2}/` | simulator installs (vq1 = ground-truth rig) |
| `.local/datasets/` | training data, captured frames, trace archives |
| `.local/models/` | model weights |
| `.local/runs/` | training runs, recorded attempts |
| `.local/logs/` | flight logs |
| `.local/docs-build/` | generated doc HTML/PDF |

Nothing under `.local/` is version-controlled. Course maps that are small, reviewed,
and version-controlled live in `deployments/vq2/course_maps/`; raw observations
belong in `.local/datasets/`.

## Course-map readiness

Course maps use schema-v1 TOML and are loaded by
`drone_racing.deployments.vq2.course_map.load_course_map`. Loading checks structure,
frames, units, normals, gate order, aperture clearance, and evidence counts.

`require_flight_ready=True` additionally requires `status = "validated"`, current
perception provenance, at least three independent runs per mapped gate, resolved
traversal normals, and every consecutive leg. The checked-in
`vq2_historical_v0.toml` is evidence only and is intentionally rejected for flight.

Do not interchange these range definitions:

- `accept_depth_m`: camera-Z depth used by the legacy active mission;
- `slant_range_m`: Euclidean camera-to-gate distance used by the map/shadow contract.
