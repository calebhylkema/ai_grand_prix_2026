# Configuration

## Deployment configs

Active VQ2 configuration lives in `deployments/vq2/configs/`:

- **`training.toml`** — development runs (sim in Training mode)
- **`qualification.toml`** — competitive runs (sim in Qualification mode;
  code-audited — keep the stack clean)

The two files are complete and separate — no layering/inheritance, so what you read
is what flies. Numerical flight values (gains, clamps, thresholds) were carried over
verbatim from the flight-validated controllers during the 2026-07-09 restructure.

The VQ2 deployment intentionally **hardcodes the official camera intrinsics**
(fx=fy=320, 640×360) — they are spec constants, not tunables.

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
