# Repository reorganization report (2026-07-09/10)

One-time migration record for the July 2026 restructure (branch
`vq2-reorg-calibrated-controller`, commits `f8eb395` → `8945482` → `937014e` → phase 4).
For current project state see [status](status.md); for the layout see
[architecture](../reference/architecture.md).

## What changed

**Before:** flat `scripts/vq2/` sibling-import scripts, mixed-in simulators and
datasets, VQ1/pre-sim code intermingled with active work, session-handoff doc sprawl.

**After:** installed src-layout package (`src/drone_racing/`, import name unchanged),
VQ2 as the only active deployment, everything historical in `archive/`, all local
artifacts under gitignored `.local/`, Diátaxis docs with one authoritative status doc.

## Old → new mapping (active code)

| Old | New |
| --- | --- |
| `scripts/vq2/fly2.py` | `src/drone_racing/deployments/vq2/mission.py` (`aigp-vq2`) |
| `scripts/vq2/fly.py` (infrastructure) | `src/drone_racing/deployments/vq2/runtime.py` (extracted verbatim) |
| `scripts/vq2/fly.py` (whole controller) | **still at `scripts/vq2/fly.py`** — see deferred items |
| `scripts/vq2/gates.py` | `src/drone_racing/deployments/vq2/perception.py` |
| `scripts/vq2/estimator.py` | `src/drone_racing/deployments/vq2/attitude.py` |
| `scripts/vq2/eskf.py` | `src/drone_racing/deployments/vq2/eskf.py` |
| `scripts/vq2/guidance.py` | `src/drone_racing/deployments/vq2/guidance.py` |
| `scripts/vq2/control_check.py` | `src/drone_racing/deployments/vq2/calibration.py` (`aigp-calibrate`) |
| `scripts/vq2/course_map.py` | `src/drone_racing/deployments/vq2/mapping.py` |
| `scripts/vq2/recon.py` | `src/drone_racing/deployments/vq2/diagnostics.py` (`recon` subcommand) |
| `scripts/vq2/detect_lab.py` | archived; replaced by `diagnostics.py lab` subcommand |
| `scripts/vq2/test_control.py` | `tests/unit/vq2/test_control.py` |
| `scripts/vq2/test_estimation.py` | `tests/unit/vq2/test_estimation.py` |
| `scripts/vq2/guidance_test.py` | `tests/unit/vq2/test_guidance.py` (rewritten as pytest) |
| hardcoded fly2 constants | `deployments/vq2/configs/{training,qualification}.toml` + typed `config.py` (values verbatim) |
| simulators / datasets / models / runs / logs in-tree | `.local/` (gitignored; nothing deleted) |

All moves used `git mv` — rename detection is 96–100 % throughout, history intact.

## Archived (not deleted)

- `archive/pre_sim/` — pre-simulator stack experiments
- `archive/vq1/` — the complete VQ1 round (controller, findings)
- `archive/vq2_experiments/race_v1/` — superseded VQ2 race controller + detect_lab
- `archive/rounds/vq1/code/mavsdk_adapter.py` (+ its tests) — the legacy MAVSDK
  adapter, superseded by the pymavlink adapter since 2026-06-01; archiving it also
  dropped the `mavsdk` dependency. `keyboard` (no active users) and `albumentations`
  (unused; unbuildable transitive dep on Windows/3.14) were dropped at the same time,
  and `ultralytics` moved out of core deps (it lives in the `perception` extra;
  imports are lazy).
- `archive/` is excluded from ruff, mypy, and pytest; it is reference only.

## Behavior guarantee

No flight algorithm, gain, sign convention, coordinate frame, safety clamp,
failsafe, or the calibrated command boundary was changed. `runtime.py` carries a
provenance docstring; the config TOMLs carry the fly2 tuning values verbatim
(`accept_turn` stored in degrees, converted to radians at the load boundary only).

## Command changes

| Task | Command |
| --- | --- |
| Fly the mission | `uv run aigp-vq2 --config training --attempts N --secs S` |
| Verify calibration | `uv run aigp-calibrate --test rates --calib` |
| Mine the course map | `uv run python -m drone_racing.deployments.vq2.mapping` |
| Telemetry recon / perception lab | `uv run python -m drone_racing.deployments.vq2.diagnostics recon\|lab` |
| Legacy reactive controller | `uv run python scripts/vq2/fly.py` (until archived) |

## Verification results (2026-07-10)

- `ruff check` + `ruff format`: clean (archive excluded)
- `pytest`: **466 passed** (baseline before reorg: 444; migrated VQ2 unit tests
  added, the 26 legacy MAVSDK-adapter tests archived with their module)
- `mypy src/drone_racing`: **11 errors** (was 23 pre-reorg; archiving the
  superseded MAVSDK adapter removed 12), all in the pre-sim lineage library
  (numpy/cv2 stub friction); zero in the VQ2 deployment
- `uv build`: sdist + wheel build clean
- All five console scripts (`aigp-vq2`, `aigp-calibrate`, `aigp-replay`,
  `aigp-train-detector`, `aigp-benchmark`) pass `--help`
- Both deployment configs load through the typed loader
- No `sys.path` hacks and no old sibling imports outside `archive/`
- Coverage 69 % measured / gate at 65 (see debt below)

## Remaining structural debt

1. **fly.py archival is gated on a live sim regression.** `scripts/vq2/fly.py` runs
   on package imports and stays until a gate-0 run reproduces the baseline
   (100 % thread @ ~2.2 s), proving the runtime.py extraction. Then it moves to
   `archive/vq2_experiments/reactive_fly/`.
2. **11 pre-existing mypy errors** in `utils/transforms`, `perception/neural`,
   `perception/classical`, `logging/flight_logger`, `sim/mavlink_adapter`,
   `control/rl_env`. CI runs mypy non-blocking until burned down.
3. **Coverage gate at 65 % (target 80).** The gap is the untested pre-sim lineage
   library (`training/`, `cli/`, sim adapters). Live-sim-loop VQ2 modules
   (mission/runtime/calibration/diagnostics) are omitted from measurement — they
   need the running simulator.
4. **Flat legacy tests** (`tests/test_*.py`) not yet bucketed into
   `unit/ | integration/ | simulation/`.
5. **mypy relaxations for `drone_racing.deployments.vq2.*`** (untyped defs,
   any-generics, return-any) — converge to package strictness over time; do not let
   it spread.
6. CI runs on ubuntu and cannot exercise the Windows-only simulator; simulation
   tests are excluded there by marker.
