# NEXT SESSION — read this first (standing handoff doc, keep updated)

(Fast catch-up handoff. Deeper detail persists in the memory dir — see
`project_vq2_rate_calibration.md` ⭐, `feedback_verify_fundamentals_first.md`,
`project_vq2_session_2026-07-01.md`, `project_vq2_playbook.md`, `project_vq2_drop_findings.md`.)

## What this project is
Python autonomy stack for the **AI Grand Prix 2026** autonomous FPV drone race. We are in
**Virtual Qualifier 2 (VQ2)**, live since 2026-06-30, sim build v1.0.3379, spec VADR-TS-003.
VQ2 removed ALL ground-truth telemetry (ATTITUDE / LOCAL_POSITION_NED / ODOMETRY / GATE_INFO).
**We fly from monocular camera + IMU only.** Objective: fastest valid completion → qualify for the
Sept physical round. VQ2 work is in `scripts/vq2/`. Keep the sim in **Training** for dev.

## ⭐⭐ THE BREAKTHROUGH (2026-07-01) — the multi-day root cause
For days the drone would thread gate 0, see gate 1, carry its momentum, tumble, and crash — zero
progress past gate 0. We finally measured the control primitives directly (gyro = ground truth for
body rate) with the probe `scripts/vq2/control_check.py --test rates`:

**Rate commands are INVERTED and ~2.5x scaled.**  `actual_gyro ≈ GAIN * commanded`, with
`GAIN = (roll -2.55, pitch -2.55, yaw -2.35)`. Linear (verified ±0.4 and ±0.8 rad/s), no cross-coupling.
So commanding +roll to correct rightward drift rolls the drone the OPPOSITE way 2.5x too hard → every
correction diverges → tumble. This is why any turn / the geometric controller (which computes
physically-correct rates) always tumbled, and why race.py is full of hand-tuned sign flips (YAW_SIGN,
GYRO_INT_SIGN=-1, ROLL_X_SIGN=-1, THR_SIGN=-1) that were partially compensating this by accident.

**FIX (verified):** invert at the command boundary — `cmd = desired / GAIN` (negate + scale ~0.4).
`control_check.py --test rates --calib` then shows scale +0.98 (correct sign, ~unity) on all 3 axes.
Also confirmed **hover thrust ≈ 0.30**.

## 🗂️ 2026-07-01 (later): WORKSPACE REORGANIZED — new paths
A full audit + reorganization pass was done. If a path from memory 404s, check here:
- `scripts/` is now **by phase**: `scripts/vq2/` (⭐ active), `scripts/vq1/` (was
  `scripts/run_race.py` + `scripts/sim_v1/*`), `scripts/pre_sim/` (unchanged).
- `docs/specs/` holds the official material: `VADR-TS-001_Technical_Spec_VQ1.pdf`,
  `VADR-TS-003_Technical_Spec_VQ2.pdf` (was `VQ2/Technical_Specifications.pdf`),
  `vq1_spec.md`, `vq2_spec.md`, `vq2_spec_text.txt`.
- `docs/PLAN.txt` + `docs/PROGRESS.txt` → merged into **`docs/ROADMAP.txt`** (includes the
  ordered SPEED roadmap: calibrated control → turn trajectories → racing line/MPCC →
  100 Hz commands → cruise → perception latency → RL last).
- Root `MANIFEST.md` → **`docs/CODE_MAP.md`** (per-file map, updated).
- Deleted: `docs/CHATGPT_RESEARCH.txt` (fully absorbed into RESEARCH.txt), `docs/GSD_SETUP.txt`
  (GSD was never installed).
- `scripts/vq2/_archive/` holds superseded scripts: `control_probe.py` (→ control_check.py),
  `eskf_synthetic_test.py` (→ test_estimation.py, same math as pytest).
- Captured frames: `VQ2/probe_frames`, `VQ2/recon_frames` → `data/vq2/…` (recon.py updated).
- `VQ2/` now contains ONLY the sim install (gitignored via anchored `/VQ2/`). `SIMULATOR/` =
  the OLD VQ1 sim v1.0.3364 — **keep it: it still broadcasts ground truth = estimator
  validation rig.**
- Fixed during audit: `config/default.toml` camera now matches the VQ2 spec (fx=fy=320,
  tilt −20 = 20° UP; was 300/15° fwd). `tests/test_run_race_mpcc_handoff.py` path updated.
  CLAUDE.md: `control/safety.py` never existed — the MUST-NOT rule now names the inline
  failsafe logic instead. Full fast suite: **448 passed** after everything.
- ⚠️ NOTHING IS COMMITTED YET — the whole VQ2 stack + reorg is staged/untracked. First
  action next session: review `git status`, then commit (branch off master first).
- Disk note: `runs/` is **8.8 GB / 294k files** (old PPO checkpoints + recorded frames) —
  prune when convenient; keep `best_model` (@400k) per memory.

## Other findings still valid from earlier 2026-07-01 (pre-breakthrough)
- ESKF ATTITUDE DIVERGES live (CSV est_pitch −18°→−60° during gate 0) because gate-position updates
  corrupt attitude + the gravity update is gated off during thrust. It poisoned the geometric guidance.
  (Real underlying issue was the rate inversion above; the ESKF is a secondary problem.)
- We decoupled gate≥1 guidance state (accel attitude + PnP-direct pos/vel, bypassing the ESKF) and added
  a PnP temporal-consistency filter (reject far/jumpy gate detections) in `race.py`. These are still in
  race.py but race.py itself is now suspect (tuned to the broken rate mapping) — see next steps.

## Where the code is RIGHT NOW
- `scripts/vq2/control_check.py` — probe suite, our source of truth for control primitives.
  `--test rates [--rate R] [--calib]`, `--test hover`, `--test translate`. Calibration constants
  live here: `RATE_GAIN=(-2.55,-2.55,-2.35)`, `desired_to_cmd()`, `CALIBRATE` flag.
- `scripts/vq2/race.py` — the OLD controller. Threads gate 0 (6/6), tumbles at gate 1. Tuned around
  the BROKEN rate mapping. Do NOT bake calibration into it blindly — it would double-compensate the
  existing sign hacks and break gate 0. Reference/regression only.
- `eskf.py` / `estimator.py` / `guidance.py` unchanged in logic; guidance.py computes CORRECT rates
  (will finally work once fed calibrated commands + a sane attitude).

## ✅ UPDATE (2026-07-01, later): fly.py BUILT AND LIVE-TESTED — calibrated base PROVEN
`scripts/vq2/fly.py` (fresh controller, calibration at ONE boundary via
`control_check.send_att` with `cc.CALIBRATE=True`) was built and flown live. Results:

- **The calibrated attitude loop is STABLE.** Level-hover + point-and-go fly with
  tilt_max = 17.8° (just the platform's own rest pitch — never exceeded) and near-zero
  roll gyro RMS. Days of tumble are OVER. Failsafes (tilt-abort 60°, collision-abort,
  stale-live) all verified firing correctly.
- **Gate 0 threads consistently at t≈2.2s** (~0.7s after the stabilize window) — with
  physics-correct math, no per-gate tuning needed.
- **The instrumented gate-1 failure:** ~3s after gate 0, 3–6 collisions, ALL
  `COLLISION id=1002 (ENVIRONMENT)`, threat 1 — while attitude is PERFECT and the drone
  is ALIGNED with a red blob (last sighting bearing +2..+12°) but **`range=nan`: the
  PnP quad does NOT resolve on the gate-1 approach — we steer on the blob fallback with
  no metric depth** and fly into environment (pillar/wall/floor; gate 1 is the low/right
  gate, so the cy-thrust servo may also be descending us into the floor).
- Tried and measured: bank-toward-gate (KP_ROLL_BEAR=0.25) made it WORSE (tilt spikes;
  now OFF pending a `--test translate --calib` sign check); slower pitch (−0.07) and a
  1.2s post-gate pitch-up brake did NOT stop the ENV hits alone.

## ✅ UPDATE 2 (2026-07-02): perception SOLVED, obstacle identified — the frontier moved

Committed: branch `vq2-reorg-calibrated-controller` (reorg + fly.py in 8fd2f90, then
this session's perception work). `runs/` pruned 8.8 GB → 0.5 GB (kept trained YOLO
weights `runs/pose/.../best.onnx|pt`, RL `runs/rl_v1/`, all vq2 recordings).

### What was found by LOOKING at the captured frames (Claude can read PNGs!)
- `runs/vq2_collisions/*.png` (race.py's dumps, filename = PnP pose) show the truth:
  the gates cast a huge diffuse red GLOW on nearby walls/pillars/floor. **Both the
  blob centroid AND naive quad fits lock onto glow** → race.py/fly.py steered INTO
  glow-lit walls. Confirmed frame-by-frame. Far tiny gates also gave garbage PnP
  locks (the old x+13m "poses").
- **FIX SHIPPED — `scripts/vq2/gates.py`:** ring-validated detection (a gate is a red
  RING around a non-red core; glow is SOLID ~0.9 fill vs gate ~0.4; border-touching
  contours = partial gates rejected; PnP only from quads ≥800 px²; minAreaRect
  fallback for banner/perspective). Validated offline in `scripts/vq2/detect_lab.py`
  over 1118 real frames. fly.py now steers ONLY on validated gates (0.8s persistence,
  else search) — no more glow-chasing, and its sighting logs are now trustworthy.
- **The remaining blocker is now PHYSICAL, not perceptual:** a huge dark 3D-scanned
  sloped structure sits right of the gate-0 exit line (see `data/vq2/fly_collisions/`
  — fly.py dumps a frame on every collision now). The sim LAUNCHES the drone off the
  platform (~6+ m/s; gate 0 crossed 0.6s after stabilize); ~1.2s later it skims that
  structure → ENV collisions at t≈3.3s. Consistent 100% of attempts.

### Tried and measured (don't redo)
- Post-gate pitch-up brake +0.14/1.2s and +0.22/2.2s: collision time unchanged (~3.3s).
- Nose-up during the launch window (STAB_PITCH +0.10): drone hits the gate-0 FRAME
  instead of threading (3/3) — ride the launch LEVEL (gate 0 = 100%), brake after.
- Range-gated descent (VERT_RANGE_M=7, in fly.py): correct idea (stops early descent
  toward the far low gate) but the dark structure is hit at cruise height too.
- Bank-toward-gate 0.25: worse (off; sign unverified).

### → NEXT SESSION, concrete order
1. **Beat the dark structure after gate 0.** Ideas, best first:
   a. Watch one run with eyes on the sim (the structure's exact geometry decides this);
   b. steer a fixed OPEN-SPACE heading for ~1.5s after gate 0 (continue gate-0's exit
      direction, do NOT turn right toward gate 1's bearing yet — the straight-line
      path to gate 1 clips the structure), THEN acquire gate 1;
   c. use the optical-flow looming net (race.py `compute_looming`) to veto headings;
   d. climb over it (hold extra thrust after gate 0, descend only at range<7m — the
      range-gated descent is already in fly.py).
2. **Velocity estimate** (ESKF or flow) so braking is closed-loop, not timed.
3. **Metric turn trajectory** on ESKF + `guidance.py` once past the structure.
4. Then the speed roadmap in `docs/ROADMAP.txt` (racing line/MPCC → 100 Hz → cruise).

### The offline lab loop (use it — it needs no sim)
`uv run python scripts/vq2/detect_lab.py [--annotate N] [--glob 'g1_*.png']` scores
detector variants over the captured frames; `mask_peek.py <frame>` visualizes masks;
annotated output lands in `data/vq2/lab/`. Claude can Read the PNGs to judge quality.

## Norms
Commit only when the user asks; branch first if on master. Run
`uv run ruff format . && uv run ruff check --fix .` before any commit. Sim in Training (not
Qualification) for dev — competitive runs are code-audited.

## How to run
From repo root `C:\Users\caleb\Documents\ai_grand_prix` (sim in Training, a run live):
- Verify control primitives: `uv run python scripts/vq2/control_check.py --test rates` (raw, shows the
  inversion), then `--test rates --calib` (shows the fix). `--test hover` for hover thrust.
- Old controller (for reference/regression): `uv run python scripts/vq2/race.py --attempts 5 --max-s 30`.
- Estimation unit tests: `uv run pytest scripts/vq2/test_estimation.py -q`.
- Full fast suite: `uv run pytest -m "not simulation" -q`.
