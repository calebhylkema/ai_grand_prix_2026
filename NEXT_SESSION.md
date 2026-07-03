# NEXT SESSION — read this first (standing handoff doc, keep updated)

Fast catch-up handoff. Deeper detail in the memory dir — see
`project_vq2_rate_calibration.md` ⭐, `project_vq2_session_2026-07-01b.md` ⭐,
`project_workspace_reorg_2026-07-01.md`, `feedback_verify_fundamentals_first.md`,
`project_vq2_playbook.md`, `project_vq2_drop_findings.md`.

## What this project is
Python autonomy stack for the **AI Grand Prix 2026** autonomous FPV drone race. We are in
**Virtual Qualifier 2 (VQ2)**, live since 2026-06-30, sim build v1.0.3379, spec VADR-TS-003.
VQ2 removed ALL ground-truth telemetry (ATTITUDE / LOCAL_POSITION_NED / ODOMETRY / GATE_INFO).
**We fly from monocular camera + IMU only.** Objective: fastest valid completion → qualify for the
Sept physical round. VQ2 work is in `scripts/vq2/`. Keep the sim in **Training** for dev.

## 🎯 CURRENT STATE (2026-07-02 pm) — where we actually are
- **Gate 0 threads 100% at t≈2.0s.** Flight is stable (tilt = platform's 17.8° rest pitch
  until a terminal collision). The multi-day tumble era is OVER.
- **Active controller is `scripts/vq2/fly.py`.** `race.py` is reference/regression only.
- **The "gate-0-exit structure" hypothesis was WRONG — it was a FLOOR SINK.** The old
  collision at t≈3.3s was the drone *sinking into the floor* right after gate 0, NOT a
  lateral structure. Proven by a new trajectory recorder (below): the gate climbed out the
  top of the frame while the floor filled the view. The vertical servo that should have
  climbed never fired (it was range-gated on PnP range, which is ALWAYS nan on 3379).
- **FIXED this session** → the drone now reaches **gate 1's vicinity at t≈4.5–5.5s** (was
  3.3s), centring gate 1 in both axes and flying through its plane (bearing sweeps +35°→0→
  −36° as we pass; gate area explodes then a new small gate is acquired). Big, consistent
  progress. See "What changed" below.
- **NEW frontier = a DIFFERENT obstacle at/just past gate 1** (`COLLISION 1002 ENV
  threat2`, was threat1) at t≈5–5.6s, which knocks the drone tumbling (tilt→80–97° is the
  RESULT of the impact, not the cause). Also `active_gate` never leaves 1 → gate-1 pass may
  not be registering cleanly (we may be clipping its frame on the way through).

## ⭐ What changed in fly.py this session (2026-07-02 pm) — all committed-worthy
1. **Self-diagnosis tooling (the user's ask: "solve it from the drone's own inputs"):**
   - **Trajectory recorder** — dumps a camera frame every 0.15s to `data/vq2/fly_trace/`
     (cleared each run), with attitude / vz / gate cy·area·bearing baked into the filename.
     Reading the frame SEQUENCE into a collision reconstructs the approach geometry with NO
     human watching the sim. This is how the floor-sink was found. USE THIS FIRST next time.
   - **IMU vz** (VelocityEstimator) logged in the summary — but it is **DIAGNOSTIC ONLY**:
     it read a CLIMB while the camera showed us hitting the floor → integrated accel is too
     biased through the launch to drive thrust. Do NOT servo thrust on it (tried, broke gate 0).
2. **Continuous visual vertical servo** (`vertical_thrust`): thrust is a PD on the gate's
   image ROW (cy→CY_TARGET 130), running EVERY loop (not range-gated). Climb always allowed
   (safe); descend only when the gate is big/close (`DESCEND_AREA`). D term damps on d(cy)/dt
   (`KD_THR_PX`) to stop over-climb into the ceiling. `NO_GATE_CLIMB` bias when blind.
   `KP_THR_PX 0.10→0.20`, `THR_MAX 0.36→0.45`. This is what stopped the floor sink.
3. **"Aim then go"**: forward pitch only when the gate is centred in BOTH axes (bearing AND
   cy); otherwise nose-up brake (`TURN_BRAKE_PITCH`) and let yaw+bank+climb centre it first.
   Stops sliding past a side gate and passing UNDER a high gate (`CY_ALIGN_BAND`).
4. **Bank-toward-gate RE-ENABLED** (`KP_ROLL_BEAR 0→0.30`, `MAX_ROLL_TGT 0.12→0.25`): yaw
   rotates where we POINT, banking rotates where we GO — needed to turn onto gate 1 against
   launch momentum. Verified correct sign in fly_trace (rolled right toward a right gate).
5. **BRAKE_SECS and HOLD_HEADING_SECS set to 0** (disabled): they were built on the wrong
   "lateral structure" theory and forced flat thrust during the sink. Constants retained.

## ⭐⭐ The multi-day root cause (2026-07-01) — rate calibration
The drone tumbled at every turn for days. Measured the control primitives directly (gyro =
ground truth for body rate) with `scripts/vq2/control_check.py --test rates`:

**Rate commands are INVERTED and ~2.5x scaled.** `actual_gyro ≈ GAIN * commanded`,
`GAIN = (roll -2.55, pitch -2.55, yaw -2.35)`. Linear (verified ±0.4 and ±0.8 rad/s), no
cross-coupling. Commanding +roll to fix rightward drift rolled the drone the OPPOSITE way 2.5x
too hard → every correction diverged → tumble. race.py's sign-flip hacks (YAW_SIGN, GYRO_INT_SIGN,
ROLL_X_SIGN, THR_SIGN) were accidental partial compensation.

**FIX (verified):** invert at the command boundary — `cmd = desired / GAIN`. `--test rates --calib`
then shows scale +0.98 (correct sign, ~unity) on all 3 axes. **Hover thrust ≈ 0.30.**
`fly.py` applies this at ONE boundary (`cc.CALIBRATE=True` → `control_check.send_att`); every rate
in fly.py is a *desired actual* body rate. Do NOT retrofit this into race.py (would double-compensate
its sign hacks and break gate 0).

## ⭐ Perception glow-trap SOLVED (2026-07-02) — gates.py
Reading the captured collision frames (`runs/vq2_collisions/*.png`, filename = race.py's PnP pose)
showed the truth: **gates cast a huge diffuse red GLOW on nearby walls/pillars/floor, and both the
blob centroid AND naive quad fits lock onto the glow** → the drone steered into glow-lit walls.
Far tiny gates also gave garbage PnP (the old x+13m fly-off "poses").

- **`scripts/vq2/gates.py`** — glow-resistant detection. A gate is a red RING around a non-red core
  (glow reads solid, core-fill ~0.9, vs a real gate ~0.4); border-touching contours rejected
  (partial gate → garbage PnP); PnP only from quads ≥800 px²; minAreaRect fallback when the
  AI-GP banner / perspective breaks the strict 4-vertex fit. Returns `GateTarget` (cx, cy, area,
  bearing, pose_cam, `.range_m`). `nearest()` = best steering target.
- Validated OFFLINE over 1118 real frames with **`scripts/vq2/detect_lab.py`** (no sim needed).
- `fly.py` now steers ONLY on validated gates (0.8s persistence, else scan) — no glow-chasing, and
  its sighting logs are trustworthy. Vertical descent is range-gated (<7m) so a far LOW gate can't
  trigger an early descent into whatever's between us and it.

## 🗂️ Workspace layout (reorganized 2026-07-01) — if a memory path 404s, look here
- `scripts/` is **by phase**: `scripts/vq2/` (⭐ active), `scripts/vq1/` (was `scripts/run_race.py`
  + `scripts/sim_v1/*`), `scripts/pre_sim/` (unchanged). `scripts/vq2/_archive/` = superseded probes.
- `docs/specs/` = official material (VADR-TS-001/003 PDFs, vq1/vq2 spec refs).
- `docs/ROADMAP.txt` = merged PLAN+PROGRESS incl. the ordered SPEED roadmap.
  `docs/CODE_MAP.md` (was root MANIFEST.md) = per-file map. Deleted CHATGPT_RESEARCH/GSD_SETUP.
- `SIMULATOR/` = OLD VQ1 sim v1.0.3364 — **KEEP: still broadcasts ground truth = estimator
  validation rig.** `VQ2/` = the VQ2 sim install only (gitignored `/VQ2/`, anchored).
- `config/default.toml` camera fixed to VQ2 spec (fx=fy=320, tilt −20 = 20° UP).

## Where the code is RIGHT NOW
- **`scripts/vq2/fly.py`** — ACTIVE controller. Stages `--stage hover|forward|race`, `--secs`,
  `--attempts`. Calibrated rates, attitude P-loop (KP=4, MAX_RATE=2), gates.py perception,
  continuous visual vertical PD servo, aim-then-go + bank turning, failsafes (tilt-abort 60°,
  collision-abort ≥3, stale-live). Dumps a frame per collision to `data/vq2/fly_collisions/`
  AND a full trajectory (every 0.15s) to `data/vq2/fly_trace/` (cleared each run) — READ THIS
  to self-diagnose. `vz=[min,max]` in the summary is a logged-only IMU descent estimate. This
  is where tuning happens.
- `scripts/vq2/gates.py` — perception (above).
- `scripts/vq2/control_check.py` — control-primitive probe + calibration source of truth
  (`RATE_GAIN`, `desired_to_cmd()`, `CALIBRATE`, `Shared.last_collision`). `--test rates|hover|translate`.
- `scripts/vq2/detect_lab.py`, `mask_peek.py`, `_lab_gates_check.py` — offline perception lab.
- `scripts/vq2/eskf.py` / `estimator.py` / `guidance.py` — validated but not yet wired into fly.py.
  guidance.py computes physically-correct rates (finally usable now that rates are calibrated) —
  the metric turn-trajectory path once we're past the structure.
- `scripts/vq2/race.py` — OLD controller, reference only (threads gate 0, tumbles at gate 1;
  tuned to the broken rate mapping).

## Tried and measured (DON'T redo)
- The "gate-0-exit structure" was a misdiagnosis — it was a FLOOR SINK (see above). The
  post-gate pitch-up brake / open-space heading-hold / STAB_PITCH nose-up experiments were
  all chasing that phantom. Brake + hold are now DISABLED (=0).
- IMU-integrated vz to hold altitude: BROKE gate 0 (estimate biased through launch). vz is
  logged-only now. The camera (gate image row) is the reliable height cue.
- Range-gated vertical servo (`VERT_RANGE_M`): never fired because PnP range is always nan
  on 3379 → replaced with a continuous image-row PD servo. Removed.

## → NEXT SESSION, concrete order
0. **Read the fly_trace first.** `uv run python scripts/vq2/fly.py --stage race --attempts 1`
   then `ls data/vq2/fly_trace | sort` and Read the last ~8 PNGs (filenames carry
   t/roll/pitch/vz/gate). This shows the approach into the terminal collision WITHOUT eyes
   on the sim — it is how the floor sink got solved. Do this before changing anything.
1. **Beat the threat2 obstacle at/just past gate 1.** From this session's traces we reach
   gate 1 centred (cy≈130, bearing≈0) at t≈4s then hit ENV threat2 at ~5s. Likely we clip
   gate 1's own FRAME on the way through (close-range bearing/cy swing hard) or hit a
   structure immediately beyond it. Ideas: (a) once the gate is BIG+centred (area>~4000)
   stop steering and commit straight through (freeze roll/yaw so we don't over-bank into the
   post); (b) confirm whether threat2 = gate frame (GATE would be 1001 — it's 1002 ENV, so a
   real structure past gate 1); (c) reduce bank/turn-brake authority at close range.
2. **Why doesn't `active_gate` leave 1?** Confirm we actually pass gate 1 vs clip it. If we
   clip, the both-axes centring band (`CY_ALIGN_BAND`, bearing) may need tightening so we
   thread the true opening, not the edge.
3. **Metric turn trajectory** on ESKF + `guidance.py` once past gate 1 — arrive centred in
   position AND velocity (both validated; guidance has correct rates now).
4. Then the speed roadmap in `docs/ROADMAP.txt` (racing line/MPCC → 100 Hz → cruise → RL last).

## Git / status
- Branch **`vq2-reorg-calibrated-controller`** (off master). Committed: `8fd2f90` (reorg + fly.py),
  `9b9622f` (gates.py perception + offline lab). Working tree clean except ongoing tuning.
- `runs/` pruned 8.8 GB → 0.5 GB (kept `runs/pose/` YOLO best weights, `runs/rl_v1/`, all vq2
  recordings). `data/vq2/{fly_collisions,lab,recon_frames,probe_frames}` are the frame corpora.
- Not yet merged to master — do that once the structure is beaten and a full course completes.

## Norms
Commit only when asked; branch first if on master. `uv run ruff format . && uv run ruff check --fix .`
before any commit. Sim in **Training** (not Qualification) for dev — competitive runs are code-audited.

## How to run (repo root `C:\Users\caleb\Documents\ai_grand_prix`, sim in Training, a run live)
- Fly: `uv run python scripts/vq2/fly.py --stage race --secs 60 --attempts 3`
  (also `--stage hover` / `--stage forward` to isolate layers).
- Offline perception (no sim): `uv run python scripts/vq2/detect_lab.py --annotate 8 --glob 'g1_*.png'`
  → writes annotated frames to `data/vq2/lab/` (Claude can Read the PNGs to judge quality).
- Verify control primitives: `uv run python scripts/vq2/control_check.py --test rates` (raw inversion),
  then `--test rates --calib` (the fix); `--test hover`; `--test translate`.
- Tests: `uv run pytest scripts/vq2/test_estimation.py -q`; full fast suite `uv run pytest -m "not simulation" -q`.
