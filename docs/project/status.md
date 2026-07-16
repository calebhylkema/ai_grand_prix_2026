# Project Status — VQ2

> **This is the single authoritative status document.** Update it in place; do not
> create new session-handoff files. Last updated: **2026-07-15**.

## 2026-07-16 — Roll-observer + arbitration (attempt-2 roll_invalid was a multi-observer disagreement) (UNCOMMITTED)

**Two live attempts of the 2026-07-15(b) build, then a full estimator audit + a dedicated bank
observer. Nothing committed/reset/stashed. Post-pass ESKF velocity authority, MPCC, map, RL all
OFF. Calibrated boundary, arming, global thrust clamp, and the 45°/60° roll+tilt backstops
untouched.**

### Live attempts (logical handoff + geometry bank)
- **Attempt 1** (`--experimental-visual-handoff`, NO `--record` — my omission, now impossible):
  gate 0 threaded (t2.2s), visual PRETURN did NOT latch → legacy BRAKE/SCAN → tilt-abort tumble
  (roll −103°), coll=1. No recording.
- **Attempt 2** (`run_20260716_001807`, recorded, **live==replay exact**): the visual handoff
  ENGAGED — `idle→preturn→turn→reacquire_arc→safe_exit`; the **logical handoff fired live**
  (established raw0 → gap → reacquired raw3 @15°); geometry bank commanded 38°; **coll=0**, no
  tumble. It SAFE_EXITed on `roll_invalid`.

### ⭐ Attitude audit (Phase A/B) — CORRECTS my first read of attempt 2
My initial "control-att ~40° true, shadow ESKF over-reads to 47°" was WRONG (I'd only looked at
the tail t≥+1.4s). Full audit (raw gyro sign, command-integral bank, offline no-flip quaternion
integration — three independent signals) shows:
- **The drone REALLY over-banked to ~48°** (right, monotonic to a peak @+0.77s, then unwound) —
  a genuine roll-rate-inertia overshoot of the 38° geometry target, past the 45° validity.
  Confirmed: raw gyro roll rate **+0.97 rad/s** = commanded `RATE_GAIN·cal` **+1.0**; the no-flip
  quaternion integration matches the control attitude within ~1° across the whole turn.
- **The shadow ESKF roll is sign-DEFECTIVE for roll**: it applies `GYRO_SIGN=−1` (correct for the
  ESKF's velocity axes, WRONG for the roll axis here) → reads the wrong sign during the right bank,
  then a late fix-driven reversal to +45°. **That spurious reversal is what tripped `roll_invalid`**
  — the wrong signal, at the wrong time; the real +48° over-bank earlier went undetected by that
  gate. Euler-coupling is negligible (pitch only ~−18°).

### Fix (Phases C–F, offline-validated)
- **C — isolated bank observer** (`visual_handoff.py`): its OWN scalar-first quaternion, seeded
  from the control attitude, propagating the raw body gyro (validated no-flip roll sign), NO accel
  levelling, optional slew-limited PnP roll correction (gyro-only in the live path — PnP roll is
  too sparse/inconsistent per the audit), reset per attempt, fully decoupled from the shadow ESKF.
- **D — fail-closed roll arbitration**: primary = observer; independent envelope = command-integral
  bank; control-att + shadow ESKF are cross-checks only. Fail closed on observer unavailable /
  stale / gyro-gap / nonfinite / `observer>45°` / `integral>cap` / persistent OVER-envelope
  disagreement (asymmetric — over-bank is the danger; under-reading is safe). `shadow_roll_degraded`
  is RECORDED when the shadow alone reverses while observer+integral agree, but it no longer
  terminates the turn. Roll FEEDBACK now uses the observer, not the sign-defective shadow.
- **E — derate**: geometry command cap **38→34°**, feedforward cap **38→35°** (record
  `bank_geometry_raw` + `bank_authority_cap=34°` + clamped). Validity 45° / abort 60° / roll-rate
  ±1.0 UNCHANGED. Ladder: 34 ≤ 35 < 45 = 45 ≤ 60.
- **F — no more unrecorded authority runs**: `--experimental-visual-handoff` now REQUIRES
  `--record`; per-tick `preturn_reject` reason recorded + an end-of-attempt console histogram.

### Verification (Phase G)
19 new tests (14 observer + 5 arbitration + 2 CLI, minus overlap) + reworked roll tests. **VQ2:
429 pass.** Full suite: 872 pass + the documented MPCC flake (a different MPCC test flaked this
run under load; **passes in isolation**; MPCC untouched). **Ruff/format/Mypy clean; /review PASS.**
Exact-replay preserved (recorder roundtrip green; new `ctrl_vh_body_gyro` + observer columns
recorded, control roll/pitch/yaw reconstructed from `ctrl_att`).

### Phase H — next recorded live attempt: PENDING
34° command cap + 35° ff + observer feedback + required recording; decision rules per the plan.
Gate 1 not yet threaded live.

## 2026-07-15 (b) — Logical gate-1 handoff + geometry-derived corner bank (UNCOMMITTED, offline-validated; live campaign pending)

**Built + fully validated offline this session, all Training-only/log-gated behind the existing
`--experimental-visual-handoff` flag. Nothing committed/reset/stashed/discarded. ESKF post-pass
velocity authority, MPCC, map authority, RL remain OFF. The calibrated command boundary,
arming, rate/thrust clamps, and normal aborts are untouched.**

### Phase 0 — speed reconciliation (CORRECTS the earlier "ESKF over-reads ~2×" claim)
Reconstructed per-run from the six v9 recordings (no live attempt used). The recorded
`ctrl_vh_spd_exit` ≈ **6.9–7.0 m/s is a STALE PnP closure** — it is the closure evaluated at the
*last fresh gate-0 metric fix* (~t1.5 s), NOT the physical pass speed. After that fix the gate
clips to bearing-only and the drone keeps accelerating: the ESKF *velocity*, the ESKF
position-derivative `|p|`-rate, and a fresh closure would all **rise together** past t1.5 s — so
there is no 2× over-read; the earlier "over-reads ~2×" was an averaging artifact of comparing a
stale closure against a live velocity and **is withdrawn**. Reconciled physical pass speed:
**v_exit ≈ 9.0 ± 1.5 m/s** (conservative upper CI ~10.5). Terminology fix applied in code
comments: `ctrl_vh_spd_exit` is labelled a *stale PnP closure*, not the physical pass speed. Raw
ESKF velocity stays diagnostic-only post gate 0.

### Phase 1 — LOGICAL gate-1 identity above the churning raw tracker ids (`visual_handoff.py`)
Raw track ids are **not stable physical labels** (they churn 0→2→4 across the pre-turn). Added a
logical identity keyed on the **ESKF-independent world azimuth = `control_yaw + bearing`**, where
`control_yaw` is the control-attitude gyro-integrated yaw (`fl.att.euler[2]`, world-NED, +right —
**NOT** shadow-ESKF yaw). A fixed gate holds this azimuth ~constant as the drone yaws, so a
different raw id is accepted as the SAME physical gate only via a guarded handover: existing
logical provisional established, right-side/confirmed (≥3 hits), within the 0.35 s gap, wrapped
azimuth residual ≤15°, plausible azimuth rate. On acceptance it preserves the streak, swaps the
raw id, records `logical_provisional_reacquired`, and does NOT restart PRETURN. Rejections record
the residual + reason (`rejected:imposter|gap|unconfirmed|rate|wrong_side`). New immutable input
`control_yaw_rad` is recorded (`ctrl_vh_ctrl_yaw`) and reconstructed for exact replay (older runs
fall back to the identical `ctrl_att[:,2]` signal). PRETURN now holds through raw-id churn/gaps
up to 0.35 s (revocation gap falls back to the raw provisional-sighting age before the logical id
is established, so pre-logical/NaN-yaw behaviour is preserved).

### Phase 2 — offline replay of all 6 recordings (deterministic)
| run | rows | determ | logical timeline (raw→reason@resid°) | max PRETURN gap | reached TURN |
|-----|-----:|:------:|--------------------------------------|----------------:|:------------:|
| 193148 | 248 | ✓ | est raw0 → gap → **reacquired raw2 @11°** | 0.200 s | ✓ |
| 193839 | 154 | ✓ | est raw0 → gap → **reacquired raw2 @14°** | 0.200 s | ✓ |
| 194346 | 517 | ✓ | est raw0 → **reacq raw2 @12°** → rej imposter @24° | 0.267 s | (refused) |
| 194949 | 161 | ✓ | est raw0 → **rej imposter @21°/gap** | 0.551 s (>0.35 → correctly revoked) | (churn) |
| 195643 | 1220 | ✓ | est raw0 → **reacq raw2 @10°** → rej imposter @36° | 0.317 s | ✓ |
| 195951 | 1233 | ✓ | est raw0 → **rej raw2 @20°** → **reacq raw4 @6°** | 0.334 s | ✓ |

All 6 are byte-deterministic (replay twice → identical). **Run 195951 refutes the earlier
"track 4 = imposter" framing:** there, track 2 sits at world-azimuth 21.7° (20.4° off the logical
gate → correctly *rejected*) while track 4 sits at 48.6° (6.4° off → correctly *accepted*). The
world-azimuth continuity test — not the raw id — is the authority, per design. PRETURN survives
gaps ≤0.35 s (0.200/0.267/0.317/0.334 held into TURN) and correctly reverts the 0.551 s gap.

### Phase 3 — geometry-derived corner bank (`visual_handoff.py`)
The gate-0→1 corner is a coordinated turn of radius **R = (x²+y²)/(2y) = 10.39 m** (mined map
offsets x≈9.64 fwd, y≈6.52 right). Bank = `atan(v²/(g·R))`, frozen once at TURN entry from a
**conservative pre-pass speed constant** (`corner_bank_speed_mps=9.5`, the Phase-0 upper-CI —
NOT the stale 6.9 closure) and **clamped to the reviewed 38° max**. At v=9.5 the raw bank is
41.5° → clamps to 38°; it scales down cleanly if a later attempt brakes (7.0→25.7°, 7.7→30.2°,
8.5→35.3°). Used as the TURN bank ceiling (bearing schedule ramps up to it); **REACQUIRE_ARC
preserves the frozen geometry bank**; CAPTURE tapers off it; PRETURN unaffected (≤20°).
Authorized bounded raises: feedforward cap **35°→38°**, local nose-level thrust **0.36→0.38**
(global calibrated clamp 0.45 UNCHANGED). Roll-validity **45°**, tilt-abort **60°**, roll-rate
clip **±1.0 rad/s** ALL UNCHANGED; 38° geometry max < 45° validity < 60° abort.

### Phase 4 — verification
- **19 new unit tests** (logical establish/reacquire/imposter/gap/hits/±π-wrap/streak/reset/
  no-yaw; PRETURN gap-survival + logical-reacquisition promotion; geometry bank formula/38°-clamp/
  frozen/ceiling/REACQUIRE-preserve/validity/frozen-vs-closure).
- **VQ2 suite: 390 pass.** Full suite (`-m "not simulation"`): **852 pass, 1 fail = the documented
  MPCC-latency timing flake** (p95 inflated under full-suite parallel load) — **passes in isolation**
  (`test_mpcc_update_latency` ✓ alone), unrelated to these changes (no MPCC touched). Effective 853/853.
- **Ruff check: clean. Ruff format (targeted, changed files only): clean. Mypy `src/drone_racing`:
  Success, no issues (68 files). `/review`: PASS** (all bank paths clamped ≤38°; NaN/inf guards on
  the geometry + azimuth math; no safety threshold removed; calibrated boundary untouched).

### Phase 5 — live Training campaign: **PENDING** (staged, not yet run)
Code is complete and offline-validated; the live 8-attempt campaign (logical handoff + 38° geometry
bank, no aggressive brake, existing 1.2 s REACQUIRE_ARC) needs the sim window focused + Backspace
restarts. Sim confirmed up (heartbeat sys 1). **Gate 1 not yet threaded live this session.**

## 2026-07-15 — Gate-0 track-steal FIXED + exit-speed control; real corner speed ~8.5 m/s (UNCOMMITTED)

**Three new pieces built + validated this session, all Training-only/log-gated behind the
existing `--experimental-visual-handoff` flag. Gate 1 still NOT threaded (0 passes). Nothing
committed. ESKF post-pass velocity authority, MPCC, map authority, RL remain OFF.**

**1. Active gate-0 track-steal FIXED (`perception.py`, Phase 1).** Root cause (audited
frame-by-frame across the four v9 runs): the genuine gate-0 track (posed ~5 m/1°) was
absorbing a downstream ~21 m/+38° gate — the association cost is image-centroid based and the
two gates are image-adjacent (cx 350 vs 383), so the 5→21 m / 1°→38° jump was ungated. Both
were published as the same active `track 1`, driving the PRETURN `revoke_gate0_misaligned`
oscillation. Fix: a `GateTrack.min_range` memory + `_identity_jump` guard — a track that posed
CLOSE cannot rebind to a candidate whose metric range jumps to >15 m or bearing jumps from
near-centred to >30°; the match is dropped (recorded `active_identity_jump`) and the far gate
spawns its OWN track (it is the real gate-1 provisional). Live-confirmed: **0 active far-metric
jumps, NIS rejections 6→2, NIS max 2800→27.** Regression tests added (6).

**2. PnP-closure speed estimate + ESKF-speed validation (`speed.py`, Phase 2).**
`pnp_closure_estimate` = robust median pairwise `-Δrange/Δt` of genuine, accepted, near-centred
(±10°) gate-0 metric obs (dedup by frame arrival, reject range increases / impossible steps).
`validate_gate0_speed` gates the scalar ESKF horizontal speed on anchor==0, fresh fix, low
velcov, and PnP agreement — **fails closed**; a stale-fix "coast" is deliberately refused (see
finding). 17 unit tests.

**3. Gate-0 exit-speed controller (`visual_handoff.py`, Phase 3).** A bounded, Training-only
nose-up PITCH-only overlay during the anchored gate-0 approach (PRETURN keeps roll/yaw; thrust
stays legacy). Engages above a speed threshold, `pitch=clip(kp·(speed−target),0,0.60)`,
cumulative pitch-impulse budget (22°), revokes on misalignment/validation failure, and at the
advance captures the exit speed + REFUSES the corner (bounded SAFE_EXIT) if it is too fast.
Recorder schema-v9 `ctrl_vh_spd_*` + exact `replay_visual_decisions` (all live runs replay
byte-for-byte). ~13 controller tests.

**⭐ KEY MEASURED FINDING — the ESKF VELOCITY over-reads ~2× on the gate-0 approach.** The
reliable ESKF *position* (|p| to gate 0: 11.1 m→0.3 m) and the PnP closure both give a real
corner-entry speed of **~8.5 m/s**, while the ESKF *velocity* reads **11–12 m/s** (velcov grows
from ~0.5 to ~3 as gate 0 clips to bearing-only and the filter coasts). So the drone is NOT
11–12 m/s (corrects the prior assumption) — it is ~8.5 m/s. That is still marginally above the
**7.7 m/s a 30° bank needs for the R≈10.4 m corner**, and a 30° bank + 1.2 s REACQUIRE_ARC only
turns ~52° of the ~90° corner at 8.5 m/s → it under-turns / flies past.

**Live campaign (6 attempts, all schema-v9, exact live==replay):**
1. `193148` — brake didn't engage (gated on active_gate==0, but the metric window precedes the
   race counter setting gate 0) → re-gated on anchor==0.
2. `193839` — brake engaged 0.58 s (+0.60 rad/s, proven physical via raw gyro) but speed still
   rose; corner refused (`exit_speed_too_high`), **coll=0**.
3. `194346` — earlier/lower target braked harder but pitched the camera up and BROKE the gate-1
   provisional confirmation → PRETURN never engaged → legacy tumbled.
4. `194949` — bigger impulse cap, PRETURN engaged, clean refuse, **coll=0**. Established the
   drone's nose-down TRIM (~−18°) means a ≤15° pitch-up only reaches ~−6° absolute (still
   accelerating).
5. `195643` — coast removed + closure-based speed + permit ≤9.5; PRETURN engaged briefly then
   the provisional identity churned (track 0→lost→2→4) → IDLE → legacy tumbled.
6. `195951` — refuse 8.5; PRETURN again failed to engage → legacy tumbled.

**Honest outcome: gate 1 NOT threaded, 0 consecutive passes.** Two blockers: (a) the drone is
~8.5 m/s — marginally too fast for a bounded 30° corner — and the launch-limited nose-up brake
(fighting the −18° trim) cannot reliably reach ~7 m/s; (b) the identity-jump fix, while
correctly stopping the active-gate-0 steal, **destabilised the provisional gate-1 confirmation**
in some runs (track churn), so PRETURN engagement is now non-deterministic — when it engages the
control cleanly refuses the corner (SAFE_EXIT, no collision, attempts 2/4); when it doesn't,
legacy BRAKE→SCAN tumbles (attempts 3/5/6). **390 vq2 + 833 full-suite pass, Ruff/format/Mypy
clean, `/review`=PASS.**

**Recommended next (separate):** (i) stabilise the provisional gate-1 track through the
identity-jump change so PRETURN engages reliably; (ii) the corner needs ~7 m/s — either an
earlier/gentler brake that does NOT pitch the camera off the provisional, or accept a slightly
larger post-pass bank (the drone needs ~36° at 8.5 m/s, vs the 30° cap). Do NOT trust the
coasted ESKF velocity (it over-reads ~2×); the PnP closure / ESKF position are the reliable
speed measures.

## 2026-07-14 — Visual gate-0→1 handoff: mechanism validated; blocked by fly-past momentum (UNCOMMITTED)

**A Training-only visual handoff controller was built, fully tested, and live-validated
across 8 attempts. The MECHANISM works (engage → same-id promotion → bounded bank →
bounded REACQUIRE_ARC that continuously tracks and re-locks gate 1 → clean termination;
exact live==replay every run; ZERO collisions attempts 3–8). Gate 1 was NOT threaded: the
drone flies PAST gate 1 and none of the authorized control levers can redirect it. Nothing
committed. ESKF velocity/position guidance, MPCC, map authority, and RL remain OFF.**

**CORRECTED root cause (supersedes the earlier "gate 1 absent from the FOV for ~3 s"
claim, which was wrong — it read only the first 20 gate-1 frames).** Gate 1 is tracked and
reacquired throughout. The real blocker: after the pass the drone's velocity carries it
**forward past gate 1** — the gate's *world azimuth* (yaw heading + bearing) sweeps
**+42°→+93°** (perception-confirmed, independent of the ESKF) while the drone yaws right to
chase; it never comes within the 3–16 m fusion band (slant stays NaN / bearing-only). With
ESKF horizontal speed ~12→16 m/s a ≤30° bank gives R≈35 m vs a ~9 m gate — the classic
"yaw points the nose but does not steer the velocity" under-turn. Redirecting/decelerating
that momentum needs velocity-vector control (ESKF velocity guidance / closed-loop braking),
which is off-limits for this task.

**Controller — `deployments/vq2/visual_handoff.py` (new, pure/deterministic, Training-only
authority):** states IDLE→PRETURN→TURN→CAPTURE→COMMIT + **REACQUIRE_ARC** + SAFE_EXIT.
PRETURN overlays right roll+yaw only; TURN replaces BRAKE/SCAN with a bearing-scheduled
right bank + image-row thrust; **REACQUIRE_ARC** is a bounded (≤1.2 s) right-arc that keeps
turning through a confirmed-target loss and re-locks gate 1 (original id, or a guarded
NEW id: race-1 + right-turn + ≥3 hits + bearing-continuity imposter guard); SAFE_EXIT emits
a bounded level/brake for `safe_exit_hold_s` (0.40 s) then signals `mission.run()` to END
the attempt (fixes the prior SAFE_EXIT-holds-thrust-for-seconds bug) — never resumes the
blind SCAN. Bank feedback uses ONLY the sign-corrected shadow-ESKF roll channel (see below)
+ a feedforward commanded-bank hard cap; emits desired ACTUAL rates through the unchanged
calibrated `Flight.command`. A bounded open-loop corner brake was added (attempts 7–8) and
left at a conservative bound; it did not resolve the fly-past.

**Roll-convention audit (independent):** PnP `vision_roll` correlates **positively** with
control-att roll (+0.80/+0.21) but **negatively** with shadow-ESKF roll (−0.63/−0.76) →
the ESKF roll IS sign-inverted vs the command frame (`eskf_roll_feedback_sign=−1` correct;
the old "agrees to 1.49°" was a near-zero-magnitude artifact). The feedforward gain was
corrected from an inferred 1.5 to the **measured** command→actual roll-rate ratio ~0.9–0.95
(use 1.0). Hard bank cap 35° + roll-validity 45° + 60° tilt abort unchanged.

**Files changed (all uncommitted):** new `visual_handoff.py`; `gate_relative.StructuredTarget`
+6 image-geometry fields (defaults keep v7/v8 replay valid); `shadow.py` populates them;
`recorder.py` → **schema v9** (`ctrl_vh_*` decision + reacquisition diagnostics + active/
provisional geometry) with exact `replay_visual_decisions`; `mission.py` wires
`--experimental-visual-handoff` (Training TOML `permit_visual_handoff` + CLI flag;
Qualification rejects; mutually exclusive with `--experimental-transition-authority`;
byte-identical when off) and terminates on the visual `terminate` signal; `config.py` +
TOMLs add the permission. **354 vq2 tests + 798 full-suite pass, Ruff + format + Mypy clean,
safety `/review` = PASS** (calibrated boundary/RATE_GAIN/clamps/arming/aborts untouched;
safety aborts retain priority; all outputs finite+clamped; SAFE_EXIT termination is a net
safety improvement).

**Offline evidence:** exact `replay_visual_decisions` reproduces the selected command +
state byte-for-byte on all six live v9 captures (live==replay). The revised controller was
also replayed against attempt-4 observations to prove it enters REACQUIRE_ARC (not level),
keeps positive bank/yaw, cannot exceed 1.2 s, and SAFE_EXIT terminates after 0.40 s (logic
only — it cannot predict the altered physical trajectory).

**Live campaign — all 8 attempts (all schema-v9, exact live==replay, coll=0 from #3 on):**
1. `202828` never engaged (gate-0 target de-selects as the provisional confirms) → gate-0
   proximity **latch** + provisional-gap tolerance.
2. `203538` engaged (388 rows) but spurious `identity_mismatch` at the advance (active
   target lags the race counter ~1 frame) → enter TURN unless a *valid, different* id appears.
3. `203831` TURN engaged, no collision, roll ran to a 60° tilt abort (ESKF roll sign-inverted
   + under-reads) → sign-correct + **feedforward bank cap**.
4. `204817` roll bounded ~20°, but TURN lasted ~5 frames then leveled and gave up; SAFE_EXIT
   held thrust ~3 s → add **REACQUIRE_ARC** + guarded reacquisition + SAFE_EXIT termination;
   correct the ESKF-roll gain to the measured 1.0.
5. `211830` (REACQUIRE_ARC) **reacquired a guarded NEW gate-1 id and resumed TURN**, tracked
   it — but the bearing/world-azimuth kept growing: it under-turns and flies past. Clean
   terminate `visual_handoff:roll_invalid`, tilt 29°.
6. `212207` +5° bank (under-turn rule) → still flies past; bank is NOT the limiter (speed is).
   Terminate `reacquire_timeout`, tilt 34°.
7. `212746` bounded corner brake (0.30 rad/s/1.0 s) → nose-down −27°→−10° but **no
   deceleration** (speed 12→17), still flies past.
8. `212939` stronger brake (0.45 rad/s/1.2 s) → pitch to level (+1°), still **no
   deceleration** (11→16), still flies past. Brake reverted to the conservative bound.

**Acceptance NOT met.** Gate 1 was never threaded; **0 consecutive gate-0→1 passes.** The
handoff mechanism is fully validated but the drone flies past gate 1, and the four authorized
evidence-driven attempts (bank +5°, brake 0.30, brake 0.45, baseline arc) do not stop it.

**Recommended next step (separate, explicitly-authorized):** the gate-1 corner needs the
velocity vector redirected/decelerated so the drone actually flies TO gate 1 (and slows
enough for PnP to resolve an in-band metric pose). That requires closed-loop velocity
guidance — a reviewed, authorized use of the shadow-ESKF velocity (currently off-limits) —
or a slower gate-0-thread corner entry. Do NOT weaken perception/association/NIS/range/
right-turn/slant-band guards; the perception is tracking gate 1 correctly.

## 2026-07-13 rework checkpoint

- **Pre-pass gate-1 handoff implemented; first authority-off capture found and fixed one
  confirmation-publication defect (2026-07-13).** `GateTracker` now performs one
  association pass while publishing distinct active and correct-turn-side provisional
  identities. The provisional is direction-only, cannot steal gate 0, is never fused
  before promotion, and retains a far pose without fusion until inside the unchanged
  3–16 m band. Schema-v8 run `run_20260713_185343.npz` proved the visual handoff itself:
  provisional track **3** was seen for 10 consecutive frames and promoted as active track
  **3** at gate 0→1. It also exposed that `ShadowWorker` reset published `track_hits`
  **10→1** when the role changed, so the transition policy returned `no_target` on its one
  pre-45°-tilt row. Fixed by publishing the tracker's authoritative streak across
  promotion; replay now reports both `identity_match` and `confirmation_preserved`, with
  regression tests proving an already-confirmed bearing-only target is immediately
  transition-eligible. No far pre-pass pose was fused; gate-1 fixes occurred later on an
  in-band track at 12.5–12.8 m. The one run was not repeated: legacy SCAN collided and its
  IMU clock repeated 860 timestamps (322 positive 33 ms intervals), so it is not an
  authority-trial acceptance run. Valid evidence: 60.05 Hz scheduler, 598/598 exact
  guidance and authority replay, authority requested on 0 rows, clean gate-0 PnP/ESKF roll
  agreement (1.49° median / 5.96° p95), finite/PSD covariance, and same-id handoff.
  Combined post-fix validation: **734 tests**, Ruff, formatting, and Mypy clean.
  Experimental transition authority remains off; a new clean authority-off confirmation
  requires separate authorization and must still precede any authority trial.
- Camera extrinsic is resolved at **20° up**.
- Structured perception/PnP, tracker association guards, dual-clock schema-v3
  recording, arrival-order replay, and the background shadow worker are implemented.
- **Phase 4A — synchronized control observability implemented (2026-07-13, log-only).**
  Recorder schema bumped to **v4**: a control-loop stream (`ctrl_*`) captures, per command
  and stamped with the shared `time.monotonic_ns()` receipt clock, the mission phase +
  active gate, target mode/bearing/slant/freshness, and the command boundary at three
  SEPARATE levels — requested / post-clamp desired ACTUAL rates / calibrated raw output —
  so the sim rate inversion+scale can never be applied twice in replay, plus per-axis
  saturation, active attitude, and an ATOMIC shadow-ESKF snapshot (`ShadowState`: anchor,
  p, v, euler, cov-diag, fix freshness). `command()` still sends exactly the clamped rates
  (behaviour unchanged; test-locked). New `recorder.event_timeline` merges IMU/fix/frame/
  control by the monotonic clock; alignment + schema-v4 + command-telemetry tests added.
  **Phase 4A live capture DONE (2026-07-13):** one 10 s instrumented Training run
  (`run_20260713_102447.npz`, schema v4) — all validity checks pass (ctrl stream
  monotonic, 484 rows spanning the attempt, IMU/fix/frame/control interleave, single
  calibration, ESKF finite+PSD, worker terminated before save, loop rate **48.5 Hz** vs
  the ~53 Hz baseline: no material regression). **Root cause of the post-gate-0 failure
  (measured, not saturation/attitude/sign):** the legacy G0→BRAKE(1.6 s blind flare)→SCAN
  (5.2 s, 107° yaw, no target visible) sequence deliberately goes open-loop-blind after
  gate 0, discarding the gate-1 metric sighting it already has at gate-0 exit (slant 31 m,
  bearing +35° right), over-rotating past it, and re-acquiring gate 1 only at ~5 m
  point-blank in the final 0.85 s — 0 rate/thrust saturation and attitude tracked
  throughout. It is a trajectory/guidance-strategy failure, not a control-actuation one.
- **Gate-1 metric-perception gap closed (2026-07-13, structured pipeline, log-only).**
  Root cause was NOT a missing 4th side: the corner *orderer* (`_order_quad`, sum/diff
  heuristic) collapsed a diamond-oriented (~45°-rotated) quad — the gate's view through
  the banked turn — into a duplicate corner → zero-length side → "degenerate" → bearing
  only. Fix: `_order_quad_robust` (angular sort, byte-identical for axis-aligned so gate
  0 is unchanged), used only in `_quad_from_contour`; legacy `detect()` untouched. Second
  fix: fit the inner-opening corners when the inner hole is fully in-frame even if the
  OUTER frame is border-clipped (11/13 clipped gate-1 views keep a full inner opening) —
  real corners, no extrapolation. Offline (captured raw frames): gate-1 metric obs
  **12→35**; **93%** of frames with 3–4 usable sides now metric; all right-side. Live
  clean run: gate-1 fused **9× (all position_only, in-band 13–14 m), NIS median 0.84 /
  max 2.4, 100% accepted, 0 wrong-gate fused, range continuous (≤0.24 m/step)**; control
  byte-identical (gate 0→1 @2.2 s, tilt 19.9°, coll=0); perception 3.2 ms mean / 5.0 ms
  p95. NIS gate (16) and the 3–16 m band were NOT changed. Fusibility is trajectory-
  limited: the drone brakes/scans and lingers at ~13–16 m from gate 1, so metric obs
  beyond 16 m are (correctly) refused by the unchanged band — a control/approach matter,
  not perception. `--save-frames` diagnostic added to the mission runner. Uncommitted.
- **Shadow-ESKF root cause found & fixed (2026-07-13, log-only).** The divergence
  (Attempt A: NIS median 49.6, 1/25 fixes accepted, velocity → ±10 m/s) was an inverted
  gyro convention vs our FRD frame; `RATE_GAIN=(-2.55,-2.55,-2.35)` independently shows
  the command-response convention is inverted on all axes. The ESKF *integrates*
  gyro (control uses accel leveling, so control was immune), so launch thrust projected
  "up" instead of "forward" and every gate update rightly failed χ²(3). Fix:
  `recorder.GYRO_SIGN=(-1,-1,-1)` applied ONLY at ESKF intake (live worker + offline
  replay; raw gyro still recorded; control's raw `sh.imu` untouched). Attempt B (live):
  **NIS median 0.8, 26/27 accepted, filter finite+PSD, gate unchanged at 16, no
  covariance relaxation**; control byte-identical (gate 0→1 @2.1 s, tilt 19.9°, coll=0);
  live==replay confirmed. Ruled out: camera latency (3 ms) and PnP covariance
  (lateral ≈5 cm, depth ≈0.39 m — both physically correct). A schema-v7 eight-sign
  landmark replay now makes the evidence boundary explicit: pitch `−1` gives NIS median
  1.11 and 97.2% acceptance versus pitch `+1` at 59.86 and 2.8%; that maneuver does not
  independently excite roll/yaw enough to distinguish their signs. Changes are
  **uncommitted**.
- **Phase 4B — gate-relative velocity-aware shadow policy implemented (2026-07-13,
  strict shadow / log-only).** New `gate_relative.py`: a pure, map-free policy (does NOT
  require `CourseMap.require_flight_ready`) that consumes only validated live info (active
  race gate, matching ESKF anchor, gate-relative position `−p`, ESKF velocity, attitude,
  covariance, fix freshness, confirmed structured target mode/bearing/slant), tracks a
  **nonzero, bounded closure objective** (velocity-tracking toward the gate, NOT the
  legacy zero-velocity-at-gate spring in `guidance.geometric_control`, which was audited
  and left unwired), turns toward the target, and converts through the single canonical
  `follower.acceleration_to_rate_thrust` boundary. The ESKF sign is established from
  `reanchor` (`p = drone − gate`, so to-gate `= −p`) and regression-locked. **Fails closed
  with a recorded reason** on uninitialised/anchor-mismatch/no-target/bearing-only/stale/
  uncertain-covariance/wrong-turn/out-of-band/excessive-tilt/near-gate-handoff/non-finite.
  Never feeds `Flight.command` — command output is **byte-for-byte unchanged** (test-locked).
- **Phase 4B live integration — worker-owned structured target + guidance + TRANSITION_ACQUIRE
  (2026-07-13, strict shadow / log-only).** Guidance moved ENTIRELY into `ShadowWorker`:
  it builds an immutable `StructuredTarget` from its SELECTED `GateTracker` observation
  (real track id, race-gate association, continuity/hits, mode, LOS = metric-pose
  direction, disposition, live `expected_turn_sign` = +1 for gate 1, and separated
  `wrong_turn` / `association_refused` / `out_of_band` flags — NOT the legacy detector),
  evaluates the policy, and republishes estimator state + target + `GuidanceDecision`
  **together each tick under one lock** (`coherent_snapshot`) so the command loop can never
  pair a fresh state with a stale target. The command loop performs NO perception/ESKF/
  guidance — it only reads and records. New **`TRANSITION_ACQUIRE`** mode bridges the window
  after the race gate advances but before the ESKF reanchors: it uses target DIRECTION only
  (never far-range PnP scale — `distance` stays `None`, the 31 m obs is never fused),
  tracks a bounded approach velocity along the confirmed LOS, and **PRESERVES useful
  existing closure (≈10 m/s) instead of braking to the anchored cruise cap**, clamped to a
  reviewed shadow safety-speed ceiling (12 m/s). It fails closed on target loss / wrong-side
  / association-refused / stale / uncertain-covariance / excessive-tilt / FOV / transition
  timeout. Recorder schema → **v6** (`ctrl_sg_*` guidance decision + `ctrl_st_*` structured
  target, both separate from legacy `ctrl_tgt_*`). Clean live run `run_20260713_142154.npz`:
  **IMU clean (dup=0 gap=0 — the prior 877-dup/300-gap corruption did NOT repeat)**, ESKF
  finite/PSD, loop **47.8 Hz** (worker-owned guidance did not slow the command loop),
  decision latency median 1.6 ms; real structured track ids reached guidance (`sg_track ==
  st_track`), `TRANSITION_ACQUIRE` fired at the gate advance (BRAKE, active=1/anchor=0,
  track=2) with preserved closure and no far-range fusion, and it fail-closed for the rest
  of the gate-1 phase as the legacy trajectory lost the target. Active control byte-identical
  (the run's jet collision is a legacy-trajectory variation, not shadow-attributable).
- **Phase 4C pre-authority hardening complete; one bounded Training experiment stopped
  at the evidence gate (2026-07-13).** Recorder schema **v7** now records exact immutable
  state/target/decision sequence identities and every input used by the worker policy;
  both guidance replay and authority-mux replay reproduce the corresponding live run
  **598/598 rows exactly**. The command loop now uses an absolute-deadline, high-resolution
  scheduler. Its clean authority-off baseline (`run_20260713_145539.npz`) achieved
  **60.06 Hz**, release-dt p50/p95 **16.68/17.09 ms**, no sub-half-period catch-up bursts,
  clean IMU/queues, and no collision. A Training-only, default-off transition authority
  mux was added behind both `permit_transition_authority=true` and the explicit
  `--experimental-transition-authority` flag; Qualification rejects the flag. Gate 0,
  anchored guidance, calibration, clamps, failsafes, and normal commands remain legacy.
  The single permitted authority run (`run_20260713_145648.npz`) was mechanically clean
  (60.08 Hz, no collision, finite/PSD, clean IMU, exact replay), but authority engaged for
  only **4 rows / 49 ms**: gate 1 was absent or unconfirmed immediately after gate 0,
  briefly became eligible around 1.3 s later, then disappeared. There was no gate-1
  reanchor and no completion. The acceptance criterion therefore **failed**, no second
  authority run was made, and normal authority remains off.
- **Handoff visual identity passed live; confirmation preservation is fixed offline and
  awaits a replacement authority-off run.** The latest capture promoted the same track
  but exposed the 10→1 published-hit reset described above. Combined ESKF body tilt first
  exceeded the policy's 45° limit at 2.108 s, one frame after promotion, during legacy
  BRAKE. Static-gate sign replay strongly confirms the pitch convention; the legacy active
  attitude is only a comparison signal, not ground truth. Do not extend the 2 s transition
  window, loosen covariance/association/NIS gates, raise closure limits, or run authority
  until schema-v8 confirmation preservation and exact replay pass in a clean replacement
  authority-off capture.
- **Gate 1 HAS a fusible metric pose** (corrects the earlier "no fusible metric pose"
  note): the structured pipeline fuses gate 1 (9× in the clean run, NIS median 0.84, 100%
  accepted, in-band 13–14 m). The remaining gap is **estimator-feeding / trajectory, not
  perception**. Raising the conservative closure cap (anchored 4 m/s; transition preserves
  up to the 12 m/s shadow ceiling) toward racing speed is a separate reviewed pre-authority
  step — NOT the next milestone. The next milestone is the trajectory fix that lets the
  transition controller bridge to the first trustworthy gate-1 anchor.
- The historic gate-0→1 map (three legacy co-visible pairs) is now checked in as
  `vq2_historical_v0.toml` and intentionally fails the flight-readiness gate.
- Offline planning now has an open arc-length reference, monotonic local progress,
  gate aperture/direction checks, acceleration/braking limits, a VQ2 NED
  rate/thrust dynamics contract, and shadow trace evaluation.
- Legacy MPCC command authority is disabled by default. It is not connected to VQ2.

Immediate order: preserve and use the failed capture as evidence; do not retry
automatically → after explicit authorization, run one replacement schema-v8 authority-off
capture to verify both same-id promotion and confirmation preservation with clean IMU/no
collision → only then review whether a second bounded Training authority trial is
justified. Independent map evidence follows; do not tune MPCC or train RL before
repeatable gate-0/gate-1 completion.

## Where we are

**Phase:** VQ2 (live since 2026-06-30, sim v1.0.3379, spec VADR-TS-003). Deadline
mid–late July 2026 (~20 days remaining as of 2026-07-09). Objective: bank a valid
full-course time in Qualification mode → qualify for the September physical round.

**Active runner:** `aigp-vq2` (`drone_racing.deployments.vq2.mission` — the legacy
phased executor with depth/turn association, not the new trajectory planner).
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
- **Historical map evidence** (three co-visible legacy-pipeline pairs): robust median
  gate-1 delta is 9.64 m forward / 6.52 m right / 0.65 m up from gate 0 (the original
  ADR reported the approximate mean as 9.2/6.4/0.6). The 8.6 m pad
  distance, downstream gate identities, and gate-1 traversal normal are unresolved.
  This artifact is useful offline but explicitly rejected for flight.
- Perception: glow-resistant ring-test gate detection; ridge-walk cyan-line aim;
  offline validation harness over 1000+ real frames.
- IMU frames at 30 Hz stable; JPEG decode fine; comms proven against
  PyAIPilotExample-v2 (which itself does nothing — wire-protocol demo only).

## What is broken (current blocker)

**Gate 1 has never been threaded.** All 8 batches on 2026-07-09 failed identically in
the THREAD phase with an **orbit signature**: after a clean acquisition, gate bearing
drifts +34→+42° while yawing right at MAX_YAW with bank clamped → "target lost" →
re-scan → collide with floor clutter. Different thrust profiles did not change it.
The low-level calibrated command boundary is not implicated. Perception association,
estimator consistency, map confidence, and turn planning remain unvalidated in that
order; do not skip directly to gain tuning.

## ✅ Resolved — camera tilt is 20° UP (2026-07-10)

- **Spec (VADR-TS-003 §3.8):** "camera is tilted upwards by 20°." **This is correct.**
- The code implemented **20° DOWN** (`_TILT = −20°` in `deployments/vq2/perception.py`)
  — **wrong**, being flipped to `+20°`.
- The 07-03 "DOWN" reading (floor filling the frame at "level" hover) was **confounded
  by the rest pitch**: the VQ2 hover pose is already nose-down ~18° (attitude filter
  reads −17.8° from the start accel `(-3,0,-9.34)`), so a 20°-up camera on an 18°-down
  body points near-horizontal and the floor fills the lower frame anyway. Physics
  agrees: forward flight needs nose-down pitch, so a 20°-up mount looks straight ahead.
- **Consequence:** prior PnP-derived **bearing, `range_m`, `up_body`/`vision_roll`**
  were geometrically inverted (camera-frame `pose_cam` was fine). Any `mission.py`
  tuning leaning on those body-frame quantities is suspect — re-baseline gate 0 after
  the flip. A projection/sign unit test now guards the correct sign.
- Corrected and guarded by sign/projection tests. VQ1 remains available as an
  independent ground-truth cross-check if absolute scale is still disputed.

## Next technical milestones (ordered)

1. With separate authorization, capture and replay one replacement schema-v8 authority-off
   gate-0→1 attempt; require same-id provisional promotion **with confirmation preserved**,
   no far-pose fusion, clean ESKF/queues/timing, and no collision.
2. If the replacement handoff passes, review one separately authorized, bounded
   Training-only transition trial; if it fails, diagnose the saved run without rerunning.
3. Require five consecutive gate-0→1 completions before extending authority.
4. Collect corrected-pipeline co-visible evidence and promote a validated gate-0→1 map.
5. Shadow-evaluate the open course reference; then consider conservative planning/control.
   Full-course completion comes before MPCC speed optimization or RL.

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
