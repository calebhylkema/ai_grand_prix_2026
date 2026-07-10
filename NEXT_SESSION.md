# NEXT SESSION — read this first (standing handoff doc, keep updated)

## ⚠️ CRITICAL DISCOVERY (2026-07-09 end-of-day)

**READ IMMEDIATELY:** `DIAGNOSTIC_2026_07_09.md` — We found a fundamental question:

> **The spec (VADR-TS-003 §3.8) says: "The camera is tilted upwards by 20° upwards."**
> **But our code (gates.py line 51) implements: `_TILT = math.radians(-20.0)` ← DOWNWARD**

**Impact:** ALL PnP poses are geometrically inverted. If wrong, we're detecting **the wrong objects** in gate 1 threading. This may explain why 10 days of control tuning didn't help—it's a **perception fundamentals** problem.

**Next session MUST:**
1. Validate camera tilt (flip sign, test gate 0, see if it breaks or fixes)
2. Visualize detections vs. expected geometry
3. Use VQ1 sim as ground-truth rig
4. Simplify controller back to fly.py gate-0 logic, clone for gate-1

**Status:** 8 batches flown 2026-07-09, all failed in THREAD phase with orbit signature. Gate 0 works 100%. Problem is not control—it's **perception or path planning.**

---

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

## 🎯 CURRENT STATE (2026-07-09) — STRATEGIC PIVOT: map-and-plan (READ THIS FIRST)
**User mandate (10 stalled days): "something needs to change… feel free to scrap stuff."
Reactive fly.py tuning is DONE. New architecture: mine a COURSE MAP offline from saved
traces, fly a phased plan, use the map to gate data association.** ADR:
`docs/adr/0003-course-map-and-plan.md`. Memory: `project_vq2_course_map_pivot.md` ⭐⭐.

- **`scripts/vq2/course_map.py`** — offline miner: co-visible PnP gate pairs from
  fly_trace archives → relative course geometry (no odometry needed). **Map v0:
  pad→gate 0 = 8.6 m dead ahead; gate 1 = 9.2 m fwd / 6.4 m RIGHT of gate 0 (35°,
  11.2 m), +0.6 m up; post-crossing z 21–23 m sightings = gate 2 (imposters);
  return-leg gate pair at z 12–14 ~19° LEFT of the exit heading (trap).** The old
  "pad 2–3 m before gate 0 / gate 1 at 19–21 m" AV note was a misread — corrected.
- **`scripts/vq2/fly2.py`** — phased mission executor on fly.py's proven primitives
  (Flight, calibrated rates, vertical_thrust, trace tooling; fly.py untouched as
  fallback): STAB → G0 thread → on gate-pass event BRAKE (deep flare + climb out of
  the floor clutter) → SCAN right → map-gated accept (`ACCEPT_Z` 3–16 m AND
  `ACCEPT_TURN` +25…110° of scan-start heading) → THREAD (CY servo + bank-on-bearing
  + recenter brake + pass-commit) → repeat per gate.
- **Batch findings (N=4 each, all reproducible):** (1) range-only accept latched the
  LEFT return-leg pair → ENV/threat2; ACCEPT_TURN direction gate fixed it. (2) Open-
  loop +0.04 brake climb coasted to the CEILING; the z 15.1–15.6 / cy≈61 acquisition
  is an ELEVATED far-gate IMPOSTOR (not gate 1) — ACCEPT_Z max now 13. (3+5+6) NO
  blind vertical works: +0.02 and plain hover both ended at the FLOOR (the gate-0
  crossing carries a real ~−2 m/s sink), and vz-null on the vision-anchored
  integrator flew INTO the floor (launch pins vz at −3 fiction; IMU vz is never
  control-grade near the launch). **The fix that stuck (batch 7): arrest-then-look
  brake — 1.0 s blind flare at hover+0.04 arrests the sink, then nose to +0.10 so the
  CYAN RACING LINE returns to view and `fly.line_thrust` owns height closed-loop
  (brake tail + all of SCAN). 4/4 survived the corner; acquisitions at z 9.8–12.6.**
  (7) New failure mode = THREAD ORBIT: bearing +34→+42° at MAX_YAW, bank clamped
  +14°, gate out the right frame edge, "target lost" → re-scan → collide. Source =
  self-commanded BACKWARD accel (SCAN_HOLD_PITCH +0.05 for 3.5 s ≈ −1.7 m/s, plus
  far-branch RECENTER_PITCH). Batch 8 = level scan + level far-converge (RECENTER
  only in the close branch). Check newest prev_* traces + task output first.
- Gate-2 recon for free: close-approach frames show gate 2 just RIGHT-BEHIND gate 1.

## 🎯 EARLIER STATE (2026-07-05, late) — reactive fly.py era (fallback)
**Shadow-mode ESKF is WIRED and VALIDATED in fly.py (`Flight.kf`). Its cruise-lie flaw
was found and fixed. Two kf-speed blind-arc variants (AK/AL) were flown and did NOT beat
AH's fixed arc — the reactive architecture is saturated; the next lever is the metric
REFERENCE ARC (world-frame path integrated at last good fix, tracked by accel_guidance).**

- **Shadow wiring (kept, active):** `Flight.kf = ESKF()` fed in `step_sensors`
  (predict every IMU step); gate fixes fused at the validated-PnP site (`update_gate`,
  chi2-gated) with `reanchor` on active-gate change (`fl.kf_needs_anchor`). LOG-only
  for steering except where noted. Trace filenames now carry `K{h}z{vz}I{h}` = kf
  horizontal speed + kf vz vs the simple integrator's horizontal speed; the run summary
  prints `kf_att=(r,p)/att=(r,p)` as an attitude sanity check.
- **⭐ ESKF cruise-lie found & fixed (shadow batch 1):** `update_gravity` in flight
  passes its own |‖a‖−G|>1.5 gate during cruise (|a|≈1.05g) and erased ~12° of real
  nose-down pitch — kf speed read 0.3–0.8 m/s mid-corner vs the vision-anchored
  integrator's 4–6. Same lesson as AttitudeEstimator's `test_in_flight_accel_is_ignored`;
  its synthetic test never had a cruise regime. **Fix: update_gravity is ground-only**
  (`if not self.att.in_flight`). Batch 2 after the fix: kf tracks the integrator within
  ~20–25% and kf vz matches integrator vz sample-for-sample; kf_att ≈ att through
  clean flight. Corner-entry speed measured: **~1.5–2.6 m/s at latch time** (both
  estimators agree) — much slower than the ~3 m/s the arc constants assumed.
- **Config AK (radius-pinned blind_arc_cmd on kf speed): 0/3, WORSE (deaths t3.3–4.4),
  REVERTED.** At slow entry the coordinated arc softens bank (4°) AND yaw (0.5 rad/s) —
  but the blind second's real job is swinging the CAMERA around the corner to re-acquire
  gate 1. AH's fixed 20°+MAX_YAW survives by doing exactly that. Slower entry must never
  mean a slower head-turn.
- **Config AL (MAX_YAW always + bank coordinated to it, tanφ=v·ψ̇/g on kf speed): 0/6,
  a WASH vs AH (deaths t3.1–5.3; best = tracked gate-1 continuously to t5.3, died at
  11m with it in view), REVERTED.** AH keeps the fixed arc + its 2 gate clips.
- **Config AM (KEPT, live): position-closed blind reference arc.** At blind entry,
  capture `kf.p`/`kf.v`; integrate a fixed-R circle (TURN_REF_R=2.8, turn_dir) at the
  measured entry speed, decelerating at ARC_DECEL=2.0 (the brake, encoded in the
  reference), climbing on the AE1 line-aim cue; track POSITION with `accel_guidance`
  in the kf's gate-anchored world; **yr stays MAX_YAW** — path geometry and camera
  swing are decoupled (the capability no attitude shaping had). The latch brake
  (TURN_LATCH_BRAKE pitch) moved INTO the ribbon-tracking branch; the blind branch's
  pitch comes from the tracker. State = `arc_ref` (reset on fresh sighting + gate
  change). **Results over 6: 2 GATE-kind contacts (t5.0; t4.1 after tracking b−2°
  dead-ahead) + closest ranged sighting ever (6.0m, t5.0). ≥ AH's rate (2/9), kept.**
- **Config AN (KEPT, live): hot-entry brake on the kf's measured speed KILLED the
  fast-entry mode.** The dominant AM-era failure (3/6: corner entry at K≈3.5–4 m/s ⇒
  threat2 death ~t3.1) is gone: while latched and `‖kf.v[:2]‖ > V_ENTRY_MAX=2.6`,
  `pitch_target = max(pitch_target, HOT_BRAKE_PITCH=0.35)` whatever the steering
  branch wanted. AN over 6: corner deaths t4.5/4.7/5.1/5.5/**6.2 (deepest ever, died
  LEVEL at tilt 17.8°)** + one unrelated gate-0 fluke — zero t3.1 deaths. Note: the
  OLD unconditional post-gate brake (BRAKE_SECS, "no help") was tried before the
  latch/climb/AM stack existed and without speed measurement — the conditional,
  in-latch version is what works.
- **⭐ Remaining failure = corner-EXIT placement:** all AN deaths are threat1/threat2
  at t4.5–6.2, mostly gentle (level, low tilt). We now survive the corner and die in
  the terminal corridor — see NEXT items 2–3 (tune AM's R/decel from traces;
  terminal-corridor height shaping).
- 40 unit tests green. Everything below (AG/AH state) still accurate.

## 🎯 EARLIER (2026-07-05) — AG/AH, kept for context
**Configs AG/AH (6+3 attempts): the terminal approach now WORKS — four head-on gate-1
approaches (b−10°@5.5m, +7°@7.7m, −8°, +8°; sightings held to t4.3–5.5). The binding
constraint moved to BLIND-ARC PLACEMENT: the open-loop corner arc lands anywhere within
a few-metre spread, in a corridor a few metres wide (Station-19 bay + jet + gate).**

- **AG (blind climb with real authority — TURN_BLIND_KVZ_CLIMB=0.15, TURN_BLIND_CLIMB_CAP
  =0.12):** AF's climb had cap-saturated to ~zero net climb at 20° bank; with authority
  the drone now CLIMBS through the corner (vz −2.0..−2.5 sustained) and arrives AT
  gate-1 height. Immediate effect: first-ever centred close approaches. New failure:
  vertical BANG-BANG through the gate row at close range (AG3: gate cy 73→255 in 0.4s,
  vz −3.0 = overshot above the gate, then descended onto the jet parked under it,
  15 grinding collisions).
- **AH (pass-through commit freezes THRUST too — PASS_KVZ=0.08, PASS_VZ_CAP=0.05):**
  the commit froze pitch/roll/yaw but the row servo kept the corner climb through the
  coast (that's what ran vz to −3.0). Now the coast is hover±bounded-vz-damp (vision-
  anchored vz only). AH1 tracked head-on to t5.5 (deepest ever), AH5 head-on b−8 at
  t4.3 — but both still hit the threats flanking the corridor.
- **⭐ THE REMAINING PROBLEM (AH5 frame 022 is the picture):** 0.1s before impact we're
  banked INSIDE Station 19's bay, beside the placard. The blind arc (fixed 20° bank +
  max yaw at whatever speed the run happened to have) has run-to-run radius variance of
  metres — R=v²/(g·tanφ) with v varying ±30% swings R ±70%. Gate 1 sits directly above
  the jet; the corridor tolerance is ~2–3m. Open-loop cannot repeat that. **This is the
  ESKF+guidance wiring task, now precisely scoped** (see NEXT).
- **⚠️ Trace-archiver midnight bug FIXED:** `prev_%H%M%S` names sort lexically, so
  post-midnight dirs sorted FIRST and the keep-last-6 cleanup deleted the NEW traces
  (lost AG1–AH2, incl. AH1's t5.5 run). Now `prev_%Y%m%d_%H%M%S` + ctime-sorted cleanup.
- Failure taxonomy update: corner-loss runs (~50–60%) still die on threat2 t4.2–5.2;
  successful-corner runs now die at the CORRIDOR (threat1/threat2 flanking the final
  approach) instead of short of it. AH6 clipped the GATE blind (3rd gate contact).

## 🎯 PREVIOUS STATE (2026-07-04) — kept for context
**The gate-0→1 corner campaign (configs W→AE, 24 attempts). Corner HEADING is solved;
corner ALTITUDE is the one remaining lever. Gate 1 was physically CLIPPED twice (AA1, AE1).**

- **The corner, mapped:** post-gate-0 the racing line is a steep **climbing right turn
  (~100–130°) up to elevated gate 1 (~2m+ above the gate-0 exit)**. The cyan ribbon tube
  sits at the right shoulder on exit; a **distant post-gate-1 course line runs across the
  frame top** (a wrong-branch trap for line detection); the parked **fighter jet (threat2)
  sits inside the corner directly UNDER gate 1's approach** — gate-1's red glow appears
  ABOVE it. Camera 20° down + flying ~2m below the path ⇒ ALL cues exit the frame top
  ~1s after the gate-0 pass: the corner necessarily has a blind second.
- **Latched coordinated turn shipped (fly.py):** gate >TURN_ENTER_BEAR(0.35) off-axis and
  not close ⇒ committed turn (TURN_LATCH_S per sighting, TURN_BLIND_S=1.0 stale extension).
  Tracks the ribbon when line_fresh, fixed arc otherwise. Key tuned values, each traced to
  a crash (citations in code comments):
  - **TURN_BANK=0.35** (~20°, latch-only): at MAX_ROLL_TGT(14°) lateral accel g·tan14°≈2.6
    m/s² turns the velocity vector far slower than the nose → understeer into station bays (W3).
  - **TURN_LATCH_BRAKE=0.24** nose-up inside the latch: R=v²/(g·tanφ) — slow through the
    corner. **AD3 broke the understeer for the first time (gate bearing reached −17°).**
  - **Blind latch = vz-null hold** (TURN_BLIND_KVZ=0.05), NOT the frozen-stale-pixel row
    servo (that rode max climb into trusses, W3). Gate-row thrust needs a sighting fresher
    than TURN_ROW_FRESH_S=0.3.
  - **Latch expiry while blind ⇒ blind_brake** (yr=0, kill momentum), not SEARCH_YAW
    scan-away (X3 sank into the jet while scanning). Brake nulls velocity ONLY while the
    estimate has a vision anchor fresher than **VEL_TRUST_S=1.5** (vel_fix_t); stale ⇒
    level attitude and let drag work (AB3: "brake" pitched −15° chasing a fiction and sank
    1.2 m/s into the jet).
  - **RELATCH_CALM_S=1.2** after blind brakes: a coordinated turn is a SPEED maneuver —
    re-latching from braked hover banked us straight into the jet beside us (Z3).
    Re-acquisition from hover must be aim-then-go.
  - **DR_DIVERGE_M=2.0**: dead-reckon pursuit aborts if the remembered point gets 2m
    farther than its closest approach (AD3 chased a phantom fix that receded 26→39m).
- **⭐ Perception root cause found & fixed (gates.py detect_line), validated offline with
  `_line_probe.py`:** (1) one image row can hold TWO course branches — band-mean at the
  median row mixed them into aim "+8° dead ahead" while the line broke hard right (Y3
  frame 013, the crucial first second of the corner). Fix = **ridge-walk aim**
  (LINE_WALK_STEP/LINE_WALK_LAT): walk up from the near point keeping only laterally
  continuous pixels. (2) At close range the ribbon tube is FAT and failed the elongation
  test — fix = **near-blob exception** (LINE_NEAR_BLOB_PX=800, must be anchored to frame
  bottom/lower side edges; holograms float mid-frame).
- **Failure taxonomy (stable across configs):** ~75% corner-exit deaths on threat2 at
  t4.7–5.3 — now GENTLE (tilt~21°, banked flight into the jet, no more tumbles); ~25%
  early-loss mode (gate-1 sighting lost t2.1–2.2 → die on threat1 t3.3–3.9) — UNDIAGNOSED.
- **⭐ AE1 endgame analysis (the second gate clip):** heading through the corner is now
  right — after the blind second, gate 1 was re-acquired at t4.23 (b+36, 12.1m, genuine).
  But the trace shows the frame FILLED with the jet's wing at that moment, floor markings
  large in the mid-blind frames, vz swinging to +1.3 (sinking): **the blind vz-null holds
  altitude while the racing line climbs ~2m — we exit the corner at jet height and hit
  the jet fuselage with gate-1's glow visible just above.** The last fresh line-aim row
  before going blind was pinned at y=5 (frame top = "climb hard") — the cue for the fix
  is already in hand.
- Trace tooling: `data/vq2/fly_trace/` now ARCHIVES each attempt to `prev_HHMMSS/`
  (keeps last 6) instead of wiping — AA1's gate-clip trace was lost to the old behavior.
- 37 unit tests green (`uv run pytest scripts/vq2/test_control.py scripts/vq2/test_estimation.py -q`).

## 🎯 PREVIOUS STATE (2026-07-03) — kept for context, partially superseded
- **The user's diagnosis was CONFIRMED and fixed: the controller treated yaw as if it
  steered.** Yaw only points the nose; only tilting (roll/pitch) redirects travel. fly.py
  now has a metric acceleration→attitude outer loop (`accel_guidance`): velocity-capped
  law (v_des = KV_POS·p_rel capped at V_MAX=2.5, a = KA_VEL·(v_des−v)) → roll/pitch
  setpoints (a = g·tanθ). Velocity comes from finite-diffed PnP poses (gate static ⇒
  d(p_rel)/dt = −v), EMA'd, with a POSE_JUMP_M guard against target switches.
- **⭐⭐ THE CAMERA POINTS ~20° DOWN, NOT UP.** Verified from level-hover trace imagery
  (horizon at y≈64, floor fills the frame; far gates detected exactly at that line). The
  spec/config note claiming "20° up" is WRONG for build 3379. `gates.CAM_TO_BODY`'s MATH
  was already correct (its comment lied — now fixed). ⇒ **PnP body poses are trustworthy**,
  including relative HEIGHT. The "PnP range always nan" note was only about far/small gates.
- **Consequence chain fixed:** CY_TARGET image-row servo conflated angle with height
  (cy 130 for a 30m gate = demand to fly ~6m above it → pegged climb → ceiling hits).
  Replaced for posed gates by **metric vertical** (hold gate Z_DES_REL=0.5m below us via
  PnP z, v_rel-damped). cy servo remains the pose-less fallback.
- **Other structural fixes (each traced to a live crash):** target continuity
  (`pick_target`, TARGET_MATCH_PX) stops mid-approach hops to far gates; STEER_MIN_AREA
  stops banking at tiny gates seen THROUGH the current gate; **blind = brake-and-climb**
  (null frozen v_rel) instead of scan-and-drift — this ended the tumble era completely
  (post-fix runs abort level, tilt_max ≈ rest pitch); **pass-through commit**
  (PASS_RANGE_M/PASS_AREA): inside ~4m, hands off and coast through the plane —
  close-range PnP lateral is unreliable (centred gate 0 read bearing +26°) and steering
  at a close gate brushes its frame.
- **Where it stands now: gate 0 threads; flight is stable and level; the drone brakes to
  ~hover when blind. The remaining blocker is NOT control — the straight line from
  gate-0 exit to gate 1 crosses a PILLAR/SCAFFOLD field** (trace frame 021: pillar dead
  centre, gate at +40° behind it). We stop safely mid-field, then creep into scaffold
  (coll 5-6 while level/parked). This is the obstacle-avoidance frontier the playbook
  predicted.
- **→ NEXT: follow the CYAN RACING LINE.** The sim paints a cyan line on the floor
  (visible in many fly_trace frames, e.g. 024 of the 07-02 trace, 010 of 07-03) — a
  camera-visible, obstacle-free path between gates. Detect it (HSV cyan mask, fit its
  direction in the lower half of the frame) and steer along it between gate passes;
  gates become confirmation/pass targets instead of the sole steering cue. This likely
  beats any pillar-avoidance heuristic.
- 16 unit tests green (`uv run pytest scripts/vq2/test_control.py -q`), incl. the
  guidance law's brake-when-hot property. `scripts/vq2/_pnp_probe.py` = offline PnP
  probe over fly_trace frames (how the camera-down discovery was made).

## 🎯 PREVIOUS STATE (2026-07-02 pm) — kept for context, partially superseded
- **Gate 0 threads 100% at t≈2.0s.** Flight is stable (tilt = platform's 17.8° rest pitch
  until a terminal collision). The multi-day tumble era is OVER.
- **Active controller is `scripts/vq2/fly.py`.** `race.py` is reference/regression only.
- **The "gate-0-exit structure" hypothesis was WRONG — it was a FLOOR SINK.** The old
  collision at t≈3.3s was the drone *sinking into the floor* right after gate 0, NOT a
  lateral structure. Proven by a new trajectory recorder (below): the gate climbed out the
  top of the frame while the floor filled the view. The vertical servo that should have
  climbed never fired (it was range-gated on PnP range, which is ALWAYS nan on 3379).
- **FIXED this session** → the drone now reaches **gate 1's vicinity at t≈4.5–6.1s** (was
  3.3s), stable throughout. Big, consistent progress. See "What changed" below.
- **BUT we still MISS gate 1** — `active_gate` = the NEXT-target gate index (RACE_STATUS
  `gate`: 0 while parked, →1 on takeoff, increments as gates are passed). It **never leaves
  1**, so gate 1 is never actually threaded. The trace shows why: on final approach the gate
  sweeps from bearing +5°→−35° while its area explodes — i.e. **it slides to our LEFT and we
  pass on its RIGHT edge.** We overshoot laterally; centring doesn't hold through the last
  ~1 m. Then we hit `ENV threat2` (a floor/structure) during the ensuing hard LEFT turn
  toward the next gate, and the impact knocks us tumbling (tilt→60–97° is the RESULT).

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
6. **Tilt-compensated thrust** (`Flight.tilt_comp`, `TILT_COMP_MIN_COS`): scale thrust by
   1/(cosR·cosP) using the reliable attitude estimate, so banking a turn doesn't bleed
   altitude — the sink recurred on the hard left turn past gate 1. Marginal help so far.
7. **Lateral-drift proxy** `drift = dβ/dt + gyro_z` computed every loop + a drift-damped bank
   term (off by default, see NEXT step 1). Trace filenames now log `gz` (gyro yaw) too.
8. **Unit tests** `scripts/vq2/test_control.py` (9, all green) lock in `vertical_thrust` +
   `Flight.tilt_comp`. Run: `uv run pytest scripts/vq2/test_control.py -q`.

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
0. **Read the fly_trace first** (now archived per attempt in `data/vq2/fly_trace/prev_HHMMSS/`,
   last 6 kept; current attempt in the root). Filenames carry t/roll/pitch/vz/gyro-z/gate/
   held-target/line-aim. This is how every root cause this week was found — do it before
   changing anything.
1. ~~Kill the FAST-ENTRY mode~~ **DONE (config AN, kept — see CURRENT STATE).**
2. **Tune the AM reference arc (KEPT, live — see CURRENT STATE):** TURN_REF_R=2.8 and
   ARC_DECEL=2.0 are first guesses. Read the archived AM traces (K/I fields + frames)
   to see where the tracked arc actually lands vs the corridor; adjust R/decel/climb.
   Do NOT go back to attitude-level shaping (AI/AK/AL all failed — see fly.py comment
   in the blind branch). NOTE: kf attitude is gyro-only in flight — fine over ~6s;
   consider `correct_roll`-style vision fixes into the kf for longer runs.
3. **Terminal-corridor shaping (after 1–2):** gate 1 sits directly above the jet; the
   final 5–8m must be flown AT gate height, not climbed late (AH5: head-on but at jet
   height = hit the jet 5m short). The commit freeze (AH) handles the last 4m; the
   approach height between blind-exit and commit is the gap.
4. **Diagnose the corner-loss mode** (gate-1 sighting lost t2.1–2.5 → threat death;
   partially explained by fast entry, see 1). Frame-by-frame an archived corner-loss
   trace: late latch entry, or run-to-run gate-0 exit geometry?
5. Then the speed roadmap in `docs/ROADMAP.txt` (racing line/MPCC → 100 Hz → cruise → RL last).

## Git / status
- Branch **`vq2-reorg-calibrated-controller`** (off master). Latest commits: `9a11570`,
  `5f6dee7`, `6a3b732`. **UNCOMMITTED (2026-07-04 + 07-05, commit when the user asks):**
  the whole corner campaign — fly.py (latched turn: TURN_BANK/TURN_LATCH_BRAKE/blind
  vz-null/blind-brake/RELATCH_CALM/VEL_TRUST/DR_DIVERGE + trace archiving + AG/AH climb
  & commit-freeze + shadow ESKF wiring + K/I trace logging), eskf.py-adjacent: the
  ground-only update_gravity call site lives in fly.py (eskf.py itself unchanged),
  gates.py (detect_line ridge-walk + near-blob), test_control.py (+blind_arc tests),
  this file. `scripts/vq2/_line_probe.py` (untracked) = offline
  detect_line harness → `data/vq2/lab_line/` — KEEP, it found the corner's perception root
  cause twice.
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
