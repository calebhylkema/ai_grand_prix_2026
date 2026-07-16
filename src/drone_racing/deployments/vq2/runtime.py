"""Flight-validated VQ2 runtime: calibrated command boundary, estimators, failsafes.

Extracted VERBATIM (2026-07-09 restructure) from the flight-validated reactive
controller ``fly.py`` — behavior, gains, signs, and safety logic are unchanged.
This module is the shared infrastructure under any VQ2 mission:

- ``Flight``: one armed attempt — sensor stepping (re-arm every loop!), Mahony
  attitude + velocity estimators, shadow ESKF, tilt/collision/finish failsafes,
  the single calibrated command boundary (``command``), tilt-compensated thrust,
  and the level-rate attitude P loop.
- Target persistence/selection helpers (``pick_target``, ``target_switch_ok``,
  ``metric_fix_ok``), metric accel guidance, blind-arc math, the visual vertical
  servos (``vertical_thrust``, ``line_thrust``), and the trace-frame recorder.

The constants block carries the full batch-tuning history from the reactive
campaign — many constants are used only by the archived reactive branch stack,
but they are kept (with their hard-won comments) as the tuning record and
because the locked-in unit tests reference them.

Everything routes through ONE boundary: ``calibration.send_att`` with
CALIBRATE=True maps desired ACTUAL body rates onto the sim's inverted/scaled
rate loop (actual_gyro = RATE_GAIN * commanded). See ``calibration.py``.
"""

from __future__ import annotations

import math
import time
from typing import TYPE_CHECKING

import cv2
import numpy as np

from drone_racing.deployments.vq2 import calibration as cc
from drone_racing.deployments.vq2 import perception as gates
from drone_racing.deployments.vq2.attitude import AttitudeEstimator, VelocityEstimator
from drone_racing.utils.workspace import Workspace

if TYPE_CHECKING:
    from pathlib import Path

    from drone_racing.deployments.vq2.shadow import ShadowWorker

_WS = Workspace.locate()
COLL_DUMP_DIR = _WS.datasets / "vq2" / "fly_collisions"

# Calibration is ALWAYS on for this controller: cc.send_att maps desired->raw rates.
cc.CALIBRATE = True

CMD_HZ = 60.0
HOVER_THRUST = 0.30  # measured (control_check --test hover)

# --- IMU vertical-velocity estimate: DIAGNOSTIC ONLY (do not drive thrust with it) ---
# 2026-07-02: tried servoing thrust to hold this vz≈0 to stop the post-gate "floor"
# sink. It made things WORSE (broke gate-0 threading) because the estimate is badly
# biased through the violent sim launch — it read a CLIMB (vz≈-1 to -2) while the camera
# frames showed us skimming the floor. Integrated accel is untrustworthy over the launch
# window; the reliable height cue is the CAMERA (gate image row). vz stays logged only.
VZ_CLAMP = 3.0  # clamp the logged value so an impact spike doesn't dominate the range

# --- trajectory recorder (self-diagnosis without a human watching the sim) ---
# Dump a camera frame every TRACE_DT with attitude / vz / gate-detect baked into the
# filename, so the frame SEQUENCE into a collision can be read back to reconstruct the
# approach geometry. Cleared at the start of each race run.
TRACE_DIR = _WS.datasets / "vq2" / "fly_trace"
TRACE_DT = 0.15  # seconds between trace frames

# --- outer attitude P loop (desired body rate from attitude error) ---
KP_ATT_ROLL = 4.0  # rad/s of desired rate per rad of roll error
KP_ATT_PITCH = 4.0
MAX_RATE = 2.0  # clamp desired roll/pitch rates (rad/s) — gentle by design

# --- forward flight (stage b) ---
PITCH_FWD = -0.05  # rad, gentle nose-down target => slow forward drift (SLOW first).
#                    Tried -0.03 (slower) to fix the gate-1 miss: NO help (bearing still
#                    ends -37°). The rightward miss is independent of forward speed — yaw/bank
#                    can't redirect the launch-direction momentum. Needs velocity guidance.
KP_YAW_BEAR = 1.5  # desired yaw rate per rad of bearing to the gate (body frame)
MAX_YAW = 1.0  # rad/s
SEARCH_YAW = 0.4  # rad/s slow scan when no gate visible
BRAKE_BEAR = 0.22  # rad (~13°) — taper forward pitch to 0 as |bearing| approaches this.
#                    Was 0.5 (28°). NECESSARY BUT NOT SUFFICIENT for the gate-1 miss:
#                    tightening it (turn-then-go) did not fix it — bearing still ends at
#                    -36..-39° (gate to our LEFT) because the real problem is uncontrolled
#                    LATERAL (rightward) momentum that forward-braking can't cancel and
#                    moderate bank can't null in time (aggressive bank tumbles). Detection
#                    is accurate (frame 025 t4.10: gate clearly left, on the racing line).
#                    The true fix needs a VELOCITY signal (optical flow / ESKF+guidance) to
#                    null lateral drift — see NEXT_SESSION step 3. Kept tight; it's sound.
TURN_BRAKE_PITCH = +0.15  # rad nose-up ACTIVE brake when the gate is far off-axis
#                           (|bearing| > BRAKE_BEAR). Gate 1 is a hard right turn; the
#                           launch momentum carried us PAST it (bearing pinned +35-40°)
#                           while yaw ran at max. Kill forward speed so the turn lands.
CY_ALIGN_BAND = 60.0  # px — "aim then go": only commit forward pitch when the gate is
#                       vertically centred too (|cy-CY_TARGET|<band). A high gate climbed
#                       out the TOP of frame (cy 33 -> lost) because we approached faster
#                       than we climbed and passed UNDER it — brake+climb to centre first.
KP_ROLL_BEAR = 0.30  # bank-toward-gate RE-ENABLED 2026-07-02: yaw alone rotates where we
#                      POINT, not where we GO — with ~6 m/s launch momentum the drone slid
#                      straight past gate 1's right-turn (bearing pinned +38°, fly_trace).
#                      Banking tilts the thrust vector to pull the velocity into the turn.
#                      The earlier "worse" test was pre-calibration + with the floor-sink;
#                      flight is stable+calibrated now. Roll sign follows the calibrated
#                      boundary (desired actual rate); watch fly_trace for the sign.
MAX_ROLL_TGT = 0.25  # rad (~14 deg) bank clamp — enough to actually turn, still bounded.
#                      Tried proximity-scaled bank (up to 3x) + 20° clamp to fix the gate-1
#                      right-miss: WORSE (roll excursions -80°, tumbled EARLIER, still missed).
# --- metric acceleration guidance (2026-07-03) ---
# THE missing outer loop (user-diagnosed): on a quadrotor YAW DOES NOT CHANGE THE
# DIRECTION OF TRAVEL — only tilting the thrust vector (roll/pitch) creates horizontal
# acceleration. The old logic yawed at the gate and gated a FIXED forward pitch on nose
# alignment ("aim then go"), so a cross-track gate was never actually accelerated toward.
# Fix: PD on the PnP-metric relative position (gate is static ⇒ finite-diff of its pose
# = -our velocity) → desired horizontal acceleration → roll/pitch setpoints (a = g·tanθ).
# PnP poses ARE available on 3379 (probed offline over the whole recorded gate-1
# approach; the "range always nan" note was about far/small gates only).
KV_POS = 0.8  # (m/s per m) desired-speed gain: v_des = KV_POS * p_rel, capped at V_MAX
V_MAX = 2.5  # m/s cap on desired approach speed — SLOW first (user directive). A raw
#              accel-PD arrived hot: with kp·|p_rel| >> MAX_ACC the clamp erased the
#              velocity damping (live 2026-07-03, tumble at t≈3.4s). Capping desired
#              SPEED makes the law brake whenever actual speed exceeds V_MAX.
KA_VEL = 2.0  # m/s^2 per m/s of velocity error (a = KA_VEL * (v_des - v))
MAX_ACC = 3.5  # m/s^2 clamp on desired horizontal acceleration
MAX_TILT_CMD = 0.30  # rad (~17°) clamp on accel-derived roll/pitch setpoints
G = 9.81
VIS_ROLL_GAIN = 0.5  # blend fraction per sighting for the gate-uprightness roll fix.
#                      Run-3 crash (2026-07-03): real bank ratcheted to 41° while the
#                      filter read +5° — in flight the accel reads pure body-z thrust
#                      ("level" at any bank), so its correction ERASES real bank from
#                      the gyro integral. A world-upright gate's in-plane rotation via
#                      PnP rvec is the only absolute bank reference we have (validated
#                      on synthetic renders: ±1° out to 40°).
VIS_ROLL_MAX_BEAR = 0.25  # rad — accept the vision roll fix only near head-on: oblique
#                           gate projections (the whole gate-1 corner) are where planar
#                           PnP's ambiguity flips up_body and poisons the roll estimate.
VREL_EMA = 0.35  # smoothing for the finite-diff relative velocity
VREL_MAX_GAP = 0.6  # s — don't finite-diff across a detection dropout longer than this
VREL_MIN_GAP = 0.05  # s — nor over a gap so short that PnP jitter swamps the motion
POSE_JUMP_M = 3.0  # m — a bigger jump in the gate's world position means the TARGET
#                    SWITCHED (near gate border-clipped -> nearest() found a farther one);
#                    differencing across it poisoned v_rel with a huge phantom velocity
#                    (live 2026-07-03). Re-anchor instead.
STEER_MIN_AREA = 800.0  # px^2 — don't BANK toward a gate too small/far for PnP; yaw-track
#                         it only. Steering at a tiny corridor gate seen THROUGH gate 0
#                         banked us right while still threading gate 0 (live 2026-07-03).
PASS_RANGE_M = 4.0  # inside this PnP range, COMMIT: stop steering, coast straight
#                     through the gate plane. Close-range PnP lateral is unreliable
#                     (a centred gate 0 at ~5m read bearing +26°) and braking/banking
#                     AT the gate brushed its frame (live 2026-07-03, coll ×4 at the
#                     gate-0 pass). Threading close = hands off, fly through.
PASS_AREA = 2800.0  # px^2 fallback commit trigger when PnP is absent
BLIND_BRAKE_SECS = 1.2  # after losing the target, spend this long killing momentum
BLIND_CLIMB = 0.02  # small extra thrust while blind-braking (was 0.06: still hit the
#                     ceiling zone; the metric-z vertical below removes the need to
#                     climb hard blind). Blind must mean STOP, not drift+scan.
# --- dead-reckon pursuit (2026-07-04, research: TU Delft ADR "state prediction feed-
# forward through blind segments"; MonoRace flies the estimate between sightings) ---
# Run H: gate 1 left the frame at bearing +40° WITH a fresh PnP fix (range 31m) that we
# threw away — 3.4s of blind-brake+scan later we hit an obstacle. A lost gate's last
# world-relative position is still the target: propagate it with the IMU velocity
# (vision-anchored at every PnP sighting) and keep flying metric guidance toward it,
# yawing to its PREDICTED bearing (which is also what re-acquires it).
DR_PURSUE_SECS = 2.5  # pursuit budget after losing a PnP-fixed gate; then brake+scan
DR_ARRIVE_M = 1.8  # m horizontal — reached the remembered point still blind: brake+scan
DR_DIVERGE_M = 2.0  # m — abort the pursuit if the remembered point gets this much FARTHER
#                     than its closest approach: the propagation velocity is fiction and
#                     we are demonstrably not closing (AD3: chased a phantom "26 m" fix
#                     that RECEDED to 39 m over 2.4 s while the drone sank 3 m).
DR_MAX_AREA = 1500.0  # px^2 — only pursue gates lost while still FAR: close-range PnP
#                       range is unreliable (run I: a2340 read 15.7m, truly ~6m)
VEL_VIS_ANCHOR = 0.25  # per-sighting gain pulling IMU velocity toward the PnP-derived
#                        velocity (−v_rel): bounds integration drift, MonoRace-EKF-lite
FIX_RANGE_MAX = 25.0  # m — config AP (07-05) set 20: PnP fixes beyond are BEARING-ONLY.
#   Early-corner-loss mode root cause: post-gate-0, gate 1 shows as a tiny frame-top blob
#   (a~650px²) and PnP ranged it at 28m. Those fixes reanchored the kf 28m
#   out, fed range-jitter fiction into vel.anchor (integrator jumped 1.6→3.3 m/s in one
#   frame while the chi2-gated kf held 1.3), set vel_fix_t so the blind brake TRUSTED the
#   fiction, and thr_m servoed altitude to a phantom. Same disease as AD3 (phantom chased
#   26→39m) and run I (a2340 read 15.7m, truly ~6m). Far gates still steer via the pixel
#   branch (bearing/cy/turn-latch); metric machinery waits for a close, trustworthy fix.
#   Config AW (07-09) raised 20→25: offline probe (_pnp_probe.py on AV traces) showed the
#   ring-validated a≥800 corner sightings are HONEST — PnP z agrees with the yaw-
#   foreshortening-free bbox-height range 864/bh within ~15% (a1346: z21.1/zh18; a1624:
#   z19.2/zh15.7; a2798: z13/zh13.1). Gate 1 truly IS ~19–21 m from the corner entry
#   (the old "true 12–15m" belief was wrong — the jet at ~6–8 m is what we kept hitting).
#   20 m rejected exactly the fixes that would re-anchor vz/kf at corner entry and start
#   the metric climb; the tiny-blob jitter that motivated 20 is now filtered by the
#   config-AS area gate (metric_fix_ok area ≥ STEER_MIN_AREA), not by range.
VEL_FIX_RANGE_MAX = 10.0  # m — config AX (07-09): fixes may feed the ESTIMATORS (vel
#   anchor, kf fix, vel_fix_t) only when CLOSE. AW batch: with 19–25 m fixes accepted,
#   kf exploded WORSE while "anchored" (K 3.6→10.4) — a890→a1414 in 0.16 s reads as
#   23 m/s closure, impossible; at range, frame-to-frame Δz is viewing-angle change,
#   not motion. Noise math: z = fW/w ⇒ σz ≈ z²/(fW)·σw ≈ 0.46 m/px at z=20 — one px of
#   quad noise differenced at frame rate is ~14 m/s of velocity fiction. At z<10 the
#   same px is ~0.12 m ⇒ tolerable. POSITION consumers (thr_m, accel_guidance,
#   dr_target) keep the full FIX_RANGE_MAX — ±1 m position error is harmless there.
# --- metric vertical (2026-07-03): hold the gate's PnP relative HEIGHT, not its image
# row. The cy servo (CY_TARGET 130) conflates angle with height; historical ceiling
# collisions showed that image row alone is not a metric height target. The camera is
# now correctly modeled as 20° up, so old down-camera geometry must not be reused.
Z_DES_REL = 0.5  # m — keep the gate slightly BELOW us (down+); pass just above centre
KZ_P = 0.05  # thrust per m of height error
KZ_D = 0.08  # thrust per m/s of relative vertical rate (v_rel z damps the climb).
#              0.03 couldn't balance MET_THR_BAND: T3 dove at a gate ~4 m below at a
#              runaway 3 m/s (KZ_P saturated the band, damping too weak to null it) and
#              overshot below the gate into the under-gate furniture. 0.08 caps the
#              vision-measured closure rate at ~1 m/s per band-saturated meter.
KVZ_MET = 0.02  # thrust per m/s of IMU vz (NED): REAL vertical damping. v_rel z alone
#                 is PnP-diffed — laggy and reset to 0 on every dropout — so run 6 climbed
#                 at 3 m/s toward a high gate 1 and overshot 3m above it (cy 27→315),
#                 border-clipped the ring, and ended chasing a 30m gate into the pillars.
MET_THR_BAND = 0.08  # cap metric thrust to hover±this: approach height changes should be
#                      gentle; the launch/no-gate servos keep their full authority
Z_ALIGN_BAND = 1.2  # m — metric "vertically aligned" band for committing forward

# --- lateral drift damping (2026-07-02): the gate-1 miss is uncontrolled rightward drift.
# A STATIC gate's de-rotated bearing rate is a lateral-velocity proxy: drift = dβ/dt + gyro_z
# (rad/s). Validated OFFLINE from a gyro-logged trace — it is cleanly, monotonically negative
# (rightward drift) from the moment the gate centres. Bank to cancel it: roll += KD * drift
# (drift<0 => extra LEFT bank). Applied only during the approach; near the pass the bearing
# rate is parallax-dominated (would over-bank into a tumble), so gate it off by area.
KD_DRIFT_BANK = 0.0  # DISABLED by default. The drift SIGNAL is validated (offline: cleanly,
#                      monotonically negative through the gate-1 approach = rightward drift)
#                      and the sign is right (0.12 pulled bearing -37°→-11°). BUT bank alone
#                      can't thread gate 1: mild gain is neutral (bearing still ~-35°), strong
#                      gain tumbles (roll -103°). The approach is too FAST for the lateral
#                      window. Real fix = slower, velocity-nulled approach (ESKF + guidance);
#                      this drift proxy is the vision half of that velocity signal. Machinery
#                      stays wired (drift is computed + traced) so it's ready to turn back on.
DRIFT_TERM_CLAMP = 0.10  # rad — cap the drift term so it can't dominate/oscillate the bank
DRIFT_MAX_AREA = 3000.0  # only damp drift below this area (approach, not the parallax pass)
# vertical servo on thrust from the gate's image row (proven values from race.py —
# thrust was never part of the inversion bug; hover 0.30 re-confirmed on 3379)
CY_TARGET = 130.0
KP_THR_PX = 0.20  # was 0.10: too weak — a gate high in the frame (cy≈28) commanded only
#                  ~0.357 thrust, not enough to arrest the launch-arc descent to the floor
KD_THR_PX = 0.05  # visual D term: damp on d(cy)/dt (px/s, normalised). Climbing makes the
#                  gate move DOWN the frame (cy rises); without damping the P servo drove
#                  from floor to the CEILING trusses in ~1.5s (proven in fly_trace). This
#                  bleeds thrust as the gate approaches centre so we level off, not overshoot.
THR_MIN, THR_MAX = 0.25, 0.45  # was 0.36: give the climb servo real authority to recover
# The vertical servo now runs CONTINUOUSLY off the gate's image ROW (PnP range is always
# nan on 3379, so the old range gate meant it NEVER fired — the drone sank into the floor
# while the gate climbed out the top of the frame; proven in data/vq2/fly_trace).
# Asymmetric safety: a gate above center (cy<target) => CLIMB, always allowed (moves away
# from the floor). A gate below center (cy>target) => DESCEND, allowed only when the gate
# is clearly CLOSE (large area) so a far LOW gate can't pull us into the floor.
DESCEND_AREA = 2500.0  # px^2 — only descend toward a gate at least this big in frame
DESCEND_MAX_BEAR = 0.35  # rad — config AO (07-05): descent ALSO requires rough alignment.
#                          The t6.2 deepest run reached gate 1 at a15159 but b+32°
#                          (beside it, looking down at it): the row read cy279 = "gate
#                          low", the area test passed, and the servo dove at THR_MIN for
#                          ~1 s (vz +2.3) onto the jet parked UNDER the gate. Off-axis at
#                          point-blank range the image row is geometry, not height —
#                          hold altitude and let yaw/lateral re-align first.
NO_GATE_CLIMB = 0.03  # thrust bias above hover when no gate is visible: forward flight
#                       bleeds vertical lift, so hold a mild climb rather than sink blind
TILT_COMP_MIN_COS = 0.5  # tilt-compensate thrust by 1/(cosR·cosP) so BANKING a turn (which
#                          tilts the thrust vector) doesn't bleed altitude — the sink
#                          recurred on the hard left turn past gate 1 (fly_trace). Uses the
#                          RELIABLE attitude estimate, not the biased IMU vz. Bounded so a
#                          near-90° tumble can't command infinite thrust.

# post-gate brake: the launch/cruise carries real momentum and we have no velocity
# estimate yet, so after each gate pass pitch UP briefly to bleed speed before the turn
# (live finding 2026-07-01: collisions near gate 1 happen with PERFECT attitude — the
# failure is entry speed, not stability).
BRAKE_SECS = 0.0  # OFF. Tried a 1.2 s post-gate-0 speed-kill brake (+0.15 nose-up) to slow
#                   the gate-1 approach: NO help (bearing still ends -37°, gate lost earlier).
#                   The gate-1 miss is not a forward-speed problem — needs velocity guidance.
BRAKE_PITCH = +0.15  # (retained; unused while BRAKE_SECS=0)

# post-gate OPEN-SPACE heading hold (2026-07-02): braking alone never changed the
# ~3.3s collision — the drone rides the launch momentum straight into a dark 3D
# structure that sits ON the line from the gate-0 exit toward gate 1. The straight-line
# turn toward gate 1's bearing is what clips it. Instead: after passing gate 0 keep the
# gate-0 EXIT heading (yaw held, wings level, no gate steering) for HOLD_HEADING_SECS so
# we coast through the open corridor, THEN acquire gate 1 from past the structure.
HOLD_HEADING_SECS = 0.0  # DISABLED with the brake (same disproven hypothesis): just
#                          point-and-go continuously; the vertical servo holds height.
HOLD_PITCH = -0.03  # (retained; unused while HOLD_HEADING_SECS=0)

# post-gate POP-UP (2026-07-04, runs E–S): the gate-0→1 segment is a climbing right
# corner — the ribbon is an ELEVATED arc (~2 m above the gate-0 exit height) with
# threat1 (parked drone) on the floor straight ahead and threat2 (Station-19 jet)
# filling the corner inside at low/mid height. Every low approach died on one of them,
# and every image cue we steer by is degenerate from 2 m BELOW the path: the ribbon
# hugs rows 0–40 (near the horizon), its row servo pegs the climb cap, and its
# near/aim bearings barely respond to our lateral offset. So climb FIRST: brake +
# fixed-thrust pop-up straight after the gate pass, then run the corner logic AT path
# height, where the ribbon reads level (rows ~110–180), the gate is head-on for PnP,
# and both threats are below us.
POST_GATE_CLIMB_S = 0.7  # s of pop-up after each real gate pass (T3: 1.0 s stacked on
#                          the climb-biased servos left us ~4 m over the path — the
#                          pop-up only needs to BREAK the floor-hugging; cues finish it)
POST_GATE_CLIMB_THR = 0.07  # thrust above hover during the pop-up (≈ +2 m incl. coast)

# --- failsafes ---
TILT_ABORT = math.radians(60.0)  # attitude estimator says we're tumbling
COLL_ABORT = 3  # collision messages before aborting the attempt
STAB_SECS = 1.5  # initial window before point-and-go
STAB_PITCH = 0.0  # LEVEL through the sim's launch. Tried +0.10 nose-up to bleed the
#                   launch momentum before gate 0 (2026-07-02): the drone then hit the
#                   gate-0 FRAME instead of threading (3/3). Ride the launch level
#                   (gate 0 threads 100%), brake after.


TARGET_MATCH_PX = 90.0  # continuity radius: a fresh detection within this many px of the
#                         held target IS the held target. Without it, the near gate getting
#                         border-clipped at frame top made nearest() hop to a FARTHER gate
#                         in the corridor whose approach line crossed the pillar row (live
#                         2026-07-03: 15m gate -> 31m gate -> ENV collision).
# Config AV (07-09): identity continuity for the RE-LOCK case TARGET_MATCH_PX can't
# catch (the new sighting is a different gate far away in the image). AU attempt 3
# reached gate 1 AT HEIGHT (a7565, b+32°) and then the detector re-locked gate 2
# down the hall (a832, b−9°) in ONE frame: metric guidance dove for gate-2's
# floor-level height (vz +3 sink), steering slid left past the hoop, and the kf ate
# a 3 m→20 m pose teleport (K 8.9→17 while I read 2). A point-blank gate does not
# teleport: both cuts must hold for the sighting to be a DIFFERENT physical gate
# (same-gate area collapses from clipping don't jump bearing; ego-yaw at max whip
# moves ~0.16 rad/frame). Blocked ⇒ treat as dropout: held freezes, dr pursuit
# chases the last real fix, and TARGET_HOLD_S expiry hands over naturally. The
# gate-0 pass flip (a3010→a288, Δb 0.21 rad) stays ALLOWED by the bearing cut.
SWITCH_GUARD_AREA = 2500.0  # held must be point-blank for the guard to apply (px²)
SWITCH_AREA_RATIO = 0.4  # new sighting under this fraction of held.area…
SWITCH_GUARD_BEAR = 0.35  # …AND farther than this in bearing (rad) ⇒ different gate


# Perception: gates.nearest() returns ONLY ring-validated gates. The former blob
# fallback steered into glow-lit walls (root cause of the gate-1 ENV collisions —
# confirmed frame-by-frame in runs/vq2_collisions); glow is never a target now.
# A short persistence window bridges detection dropouts between valid sightings.
TARGET_HOLD_S = 0.8

# --- cyan racing-line follower (2026-07-03): spec §3.4 "course guidance structures".
# The sim renders a cyan path threading the gates: a wide floor wedge at launch and an
# elevated ribbon at gate height between gates. fly_trace 07-03 frame 017 shows it
# arcing LEFT around the pillar field straight to gate 1 — an obstacle-free path exactly
# where straight-line gate chasing crosses pillars. Validated offline over the trace
# (_line_probe.py: tracked continuously through the whole gate-1 approach, incl. the
# frames where gate perception went blind BEFORE we drifted into the scaffold).
# Engaged when there is no steerable gate: blind (after the brake) or gate < STEER_MIN_AREA.
# Steer at the AIM point (pure-pursuit mid-path lookahead) — the far tip flies the chord
# across the path's arc (run-3 crash), and the wide near-field wedge skews the centroid
# (offline frame 002: centroid −20° vs apex at gate).
LINE_HOLD_S = 0.8  # persistence, same pattern as gate targets
LINE_PITCH = -0.04  # gentle forward drift along the path (slower than the gate chase)
LINE_ALIGN_BEAR = 0.30  # rad — pitch forward only when roughly nose-aligned with the path
# --- latched coordinated turn (2026-07-04, runs E–J) --- every attempt understeers the
# gate-0→1 right corner the same way: bearing grows +20°→+40° while yaw runs at max,
# because no branch holds a bank for more than ~0.5 s — metric expires, ribbon zeroes
# roll, bearing-fallback brakes — and a ~90° corner at 3–4 m/s needs ~2 s of CONTINUOUS
# ~20° bank (R = v²/(g·tanφ) ≈ 5 m). When the bearing is large and the gate is not yet
# close, LATCH a committed constant-bank turn toward it and hold it through sighting
# dropouts until the bearing closes. This is how the corner is actually flown.
TURN_ENTER_BEAR = 0.35  # rad (~20°) — off-axis this far = corner: commit to the turn
TURN_EXIT_BEAR = 0.15  # rad — roughly nose-on again: hand back to normal guidance
TURN_LATCH_S = 0.5  # per-sighting latch: keeps the bank through dropouts/branch flips
TURN_MIN_AREA_FRAC = 0.4  # run K: mid-latch the corner gate (a1618) left the frame right
#                           and detection handed the target to a DOWNSTREAM gate straight
#                           ahead (a316, 23 m) — the controller abandoned the corner and
#                           flew 23 m into threat1. While latched, only a comparably-sized
#                           sighting can be the corner gate; reject far ones.
TURN_BLIND_S = 1.0  # corner gate stale mid-latch: it exited frame IN the turn direction,
#                     so keep the committed turn this much longer — it re-enters the frame
#                     as the yaw comes around (braking here = run-H's 3.4 s blind drift).
VEL_TRUST_S = 1.5  # s — how long after a vision anchor the velocity estimate may be
#                    actively steered against (blind brake). Beyond it: level + drag.
RELATCH_CALM_S = 1.2  # s after a blind brake before the turn latch may re-engage. Z3:
#                       corner exit reached STABLE (brake-hold worked, gate 1 re-acquired
#                       at b+25°) — but 0.44 rad > TURN_ENTER re-latched instantly and
#                       banked 20° into the jet beside us. A coordinated turn is a SPEED
#                       maneuver; braked-to-hover means aim-then-go (the gate-0 recipe).
TURN_BANK = 0.35  # rad (~20°) — latch-only bank clip. W3: at MAX_ROLL_TGT (14°) the
#                   lateral accel is only g·tan14° ≈ 2.6 m/s², so the VELOCITY vector
#                   turned far slower than the nose (gz ~0.95 vs achievable ψ̇=a/v ~0.65)
#                   and the drone translated outside through the corner into the station
#                   bays ("Station 18" sign seen from BEHIND at t5.0). 20° buys +40% a_lat.
TURN_LATCH_BRAKE = 0.24  # rad nose-up pitch INSIDE the latch (vs TURN_BRAKE_PITCH 0.15
#                          AJ (07-05): 0.30 was NOT better (0/3 deep runs, sightings lost
#                          slightly EARLIER) — the brake direction is exhausted at 0.24.
#                          elsewhere). AB/AC: yaw saturates at MAX_YAW but path curvature
#                          is a_lat/v² — at corner-entry speed the arc runs wide of the
#                          ribbon and every cue exits frame by ~t3.4. Braking at ~2.4 m/s²
#                          tightens the arc QUADRATICALLY and keeps the ribbon in view
#                          longer. Slow through the corner is the whole ballgame.
TURN_REF_R = 2.8  # m — the METRIC blind arc's reference radius. AH5 (07-05): the fixed
TURN_VREF_MIN = 0.8  # 20°-bank blind arc has run-to-run radius variance of metres
#                      (R = v²/g·tanφ: v±30% ⇒ R±70%) and landed us banked INSIDE
#                      Station 19's bay, in a ~2–3m corridor. With the vision-anchored
#                      speed measured, pin the RADIUS instead: bank φ = atan(v²/(R·g)),
#                      yaw rate ψ̇ = v/R — a coordinated arc of the same geometry at any
#                      entry speed. 2.8m ≈ what the successful 20°-bank runs carved at
#                      their typical post-brake ~3 m/s. Config AI flew this on the simple
#                      integrator's speed = sharply worse (underread ⇒ near-straight arc
#                      into the threats; 0/3 past t3.2). Config AK re-wired it to the
#                      shadow ESKF's speed after the ground-only update_gravity fix made
#                      the kf track the vision-anchored velocity (batches 1–2, 07-05).
ARC_DECEL = 2.0  # m/s² — config AM: the blind REFERENCE arc decelerates like the latch
ARC_VMAX = 2.0  # brake did (g·tan(0.24)≈2.4), so the position tracker itself commands
#                 the nose-up brake while chasing it; entry speed clipped to ARC_VMAX.
#                 Config AQ (07-08): 3.5→2.0 — the AO batches fixed the sink but died
#                 t4.2–4.9 arriving BANKED into the jet: lateral arc placement varies
#                 metres run-to-run because R tracks v² through a ~2–3 m corridor.
#                 A slow reliable pass beats a fast random one (VQ2 = qualify first),
#                 so the blind reference now starts no hotter than 2 m/s.
V_ENTRY_MAX = 1.8  # m/s — config AN: runs entering the corner at kf≈3.5–4 died on
HOT_BRAKE_PITCH = 0.35  # threat2 at ~t3.1 (3/6 across the AM batches, config-
#                         independent), while 1.5–2.6 m/s entries reached t4.1–5.5 and
#                         clipped the gate. While latched AND the kf reads hotter than
#                         V_ENTRY_MAX, brake at ≥HOT_BRAKE_PITCH (~20° nose-up,
#                         g·tan20°≈3.6 m/s²) whatever the branch wanted — R=v²/(g·tanφ).
#                         Config AQ (07-08): 2.6→1.8 — kill the corner speed for a
#                         tight repeatable arc (the AO lateral-spread finding above).
#                         Only trusted while the kf is vision-anchored (< VEL_TRUST_S):
#                         trace 233117's blind kf overspeed latched this brake for 2 s
#                         of fictitious 12.8 m/s while the drone actually hovered/sank.
TURN_BLIND_KVZ = 0.05  # blind mid-latch thrust: NULL vz instead of chasing pixels. W3:
#                        the row servo kept serving the FROZEN stale cy=28 (max climb)
#                        through the whole blind second → rose past the gate into the
#                        truss field (frame 026), where re-acquisition is impossible.
#                        Post-VIS_ROLL-guard the vz estimate tracked the frames honestly
#                        (−1.8 climb → +1.6 sink, no ±3 pinning), so a soft vz-null hold
#                        is now trustworthy. The pop-up already bought the elevation.
TURN_ROW_FRESH_S = 0.3  # latch row servo only on RECENT sightings — a corner sweeps the
#                         gate across the frame fast, so pixels older than this lie.
TURN_BLIND_VZ = -1.0  # AE1 (07-04): the blind vz-null HELD ALTITUDE while the racing line
LINE_AIM_TOP_PX = 40.0  # climbed ~2 m to the elevated gate 1 → exited the corner at jet
LINE_CLIMB_MEM_S = 2.5  # height and hit threat2's fuselage with gate-1's glow just above
#                         (trace 025: frame filled with the wing; mid-blind frames show
#                         floor markings large = LOW). The cue was in hand: the last fresh
#                         line aim row was pinned at the frame TOP (y=5, "climb hard")
#                         right before the cues went blind. So: if the last line sighting
#                         (within LINE_CLIMB_MEM_S) aimed above LINE_AIM_TOP_PX, the blind
#                         hold targets a CLIMB (vz→TURN_BLIND_VZ NED) instead of vz=0.
TURN_BLIND_KVZ_CLIMB = 0.15  # AF (07-04): the climb ENGAGED but SATURATED — gain 0.05 ×
TURN_BLIND_CLIMB_CAP = 0.12  # 1 m/s error = +0.05 thrust = exactly the cap, and at 20°
#                              bank that is ~zero net climb: vz decayed −1.4→+0.2 through
#                              the blind second and AF died on the jet with AE1's exact
#                              terminal frame. When the climb cue is engaged, the servo
#                              gets REAL authority (gain 0.15, cap hover+0.12). Still not
#                              W3/run-N's open-loop ride: the honest vz feedback throttles
#                              back as vz reaches −1.0, and blind time is bounded
#                              (~1.2 s) ⇒ worst-case ~1.5 m of climb, under the trusses.
TURN_CLIMB_CAP = 0.05  # run N: full-authority row servo (gate at cy30 → thr 0.41) climbed
KVZ_TURN = 0.03  # ....  ~3 m/s and overshot the ~2 m-elevated corner gate into the trusses.
#                  The corner wants a steady ~1.2 m/s climb: cap above hover + damp on vz.
COORD_BANK = 0.30  # rad of bank per rad/s of commanded yaw — COORDINATED TURN while
#                    line-following. Runs E/F/G (07-04) all understeered the gate-1 right
#                    turn the same way: the ribbon branches commanded roll_target=0, so
#                    yaw swung the NOSE along the arc while momentum carried the drone
#                    straight off the outside (bearing pinned +18…+36°, slid left into
#                    the station). Yaw does not steer; φ = atan(v·ψ̇/g) ≈ (v/g)·ψ̇ does —
#                    at our ~3 m/s that is 0.3·ψ̇, clipped to MAX_ROLL_TGT.
KXT_BANK = 0.5  # rad of bank per rad of near-point bearing — CROSS-TRACK correction.
#                 Runs R1–R3 (2026-07-04): pure pursuit on the mid-path aim point holds
#                 heading but cuts corners; every arc sliced the inside of the 0→1 right
#                 turn through the Station-19 parked jet (threat2). Yaw follows the aim
#                 point (heading along the path); bank ALSO pulls toward the ribbon's
#                 near point, nulling the lateral offset that pure pursuit leaves.
LINE_CY_TARGET = 110.0  # px — hold the AIM row here (empirical: threading gate 0 at the
#                         right height put the aim row at 72–124). The old servo drove the
#                         FAR-point row, but a horizontal path's far tip sits near the
#                         horizon at ANY height — an unsatisfiable target that pegged max
#                         climb the whole run-5 crossing (vz −3 m/s) and delivered us 3m+
#                         OVER gate 1 into a blind descent.
KP_THR_LINE = 0.10  # gentler than the gate-row servo (path height is a soft cue)
KVZ_LINE = 0.02  # thrust per m/s of IMU vz (NED, climb<0): damps the climb so the line
#                  servo physically can't run away (drifty long-term, fine as a damper)
LINE_CLIMB_CAP = 0.06  # max thrust above hover while line-following (soft cue, soft cap)
LINE_DESCEND_CAP = 0.04  # run P: arriving ~1.5 m over the ribbon, the servo dove at
#                          THR_MIN (vz +3 m/s) into the corner furniture — the descent
#                          needs the same "soft cue, soft cap" as the climb (~1 m/s).
LINE_DESCEND_AREA = 2000.0  # px^2 — only descend toward a big/near path (floor-sink guard)
LINE_PREF_AREA = 700.0  # px^2 — while the target gate is smaller (farther) than this,
#                          the RIBBON steers, not the gate: the straight line at a far
#                          gate is exactly what crosses the pillar field / arrives with
#                          uncorrectable lateral momentum (live 07-03: metric chase ended
#                          bearing −26° sliding past gate 1). The ribbon's arc IS the
#                          correct approach line; the gate takes over close-in.
#                          Run-F (07-04): 2000 was far too late a handover — ribbon
#                          pure-pursuit only aligns heading to the ribbon median, it
#                          never nulls cross-track to the gate: bearing sat at +18°
#                          from a791 all the way to a1717 and we passed the gate's
#                          right shoulder. Metric guidance (which does null it, with
#                          velocity damping) must own the final ~1.5 s.
LINE_VETO_AREA = 2000.0  # px² — config AR (07-08): below this (gate still far), a held
LINE_GATE_VETO_BEAR = 0.4  # gate whose bearing contradicts the ribbon's FAR bearing by
#                            more than this (rad) is the WRONG gate — post-gate-0 the
#                            detector locks a downstream gate near b≈0 down the hall
#                            (a510–1328, "range" 21–31 m) while the ribbon's far tip
#                            points +31…+42° right at the corner; metric guidance then
#                            ACCELERATES straight at it (AQ batch: K 2.0→5.0 in 0.7 s =
#                            the whole "hot entry" mode + the old early-loss/threat1
#                            deaths). The ribbon steers instead — it is the course's own
#                            guidance structure and threads every gate. Genuine gate-1
#                            locks agree with the far tip within ~17° (AQ attempt 1) and
#                            keep gate steering; close gates (≥ this area) are exempt so
#                            the final-approach handover (run-F lesson above) is intact.
PASS_MAX_BEAR = 0.35  # rad (~20°) — only COMMIT through the gate plane if roughly centred;
#                       a big/close gate at a large bearing is already MISSED, and
#                       committing straight then flies into whatever sits beside the gate
#                       (live 07-03: commit at −26° → ENV collision → tumble). Brake instead.
PASS_MAX_CY = 100.0  # px — the VERTICAL commit gate. Run 7: gate 1 close (a3368) at
#                      bearing +13° passed the lateral check, but cy=260 (130px below
#                      target = ~2.5m too high) — "straight and level" sailed over the
#                      top bar into the ceiling trusses (coll ×5). Vertically off = missed.
PASS_KVZ = 0.08  # thrust per m/s of vz during the commit coast — AG3 (07-05): the commit
PASS_VZ_CAP = 0.05  # froze pitch/roll/yaw but NOT thrust, and the corner-climb servo ran
#                     vz to −3.0 through the coast → rose over the gate onto the jet
#                     parked under it. The coast is level: hover ± a bounded vz damp
#                     (vision-anchored vz only — it pins ±3 when poisoned).


class Flight:
    """One armed attempt; owns the estimator and the failsafe bookkeeping."""

    def __init__(self, sh: cc.Shared, conn, tgt_sys: int, tgt_comp: int, lock) -> None:
        self.sh, self.conn, self.lock = sh, conn, lock
        self.tgt = (tgt_sys, tgt_comp)
        self.att = AttitudeEstimator()
        self.vel = VelocityEstimator()
        self.vz = 0.0  # world-frame descent rate (m/s, +down), IMU-integrated
        self.gyro_z = 0.0  # latest body yaw rate (rad/s), for bearing de-rotation
        self.body_gyro = (0.0, 0.0, 0.0)  # latest raw body rates (rad/s), for the bank observer
        self.vz_min = 0.0  # most-negative (climb) seen
        self.vz_max = 0.0  # most-positive (sink) seen
        self.t_prev = time.time()
        self.last_arm = 0.0
        with sh.lock:
            self.coll0 = sh.collisions
        self.tilt_max = 0.0
        self.gyro_sq_sum = np.zeros(3)
        self.n = 0
        # Optional reference to the log-only shadow worker (set by mission), read ONLY for
        # diagnostics (trace filenames + summary). The command loop does no shadow work;
        # all shadow ESKF/perception runs on ShadowWorker's own thread.
        self.shadow: ShadowWorker | None = None
        # Last command-boundary telemetry (Phase-4A observability, diagnostic only). Set by
        # command(); keys req/des/cal each = (thr, roll, pitch, yaw) — requested,
        # post-clamp desired ACTUAL rates, and calibrated raw command — kept SEPARATE so
        # the inversion/scale cannot be applied twice in replay. sat = per-axis clamp flags.
        self.last_cmd: dict[str, tuple[float, float, float, float]] | None = None

    # -- per-loop primitives -------------------------------------------------
    def step_sensors(self) -> tuple[np.ndarray, np.ndarray] | None:
        """Re-arm, read IMU, advance the attitude + velocity estimators. None if no IMU yet.

        Control-only: the shadow ESKF/perception is handled entirely by ShadowWorker on a
        separate thread, so this stays off the shadow path and holds ~60 Hz.
        """
        now = time.time()
        if now - self.last_arm > 0.4:
            cc.arm(self.conn, *self.tgt, self.lock)
            self.last_arm = now
        with self.sh.lock:
            imu = self.sh.imu
        if imu is None:
            return None
        dt = min(0.05, now - self.t_prev)
        self.t_prev = now
        acc = np.array(imu[:3])
        gyro = np.array(imu[3:])
        self.gyro_z = float(gyro[2])  # body yaw rate — de-rotates gate bearing drift
        self.body_gyro = (float(gyro[0]), float(gyro[1]), float(gyro[2]))  # raw body rates (rps)
        self.att.update(gyro, acc, dt)
        self.vel.predict(acc, self.att.R, dt)
        self.vz = float(np.clip(self.vel.v[2], -VZ_CLAMP, VZ_CLAMP))
        self.vz_min = min(self.vz_min, self.vz)
        self.vz_max = max(self.vz_max, self.vz)
        roll, pitch, _ = self.att.euler
        self.tilt_max = max(self.tilt_max, abs(roll), abs(pitch))
        self.gyro_sq_sum += gyro**2
        self.n += 1
        with self.sh.lock:  # publish control attitude for the shadow worker's comparison log
            self.sh.active_att = self.att.euler
        return acc, gyro

    def command(self, thr: float, rr: float, pr: float, yr: float) -> None:
        """Single calibrated command boundary (desired ACTUAL body rates in, rad/s)."""
        rr_c = float(np.clip(rr, -MAX_RATE, MAX_RATE))
        pr_c = float(np.clip(pr, -MAX_RATE, MAX_RATE))
        yr_c = float(np.clip(yr, -MAX_YAW, MAX_YAW))
        # observability only — capture the three levels separately (behaviour unchanged:
        # the same clamped rates are sent). cal = what send_att emits after calibration.
        self.last_cmd = {
            "req": (thr, float(rr), float(pr), float(yr)),
            "des": (thr, rr_c, pr_c, yr_c),
            "cal": (thr, *cc.desired_to_cmd(rr_c, pr_c, yr_c)),
            "sat": (0.0, float(rr != rr_c), float(pr != pr_c), float(yr != yr_c)),
        }
        cc.send_att(self.conn, *self.tgt, self.lock, thr, rr_c, pr_c, yr_c)

    def tilt_comp(self, thr: float) -> float:
        """Scale thrust by 1/(cosR·cosP) so a banked/pitched turn holds altitude."""
        roll, pitch, _ = self.att.euler
        c = math.cos(roll) * math.cos(pitch)
        return float(np.clip(thr / max(TILT_COMP_MIN_COS, c), THR_MIN, THR_MAX))

    def level_rates(
        self, pitch_target: float = 0.0, roll_target: float = 0.0
    ) -> tuple[float, float]:
        """Desired roll/pitch rates that P-track (roll_target, pitch_target) attitude."""
        roll, pitch, _ = self.att.euler
        return (KP_ATT_ROLL * (roll_target - roll), KP_ATT_PITCH * (pitch_target - pitch))

    # -- failsafes -----------------------------------------------------------
    def abort_reason(self) -> str | None:
        with self.sh.lock:
            coll = self.sh.collisions - self.coll0
            live = self.sh.is_live()
            fin = self.sh.race_finish_ns
        if fin >= 0:
            return "FINISHED"
        if not live:
            return "not live"
        if coll >= COLL_ABORT:
            return f"collisions ({coll})"
        roll, pitch, _ = self.att.euler
        if abs(roll) > TILT_ABORT or abs(pitch) > TILT_ABORT:
            return f"tilt abort (roll={math.degrees(roll):.0f} pitch={math.degrees(pitch):.0f} deg)"
        return None

    def summary(self) -> str:
        with self.sh.lock:
            coll = self.sh.collisions - self.coll0
            gate = self.sh.active_gate
            last = self.sh.last_collision
        rms = np.sqrt(self.gyro_sq_sum / max(1, self.n))
        what = ""
        if coll and last is not None:
            kind = {1001: "GATE", 1002: "ENV"}.get(last[0], str(last[0]))
            what = f" last_coll={kind}/threat{last[1]}"
        ar, ap, _ = (math.degrees(v) for v in self.att.euler)
        return (
            f"tilt_max={math.degrees(self.tilt_max):.1f}deg "
            f"gyro_rms=({rms[0]:.2f},{rms[1]:.2f},{rms[2]:.2f}) "
            f"vz=[{self.vz_min:+.1f},{self.vz_max:+.1f}]m/s "
            f"coll={coll}{what} active_gate={gate} loops={self.n} "
            f"att=({ar:+.0f},{ap:+.0f})deg"  # control estimators only; shadow health is separate
        )


def pick_target(
    dets: list[gates.GateTarget], held: gates.GateTarget | None, held_fresh: bool
) -> gates.GateTarget | None:
    """The steering target for this frame, preferring continuity with ``held``.

    A detection within TARGET_MATCH_PX of the held target's image position is the
    same physical gate and wins outright; otherwise fall back to gates.nearest()'s
    rule (closest by PnP range, else largest ring).
    """
    if not dets:
        return None
    if held is not None and held_fresh:
        near = min(dets, key=lambda t: math.hypot(t.cx - held.cx, t.cy - held.cy))
        if math.hypot(near.cx - held.cx, near.cy - held.cy) < TARGET_MATCH_PX:
            return near
    with_pose = [t for t in dets if t.pose_cam is not None]
    if with_pose:
        return min(with_pose, key=lambda t: t.pose_cam[2])  # type: ignore[index]
    return max(dets, key=lambda t: t.area)


def target_switch_ok(
    held: gates.GateTarget | None, held_fresh: bool, tgt: gates.GateTarget
) -> bool:
    """Config AV: may ``tgt`` replace ``held`` as the pursued gate this frame?

    False only for the identity-switch signature — a fresh point-blank held target
    (area > SWITCH_GUARD_AREA) "replaced" by a sighting that is BOTH much smaller
    (< SWITCH_AREA_RATIO of held's area) and far away in bearing
    (> SWITCH_GUARD_BEAR): that is a different physical gate downstream, seen
    because ours clipped out of the ring test at the gate plane. Adopting it mid-pass
    steers/descends AWAY from the gate we are threading (AU attempt 3) — the correct
    reading is a DROPOUT of the held target.
    """
    return (
        held is None
        or not held_fresh
        or held.area <= SWITCH_GUARD_AREA
        or tgt.area >= held.area * SWITCH_AREA_RATIO
        or abs(tgt.bearing - held.bearing) <= SWITCH_GUARD_BEAR
    )


def metric_fix_ok(pose_cam: tuple[float, float, float] | None, area: float = np.inf) -> bool:
    """Config AP: a PnP fix is metric-trustworthy only when CLOSE (< FIX_RANGE_MAX).

    Far fixes (tiny frame-top blobs post-gate-0 ranged at ~28m, true ~12–15m) are
    bearing-only: they still steer the pixel branch but must not reanchor the kf,
    anchor the velocity integrator, or drive metric guidance. See FIX_RANGE_MAX.

    Config AS (07-08): the range test alone let a278–660 px² far gates through
    (their PnP ranges JITTER under 20 m). Around the gate-0 pass the tracker holds
    exactly such a gate, and differencing its metre-noisy poses injected garbage
    velocity anchors — every AR-batch run had fl.vz pinned at −3 ("climbing") from
    t2.0 while the frames showed the floor 2 m below, so the climb servos cut
    thrust to their clip floor and the corner arrived at JET height. Below
    STEER_MIN_AREA the PnP pose is bearing-only-grade: no metric anything.
    """
    return pose_cam is not None and pose_cam[2] < FIX_RANGE_MAX and area >= STEER_MIN_AREA


def blind_arc_cmd(v_h: float) -> tuple[float, float]:
    """Coordinated-arc (bank rad, yaw-rate rad/s) pinning radius TURN_REF_R at speed v_h.

    φ = atan(v²/(R·g)) supplies exactly the centripetal acceleration for radius
    TURN_REF_R at the measured horizontal speed; ψ̇ = v/R keeps the nose tangent.
    Both capped (TURN_BANK, MAX_YAW) — see TURN_REF_R for the AH5 rationale.
    """
    phi = math.atan2(v_h * v_h, TURN_REF_R * 9.81)
    return min(phi, TURN_BANK), min(v_h / TURN_REF_R, MAX_YAW)


def accel_guidance(
    p_rel_w: np.ndarray, v_rel_w: np.ndarray, yaw: float, elev_ok: bool = True
) -> tuple[float, float]:
    """Desired (pitch_target, roll_target) rad that tilt the thrust vector at the gate.

    ``p_rel_w`` = gate − drone in the world frame (NED-ish; yaw origin arbitrary but
    consistent with ``yaw``). ``v_rel_w`` = d(p_rel)/dt (gate static ⇒ = −our velocity),
    so +KD on it IS velocity damping. Only the horizontal (x,y) components are used —
    height stays with the proven visual vertical servo.

    Velocity-capped law: v_des = KV_POS·p_rel capped at V_MAX, a = KA_VEL·(v_des − v).
    Unlike a raw position-PD (whose clamped accel always points AT the gate and thus
    never brakes an already-hot approach), this brakes whenever actual speed exceeds
    the capped desired speed — the property that lets a fast drone actually turn.

    Small-angle inverse of a = g·tanθ. Body FRD: nose-DOWN (negative pitch) accelerates
    forward; right-wing-down (positive roll) accelerates right. ``elev_ok=False`` (gate
    far from vertical centre) forbids forward acceleration (braking still allowed) so we
    climb to a high gate before committing — keeps the old aim-then-go safety.
    """
    p = np.asarray(p_rel_w[:2], float)
    v_des = KV_POS * p
    n = float(np.linalg.norm(v_des))
    if n > V_MAX:
        v_des *= V_MAX / n
    v = -np.asarray(v_rel_w[:2], float)  # gate static ⇒ our velocity = −d(p_rel)/dt
    a = KA_VEL * (v_des - v)
    n = float(np.linalg.norm(a))
    if n > MAX_ACC:
        a *= MAX_ACC / n
    cy_, sy_ = math.cos(yaw), math.sin(yaw)
    a_fwd = cy_ * a[0] + sy_ * a[1]  # world → heading frame
    a_lat = -sy_ * a[0] + cy_ * a[1]
    if not elev_ok:
        a_fwd = min(a_fwd, 0.0)
    pitch_t = float(np.clip(-math.atan2(a_fwd, G), -MAX_TILT_CMD, MAX_TILT_CMD))
    roll_t = float(np.clip(math.atan2(a_lat, G), -MAX_TILT_CMD, MAX_TILT_CMD))
    return pitch_t, roll_t


def vertical_thrust(held: gates.GateTarget | None, cy_rate: float = 0.0) -> float:
    """Thrust (PD on the gate's image row) that keeps the gate at CY_TARGET.

    Camera image row is the reliable height cue (IMU vz is too biased through the launch).
    ``cy_rate`` = d(cy)/dt in px/s (the visual climb-rate proxy) damps the overshoot.
    No gate visible => a mild climb bias, since forward flight bleeds vertical lift.
    """
    if held is None:
        return float(np.clip(HOVER_THRUST + NO_GATE_CLIMB, THR_MIN, THR_MAX))
    err = (held.cy - CY_TARGET) / 180.0  # <0: gate high => climb; >0: gate low => descend
    if err >= 0 and (held.area < DESCEND_AREA or abs(held.bearing) > DESCEND_MAX_BEAR):
        return HOVER_THRUST  # far low gate OR off-axis gate: hold, don't dive (see AO)
    thr = HOVER_THRUST - KP_THR_PX * err - KD_THR_PX * (cy_rate / 180.0)
    return float(np.clip(thr, THR_MIN, THR_MAX))


def line_thrust(line: gates.LineTarget, vz: float = 0.0) -> float:
    """Thrust servo on the racing line's AIM row (vertical_thrust, but gentler).

    The aim (mid-path) row actually responds to our height; the far tip does not (it
    rides the horizon at any altitude — the run-5 climb runaway). ``vz`` (NED, climb
    negative) damps so the servo cannot run away vertically, and the climb is capped:
    the path is a soft height cue, never worth 3 m/s.

    Same asymmetric safety: climbing toward a high path is always allowed; descending
    is allowed only when the path is big/near in frame (a far low wedge must not pull
    us toward the floor — the same trap the gate servo's DESCEND_AREA guards).
    """
    err = (line.aim_cy - LINE_CY_TARGET) / 180.0
    if err >= 0 and line.area < LINE_DESCEND_AREA:
        return HOVER_THRUST
    thr = HOVER_THRUST - KP_THR_LINE * err + KVZ_LINE * vz
    return float(np.clip(thr, HOVER_THRUST - LINE_DESCEND_CAP, HOVER_THRUST + LINE_CLIMB_CAP))


def _trace_frame(
    fl: Flight,
    tdir: Path,
    t_rel: float,
    idx: int,
    held: gates.GateTarget | None,
    gate: int,
    metric_on: bool = False,
    line: gates.LineTarget | None = None,
    on_line: bool = False,
    dr: np.ndarray | None = None,
) -> None:
    """Save one trajectory frame with attitude / vz / gate-detect in the filename."""
    with fl.sh.lock:
        frame = fl.sh.frame
    if frame is None:
        return
    roll_d, pitch_d, _ = (math.degrees(v) for v in fl.att.euler)
    g = "none"
    if held is not None:
        g = f"cy{held.cy:.0f}a{held.area:.0f}b{math.degrees(held.bearing):+.1f}"
        if metric_on:
            g = "M" + g  # metric accel guidance active (PnP pose fresh)
    elif dr is not None:
        # dead-reckon pursuit: propagated range + world-frame offsets of the lost gate
        g = f"dr{float(np.hypot(dr[0], dr[1])):.1f}m_z{dr[2]:+.1f}"
    if line is not None:
        # racing-line sighting: aim (pure-pursuit steering) + far bearings, far row;
        # F = steering follows the line. ('*' is illegal in Windows filenames — it
        # silently killed imwrite, so plain letters only.)
        g += (
            f"_L{'F' if on_line else ''}a{math.degrees(line.aim_bearing):+.0f}"
            f"f{math.degrees(line.far_bearing):+.0f}y{line.aim_cy:.0f}"
        )
    # shadow ESKF (worker) vs simple integrator: K = kf horiz speed + kf vz,
    # I = integrator horiz speed. kf lives on the shadow worker (diagnostic read only);
    # 0 if no worker is attached.
    kf_v = fl.shadow.velocity_snapshot() if fl.shadow is not None else np.zeros(3)
    kh = float(np.hypot(kf_v[0], kf_v[1]))
    ih = float(np.hypot(fl.vel.v[0], fl.vel.v[1]))
    name = (
        f"{idx:03d}_t{t_rel:.2f}_r{roll_d:+.0f}p{pitch_d:+.0f}"
        f"_vz{fl.vz:+.1f}_gz{fl.gyro_z:+.2f}_g{gate}"
        f"_K{kh:.1f}z{kf_v[2]:+.1f}I{ih:.1f}_{g}.png"
    )
    cv2.imwrite(str(tdir / name), frame)
