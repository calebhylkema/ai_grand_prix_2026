# Critical Diagnostic — Why 10 Days of Iteration Didn't Help

## The Symptom (100% reproducible)
- **Gate 0:** Threads perfectly every run (t2.1–2.3 s, 100% success rate)
- **After gate 0:** Brake + scan works; we acquire gate 1 at z 9.8–12.6 m, bearing +28–34°
- **THREAD phase:** The gate starts orbiting (bearing +34→+42° while yawing right at MAX_YAW, bank clamped)
- **Result:** "target lost" → re-scan → collide (ENV/threat1 floor clutter)
- **Pattern:** ALL 8 batches (2026-07-09) failed identically in THREAD, despite different thrust profiles

## The Dead Ends We've Hit
1. **Open-loop brake vertical:** Batches 2/4 ran to the CEILING; batch 3 to the FLOOR
2. **Vision-anchored vz-null:** Batch 5 flew INTO the floor (launch pins integrator at −3 "climb" fiction)
3. **Racing-line servo:** Batches 6+ keep height but introduced new orbit signature
4. **Tuning SCAN pitch/THREAD pitch:** No amount of recenter brake, level scan, or far-converge logic stopped the orbit

## The Hypothesis (Not Yet Validated)

**We may be DETECTING THE WRONG OBJECT in THREAD phase.**

### Critical Spec Question: Camera Tilt Direction

**SPEC (VADR-TS-003 §3.8, verbatim):**
> "The camera is tilted upwards by 20° upwards."

**OUR CODE (gates.py, lines 43–55):**
```python
# Camera (OpenCV: x-right, y-down, z-fwd; tilted 20 deg DOWN) -> body FRD.
# 2026-07-03: the sim camera points ~20° DOWN, not up — verified from live imagery
# (a Mahony-level hover puts the horizon at y≈64 with floor filling the frame, and
# far gates are detected right at that line). The spec/config note claiming "20° UP"
# is wrong for build 3379.
_TILT = math.radians(-20.0)  # NEGATIVE = DOWN
```

**The Justification:** "verified from live imagery" — but we never re-examined whether that was a misread.

### What This Means If We're Wrong
- **All PnP poses are geometrically inverted** (gates would be above/below the real position)
- **The "gate 1" we acquire in THREAD may not be gate 1** — it could be gate 2, a phantom, or the floor structure
- **The orbit signature (bearing drifting right while yawing right)** could be us chasing something that's NOT on the planned path

## What the Spec Also Says

From vq2_spec.md:
```
## Gap analysis vs our stack (what VQ2 forces)
### Must build / change
1. **IMU-only attitude estimation (AHRS)** — no more ATTITUDE
   [✓ We have AttitudeEstimator in fly.py]
4. **Robust detector for complex env** — VQ1 classical HSV likely insufficient
   [⚠️ Our detector still uses classical HSV color masks + contour filtering]
```

## The PyAIPilotExample-v2 Baseline

The reference controller does **nothing**:
- `vision_rx.py` receives frames but has empty `process_frame()`
- `controller.py` sends constant `PITCH_RATE = -0.3`, `THRUST = 0.6`
- It demonstrates the MAVLink wire protocol only

**It proves:** The sim works, commands are honored, frames arrive. Our problem is **not the comms.**

## What We Know Works (100%)

1. **Rate + thrust control:** Calibrated at −2.55 gain (fly.py verified)
2. **Gate 0 threading:** Proven in fly.py (yaw servo + CY servo + bank + gate-0-specific logic)
3. **IMU accel/gyro:** Valid in HIGHRES_IMU
4. **Frame reception:** 30 Hz stable, JPEG decoding works

## What We DON'T Know (and Haven't Validated)

1. **Camera tilt:** Is it UP or DOWN? The code comment says DOWN, the spec says UP.
2. **PnP accuracy:** If tilt is wrong, pose_cam is wrong, and bearing/cy are computed wrong.
3. **Gate detection robustness:** Are we detecting the SAME gate consistently, or gate-hopping to nearby structures?
4. **Perception-to-control loop closure:** Do the detections match what we expect from the map? (We have no ground truth.)
5. **Is gate 1 even VISIBLE from the brake/scan altitude?** Camera 20° (up or down) + gate 0.6 m above us → geometry question.

## What Needs to Happen Next Session

### MUST VALIDATE (Week 1)
1. **Re-examine the camera tilt:**
   - Flip the sign in `gates.py` line 51 to `+20.0` (UP, per spec)
   - Run a single test on gate 0 (should still work or break consistently)
   - Check if gate 1 detection changes
   
2. **Visualize what the detector is actually finding:**
   - Add frame overlay showing detected contours, PnP poses, and bearing lines
   - Compare to expected geometry from the mined course map
   - Is the "z 12.6 m gate" really at the mapped location?

3. **Use the VQ1 sim (v1.0.3364) as a ground-truth rig:**
   - It still broadcasts ODOMETRY + ATTITUDE
   - Fly the same control logic in VQ1; compare PnP poses against ground truth
   - If PnP is correct in VQ1 (camera DOWN), it will break in VQ2 (camera UP)

### THEN SIMPLIFY (Week 2)
- **Scrap fly2.py's phased complexity** if needed
- **Copy fly.py's gate-0 logic EXACTLY** and apply it to gate 1
  - yaw servo on bearing
  - CY servo on image row
  - bank on bearing
  - No fancy "map-gated accept" or "racing line servo" — just the primitives
- **Fly gate 0 + gate 1 sequentially** with zero fancy state machines
- **Prove basic completion** before optimizing

### RESEARCH (Month-long effort, in parallel)
- YOLOv8-pose detector trained on VQ2 frames (classical HSV fails under dynamic lighting per spec)
- Proper AHRS validation (AttitudeEstimator vs ground truth)
- Racing line planning (cyan ribbon seems robust; make it the primary reference)
- Obstacle mapping from collision data

## Key Files to Review Next Session

- `docs/specs/vq2_spec.md` — camera tilt, coordinate frames, control interface
- `scripts/vq2/gates.py` lines 40–55 — camera transformation
- `scripts/vq2/fly.py` lines 730–1186 — gate threading logic (proven for gate 0)
- `VQ2/.../PyAIPilotExample-v2/mavlink_rx.py` — what's actually available from the sim
- `SIMULATOR/.../PyAIPilotExample/` — VQ1 reference (ground-truth rig)

## Summary

**Root cause is still unknown**, but the most likely culprits are:
1. **Camera tilt sign is wrong** (spec says UP, we implement DOWN)
2. **Perception is gate-hopping** (detecting structure, not the actual next gate)
3. **The "racing line servo" added in batch 6+ is masking a deeper problem** (makes height look better but doesn't fix the orbit)

**Next session action:**
- Deep research on camera tilt + PnP validation
- Visualize what we're actually detecting
- Validate against PyAIPilotExample-v2 spec
- Simplify to gate-0 logic applied to gate-1
- **Prove basic completion on a clean architecture**

**Time remaining:** 20 days. Recommend:
- Days 1–3: Diagnostic + camera tilt fix
- Days 4–7: Simplified gate-0-cloned-for-gate-1 controller
- Days 8–15: Basic course completion (all gates)
- Days 16–20: Obstacle avoidance + speed tuning
