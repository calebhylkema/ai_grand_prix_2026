# Perception

How we find gates (and the racing line) in monocular RGB, and what we learned
the hard way.

## Detection backends

1. **Classical CV** (active in VQ2): HSV thresholding → contour analysis → quad
   fitting. ~5 ms/frame. General-scene accuracy is poor (~35% in the literature) but
   the sim's highlighted gates make it viable — *with* the hardening below.
2. **Neural (YOLOv8-nano keypoint)** (built, pre-sim; planned VQ2 upgrade if
   lighting/clutter defeats classical): predicts 4–8 gate corners directly,
   ~99% accuracy, <17 ms via ONNX Runtime (2–3× faster than raw PyTorch on Jetson
   via TensorRT). The A2RL winner (MonoRace) used the heavier U-Net GateSeg +
   QuAdGate corner pipeline.

Both emit corner sets ordered clockwise from top-left, with capture (not
processing) timestamps. A tracker assigns stable gate IDs across frames
(Hungarian matching on IoU).

## Hard-won VQ2 lessons (each traced to crashes)

- **The glow trap:** gates cast a huge diffuse red glow on nearby walls/pillars/
  floor; blob centroids AND naive quad fits lock onto the glow. Fix: a gate is a
  red **ring around a non-red core** (glow reads solid: core-fill ~0.9 vs a real
  gate ~0.4). Reject border-touching contours (partial gate → garbage PnP);
  PnP only from quads ≥ 800 px².
- **One image row can hold two course branches** — averaging across it mixes them.
  The cyan-racing-line aim must ridge-walk laterally-continuous pixels from the
  near anchor, and fat near-range line blobs need an anchored-blob exception
  (holograms float mid-frame; the real line touches the frame bottom).
- **Data association is the failure mode, not detection.** The tracker once
  latched three different gates within 0.3 s (PnP z 27→21→17 m — an impossible
  35 m/s closure). Range-band + turn-direction gating from the course map is
  mandatory (ADR-0003).
- Validate detectors **offline** over captured frame corpora before flying them —
  1000+ real frames catch what synthetic tests miss.

## PnP practicalities

- OpenCV `solvePnP` with the known 1.5 m gate square; needs ≥4 non-collinear
  correspondences.
- Most reliable at 2–5 m with 6+ corners (MonoRace finding). Far: small rotation
  errors → large position errors. Close: motion blur degrades corners, and
  close-range lateral reads are unreliable (a centred gate once read bearing +26°)
  — hence pass-through commit inside ~4 m.
- When full PnP is shaky, the de-rotated method (filter attitude + PnP translation
  only) is the standard fallback.
- ADR-VINS (arXiv:2603.02742) integrates corner reprojection directly in the ESKF
  (works from 2 corners, 20.9 m/s at A2RL) — the upgrade path if PnP itself becomes
  the bottleneck.

## Key references

- MonoRace (arXiv:2601.15222) — monocular+IMU A2RL winner, 28 m/s; our closest precedent
- ADR-VINS drift-corrected VIO (arXiv:2512.20475)
- Tightly-coupled ESKF racing (arXiv:2603.02742)
- AlphaPilot lessons (arXiv:2005.12813) — use ALL visible gates, reliability first
