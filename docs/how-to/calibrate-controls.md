# Calibrate controls

The VQ2 sim's rate commands are **inverted and ~2.5× scaled**:
`actual_gyro ≈ GAIN · commanded` with `GAIN = (roll −2.55, pitch −2.55, yaw −2.35)`,
linear, no cross-coupling. Every controller in this repo assumes the calibrated
boundary (`cmd = desired / GAIN`) is in place. **Hover thrust ≈ 0.30.**

## Verify the calibration (sim in Training, run live)

```bash
# 1. Raw mapping — should show the inversion/scaling above
uv run aigp-calibrate --test rates

# 2. Calibrated mapping — should show scale ≈ +0.98 (correct sign, ~unity) on all axes
uv run aigp-calibrate --test rates --calib

# 3. Hover and translation sanity checks
uv run aigp-calibrate --test hover
uv run aigp-calibrate --test translate
```

The gyro is ground truth for body rate — the test commands known rates and compares
against HIGHRES_IMU gyro readback.

## When to re-run

- After any sim build update (the gain was measured on v1.0.3379).
- Before the physical qualifier (real-drone gains will differ; the probe carries over).
- If turns start tumbling again: the historical multi-day tumble era was exactly this
  inversion — commanding +roll to fix rightward drift rolled the opposite way 2.5×
  too hard, so every correction diverged.

## Rules

- The calibration lives at **one** command boundary
  (`drone_racing.deployments.vq2.calibration`). Never apply it twice, and never
  hand-tune sign flips downstream of it — that is how the pre-calibration
  controllers accumulated compensating hacks.
- Changes to the calibrated boundary are safety-critical — see
  [safety](../reference/safety.md).
