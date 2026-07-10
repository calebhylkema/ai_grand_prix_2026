# VQ1 round (archived)

**What it was:** Virtual Qualifier 1 (May 2026) — completion-focused round, < 10 gates,
full telemetry available (`ATTITUDE`, `LOCAL_POSITION_NED`, `ODOMETRY`, `GATE_INFO`).

- **Simulator:** AI-GP Simulator v1.0.3364 (kept locally at `.local/simulators/vq1/`
  as a ground-truth validation rig — it still exposes the telemetry VQ2 removed)
- **Original entry point:** `scripts/vq1/run_race.py`
- **Final known status:** ✅ full course completion in ~30 s (2026-06-02), using
  SET_ATTITUDE_TARGET quaternion control, lead-pursuit + roll cross-track steering,
  PD altitude, finite-difference velocity, full failsafe suite
- **Why archived:** VQ2 (2026-06-30) removed the telemetry this stack depends on;
  none of its guidance can run on IMU+camera only
- **Runnable state:** commit `2b1d566` (original layout, before the 2026-07-09 restructure)
- **CI:** not collected, not supported

`code/` also contains the ground-truth probes (`axis_sign_probe`, `hover_probe`,
`sim_probe`) used to characterize sim behavior against the VQ1 rig.
`specifications/` holds the official VADR-TS-001 PDF and its curated summary.
