# VQ2 experiments (archived)

Superseded VQ2 controllers and probes. The **active** VQ2 deployment is
`src/drone_racing/deployments/vq2/` (entry point: `aigp-vq2`).

| Directory | What it was | Why superseded |
| --- | --- | --- |
| `race_v1/` | First VQ2 controller (`race.py`, 2026-06-30): gate-0 threading 6/6, collision instrumentation. `estimate_test.py` was its live estimator-validation probe. | Guidance poisoned by diverging ESKF attitude past gate 0; replaced by the reactive controller |
| `reactive_fly/` | Reactive controller (`fly.py`, 2026-07-01→09): calibrated rates, visual vertical PD servo, aim-then-go, ESKF position arc. Gate 0 100% @ 2.2 s; never passed gate 1. Its reusable runtime (Flight, calibrated command boundary, level-rate controller, trace recorder) was extracted into the active deployment. | Replaced by the map-and-plan mission (course map + phased executor) |
| `probes/` | One-off diagnostics: `control_probe`, `eskf_synthetic_test`, `_pnp_probe`, `_line_probe`, `_lab_gates_check`, `mask_peek`, `example_probe` (annotated official PyAIPilotExample-v2 reference). | Findings recorded in docs/memory; probes kept for reference only |

All were last runnable at commit `2b1d566` (original layout). Not collected by CI.
