# Documentation

Organized by the [Diátaxis](https://diataxis.fr/) model.

## Project

- **[Status](project/status.md)** ⭐ — the single authoritative "where are we" doc
- [Roadmap](project/roadmap.md) — future work by competition phase
- [Reorg report 2026-07](project/reorg-2026-07-report.md) — migration record: old→new paths, debt

## How-to guides

- [Run VQ2](how-to/run-vq2.md) — fly the active mission against the simulator
- [Calibrate controls](how-to/calibrate-controls.md) — verify the rate/thrust boundary
- [Diagnose a flight](how-to/diagnose-a-flight.md) ⭐ — trace-first debugging workflow
- [Train the gate detector](how-to/train-gate-detector.md)

## Reference

- [Architecture](reference/architecture.md) — package layout and data flow
- [Configuration](reference/configuration.md) — configs and workspace paths
- [Coordinate frames](reference/coordinate-frames.md) — NED/FRD/camera conventions
- [Command line](reference/command-line.md) — supported `aigp-*` commands
- [Safety](reference/safety.md) — failsafes and the do-not-weaken rule
- [VQ2 specification](reference/specifications/vq2/summary.md) — curated official-spec summary

## Explanation

- [Perception](explanation/perception.md) — gate detection approaches and lessons
- [State estimation](explanation/state-estimation.md) — ESKF design and pitfalls
- [Trajectory & control](explanation/trajectory-control.md) — control ladder, MPCC, RL

## Decisions

Active architecture decision records in [decisions/](decisions/). Round-specific or
superseded decisions live with their round in `archive/`.
