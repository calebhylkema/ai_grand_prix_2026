# 0002 — VQ2 Obstacle Avoidance: racing-line precision + reactive net, NOT full 3D avoidance

Date: 2026-07-01
Status: Accepted

## Context

VQ2 adds obstacles (the spec lists vertical/horizontal obstacles, boundary elements,
terrain; in-sim we see dark "Stadium" pillars flanking gates). Our drone reliably
threads gate 0 (vision + IMU only, via the ESKF + geometric guidance) but fails at
gate 1 (a right/down turn), hitting the *environment* (`COLLISION id=1002`). The
question: should we build **full obstacle avoidance**?

We researched the SOTA for monocular autonomous FPV obstacle avoidance:

- **Depth-learned end-to-end** — Loquercio et al., *Learning High-Speed Flight in the
  Wild* (arXiv:2110.05113): best performance, but requires a depth sensor in training +
  heavy learned sim-to-real. Monocular depth is the hard, unreliable part.
- **Optical-flow reactive** — *Seeing Through Pixel Motion* (arXiv:2411.04413) and the
  optical-flow/time-to-contact literature: monocular, but the paper itself reports
  flow-based **underperforms depth (60% vs 90%)**, is **weak on head-on obstacles** (no
  cue near the focus-of-expansion), and the strong versions still need learning.
- **RL-with-random-obstacles** — A2RL 2026 (our direct analog: DCL-partnered, "single
  camera + IMU, vision-only, ~90 mph"; e.g. Curriculum RL arXiv:2602.24030): the
  competitive path, but needs the RL training pipeline.

**Two findings decided this:** (1) in racing, the **racing line through gate centers
inherently avoids the flanking obstacles** — explicit avoidance is for *dynamic* / on-line
obstacles. (2) Our *actual* dominant gate-1 failures are **perception data-association
fly-offs** (the drone chasing the largest red blob = a far/wrong gate) and **threading
precision** (clipping the 1.5 m opening / floor at the low gate) — **NOT collisions with
obstacles on the line.** Adding reactive obstacle avoidance was verified neutral for
gate 1 (10/10 still reach it; 0 extra passes) — confirming the failure is precision.

## Decision

Do **not** build full 3D / learned-depth / RL obstacle avoidance for VQ2. Instead, a
layered approach matched to our monocular, non-learned stack:

1. **Perception data-association** (implemented): track the **nearest** gate via PnP
   (`closest_gate_pose`) instead of the largest blob — stops fly-offs to far gates.
2. **Racing-line precision** (primary work): thread gate centers cleanly with the
   ESKF + geometric guidance → inherently clears the flanking pillars.
3. **Lightweight reactive safety net** (implemented): classical optical-flow *expansion*
   (time-to-contact) cue (`compute_looming`) that shifts the guidance aim-point away from
   a genuinely looming obstacle. Gate≥1, low gain, bounded — a safety layer, not primary
   navigation. Neutral in the current sim (obstacles aren't on the line) but the correct,
   sensor-appropriate mechanism for real obstacles (physical qualifier).
4. **Defer** learned depth / RL obstacle avoidance to the finals investment (aligned with
   A2RL SOTA), only if obstacles are confirmed *on* the racing line.

## Consequences

- Gate-1 effort is redirected to **threading precision** (control/estimation), the actual
  blocker — not to a heavy avoidance system that wouldn't help.
- We gain robustness (no fly-offs) and a real-obstacle safety net cheaply, with no learning
  and no depth sensor.
- For the physical qualifier, the reactive net + precise line is the qualifying-grade plan;
  learned/RL avoidance is the finals stretch goal, not a prerequisite.

Files (2026-07-09 restructure note): the original implementation lived in the
race-v1 controller, now `archive/vq2_experiments/race_v1/race.py`
(`closest_gate_pose`, `compute_looming`). The decision itself still governs: the
active deployment (`drone_racing.deployments.vq2`) relies on racing-line precision
+ map-gated association, not 3D avoidance.
