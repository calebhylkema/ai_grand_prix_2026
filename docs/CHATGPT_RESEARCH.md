# AI Grand Prix --- CHATGPT's plan

**Note:** This is just a second source. Use your judgment to see if this or your original plan is better, you dont need to use this. Use it as a reference if needed to get more info. 

------------------------------------------------------------------------

# 1. Competition Overview

The AI Grand Prix is a fully autonomous drone racing competition hosted
by Anduril Industries in partnership with the Drone Champions League
(DCL).

Teams develop Python-based autonomy stacks that navigate standardized 3D
racecourses as fast as possible.

Fastest valid run wins. No hardware advantages. No human piloting.

------------------------------------------------------------------------

# 2. Critical Technical Constraints (From Official Emails Provided)

## Simulator Provides

-   Forward-facing RGB visual stream
-   Telemetry:
    -   Starting position
    -   Velocity
    -   Orientation
-   Control interface:
    -   Throttle
    -   Roll
    -   Pitch
    -   Yaw
-   Python API (Python 3.14.2 confirmed working)
-   External libraries allowed
-   Acceleration layers permitted

## Simulator Does NOT Provide

-   No depth information
-   No engine RPM data
-   No battery SoC
-   No manual steering even for training

------------------------------------------------------------------------

# 3. Virtual Qualifier Structure

## Virtual Qualifier 1 (VQ1)

-   Desaturated environment
-   Highlighted gates
-   High signal-to-noise ratio
-   Visual aids possibly active

## Virtual Qualifier 2 (VQ2)

-   More complex environment
-   Lighting distractions
-   Lower SNR
-   Visual aids OFF

------------------------------------------------------------------------

# 4. Winning Design Philosophy

1.  Deterministic infrastructure first
2.  Latency is performance
3.  Adapter boundary protects core autonomy
4.  Finish-rate baseline always exists
5.  Aggressive mode layered on top
6.  Exploit telemetry in virtual phase
7.  Remain monocular-only capable for physical stage

------------------------------------------------------------------------

# 5. Recommended Repository Architecture

    aigp-autonomy/
    │
    ├── autonomy/
    │   ├── interfaces/
    │   ├── adapters/
    │   │   └── dcl/
    │   ├── perception/
    │   ├── estimation/
    │   ├── planning/
    │   ├── control/
    │   ├── runtime/
    │   └── utils/
    │
    ├── configs/
    ├── tools/
    ├── tests/
    ├── requirements/
    ├── docs/
    └── .github/workflows/

------------------------------------------------------------------------

# 6. System Architecture

    DCL Adapter
        ↓
    Observation (Camera + Telemetry)
        ↓
    Perception (Gate Detection)
        ↓
    Geometry (PnP Pose Estimation)
        ↓
    Gate Map
        ↓
    Planner (Racing Line + Receding Horizon)
        ↓
    Controller (NMPC / Tracker)
        ↓
    Command Translator
        ↓
    Throttle / Roll / Pitch / Yaw

------------------------------------------------------------------------

# 7. Perception Strategy

## VQ1 Mode

-   Classical CV (thresholding, contour detection, quad fitting)
-   Optimized for latency

## VQ2 Mode

-   Learned gate detector / segmentation model
-   SolvePnP using known gate geometry
-   Robust to lighting and clutter

Monocular only. No depth allowed.

------------------------------------------------------------------------

# 8. Estimation Strategy

If telemetry provides full pose: - Use passthrough

If not: - EKF/UKF - Fuse vision pose + IMU - Maintain gate map

------------------------------------------------------------------------

# 9. Planning Strategy

-   Global spline through gates
-   Curvature-based speed profile
-   Receding horizon refinement
-   Time-optimal bias
-   Constraint-aware tracking

Advanced upgrade: - MPCC-style contouring control - Bayesian
hyperparameter tuning

------------------------------------------------------------------------

# 10. Control Strategy

Phase 1: - PID/LQR baseline

Phase 2: - NMPC

Optional: - RL policy hook - Safety fallback ladder

------------------------------------------------------------------------

# 11. Infrastructure Requirements

## Determinism

-   Locked dependencies
-   Seed control
-   Reproducible logs

## CI

-   Windows + Linux builds
-   Unit + integration tests
-   Latency regression testing
-   Packaging validation

------------------------------------------------------------------------

# 12. Evaluation Metrics

Primary: - Lap time - Valid completion

Secondary: - Completion rate - Gate miss rate - Collision rate - Loop
latency (p95) - Perception latency (p95)

------------------------------------------------------------------------

# 13. Key Research & Information Sources

## Official Competition Sources

-   AI Grand Prix Official Site: https://www.theaigrandprix.com/
-   Official Rules: https://www.theaigrandprix.com/official-rules
-   Anduril Announcement:
    https://www.anduril.com/news/anduril-launches-the-ai-grand-prix-a-global-autonomous-drone-race
-   DCL Announcement:
    https://dronechampionsleague.com/theaigrandprix2026-announcement/

## Foundational Autonomous Drone Racing Research

### AlphaPilot (Autonomous Drone Racing)

RSS Paper: https://roboticsproceedings.org/rss16/p081.pdf arXiv:
https://arxiv.org/abs/2005.12813

### Swift --- Champion-Level Deep RL Drone Racing (Nature 2023)

PDF: https://www.nature.com/articles/s41586-023-06419-4.pdf

### MonoRace --- Robust Monocular AI Drone Racing

arXiv: https://arxiv.org/abs/2601.15222

### AirSim Drone Racing Lab

Paper: https://arxiv.org/abs/2003.05654 GitHub:
https://github.com/microsoft/AirSim-Drone-Racing-Lab

### MPCC / Time-Optimal Flight

arXiv: https://arxiv.org/abs/2403.17551

## Development Infrastructure References

-   pip-tools: https://pip-tools.readthedocs.io/
-   conda-lock: https://conda.github.io/conda-lock/
-   pyproject.toml Guide:
    https://packaging.python.org/en/latest/guides/writing-pyproject-toml/
-   GitHub Actions setup-python: https://github.com/actions/setup-python

------------------------------------------------------------------------

# 14. Operating Principle

Every module must answer:

-   Does this reduce lap time?
-   Does this reduce failure rate?
-   Does this reduce latency?
-   Does this improve robustness?

If not, it does not ship.
