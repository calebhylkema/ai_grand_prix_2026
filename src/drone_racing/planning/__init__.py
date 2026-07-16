"""Planning module: racing line, MPCC trajectory optimization, and dynamics."""

from drone_racing.planning.spline import (
    GatePassageReport,
    RacingLine,
    build_racing_line,
    evaluate_gate_passages,
)

__all__ = [
    "RacingLine",
    "GatePassageReport",
    "build_racing_line",
    "evaluate_gate_passages",
]

# MPCC/dynamics require casadi (optional dependency)
try:
    from drone_racing.planning.dynamics import DynamicsConfig, VQ2DynamicsConfig
    from drone_racing.planning.mpcc import MPCCConfig, MPCCSolution, MPCCSolver

    __all__ += [
        "DynamicsConfig",
        "VQ2DynamicsConfig",
        "MPCCConfig",
        "MPCCSolution",
        "MPCCSolver",
    ]
except ImportError:
    pass
