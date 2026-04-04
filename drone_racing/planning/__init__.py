"""Planning module: racing line, MPCC trajectory optimization, and dynamics."""

from drone_racing.planning.spline import RacingLine, build_racing_line

__all__ = [
    "RacingLine",
    "build_racing_line",
]

# MPCC/dynamics require casadi (optional dependency)
try:
    from drone_racing.planning.dynamics import DynamicsConfig
    from drone_racing.planning.mpcc import MPCCConfig, MPCCSolution, MPCCSolver

    __all__ += ["DynamicsConfig", "MPCCConfig", "MPCCSolution", "MPCCSolver"]
except ImportError:
    pass
