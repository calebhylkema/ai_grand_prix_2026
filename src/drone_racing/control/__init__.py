"""Control module: PID cascade, MPCC adapter, rate limiting, and safety ladder."""

from drone_racing.control.pid import PIDController, TrajectorySetpoint
from drone_racing.control.rate_limiter import CommandRateLimiter, RateLimiterConfig

__all__ = [
    "CommandRateLimiter",
    "PIDController",
    "RateLimiterConfig",
    "TrajectorySetpoint",
]

# MPCC adapter requires casadi (optional dependency)
try:
    from drone_racing.control.mpcc_adapter import MPCCController, MPCCControllerStats

    __all__ += ["MPCCController", "MPCCControllerStats"]
except ImportError:
    pass

# RL controller requires stable-baselines3 + torch (optional dependency)
try:
    from drone_racing.control.rl_controller import RLController, RLControllerStats

    __all__ += ["RLController", "RLControllerStats"]
except ImportError:
    pass
