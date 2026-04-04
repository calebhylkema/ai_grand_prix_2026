"""Command rate limiter — prevents instantaneous jumps in control outputs.

Sits between the controller (PID/MPCC/RL) and the flight controller to
enforce smooth command transitions.  Large instantaneous changes can
excite structural vibrations, trip safety monitors, or damage actuators.

Usage::

    limiter = CommandRateLimiter(config)
    for frame in loop:
        raw_cmd = controller.update(state, setpoint)
        safe_cmd = limiter.limit(raw_cmd)
        send(safe_cmd)

All rates are in units-per-second.  The limiter computes the maximum
allowed change per timestep as ``max_rate * dt``.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

from drone_racing.utils.types import DroneCommand

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class RateLimiterConfig:
    """Maximum allowed rates of change for each command channel.

    Parameters
    ----------
    max_throttle_rate : float
        Maximum throttle change per second (1/s).  Default 2.0 means
        full range (0→1) in 0.5 seconds.
    max_roll_rate_rate : float
        Maximum roll rate change per second (rad/s²).
    max_pitch_rate_rate : float
        Maximum pitch rate change per second (rad/s²).
    max_yaw_rate_rate : float
        Maximum yaw rate change per second (rad/s²).
    """

    max_throttle_rate: float = 2.0
    max_roll_rate_rate: float = 20.0
    max_pitch_rate_rate: float = 20.0
    max_yaw_rate_rate: float = 10.0


class CommandRateLimiter:
    """Limits the rate of change of drone commands.

    Parameters
    ----------
    config : RateLimiterConfig | None
        Rate limit parameters.  Uses defaults if None.
    """

    def __init__(self, config: RateLimiterConfig | None = None) -> None:
        self._cfg = config or RateLimiterConfig()
        self._last_cmd: DroneCommand | None = None

    def reset(self) -> None:
        """Clear the previous command (next call passes through unchanged)."""
        self._last_cmd = None

    def limit(self, cmd: DroneCommand) -> DroneCommand:
        """Apply rate limiting to a command.

        Parameters
        ----------
        cmd : DroneCommand
            Raw command from the controller.

        Returns
        -------
        DroneCommand
            Rate-limited command.  On the first call, the command passes
            through unchanged (no previous reference to limit against).
        """
        if self._last_cmd is None:
            self._last_cmd = cmd
            return cmd

        # Compute dt from timestamps
        dt_ns = cmd.timestamp_ns - self._last_cmd.timestamp_ns
        if dt_ns <= 0:
            # Same or backwards timestamp — return previous command
            return self._last_cmd

        dt = dt_ns * 1e-9

        # Apply rate limiting to each channel
        throttle = _clamp_rate(
            cmd.throttle, self._last_cmd.throttle, self._cfg.max_throttle_rate, dt
        )
        roll_rate = _clamp_rate(
            cmd.roll_rate, self._last_cmd.roll_rate, self._cfg.max_roll_rate_rate, dt
        )
        pitch_rate = _clamp_rate(
            cmd.pitch_rate, self._last_cmd.pitch_rate, self._cfg.max_pitch_rate_rate, dt
        )
        yaw_rate = _clamp_rate(
            cmd.yaw_rate, self._last_cmd.yaw_rate, self._cfg.max_yaw_rate_rate, dt
        )

        limited = DroneCommand(
            timestamp_ns=cmd.timestamp_ns,
            throttle=throttle,
            roll_rate=roll_rate,
            pitch_rate=pitch_rate,
            yaw_rate=yaw_rate,
        )

        self._last_cmd = limited
        return limited


def _clamp_rate(desired: float, previous: float, max_rate: float, dt: float) -> float:
    """Clamp a value so its rate of change does not exceed max_rate.

    Parameters
    ----------
    desired : float
        Target value.
    previous : float
        Previous value.
    max_rate : float
        Maximum allowed change per second.
    dt : float
        Time step (seconds).

    Returns
    -------
    float
        Clamped value.
    """
    max_delta = max_rate * dt
    delta = desired - previous
    clamped_delta = float(np.clip(delta, -max_delta, max_delta))
    return previous + clamped_delta
