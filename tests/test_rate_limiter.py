"""Tests for drone_racing.control.rate_limiter — command rate limiting.

Test strategy:
    1. First command passes through unchanged.
    2. Small change within limits passes through.
    3. Large throttle jump is clamped.
    4. Large rate jumps are clamped.
    5. Multiple steps converge to target.
    6. Backwards timestamp returns previous command.
    7. Reset allows next command to pass through.
    8. Negative direction clamping works.
"""

from __future__ import annotations

import pytest

from drone_racing.control.rate_limiter import (
    CommandRateLimiter,
    RateLimiterConfig,
    _clamp_rate,
)
from drone_racing.utils.types import DroneCommand

# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------


def _cmd(
    timestamp_ns: int,
    throttle: float = 0.5,
    roll_rate: float = 0.0,
    pitch_rate: float = 0.0,
    yaw_rate: float = 0.0,
) -> DroneCommand:
    return DroneCommand(
        timestamp_ns=timestamp_ns,
        throttle=throttle,
        roll_rate=roll_rate,
        pitch_rate=pitch_rate,
        yaw_rate=yaw_rate,
    )


# ---------------------------------------------------------------------------
# Unit: _clamp_rate
# ---------------------------------------------------------------------------


class TestClampRate:
    """Low-level rate clamping function."""

    def test_within_limit(self) -> None:
        """Small change passes through."""
        result = _clamp_rate(desired=1.0, previous=0.9, max_rate=10.0, dt=0.02)
        assert result == pytest.approx(1.0)

    def test_exceeds_limit_positive(self) -> None:
        """Large positive jump is clamped."""
        result = _clamp_rate(desired=1.0, previous=0.0, max_rate=2.0, dt=0.02)
        # max delta = 2.0 * 0.02 = 0.04
        assert result == pytest.approx(0.04)

    def test_exceeds_limit_negative(self) -> None:
        """Large negative jump is clamped."""
        result = _clamp_rate(desired=0.0, previous=1.0, max_rate=2.0, dt=0.02)
        # max delta = -0.04
        assert result == pytest.approx(0.96)

    def test_exact_limit(self) -> None:
        """Change exactly at the limit passes through."""
        result = _clamp_rate(desired=0.1, previous=0.0, max_rate=5.0, dt=0.02)
        # max delta = 0.1, desired delta = 0.1 → exact match
        assert result == pytest.approx(0.1)


# ---------------------------------------------------------------------------
# CommandRateLimiter
# ---------------------------------------------------------------------------


class TestRateLimiterBasic:
    """Basic rate limiter behaviour."""

    def test_first_command_passes_through(self) -> None:
        """First command has no previous reference → passes unchanged."""
        limiter = CommandRateLimiter()
        cmd = _cmd(1_000_000, throttle=0.8, roll_rate=2.0)
        result = limiter.limit(cmd)
        assert result.throttle == pytest.approx(0.8)
        assert result.roll_rate == pytest.approx(2.0)

    def test_small_change_passes(self) -> None:
        """Change within rate limits passes through unchanged."""
        config = RateLimiterConfig(max_throttle_rate=2.0)
        limiter = CommandRateLimiter(config)

        limiter.limit(_cmd(0, throttle=0.5))
        # 20ms later, throttle changes by 0.01 → rate = 0.5/s, well under 2.0/s
        result = limiter.limit(_cmd(20_000_000, throttle=0.51))
        assert result.throttle == pytest.approx(0.51)

    def test_large_throttle_jump_clamped(self) -> None:
        """Throttle jump from 0.5 to 1.0 in 20ms is clamped."""
        config = RateLimiterConfig(max_throttle_rate=2.0)
        limiter = CommandRateLimiter(config)

        limiter.limit(_cmd(0, throttle=0.5))
        result = limiter.limit(_cmd(20_000_000, throttle=1.0))
        # max delta = 2.0 * 0.02 = 0.04
        assert result.throttle == pytest.approx(0.54)

    def test_large_roll_rate_jump_clamped(self) -> None:
        """Roll rate jump from 0 to 8 rad/s in 20ms is clamped."""
        config = RateLimiterConfig(max_roll_rate_rate=20.0)
        limiter = CommandRateLimiter(config)

        limiter.limit(_cmd(0, roll_rate=0.0))
        result = limiter.limit(_cmd(20_000_000, roll_rate=8.0))
        # max delta = 20.0 * 0.02 = 0.4
        assert result.roll_rate == pytest.approx(0.4)

    def test_backwards_timestamp_returns_previous(self) -> None:
        """Command with earlier timestamp returns previous command."""
        limiter = CommandRateLimiter()
        cmd1 = _cmd(100_000_000, throttle=0.5)
        limiter.limit(cmd1)

        cmd2 = _cmd(50_000_000, throttle=0.9)  # Earlier timestamp
        result = limiter.limit(cmd2)
        assert result.throttle == pytest.approx(0.5)


class TestRateLimiterConvergence:
    """Rate limiter converges to target over multiple steps."""

    def test_converges_to_target(self) -> None:
        """Repeated commands at the target eventually reach it."""
        config = RateLimiterConfig(max_throttle_rate=2.0)
        limiter = CommandRateLimiter(config)

        dt_ns = 20_000_000  # 20ms
        target = 1.0
        limiter.limit(_cmd(0, throttle=0.0))

        throttle = 0.0
        for i in range(1, 100):
            result = limiter.limit(_cmd(i * dt_ns, throttle=target))
            throttle = result.throttle

        # Should have converged to target
        assert throttle == pytest.approx(target, abs=0.01)

    def test_negative_direction_converges(self) -> None:
        """Converges when moving throttle downward."""
        config = RateLimiterConfig(max_throttle_rate=2.0)
        limiter = CommandRateLimiter(config)

        dt_ns = 20_000_000
        limiter.limit(_cmd(0, throttle=1.0))

        throttle = 1.0
        for i in range(1, 100):
            result = limiter.limit(_cmd(i * dt_ns, throttle=0.0))
            throttle = result.throttle

        assert throttle == pytest.approx(0.0, abs=0.01)


class TestRateLimiterReset:
    """Reset behaviour."""

    def test_reset_clears_state(self) -> None:
        """After reset, next command passes through."""
        limiter = CommandRateLimiter()
        limiter.limit(_cmd(0, throttle=0.0))
        limiter.reset()

        result = limiter.limit(_cmd(20_000_000, throttle=1.0))
        assert result.throttle == pytest.approx(1.0)

    def test_all_channels_limited(self) -> None:
        """All four channels are independently rate-limited."""
        config = RateLimiterConfig(
            max_throttle_rate=1.0,
            max_roll_rate_rate=5.0,
            max_pitch_rate_rate=5.0,
            max_yaw_rate_rate=5.0,
        )
        limiter = CommandRateLimiter(config)

        limiter.limit(_cmd(0, throttle=0.5, roll_rate=0.0, pitch_rate=0.0, yaw_rate=0.0))
        result = limiter.limit(
            _cmd(100_000_000, throttle=1.0, roll_rate=10.0, pitch_rate=-10.0, yaw_rate=5.0)
        )
        # dt = 0.1s
        # throttle: max delta = 0.1, desired 0.5 → 0.6
        assert result.throttle == pytest.approx(0.6)
        # roll: max delta = 0.5, desired 10.0 → 0.5
        assert result.roll_rate == pytest.approx(0.5)
        # pitch: max delta = -0.5, desired -10.0 → -0.5
        assert result.pitch_rate == pytest.approx(-0.5)
        # yaw: max delta = 0.5, desired 5.0 → 0.5
        assert result.yaw_rate == pytest.approx(0.5)
