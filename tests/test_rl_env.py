"""Tests for drone_racing.control.rl_env — Gymnasium RL environment.

Test strategy:
    1. Environment creation and reset.
    2. Observation and action spaces are correct shapes.
    3. Step returns valid types and shapes.
    4. Episode terminates on crash or time limit.
    5. Reward is finite.
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

gymnasium = pytest.importorskip("gymnasium", reason="gymnasium not installed (optional [rl] dep)")

from drone_racing.control.rl_env import (  # noqa: E402
    DomainRandomizationConfig,
    DroneRacingEnv,
    RLEnvConfig,
)
from drone_racing.planning.spline import build_racing_line  # noqa: E402
from drone_racing.sim.mock_adapter import MockSimConfig  # noqa: E402
from drone_racing.utils.types import (  # noqa: E402
    CameraConfig,
    GateWaypoint,
    PlannerConfig,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _build_env(
    max_episode_steps: int = 100,
    randomization: DomainRandomizationConfig | None = None,
) -> DroneRacingEnv:
    """Build a minimal RL environment with a 4-gate circuit."""
    corners_local = np.array(
        [[-0.75, 0.75, 0.0], [0.75, 0.75, 0.0], [0.75, -0.75, 0.0], [-0.75, -0.75, 0.0]],
        dtype=np.float64,
    )

    gate_defs = [
        (np.array([10.0, 0.0, 3.0]), 0.0),
        (np.array([10.0, 10.0, 3.0]), np.pi / 2),
        (np.array([0.0, 10.0, 3.0]), np.pi),
        (np.array([0.0, 0.0, 3.0]), -np.pi / 2),
    ]

    gates: list[GateWaypoint] = []
    for i, (pos, yaw) in enumerate(gate_defs):
        ori = Rotation.from_euler("yz", [np.pi / 2, yaw]).as_quat()
        r_wg = Rotation.from_quat(ori)
        corners_w = (r_wg.apply(corners_local) + pos).astype(np.float64)
        gates.append(
            GateWaypoint(
                gate_id=i,
                position_w=pos,
                orientation_w=ori.astype(np.float64),
                corners_3d_w=corners_w,
                order=i,
            )
        )

    gate_positions = np.array([g.position_w for g in gates])
    gate_orientations = np.array([g.orientation_w for g in gates])
    racing_line = build_racing_line(
        gate_positions, gate_orientations, PlannerConfig(), num_points=100, closed=True
    )

    env_cfg = RLEnvConfig(
        max_episode_steps=max_episode_steps,
        sim_steps_per_action=10,  # Faster for tests
        randomization=randomization or DomainRandomizationConfig(),
    )
    sim_cfg = MockSimConfig(max_time_s=30.0)
    cam = CameraConfig(width=160, height=120)

    return DroneRacingEnv(
        gates=gates,
        racing_line=racing_line,
        config=env_cfg,
        camera_config=cam,
        sim_config=sim_cfg,
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestEnvCreation:
    """Environment initialisation and spaces."""

    def test_observation_space_shape(self) -> None:
        env = _build_env()
        assert env.observation_space.shape == (14,)

    def test_action_space_shape(self) -> None:
        env = _build_env()
        assert env.action_space.shape == (4,)

    def test_action_space_bounds(self) -> None:
        env = _build_env()
        np.testing.assert_allclose(env.action_space.low, -1.0)
        np.testing.assert_allclose(env.action_space.high, 1.0)


class TestReset:
    """Environment reset."""

    def test_reset_returns_obs_and_info(self) -> None:
        env = _build_env()
        obs, info = env.reset(seed=42)
        assert obs.shape == (14,)
        assert obs.dtype == np.float32
        assert isinstance(info, dict)

    def test_reset_obs_is_finite(self) -> None:
        env = _build_env()
        obs, _ = env.reset(seed=42)
        assert np.all(np.isfinite(obs))


class TestStep:
    """Environment stepping."""

    def test_step_returns_correct_types(self) -> None:
        env = _build_env()
        env.reset(seed=42)
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)

        assert obs.shape == (14,)
        assert isinstance(reward, float)
        assert isinstance(terminated, bool)
        assert isinstance(truncated, bool)
        assert isinstance(info, dict)

    def test_step_obs_is_finite(self) -> None:
        env = _build_env()
        env.reset(seed=42)
        for _ in range(10):
            action = env.action_space.sample()
            obs, reward, terminated, truncated, _ = env.step(action)
            assert np.all(np.isfinite(obs))
            assert np.isfinite(reward)
            if terminated or truncated:
                break

    def test_hover_action_does_not_crash(self) -> None:
        """Neutral action (hover) should keep the drone alive for a while."""
        env = _build_env(max_episode_steps=50)
        env.reset(seed=42)
        # Hover action: throttle=0 mapped → 0.5, zero rates
        hover = np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float32)
        for _ in range(20):
            obs, reward, terminated, truncated, _ = env.step(hover)
            if terminated or truncated:
                break
        # Should survive at least a few steps
        assert env._step_count >= 5

    def test_info_contains_progress(self) -> None:
        env = _build_env()
        env.reset(seed=42)
        action = np.zeros(4, dtype=np.float32)
        _, _, _, _, info = env.step(action)
        assert "progress" in info
        assert "contour_error" in info
        assert "speed" in info
        assert "gates_passed" in info


class TestEpisodeTermination:
    """Episode ends on truncation or crash."""

    def test_truncation_at_max_steps(self) -> None:
        env = _build_env(max_episode_steps=10)
        env.reset(seed=42)
        hover = np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float32)
        truncated = False
        for _ in range(20):
            _, _, terminated, truncated, _ = env.step(hover)
            if terminated or truncated:
                break
        assert truncated or terminated  # Should end by step 10


class TestDomainRandomization:
    """Domain randomization for sim-to-real robustness."""

    def test_disabled_is_deterministic(self) -> None:
        """With randomization disabled, two seeded resets produce identical obs."""
        env_a = _build_env(randomization=DomainRandomizationConfig(enabled=False))
        env_b = _build_env(randomization=DomainRandomizationConfig(enabled=False))
        obs_a, _ = env_a.reset(seed=123)
        obs_b, _ = env_b.reset(seed=123)
        np.testing.assert_allclose(obs_a, obs_b)

    def test_enabled_varies_across_seeds(self) -> None:
        """Different seeds produce different start observations."""
        rand = DomainRandomizationConfig(enabled=True)
        env = _build_env(randomization=rand)
        obs_a, _ = env.reset(seed=1)
        obs_b, _ = env.reset(seed=2)
        # Start pos/vel/yaw should differ
        assert not np.allclose(obs_a, obs_b)

    def test_enabled_is_reproducible(self) -> None:
        """Same seed yields identical observations even with randomization on."""
        rand = DomainRandomizationConfig(enabled=True)
        env_a = _build_env(randomization=rand)
        env_b = _build_env(randomization=rand)
        obs_a, _ = env_a.reset(seed=7)
        obs_b, _ = env_b.reset(seed=7)
        np.testing.assert_allclose(obs_a, obs_b)

    def test_start_jitter_bounded(self) -> None:
        """Sampled start position stays within the configured half-width."""
        rand = DomainRandomizationConfig(
            enabled=True,
            start_pos_jitter_m=0.5,
            start_vel_jitter_mps=0.3,
            start_yaw_jitter_rad=0.2,
            drag_scale_range=0.0,
            angular_drag_scale_range=0.0,
            imu_noise_scale_range=0.0,
        )
        nominal_env = _build_env(randomization=DomainRandomizationConfig(enabled=False))
        nominal_obs, _ = nominal_env.reset(seed=0)
        nominal_pos = nominal_obs[:3]

        env = _build_env(randomization=rand)
        for seed in range(5):
            obs, _ = env.reset(seed=seed)
            pos = obs[:3]
            vel = obs[3:6]
            assert np.all(np.abs(pos - nominal_pos) <= 0.5 + 1e-6)
            assert np.all(np.abs(vel) <= 0.3 + 1e-6)

    def test_physics_randomization_affects_dynamics(self) -> None:
        """Large drag randomization changes observed trajectories under same action."""
        rand = DomainRandomizationConfig(
            enabled=True,
            start_pos_jitter_m=0.0,
            start_vel_jitter_mps=0.0,
            start_yaw_jitter_rad=0.0,
            drag_scale_range=0.8,
            angular_drag_scale_range=0.8,
            imu_noise_scale_range=0.0,
        )
        env = _build_env(randomization=rand)
        thrust = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)

        positions = []
        for seed in (0, 1, 2):
            env.reset(seed=seed)
            for _ in range(20):
                obs, _, term, trunc, _ = env.step(thrust)
                if term or trunc:
                    break
            positions.append(obs[:3].copy())

        # At least one pair should differ (drag scaling changes trajectory)
        diffs = [
            np.linalg.norm(positions[i] - positions[j]) for i in range(3) for j in range(i + 1, 3)
        ]
        assert max(diffs) > 1e-3
