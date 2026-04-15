"""Gymnasium environment wrapping the mock simulator for RL policy training.

Observation space (14-D continuous):
    [position(3), velocity(3), orientation_euler(3),
     next_gate_relative(3), progress(1), speed(1)]

Action space (4-D continuous):
    [throttle, roll_rate, pitch_rate, yaw_rate]  — normalised to [-1, 1]

Reward:
    +progress along the racing line per step
    -contour error penalty (distance from racing line)
    -crash penalty (terminal)
    +gate passage bonus

The environment is designed to work with the MockSimAdapter for offline training,
and can later be swapped for the real simulator with minimal changes.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces
from scipy.spatial.transform import Rotation

from drone_racing.sim.mock_adapter import MockSimAdapter, MockSimConfig
from drone_racing.utils.transforms import quat_identity
from drone_racing.utils.types import (
    CameraConfig,
    ControlConfig,
    DroneCommand,
    GateWaypoint,
    Telemetry,
)

if TYPE_CHECKING:
    from drone_racing.planning.spline import RacingLine

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class DomainRandomizationConfig:
    """Per-episode randomization ranges for sim-to-real robustness.

    All values are *half-widths* — sampled uniformly from [-x, +x] (or scaled
    multiplicatively for ``*_scale_range``).  Set to 0 / 1 to disable.

    Parameters
    ----------
    enabled : bool
        Master switch.  When False, all sampling collapses to zero/identity.
    start_pos_jitter_m : float
        Half-width for additive position jitter (metres) applied to the
        nominal start position.
    start_vel_jitter_mps : float
        Half-width for additive velocity jitter (m/s).
    start_yaw_jitter_rad : float
        Half-width for yaw jitter at reset (radians).
    drag_scale_range : float
        Multiplicative range for ``drag_coeff``: sampled from
        ``[1-x, 1+x] * nominal``.
    angular_drag_scale_range : float
        Multiplicative range for ``angular_drag``.
    imu_noise_scale_range : float
        Multiplicative range for IMU noise stds (gyro + accel together).
    """

    enabled: bool = False
    start_pos_jitter_m: float = 0.5
    start_vel_jitter_mps: float = 0.3
    start_yaw_jitter_rad: float = 0.2
    drag_scale_range: float = 0.3
    angular_drag_scale_range: float = 0.3
    imu_noise_scale_range: float = 0.5


@dataclass(slots=True)
class RLEnvConfig:
    """Configuration for the RL training environment.

    Parameters
    ----------
    sim_steps_per_action : int
        Number of physics steps per RL action (controls action frequency).
    max_episode_steps : int
        Maximum steps per episode before truncation.
    gate_bonus : float
        Reward bonus for passing through a gate.
    crash_penalty : float
        Penalty for crashing (terminal).
    contour_weight : float
        Weight for contour error penalty.
    progress_weight : float
        Weight for progress reward.
    action_penalty_weight : float
        Weight for action magnitude penalty.
    max_contour_error : float
        Maximum contour error (m) before episode terminates.
    randomization : DomainRandomizationConfig
        Per-episode domain randomization for sim-to-real transfer.
    """

    sim_steps_per_action: int = 25  # 500 Hz / 25 = 20 Hz action rate
    max_episode_steps: int = 2000
    gate_bonus: float = 10.0
    crash_penalty: float = -50.0
    contour_weight: float = 0.5
    progress_weight: float = 5.0
    action_penalty_weight: float = 0.01
    max_contour_error: float = 5.0
    randomization: DomainRandomizationConfig = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if self.randomization is None:
            self.randomization = DomainRandomizationConfig()


class DroneRacingEnv(gym.Env[np.ndarray, np.ndarray]):
    """Gymnasium environment for drone racing RL training.

    Parameters
    ----------
    gates : list[GateWaypoint]
        Gate circuit (ordered).
    racing_line : RacingLine
        Pre-computed racing line through gates.
    config : RLEnvConfig | None
        Environment parameters.
    camera_config : CameraConfig | None
        Camera intrinsics (for mock sim rendering).
    sim_config : MockSimConfig | None
        Mock simulator parameters.
    """

    metadata = {"render_modes": ["rgb_array"]}

    def __init__(
        self,
        gates: list[GateWaypoint],
        racing_line: RacingLine,
        config: RLEnvConfig | None = None,
        camera_config: CameraConfig | None = None,
        sim_config: MockSimConfig | None = None,
    ) -> None:
        super().__init__()
        self._cfg = config or RLEnvConfig()
        self._gates = gates
        self._racing_line = racing_line
        self._control_cfg = ControlConfig()

        # Mock simulator — keep nominal sim params so reset() can re-derive
        # randomized variants without losing the originals.
        self._cam = camera_config or CameraConfig(width=320, height=240)
        self._nominal_sim_cfg = sim_config or MockSimConfig(max_time_s=60.0)
        self._sim = MockSimAdapter(
            camera_config=self._cam, gates=gates, config=self._nominal_sim_cfg
        )

        # Observation: 14-D continuous
        # [pos(3), vel(3), euler(3), relative_gate(3), progress(1), speed(1)]
        obs_high = np.full(14, 100.0, dtype=np.float32)
        self.observation_space = spaces.Box(-obs_high, obs_high, dtype=np.float32)

        # Action: 4-D continuous [-1, 1], mapped to DroneCommand
        self.action_space = spaces.Box(
            low=-np.ones(4, dtype=np.float32),
            high=np.ones(4, dtype=np.float32),
            dtype=np.float32,
        )

        # Episode state
        self._step_count = 0
        self._prev_progress: float = 0.0
        self._prev_gates_passed: int = 0

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        """Reset the environment to initial conditions.

        Parameters
        ----------
        seed : int | None
            Random seed.
        options : dict | None
            Additional options (unused).

        Returns
        -------
        tuple[np.ndarray, dict]
            (observation, info).
        """
        super().reset(seed=seed)

        # Apply per-episode physics randomization by rebuilding the sim with
        # a fresh MockSimConfig.  Cheap relative to an episode's worth of
        # rollouts, and avoids exposing physics mutators on MockSimAdapter.
        rand = self._cfg.randomization
        if rand.enabled:
            sim_cfg = self._sample_sim_config(rand)
            self._sim = MockSimAdapter(camera_config=self._cam, gates=self._gates, config=sim_cfg)

        # Reset sim with starting position near first gate
        if len(self._gates) > 0:
            # Start behind the first gate
            gate = self._gates[0]
            r_wg = Rotation.from_quat(gate.orientation_w)
            gate_normal = r_wg.apply(np.array([0.0, 0.0, 1.0]))
            start_pos = gate.position_w - gate_normal * 3.0  # 3m behind
        else:
            start_pos = np.zeros(3)

        start_vel = np.zeros(3)
        start_ori = quat_identity()
        if rand.enabled:
            start_pos = start_pos + self.np_random.uniform(
                -rand.start_pos_jitter_m, rand.start_pos_jitter_m, size=3
            )
            start_vel = self.np_random.uniform(
                -rand.start_vel_jitter_mps, rand.start_vel_jitter_mps, size=3
            )
            yaw = float(
                self.np_random.uniform(-rand.start_yaw_jitter_rad, rand.start_yaw_jitter_rad)
            )
            start_ori = Rotation.from_euler("z", yaw).as_quat().astype(np.float64)

        self._sim.reset(
            Telemetry(
                timestamp_ns=0,
                position=start_pos.copy(),
                velocity=start_vel,
                orientation=start_ori,
            )
        )
        self._step_count = 0
        self._prev_progress = self._racing_line.closest_progress(start_pos)
        self._prev_gates_passed = 0

        return self._get_obs(), {}

    def _sample_sim_config(self, rand: DomainRandomizationConfig) -> MockSimConfig:
        """Sample a per-episode :class:`MockSimConfig` from randomization ranges.

        Multiplicative perturbations are sampled from ``[1-x, 1+x]``; additive
        from ``[-x, +x]``.  Uses ``self.np_random`` so seeding is reproducible.
        """
        nom = self._nominal_sim_cfg

        def _scale(x: float) -> float:
            return float(self.np_random.uniform(1.0 - x, 1.0 + x))

        return MockSimConfig(
            sim_rate_hz=nom.sim_rate_hz,
            imu_rate_hz=nom.imu_rate_hz,
            camera_rate_hz=nom.camera_rate_hz,
            max_time_s=nom.max_time_s,
            gyro_noise_std=nom.gyro_noise_std * _scale(rand.imu_noise_scale_range),
            accel_noise_std=nom.accel_noise_std * _scale(rand.imu_noise_scale_range),
            gate_colour_bgr=nom.gate_colour_bgr,
            gate_thickness=nom.gate_thickness,
            drag_coeff=nom.drag_coeff * _scale(rand.drag_scale_range),
            angular_drag=nom.angular_drag * _scale(rand.angular_drag_scale_range),
        )

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        """Execute one RL step.

        Parameters
        ----------
        action : np.ndarray
            (4,) action in [-1, 1], mapped to [throttle, roll_rate, pitch_rate, yaw_rate].

        Returns
        -------
        tuple
            (observation, reward, terminated, truncated, info).
        """
        # Map action [-1, 1] → DroneCommand
        throttle = float(np.clip((action[0] + 1.0) / 2.0, 0.0, 1.0))  # [-1,1] → [0,1]
        roll_rate = float(action[1] * self._control_cfg.max_roll_rate)
        pitch_rate = float(action[2] * self._control_cfg.max_pitch_rate)
        yaw_rate = float(action[3] * self._control_cfg.max_yaw_rate)

        cmd = DroneCommand(
            timestamp_ns=self._sim.state.timestamp_ns,
            throttle=throttle,
            roll_rate=roll_rate,
            pitch_rate=pitch_rate,
            yaw_rate=yaw_rate,
        )
        self._sim.send_command(cmd)

        # Step physics multiple times per action
        for _ in range(self._cfg.sim_steps_per_action):
            self._sim.step()
            if not self._sim.is_running():
                break

        self._step_count += 1

        # Compute reward
        state = self._sim.state
        pos = state.position
        speed = float(np.linalg.norm(state.velocity))
        current_progress = self._racing_line.closest_progress(pos)

        # Progress delta (handle wraparound)
        dp = current_progress - self._prev_progress
        if dp < -0.5:
            dp += 1.0
        if dp > 0.5:
            dp -= 1.0

        # Contour error (distance from racing line)
        ref_pos, _, _ = self._racing_line.at_progress(current_progress)
        contour_error = float(np.linalg.norm(pos - ref_pos))

        # Reward components
        progress_reward = dp * self._cfg.progress_weight
        contour_penalty = -contour_error * self._cfg.contour_weight
        action_penalty = -float(np.sum(action**2)) * self._cfg.action_penalty_weight

        # Gate passage bonus
        gate_bonus = 0.0
        gates_now = self._sim.gates_passed
        if gates_now > self._prev_gates_passed:
            gate_bonus = self._cfg.gate_bonus * (gates_now - self._prev_gates_passed)
        self._prev_gates_passed = gates_now

        reward = float(progress_reward + contour_penalty + action_penalty + gate_bonus)

        # Termination conditions
        terminated = False
        truncated = False

        if not self._sim.is_running():
            terminated = True
            reward += self._cfg.crash_penalty

        if contour_error > self._cfg.max_contour_error:
            terminated = True
            reward += self._cfg.crash_penalty * 0.5

        if self._step_count >= self._cfg.max_episode_steps:
            truncated = True

        # Lap completion
        if self._sim.laps_completed > 0:
            truncated = True
            reward += self._cfg.gate_bonus * 5.0  # Big bonus for completing a lap

        self._prev_progress = current_progress

        info = {
            "progress": current_progress,
            "contour_error": contour_error,
            "speed": speed,
            "gates_passed": gates_now,
            "laps_completed": self._sim.laps_completed,
        }

        return self._get_obs(), reward, terminated, truncated, info

    def _get_obs(self) -> np.ndarray:
        """Build the observation vector from current state.

        Returns
        -------
        np.ndarray
            (14,) float32 observation.
        """
        state = self._sim.state
        pos = state.position
        vel = state.velocity
        r = Rotation.from_quat(state.orientation)
        euler = r.as_euler("xyz")
        speed = float(np.linalg.norm(vel))

        # Next gate relative position (in body frame for translation invariance)
        progress = self._racing_line.closest_progress(pos)
        lookahead = (progress + 0.02) % 1.0
        target_pos, _, _ = self._racing_line.at_progress(lookahead)
        relative_gate = r.inv().apply(target_pos - pos)

        obs = np.array(
            [
                *pos,
                *vel,
                *euler,
                *relative_gate,
                progress,
                speed,
            ],
            dtype=np.float32,
        )
        return obs

    def render(self) -> np.ndarray | None:
        """Return an RGB frame from the mock sim camera.

        Returns
        -------
        np.ndarray | None
            (H, W, 3) uint8 RGB image, or None.
        """
        result = self._sim.get_frame()
        if result is not None:
            return result[0]  # type: ignore[return-value]
        return None
