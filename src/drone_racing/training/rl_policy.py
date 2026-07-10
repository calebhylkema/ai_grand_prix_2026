#!/usr/bin/env python3
"""Train an RL policy for drone racing using PPO.

Uses the DroneRacingEnv (gymnasium wrapper around MockSimAdapter) for
offline training.  When the real simulator drops, swap the mock sim
for the real one with minimal changes.

Architecture: MLP policy (G&CNet-style, 3 hidden layers × 64 neurons).

Usage::

    uv run python -m drone_racing.training.rl_policy
    uv run python -m drone_racing.training.rl_policy --timesteps 500000 --lr 3e-4
    uv run python -m drone_racing.training.rl_policy --resume .local/runs/rl/best_model.zip

Requires the ``rl`` optional dependencies::

    uv pip install -e ".[rl]"
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from drone_racing.control.rl_env import (
    DomainRandomizationConfig,
    DroneRacingEnv,
    RLEnvConfig,
)
from drone_racing.planning.spline import build_racing_line
from drone_racing.sim.mock_adapter import MockSimConfig
from drone_racing.utils.config import load_config
from drone_racing.utils.types import (
    CameraConfig,
    GateWaypoint,
    PlannerConfig,
)

logger = logging.getLogger(__name__)


def build_gate_circuit() -> list[GateWaypoint]:
    """Create a 4-gate square circuit for training.

    Returns
    -------
    list[GateWaypoint]
        Four gates forming a ~15m square circuit at 3m altitude.
    """
    config = load_config()
    corners_local = config.gate.outer_corners_3d_array

    gate_defs: list[tuple[np.ndarray, float]] = [
        (np.array([15.0, 0.0, 3.0]), 0.0),
        (np.array([15.0, 15.0, 3.0]), np.pi / 2),
        (np.array([0.0, 15.0, 3.0]), np.pi),
        (np.array([0.0, 0.0, 3.0]), -np.pi / 2),
    ]

    gates: list[GateWaypoint] = []
    for i, (pos, yaw) in enumerate(gate_defs):
        orientation = Rotation.from_euler("yz", [np.pi / 2, yaw]).as_quat()
        r_wg = Rotation.from_quat(orientation)
        corners_world = (r_wg.apply(corners_local) + pos).astype(np.float64)
        gates.append(
            GateWaypoint(
                gate_id=i,
                position_w=pos,
                orientation_w=orientation.astype(np.float64),
                corners_3d_w=corners_world,
                order=i,
            )
        )
    return gates


def make_env(
    gates: list[GateWaypoint],
    racing_line: object,
    env_config: RLEnvConfig | None = None,
) -> DroneRacingEnv:
    """Create a DroneRacingEnv instance.

    Parameters
    ----------
    gates : list[GateWaypoint]
        Gate circuit.
    racing_line : RacingLine
        Pre-computed racing line.
    env_config : RLEnvConfig | None
        Environment parameters.

    Returns
    -------
    DroneRacingEnv
        Gymnasium-compatible environment.
    """
    cam = CameraConfig(width=320, height=240)
    sim_cfg = MockSimConfig(max_time_s=60.0)
    return DroneRacingEnv(
        gates=gates,
        racing_line=racing_line,  # type: ignore[arg-type]
        config=env_config,
        camera_config=cam,
        sim_config=sim_cfg,
    )


def train(args: argparse.Namespace) -> None:
    """Run PPO training loop.

    Parameters
    ----------
    args : argparse.Namespace
        CLI arguments.
    """
    try:
        from stable_baselines3 import PPO
        from stable_baselines3.common.callbacks import (
            BaseCallback,
            CheckpointCallback,
            EvalCallback,
        )
        from stable_baselines3.common.vec_env import (
            DummyVecEnv,
            SubprocVecEnv,
            VecEnv,
            VecNormalize,
        )
    except ImportError as e:
        msg = (
            "RL dependencies not installed. Run: uv pip install -e '.[rl]'\n"
            "Required: torch, stable-baselines3, gymnasium"
        )
        raise ImportError(msg) from e

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Build gate circuit and racing line
    gates = build_gate_circuit()
    gate_positions = np.array([g.position_w for g in gates])
    gate_orientations = np.array([g.orientation_w for g in gates])
    racing_line = build_racing_line(
        gate_positions, gate_orientations, PlannerConfig(), num_points=200, closed=True
    )

    # Environment config
    env_config = RLEnvConfig(
        sim_steps_per_action=args.sim_steps_per_action,
        max_episode_steps=args.max_episode_steps,
        randomization=DomainRandomizationConfig(enabled=args.randomize),
    )
    if args.randomize:
        logger.info("Domain randomization ENABLED (per-episode drag + start jitter)")

    # Create vectorized environments with observation normalization.
    # VecNormalize maintains a running mean/std of observations and
    # normalises them to zero-mean unit-variance before feeding the policy.
    # This dramatically improves PPO stability for observations that span
    # different scales (position in metres vs. progress in [0,1]).
    def make_train_env() -> DroneRacingEnv:
        return make_env(gates, racing_line, env_config)

    raw_train_env: VecEnv
    if args.n_envs > 1:
        raw_train_env = SubprocVecEnv([make_train_env for _ in range(args.n_envs)])
    else:
        raw_train_env = DummyVecEnv([make_train_env])

    norm_stats_path = output_dir / "vecnormalize.pkl"
    if args.resume and norm_stats_path.exists():
        vec_env = VecNormalize.load(str(norm_stats_path), raw_train_env)
        logger.info("Loaded observation normalization stats from %s", norm_stats_path)
    else:
        vec_env = VecNormalize(
            raw_train_env,
            norm_obs=True,
            norm_reward=True,
            clip_obs=10.0,
            clip_reward=10.0,
        )

    # Eval environment (shares normalization stats, but does not update them)
    raw_eval_env = DummyVecEnv([make_train_env])
    eval_env = VecNormalize(
        raw_eval_env,
        norm_obs=True,
        norm_reward=False,  # don't normalise eval rewards — we want true scale
        clip_obs=10.0,
    )
    eval_env.training = False
    eval_env.norm_reward = False

    # Policy network: G&CNet-style MLP (3 × 64)
    policy_kwargs = {
        "net_arch": [64, 64, 64],
    }

    # Create or resume model
    if args.resume:
        logger.info("Resuming from %s", args.resume)
        model = PPO.load(args.resume, env=vec_env)
    else:
        model = PPO(
            "MlpPolicy",
            vec_env,
            learning_rate=args.lr,
            n_steps=args.n_steps,
            batch_size=args.batch_size,
            n_epochs=args.n_epochs,
            gamma=args.gamma,
            gae_lambda=args.gae_lambda,
            clip_range=args.clip_range,
            ent_coef=args.ent_coef,
            policy_kwargs=policy_kwargs,
            verbose=1,
            tensorboard_log=str(output_dir / "tensorboard"),
        )

    # Callback that saves VecNormalize stats alongside model checkpoints
    # and syncs running stats to the eval env before each evaluation.
    class NormalizationCallback(BaseCallback):
        """Save VecNormalize stats periodically and sync to eval env."""

        def __init__(
            self,
            save_path: Path,
            train_env: VecNormalize,
            eval_env_ref: VecNormalize,
            save_freq: int,
        ) -> None:
            super().__init__()
            self._save_path = save_path
            self._train_env = train_env
            self._eval_env = eval_env_ref
            self._save_freq = save_freq

        def _on_step(self) -> bool:
            # Sync running stats to eval env each step (cheap — just copy arrays)
            self._eval_env.obs_rms = self._train_env.obs_rms
            self._eval_env.ret_rms = self._train_env.ret_rms

            if self.n_calls % self._save_freq == 0:
                self._train_env.save(str(self._save_path))
            return True

    save_freq = max(args.timesteps // 10, 10000)

    # Callbacks
    norm_cb = NormalizationCallback(
        save_path=norm_stats_path,
        train_env=vec_env,
        eval_env_ref=eval_env,
        save_freq=save_freq,
    )
    checkpoint_cb = CheckpointCallback(
        save_freq=save_freq,
        save_path=str(output_dir / "checkpoints"),
        name_prefix="ppo_drone",
    )
    eval_cb = EvalCallback(
        eval_env,
        best_model_save_path=str(output_dir / "best"),
        log_path=str(output_dir / "eval_logs"),
        eval_freq=max(args.timesteps // 20, 5000),
        n_eval_episodes=5,
        deterministic=True,
    )

    # Train
    logger.info(
        "Starting PPO training: %d timesteps, %d envs, lr=%.1e",
        args.timesteps,
        args.n_envs,
        args.lr,
    )
    model.learn(
        total_timesteps=args.timesteps,
        callback=[norm_cb, checkpoint_cb, eval_cb],
        progress_bar=True,
    )

    # Save final model + normalization stats
    final_path = output_dir / "final_model"
    model.save(str(final_path))
    vec_env.save(str(norm_stats_path))
    logger.info("Training complete. Model saved to %s", final_path)
    logger.info("Observation normalization stats saved to %s", norm_stats_path)

    # Also save a copy next to the best model for easy inference loading
    best_norm_path = output_dir / "best" / "vecnormalize.pkl"
    vec_env.save(str(best_norm_path))

    vec_env.close()
    eval_env.close()


def main(argv: list[str] | None = None) -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Train an RL racing policy with PPO",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--timesteps", type=int, default=500_000, help="Total training timesteps")
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate")
    parser.add_argument("--n-envs", type=int, default=1, help="Parallel environments")
    parser.add_argument("--n-steps", type=int, default=2048, help="Steps per rollout")
    parser.add_argument("--batch-size", type=int, default=64, help="Minibatch size")
    parser.add_argument("--n-epochs", type=int, default=10, help="PPO epochs per update")
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor")
    parser.add_argument("--gae-lambda", type=float, default=0.95, help="GAE lambda")
    parser.add_argument("--clip-range", type=float, default=0.2, help="PPO clip range")
    parser.add_argument("--ent-coef", type=float, default=0.01, help="Entropy coefficient")
    parser.add_argument(
        "--sim-steps-per-action",
        type=int,
        default=25,
        help="Physics steps per RL action (500Hz/25 = 20Hz)",
    )
    parser.add_argument("--max-episode-steps", type=int, default=2000, help="Max steps per episode")
    parser.add_argument("--resume", type=str, default=None, help="Path to model to resume")
    parser.add_argument("--output-dir", type=str, default=".local/runs/rl", help="Output directory")
    parser.add_argument(
        "--randomize",
        action="store_true",
        help="Enable per-episode domain randomization (start jitter + drag/IMU scaling)",
    )

    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    train(args)


if __name__ == "__main__":
    main()
