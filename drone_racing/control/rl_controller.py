"""RL policy inference controller for drone racing.

Loads a trained Stable-Baselines3 PPO checkpoint and maps the ESKF state
to throttle/roll/pitch/yaw commands, matching the observation and action
spaces defined in :mod:`drone_racing.control.rl_env`.

Designed to slot into the safety ladder alongside MPCC and PID controllers.

If trained with ``VecNormalize`` (recommended), pass the path to the
saved ``.pkl`` stats so observations are normalised identically at
inference time.

Usage::

    controller = RLController.from_checkpoint(
        "runs/rl/best/best_model.zip", racing_line, control_config,
    )
    cmd = controller.update(state)
"""

from __future__ import annotations

import logging
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from scipy.spatial.transform import Rotation

from drone_racing.utils.types import ControlConfig, DroneCommand, State

if TYPE_CHECKING:
    from stable_baselines3 import PPO

    from drone_racing.planning.spline import RacingLine

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class ObsNormStats:
    """Running mean/variance statistics for observation normalization.

    Mirrors the fields from SB3's ``VecNormalize.obs_rms`` (a
    ``RunningMeanStd`` instance).

    Parameters
    ----------
    mean : np.ndarray
        Running mean, shape (obs_dim,).
    var : np.ndarray
        Running variance, shape (obs_dim,).
    clip : float
        Clip normalised observations to [-clip, clip].
    """

    mean: np.ndarray
    var: np.ndarray
    clip: float = 10.0

    def normalize(self, obs: np.ndarray) -> np.ndarray:
        """Apply zero-mean unit-variance normalization.

        Parameters
        ----------
        obs : np.ndarray
            Raw observation.

        Returns
        -------
        np.ndarray
            Normalised and clipped observation.
        """
        normed = (obs - self.mean) / np.sqrt(self.var + 1e-8)
        clipped: np.ndarray = np.clip(normed, -self.clip, self.clip).astype(np.float32)
        return clipped


@dataclass(slots=True)
class RLControllerStats:
    """Diagnostics from the last RL inference cycle."""

    progress: float
    """Current progress along the racing line [0, 1]."""

    contour_error: float
    """Distance from the racing line (m)."""

    speed: float
    """Scalar speed (m/s)."""

    action_raw: np.ndarray
    """Raw [-1, 1] action output from the policy."""


def _load_norm_stats(path: Path, clip: float = 10.0) -> ObsNormStats:
    """Load observation normalization stats from a VecNormalize ``.pkl``.

    Parameters
    ----------
    path : Path
        Path to the ``vecnormalize.pkl`` saved by ``VecNormalize.save()``.
    clip : float
        Clip bound for normalised observations.

    Returns
    -------
    ObsNormStats
        Extracted running mean and variance.
    """
    with open(path, "rb") as f:
        vec_norm = pickle.load(f)  # noqa: S301

    obs_rms = vec_norm.obs_rms
    return ObsNormStats(
        mean=np.asarray(obs_rms.mean, dtype=np.float64),
        var=np.asarray(obs_rms.var, dtype=np.float64),
        clip=clip,
    )


class RLController:
    """RL policy controller for drone racing.

    Builds the same 14-D observation vector used during training
    (:class:`~drone_racing.control.rl_env.DroneRacingEnv`) and maps the
    4-D policy action to a :class:`DroneCommand`.

    Parameters
    ----------
    model : PPO
        Loaded Stable-Baselines3 PPO model.
    racing_line : RacingLine
        Reference racing line (same as used during training).
    control_config : ControlConfig
        Physical rate limits for action scaling.
    lookahead : float
        Progress-space lookahead for the next-gate relative position.
    deterministic : bool
        If True, use the mean action (no exploration noise).
    obs_norm : ObsNormStats | None
        Observation normalization statistics from training.  If provided,
        raw observations are normalised before feeding the policy.
    """

    def __init__(
        self,
        model: PPO,
        racing_line: RacingLine,
        control_config: ControlConfig | None = None,
        lookahead: float = 0.02,
        deterministic: bool = True,
        obs_norm: ObsNormStats | None = None,
    ) -> None:
        self._model = model
        self._racing_line = racing_line
        self._cfg = control_config or ControlConfig()
        self._lookahead = lookahead
        self._deterministic = deterministic
        self._obs_norm = obs_norm
        self._current_progress: float = 0.0
        self._last_stats: RLControllerStats | None = None

    @classmethod
    def from_checkpoint(
        cls,
        path: str | Path,
        racing_line: RacingLine,
        control_config: ControlConfig | None = None,
        *,
        lookahead: float = 0.02,
        deterministic: bool = True,
        norm_path: str | Path | None = None,
    ) -> RLController:
        """Load a controller from a saved SB3 checkpoint.

        If ``norm_path`` is not provided, this method auto-discovers a
        ``vecnormalize.pkl`` in the same directory as the model checkpoint.

        Parameters
        ----------
        path : str | Path
            Path to the ``.zip`` model file.
        racing_line : RacingLine
            Reference racing line.
        control_config : ControlConfig | None
            Physical rate limits.
        lookahead : float
            Progress-space lookahead.
        deterministic : bool
            If True, use the mean action.
        norm_path : str | Path | None
            Path to ``vecnormalize.pkl``.  If None, looks for one next to
            the model file.

        Returns
        -------
        RLController
            Ready-to-use controller.

        Raises
        ------
        FileNotFoundError
            If the checkpoint does not exist.
        ImportError
            If stable-baselines3 is not installed.
        """
        resolved = Path(path).resolve()
        if not resolved.exists():
            msg = f"RL checkpoint not found: {resolved}"
            raise FileNotFoundError(msg)

        try:
            from stable_baselines3 import PPO
        except ImportError as e:
            msg = (
                "RL dependencies not installed. Run: uv pip install -e '.[rl]'\n"
                "Required: torch, stable-baselines3"
            )
            raise ImportError(msg) from e

        model = PPO.load(str(resolved))
        logger.info("Loaded RL checkpoint: %s", resolved)

        # Auto-discover or load observation normalization stats
        obs_norm: ObsNormStats | None = None
        if norm_path is not None:
            norm_resolved = Path(norm_path).resolve()
            if norm_resolved.exists():
                obs_norm = _load_norm_stats(norm_resolved)
                logger.info("Loaded obs normalization stats: %s", norm_resolved)
            else:
                logger.warning(
                    "Norm stats not found at %s — running without normalization", norm_resolved
                )
        else:
            # Auto-discover next to the model checkpoint
            auto_path = resolved.parent / "vecnormalize.pkl"
            if auto_path.exists():
                obs_norm = _load_norm_stats(auto_path)
                logger.info("Auto-discovered obs normalization stats: %s", auto_path)
            else:
                logger.info("No vecnormalize.pkl found — running without obs normalization")

        return cls(
            model,
            racing_line,
            control_config,
            lookahead=lookahead,
            deterministic=deterministic,
            obs_norm=obs_norm,
        )

    @property
    def current_progress(self) -> float:
        """Current progress along the racing line [0, 1]."""
        return self._current_progress

    @property
    def last_stats(self) -> RLControllerStats | None:
        """Diagnostics from the most recent update."""
        return self._last_stats

    def reset(self, progress: float = 0.0) -> None:
        """Reset controller state.

        Parameters
        ----------
        progress : float
            Initial progress along the racing line.
        """
        self._current_progress = progress

    def update(self, state: State) -> DroneCommand:
        """Run one inference cycle: state -> observation -> policy -> command.

        Parameters
        ----------
        state : State
            Current ESKF state estimate.

        Returns
        -------
        DroneCommand
            Rate-mode command for the flight controller.
        """
        obs = self._build_obs(state)

        # Apply observation normalization if available
        if self._obs_norm is not None:
            obs = self._obs_norm.normalize(obs)

        action, _ = self._model.predict(obs, deterministic=self._deterministic)
        action = np.asarray(action, dtype=np.float32).ravel()

        # Map action [-1, 1] -> DroneCommand (same mapping as rl_env.py)
        throttle = float(np.clip((action[0] + 1.0) / 2.0, 0.0, 1.0))
        roll_rate = float(action[1] * self._cfg.max_roll_rate)
        pitch_rate = float(action[2] * self._cfg.max_pitch_rate)
        yaw_rate = float(action[3] * self._cfg.max_yaw_rate)

        # Clamp to physical limits
        roll_rate = float(np.clip(roll_rate, -self._cfg.max_roll_rate, self._cfg.max_roll_rate))
        pitch_rate = float(np.clip(pitch_rate, -self._cfg.max_pitch_rate, self._cfg.max_pitch_rate))
        yaw_rate = float(np.clip(yaw_rate, -self._cfg.max_yaw_rate, self._cfg.max_yaw_rate))

        # Store diagnostics
        ref_pos, _, _ = self._racing_line.at_progress(self._current_progress)
        self._last_stats = RLControllerStats(
            progress=self._current_progress,
            contour_error=float(np.linalg.norm(state.position - ref_pos)),
            speed=float(np.linalg.norm(state.velocity)),
            action_raw=action.copy(),
        )

        return DroneCommand(
            timestamp_ns=state.timestamp_ns,
            throttle=throttle,
            roll_rate=roll_rate,
            pitch_rate=pitch_rate,
            yaw_rate=yaw_rate,
        )

    def _build_obs(self, state: State) -> np.ndarray:
        """Build the 14-D observation vector from the ESKF state.

        Must match :meth:`drone_racing.control.rl_env.DroneRacingEnv._get_obs`
        exactly.

        Observation layout::

            [pos(3), vel(3), euler(3), relative_gate_body(3), progress(1), speed(1)]

        Parameters
        ----------
        state : State
            Current state estimate.

        Returns
        -------
        np.ndarray
            (14,) float32 observation.
        """
        pos = state.position
        vel = state.velocity
        r = Rotation.from_quat(state.orientation)
        euler = r.as_euler("xyz")
        speed = float(np.linalg.norm(vel))

        # Update progress tracking
        pos_progress = self._racing_line.closest_progress(pos)
        dp = pos_progress - self._current_progress
        if dp < -0.5:
            dp += 1.0
        if dp > 0.5:
            dp -= 1.0
        if abs(dp) < 0.5:
            self._current_progress = pos_progress

        # Next-gate relative position in body frame (translation-invariant)
        lookahead = (self._current_progress + self._lookahead) % 1.0
        target_pos, _, _ = self._racing_line.at_progress(lookahead)
        relative_gate = r.inv().apply(target_pos - pos)

        return np.array(
            [
                *pos,
                *vel,
                *euler,
                *relative_gate,
                self._current_progress,
                speed,
            ],
            dtype=np.float32,
        )
