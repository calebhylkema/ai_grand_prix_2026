"""Typed configuration for the VQ2 mission.

Deployment configs live at the workspace root under ``deployments/vq2/configs/``
(see docs/reference/configuration.md): ``training.toml`` and
``qualification.toml``. Each file is complete and separate — no
layering/inheritance, so what you read is what flies. Numerical flight values
were carried over verbatim from the flight-validated fly2.py controller during
the 2026-07-09 restructure.

Units: SI internally (meters, radians, seconds). The TOML files store the
scan-acceptance turn window in DEGREES (``accept_turn_deg``) for readability;
it is converted to radians here, at the load boundary.

Usage::

    from drone_racing.deployments.vq2 import config

    cfg = config.load("training")            # deployment config by name
    cfg = config.load("path/to/custom.toml")  # or any explicit path
"""

from __future__ import annotations

import dataclasses
import math
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from drone_racing.utils.workspace import Workspace

CONFIG_DIR = Workspace.locate().root / "deployments" / "vq2" / "configs"


@dataclass(frozen=True, slots=True)
class MissionParams:
    """Phased-mission control parameters (see mission.py for the tuning history)."""

    cmd_hz: float
    stab_secs: float
    # G0: launch-assisted gate-0 thread
    kp_yaw: float
    pitch_fwd: float
    align_bear: float
    cy_band: float
    # BRAKE: arrest-then-look corner entry
    brake_secs: float
    brake_pitch: float
    brake_arrest_s: float
    brake_arrest_thr: float
    brake_tail_pitch: float
    # SCAN: yaw toward the mapped next-gate bearing
    scan_yaw: float
    scan_hold_pitch: float
    # THREAD: servo onto the acquired gate
    kp_roll_bear: float
    recenter_pitch: float
    accept_z: tuple[float, float]
    accept_turn: tuple[float, float]  # radians (TOML stores degrees)
    target_hold_s: float
    lost_brake_s: float
    pass_area: float
    pass_range_m: float
    pass_commit_s: float


@dataclass(frozen=True, slots=True)
class RunParams:
    """Attempt orchestration + I/O parameters."""

    secs: float
    attempts: int
    trace: bool
    mavlink_endpoint: str


@dataclass(frozen=True, slots=True)
class VQ2Config:
    """Complete VQ2 mission configuration."""

    name: str
    mission: MissionParams
    run: RunParams


def load(name_or_path: str | Path) -> VQ2Config:
    """Load a VQ2 config by deployment name or explicit TOML path.

    Parameters
    ----------
    name_or_path : str | Path
        A deployment config name (``"training"`` / ``"qualification"``,
        resolved in ``deployments/vq2/configs/``) or a path to a complete
        config TOML.

    Returns
    -------
    VQ2Config
        Fully-typed configuration (angles in radians).
    """
    path = Path(name_or_path)
    if path.suffix != ".toml":
        path = CONFIG_DIR / f"{path}.toml"
    if not path.is_file():
        known = sorted(p.stem for p in CONFIG_DIR.glob("*.toml"))
        raise FileNotFoundError(f"No VQ2 config at {path} (known: {', '.join(known)})")

    with path.open("rb") as f:
        raw = tomllib.load(f)

    m = dict(raw["mission"])
    turn_deg = m.pop("accept_turn_deg")
    m["accept_turn"] = (math.radians(turn_deg[0]), math.radians(turn_deg[1]))
    m["accept_z"] = tuple(m["accept_z"])
    mission = MissionParams(**m)
    run = RunParams(**raw["run"])
    return VQ2Config(name=raw.get("name", path.stem), mission=mission, run=run)


def replace_run(cfg: VQ2Config, **overrides: Any) -> VQ2Config:
    """Return ``cfg`` with the given RunParams fields replaced (CLI overrides)."""
    return dataclasses.replace(cfg, run=dataclasses.replace(cfg.run, **overrides))
