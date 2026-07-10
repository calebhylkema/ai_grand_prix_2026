"""Configuration loader — reads TOML files into typed dataclasses.

Usage::

    cfg = load_config()                          # loads the packaged default.toml
    cfg = load_config("my_race_day.toml")        # loads a custom file

All config sections are returned as typed dataclasses defined in
:mod:`drone_racing.utils.types`.
"""

from __future__ import annotations

import tomllib
from dataclasses import dataclass, field
from pathlib import Path

from drone_racing.utils.types import (
    CameraConfig,
    ControlConfig,
    EstimationConfig,
    GateConfig,
    MAVLinkConfig,
    PlannerConfig,
)

_DEFAULT_CONFIG_PATH = Path(__file__).resolve().parents[1] / "config" / "default.toml"


@dataclass(slots=True)
class RacingConfig:
    """Top-level configuration container for the entire stack."""

    estimation: EstimationConfig = field(default_factory=EstimationConfig)
    camera: CameraConfig = field(default_factory=CameraConfig)
    gate: GateConfig = field(default_factory=GateConfig)
    control: ControlConfig = field(default_factory=ControlConfig)
    planner: PlannerConfig = field(default_factory=PlannerConfig)
    mavlink: MAVLinkConfig = field(default_factory=MAVLinkConfig)
    raw: dict[str, object] = field(default_factory=dict, repr=False)
    """The full parsed TOML dict, for sections not yet modelled."""


def load_config(path: str | Path | None = None) -> RacingConfig:
    """Load a TOML configuration file and return a typed :class:`RacingConfig`.

    Parameters
    ----------
    path : str | Path | None
        Path to the TOML file.  Defaults to the packaged ``default.toml``.

    Returns
    -------
    RacingConfig
        Parsed and validated configuration.

    Raises
    ------
    FileNotFoundError
        If the config file does not exist.
    """
    config_path = Path(path) if path is not None else _DEFAULT_CONFIG_PATH
    if not config_path.exists():
        msg = f"Config file not found: {config_path}"
        raise FileNotFoundError(msg)

    with config_path.open("rb") as f:
        raw = tomllib.load(f)

    estimation_raw = raw.get("estimation", {})
    camera_raw = raw.get("camera", {})
    gate_raw = raw.get("gate", {})

    control_raw = raw.get("control", {})
    planner_raw = raw.get("planner", {})
    mavlink_raw = raw.get("mavlink", {})

    estimation = EstimationConfig(**estimation_raw)
    camera = CameraConfig(**camera_raw)
    gate = GateConfig(**gate_raw)
    control = ControlConfig(**control_raw)
    planner = PlannerConfig(**planner_raw)
    mavlink = MAVLinkConfig(**mavlink_raw)

    return RacingConfig(
        estimation=estimation,
        camera=camera,
        gate=gate,
        control=control,
        planner=planner,
        mavlink=mavlink,
        raw=raw,
    )
