"""Workspace paths — the single place filesystem locations are decided.

All generated/local artifacts (datasets, models, runs, logs, simulators) live
under ``<workspace>/.local/``, which is never version-controlled. The workspace
root is resolved, in order:

1. the ``AIGP_WORKSPACE`` environment variable, if set;
2. the nearest ancestor of the current directory containing ``pyproject.toml``
   or ``.local`` (so commands work from anywhere inside the repo);
3. the current directory (a fresh ``.local/`` is created on first write).

Usage::

    from drone_racing.utils.workspace import Workspace

    ws = Workspace.locate()
    trace_dir = ws.datasets / "vq2" / "fly_trace"
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

_ENV_VAR = "AIGP_WORKSPACE"
_ROOT_MARKERS = ("pyproject.toml", ".local")


@dataclass(frozen=True, slots=True)
class Workspace:
    """Typed accessor for workspace-relative artifact paths."""

    root: Path
    """Workspace root (the directory containing ``.local/``)."""

    @classmethod
    def locate(cls, root: str | Path | None = None) -> Workspace:
        """Resolve the workspace root and return a :class:`Workspace`.

        Parameters
        ----------
        root : str | Path | None
            Explicit workspace root. When None, resolution follows the module
            docstring order (env var, marker search upward from cwd, cwd).

        Returns
        -------
        Workspace
            Workspace rooted at the resolved directory.
        """
        if root is not None:
            return cls(root=Path(root).resolve())

        env_root = os.environ.get(_ENV_VAR)
        if env_root:
            return cls(root=Path(env_root).resolve())

        cwd = Path.cwd().resolve()
        for candidate in (cwd, *cwd.parents):
            if any((candidate / marker).exists() for marker in _ROOT_MARKERS):
                return cls(root=candidate)
        return cls(root=cwd)

    @property
    def local(self) -> Path:
        """``.local/`` — root of all non-version-controlled artifacts."""
        return self.root / ".local"

    @property
    def datasets(self) -> Path:
        """Training data, captured frames, trace archives."""
        return self.local / "datasets"

    @property
    def models(self) -> Path:
        """Model weights (.pt/.onnx)."""
        return self.local / "models"

    @property
    def runs(self) -> Path:
        """Training runs and recorded flight attempts."""
        return self.local / "runs"

    @property
    def logs(self) -> Path:
        """Flight telemetry logs."""
        return self.local / "logs"

    @property
    def simulators(self) -> Path:
        """Simulator installs (``vq1/`` = ground-truth rig, ``vq2/`` = active)."""
        return self.local / "simulators"

    def ensure(self, path: Path) -> Path:
        """Create ``path`` (and parents) if missing and return it.

        Parameters
        ----------
        path : Path
            Directory to create.

        Returns
        -------
        Path
            The same path, guaranteed to exist.
        """
        path.mkdir(parents=True, exist_ok=True)
        return path
