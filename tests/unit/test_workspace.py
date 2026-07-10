"""Unit tests for drone_racing.utils.workspace — the single path-resolution point.

Every artifact path in the stack routes through Workspace, so a resolution
regression would silently scatter traces/models into the wrong tree.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from drone_racing.utils.workspace import Workspace

if TYPE_CHECKING:
    from pathlib import Path

    import pytest


def test_explicit_root_wins(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """An explicit root bypasses the env var and marker search."""
    monkeypatch.setenv("AIGP_WORKSPACE", str(tmp_path / "elsewhere"))
    ws = Workspace.locate(tmp_path)
    assert ws.root == tmp_path.resolve()


def test_env_var_used_when_no_explicit_root(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("AIGP_WORKSPACE", str(tmp_path))
    ws = Workspace.locate()
    assert ws.root == tmp_path.resolve()


def test_marker_search_walks_upward(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """From a nested cwd, the nearest ancestor with pyproject.toml/.local is the root."""
    monkeypatch.delenv("AIGP_WORKSPACE", raising=False)
    (tmp_path / "pyproject.toml").write_text("", encoding="utf-8")
    nested = tmp_path / "a" / "b"
    nested.mkdir(parents=True)
    monkeypatch.chdir(nested)
    ws = Workspace.locate()
    assert ws.root == tmp_path.resolve()


def test_dot_local_is_a_marker(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("AIGP_WORKSPACE", raising=False)
    (tmp_path / ".local").mkdir()
    nested = tmp_path / "sub"
    nested.mkdir()
    monkeypatch.chdir(nested)
    ws = Workspace.locate()
    assert ws.root == tmp_path.resolve()


def test_falls_back_to_cwd_without_markers(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("AIGP_WORKSPACE", raising=False)
    monkeypatch.chdir(tmp_path)
    ws = Workspace.locate()
    # tmp_path itself has no markers, but an ancestor may (e.g. a user-level
    # pyproject). The invariant: the result is tmp_path or one of its ancestors.
    assert ws.root == tmp_path.resolve() or ws.root in tmp_path.resolve().parents


def test_artifact_paths_hang_off_dot_local(tmp_path: Path) -> None:
    ws = Workspace.locate(tmp_path)
    assert ws.local == ws.root / ".local"
    assert ws.datasets == ws.local / "datasets"
    assert ws.models == ws.local / "models"
    assert ws.runs == ws.local / "runs"
    assert ws.logs == ws.local / "logs"
    assert ws.simulators == ws.local / "simulators"


def test_ensure_creates_and_returns(tmp_path: Path) -> None:
    ws = Workspace.locate(tmp_path)
    target = ws.datasets / "vq2" / "new_dir"
    assert not target.exists()
    out = ws.ensure(target)
    assert out == target
    assert target.is_dir()
    # idempotent
    assert ws.ensure(target) == target
