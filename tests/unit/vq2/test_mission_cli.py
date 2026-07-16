"""CLI guardrails for the VQ2 mission runner."""

from __future__ import annotations

import pytest

from drone_racing.deployments.vq2.mission import main


def test_visual_handoff_requires_record() -> None:
    # An experimental-authority attempt without --record must be refused BEFORE connecting, so no
    # authority run can go unrecorded/undiagnosable. argparse .error() raises SystemExit(2).
    with pytest.raises(SystemExit) as exc:
        main(["--experimental-visual-handoff"])
    assert exc.value.code == 2


def test_visual_handoff_with_save_frames_satisfies_record() -> None:
    # --save-frames implies recording, so it passes the record guard and proceeds to config load
    # (which here fails on a bogus config name — a DIFFERENT failure, proving the guard passed).
    with pytest.raises(FileNotFoundError):
        main(["--experimental-visual-handoff", "--save-frames", "--config", "does-not-exist"])
