"""Regression smoke test for the MPCC switchover wiring in ``scripts/run_race.py``.

``run_race.py`` is written against :class:`MAVSDKAdapter` which we can't drive
without the real simulator.  This test imports ``run_race.py``'s helpers
directly and exercises them against :class:`MockSimAdapter` to verify the
perception → tracker → mapper → racing-line → MPCC → DroneCommand path is
wired correctly end-to-end.

Why a dedicated test: the existing ``test_gate_mapper_integration`` only
covers the mapper's math.  The MPCC handoff — building a racing line from
discovered waypoints, constructing an :class:`MPCCSolver` via
``from_planner_config``, converting MPCC outputs to rate-mode commands — was
added in the same change set but never executed end-to-end.  Catching a
wiring bug here is much cheaper than discovering it when the sim drops.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from drone_racing.estimation.gate_mapper import GateMapper, GateMapperConfig
from drone_racing.perception.classical import ClassicalDetector
from drone_racing.perception.tracker import GateTracker
from drone_racing.planning.mpcc import MPCCSolver
from drone_racing.sim.mock_adapter import MockSimAdapter, MockSimConfig
from drone_racing.utils.config import load_config
from drone_racing.utils.types import DroneCommand, GateWaypoint, Telemetry

# ---------------------------------------------------------------------------
# Import the private helpers from scripts/run_race.py directly so this test
# exercises the real code path rather than a copy.
# ---------------------------------------------------------------------------
_RUN_RACE = Path(__file__).resolve().parents[1] / "scripts" / "run_race.py"
_spec = importlib.util.spec_from_file_location("_run_race_under_test", _RUN_RACE)
assert _spec is not None and _spec.loader is not None
_run_race = importlib.util.module_from_spec(_spec)
sys.modules["_run_race_under_test"] = _run_race
_spec.loader.exec_module(_run_race)

_extract_euler = _run_race._extract_euler
_mpcc_to_command = _run_race._mpcc_to_command
_build_racing_line_from_waypoints = _run_race._build_racing_line_from_waypoints
MIN_MAPPED_GATES_FOR_MPCC = _run_race.MIN_MAPPED_GATES_FOR_MPCC


def _build_gate(
    gate_id: int, position: np.ndarray, yaw: float, corners_local: np.ndarray
) -> GateWaypoint:
    orientation = Rotation.from_euler("yz", [np.pi / 2, yaw]).as_quat()
    r_wg = Rotation.from_quat(orientation)
    corners_w = (r_wg.apply(corners_local) + position).astype(np.float64)
    return GateWaypoint(
        gate_id=gate_id,
        position_w=position.astype(np.float64),
        orientation_w=orientation.astype(np.float64),
        corners_3d_w=corners_w,
        order=gate_id,
    )


@pytest.fixture()
def cfg():
    return load_config()


class TestHelpers:
    """Direct exercise of the small helpers lifted from run_race.py."""

    def test_extract_euler_identity(self) -> None:
        from drone_racing.utils.types import State

        s = State.identity()
        roll, pitch, yaw = _extract_euler(s)
        assert abs(roll) < 1e-9
        assert abs(pitch) < 1e-9
        assert abs(yaw) < 1e-9

    def test_mpcc_to_command_closes_attitude_loop(self) -> None:
        """Roll/pitch rate should be ``kp * (cmd - current)``; yaw passes through."""
        from drone_racing.utils.types import State

        state = State.identity(timestamp_ns=1_000)
        # Bias current orientation: 0.1 rad roll, 0.0 pitch, 0.0 yaw
        state.orientation = Rotation.from_euler("xyz", [0.1, 0.0, 0.0]).as_quat().astype(np.float64)

        mpcc_input = np.array([0.65, 0.3, -0.2, 0.4])  # throttle, phi_cmd, theta_cmd, psi_dot
        attitude_kp = (5.0, 5.0, 2.0)

        cmd = _mpcc_to_command(mpcc_input, state, attitude_kp, timestamp_ns=1_000)

        assert isinstance(cmd, DroneCommand)
        assert cmd.throttle == pytest.approx(0.65)
        # roll_rate = 5.0 * (0.3 - 0.1)
        assert cmd.roll_rate == pytest.approx(5.0 * (0.3 - 0.1))
        # pitch_rate = 5.0 * (-0.2 - 0.0)
        assert cmd.pitch_rate == pytest.approx(5.0 * -0.2)
        # yaw_rate passes through unchanged
        assert cmd.yaw_rate == pytest.approx(0.4)
        assert cmd.timestamp_ns == 1_000

    def test_mpcc_to_command_clamps_throttle(self) -> None:
        from drone_racing.utils.types import State

        state = State.identity()
        over = _mpcc_to_command(np.array([1.7, 0.0, 0.0, 0.0]), state, (1.0, 1.0, 1.0), 0)
        under = _mpcc_to_command(np.array([-0.5, 0.0, 0.0, 0.0]), state, (1.0, 1.0, 1.0), 0)
        assert over.throttle == 1.0
        assert under.throttle == 0.0


class TestMPCCHandoff:
    """End-to-end: mapper discovers gates → racing line → MPCC solves → cmd."""

    def test_mpcc_switchover_against_mock_sim(self, cfg) -> None:
        corners_local = cfg.gate.outer_corners_3d_array

        # 4-gate square circuit at altitude 3m.  Drone starts facing +X.
        gate_defs: list[tuple[np.ndarray, float]] = [
            (np.array([6.0, 0.0, 3.0]), 0.0),
            (np.array([6.0, 6.0, 3.0]), np.pi / 2),
            (np.array([0.0, 6.0, 3.0]), np.pi),
            (np.array([0.0, 0.0, 3.0]), -np.pi / 2),
        ]
        gates = [_build_gate(i, pos, yaw, corners_local) for i, (pos, yaw) in enumerate(gate_defs)]

        sim = MockSimAdapter(
            camera_config=cfg.camera,
            gates=gates,
            config=MockSimConfig(max_time_s=6.0),
        )
        sim.reset(
            Telemetry(
                timestamp_ns=0,
                position=np.array([2.0, 0.0, 3.0]),
                velocity=np.zeros(3),
                orientation=np.array([0.0, 0.0, 0.0, 1.0]),
            )
        )

        detector = ClassicalDetector(camera_config=cfg.camera)
        tracker = GateTracker()
        mapper = GateMapper(
            camera_config=cfg.camera,
            gate_corners_3d_local=corners_local,
            config=GateMapperConfig(min_observations=2, max_distance=20.0),
        )

        attitude_kp = (
            cfg.control.attitude_kp[0],
            cfg.control.attitude_kp[1],
            cfg.control.attitude_kp[2],
        )

        mpcc: MPCCSolver | None = None
        racing_line = None
        mpcc_success_count = 0
        commands_produced = 0

        # Slowly yaw the drone to see multiple gates.  We drive the sim with
        # a fixed gentle command (not PID) since our goal is wiring — the
        # attitude-error dynamics aren't under test.
        for _step_idx in range(4000):
            sim.step()
            if not sim.is_running():
                break

            frame_info = sim.get_frame()
            if frame_info is not None:
                image, ts = frame_info
                detections = detector.detect(image, ts)
                tracked = tracker.update(detections)
                mapper.observe_many(tracked, sim.state)

            # Hand off to MPCC as soon as we have enough waypoints — matches
            # the switchover condition in run_race.py line-for-line.
            waypoints = mapper.waypoints()
            if mpcc is None and len(waypoints) >= MIN_MAPPED_GATES_FOR_MPCC:
                racing_line = _build_racing_line_from_waypoints(waypoints, cfg.planner, closed=True)
                mpcc = MPCCSolver.from_planner_config(racing_line, cfg.planner, closed=True)

            if mpcc is not None and racing_line is not None:
                roll, pitch, yaw = _extract_euler(sim.state)
                mpcc_state = np.concatenate(
                    [sim.state.position, sim.state.velocity, np.array([roll, pitch, yaw])]
                )
                progress = racing_line.closest_progress(sim.state.position)
                sol = mpcc.solve(mpcc_state, progress)
                if sol.success and sol.inputs.shape[0] > 0:
                    mpcc_success_count += 1
                    cmd = _mpcc_to_command(
                        sol.inputs[0],
                        sim.state,
                        attitude_kp,
                        timestamp_ns=sim.state.timestamp_ns,
                    )
                    # Sanity: command is well-formed.
                    assert 0.0 <= cmd.throttle <= 1.0
                    assert np.isfinite(cmd.roll_rate)
                    assert np.isfinite(cmd.pitch_rate)
                    assert np.isfinite(cmd.yaw_rate)
                    commands_produced += 1
                    sim.send_command(cmd)
                    # Stop once we've proven the handoff works for several
                    # ticks — we're not testing the controller, just wiring.
                    if mpcc_success_count >= 20:
                        break
            else:
                # Before switchover, hold hover to avoid crashing the sim.
                sim.send_command(
                    DroneCommand(
                        timestamp_ns=sim.state.timestamp_ns,
                        throttle=0.5,
                        roll_rate=0.0,
                        pitch_rate=0.0,
                        yaw_rate=0.25,  # gentle yaw to see more gates
                    )
                )

        assert len(mapper.waypoints()) >= MIN_MAPPED_GATES_FOR_MPCC, (
            f"Mapper only accumulated {len(mapper.waypoints())} waypoints — can't test MPCC handoff"
        )
        assert mpcc is not None, "MPCC never got built despite enough mapped gates"
        assert racing_line is not None
        assert mpcc_success_count >= 5, (
            f"MPCC only succeeded {mpcc_success_count} times — handoff may be broken"
        )
        assert commands_produced >= 5
