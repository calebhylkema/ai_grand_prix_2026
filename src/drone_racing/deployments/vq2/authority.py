"""Training-only, fail-closed command mux for a bounded gate-1 transition trial."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from drone_racing.deployments.vq2.gate_relative import (
    GateApproachStatus,
    GuidanceDecision,
    StructuredTarget,
)

Command = tuple[float, float, float, float]  # thrust, desired-actual roll/pitch/yaw rates


@dataclass(frozen=True, slots=True)
class TransitionAuthorityConfig:
    """Reviewed bounds for the first gate-0-to-1 authority experiment."""

    max_window_s: float = 2.0
    max_decision_age_s: float = 0.05
    min_track_hits: int = 3
    handover_s: float = 0.20
    thrust_bounds: tuple[float, float] = (0.25, 0.45)
    rate_limits_rps: tuple[float, float, float] = (2.0, 2.0, 1.0)

    def __post_init__(self) -> None:
        values = (self.max_window_s, self.max_decision_age_s, self.handover_s)
        if any(not math.isfinite(value) or value <= 0.0 for value in values):
            raise ValueError("authority timing bounds must be finite and positive")
        if self.min_track_hits < 1:
            raise ValueError("authority requires at least one confirmed track hit")
        lo, hi = self.thrust_bounds
        if not 0.0 <= lo < hi <= 1.0:
            raise ValueError("invalid authority thrust bounds")
        if any(not math.isfinite(value) or value <= 0.0 for value in self.rate_limits_rps):
            raise ValueError("authority rate limits must be finite and positive")


@dataclass(frozen=True, slots=True)
class AuthoritySelection:
    """One auditable command-source decision before the calibrated boundary."""

    requested: bool
    source: str  # legacy | transition
    reason: str
    legacy: Command
    transition: Command | None
    selected: Command
    decision_age_s: float
    authority_elapsed_s: float
    entered: bool
    exited: bool
    slew_limited: bool


class TransitionAuthorityMux:
    """Grant only a fresh TRANSITION_ACQUIRE decision bounded Training authority.

    Gate 0 and anchored guidance always remain legacy. Invalid/stale guidance changes the
    source back to legacy immediately; only the command discontinuity is slew-limited.
    Abort bypasses the slew limiter so existing safety behavior retains priority.
    """

    def __init__(
        self, enabled: bool = False, config: TransitionAuthorityConfig | None = None
    ) -> None:
        self.enabled = bool(enabled)
        self.config = TransitionAuthorityConfig() if config is None else config
        self.reset()

    def reset(self) -> None:
        """Clear all per-attempt handover state."""
        self._source = "legacy"
        self._last_selected: Command | None = None
        self._last_ns: int | None = None

    def select(
        self,
        *,
        now_ns: int,
        transition_start_ns: int,
        active_gate: int,
        anchor_gate: int,
        state_sequence: int,
        target: StructuredTarget,
        decision: GuidanceDecision,
        legacy: Command,
        abort_reason: str = "",
    ) -> AuthoritySelection:
        """Choose a desired-actual command while enforcing every authority gate."""
        previous_source = self._source
        transition = self._transition_command(decision)
        age_s = decision.age_s(now_ns)
        elapsed_s = (
            (now_ns - transition_start_ns) * 1e-9 if transition_start_ns > 0 else float("nan")
        )
        source, reason = self._eligibility(
            now_ns=now_ns,
            transition_start_ns=transition_start_ns,
            active_gate=active_gate,
            anchor_gate=anchor_gate,
            state_sequence=state_sequence,
            target=target,
            decision=decision,
            abort_reason=abort_reason,
        )
        desired = transition if source == "transition" and transition is not None else legacy
        # Gate-0/default legacy commands are byte-identical. Slew applies only while entering
        # or leaving an authority interval. Abort/failsafe retains immediate legacy priority.
        should_slew = source == "transition" or previous_source == "transition"
        if abort_reason:
            selected, limited = legacy, False
        elif should_slew:
            selected, limited = self._slew(desired, now_ns)
        else:
            selected, limited = desired, False
        self._source = source
        self._last_selected = selected
        self._last_ns = now_ns
        return AuthoritySelection(
            requested=self.enabled,
            source=source,
            reason=reason,
            legacy=legacy,
            transition=transition,
            selected=selected,
            decision_age_s=age_s,
            authority_elapsed_s=elapsed_s,
            entered=source == "transition" and previous_source != "transition",
            exited=source != "transition" and previous_source == "transition",
            slew_limited=limited,
        )

    def _eligibility(
        self,
        *,
        now_ns: int,
        transition_start_ns: int,
        active_gate: int,
        anchor_gate: int,
        state_sequence: int,
        target: StructuredTarget,
        decision: GuidanceDecision,
        abort_reason: str,
    ) -> tuple[str, str]:
        if abort_reason:
            return "legacy", f"abort:{abort_reason}"
        if not self.enabled:
            return "legacy", "disabled"
        if active_gate != 1:
            return "legacy", "gate_not_one"
        if anchor_gate == 1:
            return "legacy", "reanchored"
        if anchor_gate != 0:
            return "legacy", "anchor_mismatch"
        if transition_start_ns <= 0:
            return "legacy", "no_transition_start"
        elapsed_s = (now_ns - transition_start_ns) * 1e-9
        if elapsed_s < 0.0 or elapsed_s > self.config.max_window_s:
            return "legacy", "window_expired"
        age_s = decision.age_s(now_ns)
        if age_s < 0.0 or age_s > self.config.max_decision_age_s:
            return "legacy", "stale_decision"
        if decision.mode != "transition" or decision.active_gate != 1 or decision.anchor_gate != 0:
            return "legacy", "wrong_decision_regime"
        if state_sequence != decision.state_sequence:
            return "legacy", "state_sequence_mismatch"
        if decision.result.status is not GateApproachStatus.TRANSITION_ACQUIRE:
            return "legacy", decision.result.status.value
        if decision.result.setpoint is None:
            return "legacy", "missing_setpoint"
        if self._transition_command(decision) is None:
            return "legacy", "invalid_setpoint"
        if not target.valid or target.active_gate_at_receipt != 1 or target.race_gate_id != 1:
            return "legacy", "wrong_target_gate"
        if target.track_id < 0 or target.track_id != decision.track_id:
            return "legacy", "track_mismatch"
        if target.sequence != decision.target_sequence:
            return "legacy", "target_sequence_mismatch"
        if target.track_hits < self.config.min_track_hits:
            return "legacy", "unconfirmed_track"
        if target.association_refused:
            return "legacy", "association_refused"
        if target.wrong_turn:
            return "legacy", "wrong_turn"
        if target.expected_turn_sign != 1.0:
            return "legacy", "turn_sign_missing"
        return "transition", "transition_acquire"

    @staticmethod
    def _transition_command(decision: GuidanceDecision) -> Command | None:
        setpoint = decision.result.setpoint
        if setpoint is None:
            return None
        command = (
            float(setpoint.collective_thrust),
            float(setpoint.body_rates_rps[0]),
            float(setpoint.body_rates_rps[1]),
            float(setpoint.body_rates_rps[2]),
        )
        return command if all(math.isfinite(value) for value in command) else None

    def _slew(self, desired: Command, now_ns: int) -> tuple[Command, bool]:
        if self._last_selected is None or self._last_ns is None:
            return desired, False
        dt = max(0.0, (now_ns - self._last_ns) * 1e-9)
        if dt == 0.0:
            return self._last_selected, self._last_selected != desired
        lo, hi = self.config.thrust_bounds
        full_spans = np.array(
            [hi - lo, *(2.0 * value for value in self.config.rate_limits_rps)],
            dtype=np.float64,
        )
        max_delta = full_spans * min(dt / self.config.handover_s, 1.0)
        previous = np.asarray(self._last_selected, dtype=np.float64)
        target = np.asarray(desired, dtype=np.float64)
        selected = previous + np.clip(target - previous, -max_delta, max_delta)
        command = tuple(float(value) for value in selected)
        return command, not np.allclose(selected, target, rtol=0.0, atol=1e-12)  # type: ignore[return-value]
