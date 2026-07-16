"""Deterministic tests for absolute-deadline VQ2 command scheduling."""

from __future__ import annotations

from drone_racing.deployments.vq2.timing import DeadlineScheduler


class FakeClock:
    """Nanosecond clock whose waiter advances virtual time exactly."""

    def __init__(self) -> None:
        self.now_ns = 0

    def __call__(self) -> int:
        return self.now_ns

    def wait(self, seconds: float) -> None:
        self.now_ns += int(round(seconds * 1e9))

    def work_ms(self, milliseconds: float) -> None:
        self.now_ns += int(round(milliseconds * 1e6))


def test_work_time_does_not_accumulate_into_period() -> None:
    clock = FakeClock()
    scheduler = DeadlineScheduler(60.0, clock_ns=clock, waiter=clock.wait, start_ns=0)
    first = scheduler.wait_next()
    assert first.released_ns == 0
    clock.work_ms(5.0)
    second = scheduler.wait_next()
    clock.work_ms(7.0)
    third = scheduler.wait_next()
    assert second.released_ns == scheduler.period_ns
    assert third.released_ns == 2 * scheduler.period_ns
    assert second.missed_periods == third.missed_periods == 0


def test_late_iteration_skips_expired_deadlines_without_burst() -> None:
    clock = FakeClock()
    scheduler = DeadlineScheduler(50.0, clock_ns=clock, waiter=clock.wait, start_ns=0)
    scheduler.wait_next()
    clock.work_ms(65.0)  # deadlines at 20, 40, 60 ms have expired
    late = scheduler.wait_next()
    assert late.scheduled_ns == 20_000_000
    assert late.missed_periods == 2
    next_tick = scheduler.wait_next()
    assert next_tick.scheduled_ns == 80_000_000
    assert next_tick.released_ns == 80_000_000
    assert next_tick.released_ns - late.released_ns >= scheduler.period_ns // 2


def test_scheduler_never_requests_negative_wait() -> None:
    clock = FakeClock()
    waits: list[float] = []

    def wait(seconds: float) -> None:
        waits.append(seconds)
        clock.wait(seconds)

    scheduler = DeadlineScheduler(100.0, clock_ns=clock, waiter=wait, start_ns=0)
    scheduler.wait_next()
    clock.work_ms(15.0)
    scheduler.wait_next()
    scheduler.wait_next()
    assert waits and all(seconds >= 0.0 for seconds in waits)


def test_invalid_frequency_rejected() -> None:
    for hz in (0.0, -1.0, float("nan")):
        try:
            DeadlineScheduler(hz)
        except ValueError:
            pass
        else:  # pragma: no cover - explicit failure message is clearer than parametrizing
            raise AssertionError(f"accepted invalid frequency {hz}")


def test_active_run_contains_no_time_sleep() -> None:
    import inspect

    from drone_racing.deployments.vq2 import mission

    assert "time.sleep(" not in inspect.getsource(mission.run)
