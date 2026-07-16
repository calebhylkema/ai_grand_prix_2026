"""Absolute-deadline scheduling primitives for the VQ2 command loop."""

from __future__ import annotations

import math
import sys
import threading
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Callable

_COARSE_WAIT = threading.Event()
_SPIN_GUARD_NS = 2_000_000


def _precise_wait(seconds: float) -> None:
    """Windows-friendly high-resolution wait without ``time.sleep``.

    ``Event.wait`` alone commonly coalesces 60 Hz releases into alternating 2/31 ms
    intervals. Sleep coarsely until just before the deadline, then spin only the final
    two milliseconds. The bounded spin trades a small amount of CPU for stable command
    spacing; the simulator is Windows-only and the measured control workload has ample
    CPU headroom.
    """
    duration_ns = max(0, int(round(seconds * 1e9)))
    deadline_ns = time.monotonic_ns() + duration_ns
    coarse_ns = duration_ns - _SPIN_GUARD_NS
    if coarse_ns > 0:
        _COARSE_WAIT.wait(coarse_ns * 1e-9)
    while time.monotonic_ns() < deadline_ns:
        pass


def _windows_waiter() -> Callable[[float], object] | None:
    """Create a Windows high-resolution waitable-timer callback when supported."""
    if sys.platform != "win32":
        return None
    import atexit
    import ctypes
    from ctypes import wintypes

    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    create = kernel32.CreateWaitableTimerExW
    create.argtypes = (ctypes.c_void_p, ctypes.c_wchar_p, wintypes.DWORD, wintypes.DWORD)
    create.restype = wintypes.HANDLE
    set_timer = kernel32.SetWaitableTimer
    set_timer.argtypes = (
        wintypes.HANDLE,
        ctypes.POINTER(ctypes.c_longlong),
        wintypes.LONG,
        ctypes.c_void_p,
        ctypes.c_void_p,
        wintypes.BOOL,
    )
    set_timer.restype = wintypes.BOOL
    wait_one = kernel32.WaitForSingleObject
    wait_one.argtypes = (wintypes.HANDLE, wintypes.DWORD)
    wait_one.restype = wintypes.DWORD
    close = kernel32.CloseHandle
    close.argtypes = (wintypes.HANDLE,)
    close.restype = wintypes.BOOL

    # CREATE_WAITABLE_TIMER_HIGH_RESOLUTION; fall back gracefully on older Windows.
    handle = create(None, None, 0x2, 0x1F0003) or create(None, None, 0, 0x1F0003)
    if not handle:
        return None
    atexit.register(close, handle)

    def wait(seconds: float) -> object:
        if seconds <= 0.0:
            return None
        due = ctypes.c_longlong(-max(1, int(round(seconds * 1e7))))  # relative 100 ns units
        if not set_timer(handle, ctypes.byref(due), 0, None, None, False):
            _precise_wait(seconds)
            return None
        if wait_one(handle, 0xFFFFFFFF) != 0:  # WAIT_OBJECT_0
            _precise_wait(seconds)
        return None

    return wait


_HIGH_RESOLUTION_WAITER = _windows_waiter()


@dataclass(frozen=True, slots=True)
class DeadlineTick:
    """One released command-loop deadline."""

    scheduled_ns: int
    released_ns: int
    lateness_ns: int
    missed_periods: int


class DeadlineScheduler:
    """Release one iteration per absolute monotonic deadline without catch-up bursts.

    Work duration does not accumulate into the period. If an iteration is more than one
    period late, expired deadlines are skipped and the next release remains in the future.
    The default waiter is ``Event.wait`` so production control code never calls
    ``time.sleep``; tests inject a deterministic fake clock/waiter.
    """

    def __init__(
        self,
        hz: float,
        *,
        clock_ns: Callable[[], int] = time.monotonic_ns,
        waiter: Callable[[float], object] | None = None,
        start_ns: int | None = None,
    ) -> None:
        if not math.isfinite(hz) or hz <= 0.0:
            raise ValueError("scheduler frequency must be finite and positive")
        self.period_ns = max(1, int(round(1e9 / hz)))
        self._clock_ns = clock_ns
        default_waiter = _HIGH_RESOLUTION_WAITER or _precise_wait
        self._waiter = default_waiter if waiter is None else waiter
        self._next_deadline_ns = clock_ns() if start_ns is None else int(start_ns)
        self._last_release_ns: int | None = None

    def wait_next(self) -> DeadlineTick:
        """Wait until the next deadline and advance past any expired periods."""
        scheduled = self._next_deadline_ns
        now = self._clock_ns()
        # Never compress two commands into a catch-up burst after a late release. Half a
        # period is the minimum spacing; missed absolute deadlines are still skipped below.
        release_target = scheduled
        if self._last_release_ns is not None:
            release_target = max(release_target, self._last_release_ns + self.period_ns // 2)
        if now < release_target:
            self._waiter((release_target - now) * 1e-9)
            now = self._clock_ns()
        # A spurious/early waiter return must not release a command before its deadline.
        while now < release_target:
            self._waiter((release_target - now) * 1e-9)
            now = self._clock_ns()
        lateness = max(0, now - scheduled)
        missed = lateness // self.period_ns
        self._next_deadline_ns = scheduled + (missed + 1) * self.period_ns
        self._last_release_ns = now
        return DeadlineTick(
            scheduled_ns=scheduled,
            released_ns=now,
            lateness_ns=lateness,
            missed_periods=int(missed),
        )
