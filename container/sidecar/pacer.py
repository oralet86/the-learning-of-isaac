"""Fixed-rate tick pacing for the control loop."""

from __future__ import annotations

import time
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Callable


class Pacer:
    """Blocks until the next tick deadline of a fixed-rate control loop.

    Deadlines advance by one period per tick so sleeping jitter does not
    accumulate. If the caller falls behind, the lateness is reported and the
    schedule restarts from the current time instead of bursting to catch up.
    """

    def __init__(
        self,
        tick_hz: float,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        """Create a pacer.

        Args:
            tick_hz: Ticks per second.
            clock: Monotonic time source in seconds.
            sleep: Function that blocks for a number of seconds.

        Raises:
            ValueError: If *tick_hz* is not positive.
        """
        if tick_hz <= 0:
            raise ValueError(f"tick_hz must be positive, got {tick_hz}")
        self._period = 1.0 / tick_hz
        self._clock = clock
        self._sleep = sleep
        self._deadline: float | None = None

    @property
    def period(self) -> float:
        """Seconds between ticks."""
        return self._period

    def reset(self) -> None:
        """Forget the schedule; the next tick is one period after it is requested."""
        self._deadline = None

    def wait_next(self) -> float:
        """Block until the next tick deadline.

        Returns:
            Milliseconds by which the deadline had already passed when this
            was called; ``0.0`` if the caller was on time.
        """
        now = self._clock()
        if self._deadline is None:
            self._deadline = now
        self._deadline += self._period
        remaining = self._deadline - now
        if remaining >= 0:
            self._sleep(remaining)
            return 0.0
        self._deadline = now
        return -remaining * 1000.0
