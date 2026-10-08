"""Test doubles shared by the sidecar tests."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from collections.abc import Sequence

    from container.sidecar.protocol import ResetMode


class FakeKeyboard:
    """Records key events instead of sending them to an X server."""

    def __init__(self) -> None:
        self.events: list[tuple[str, str]] = []
        self.flushes = 0

    def press(self, key: str) -> None:
        self.events.append(("press", key))

    def release(self, key: str) -> None:
        self.events.append(("release", key))

    def flush(self) -> None:
        self.flushes += 1


class FakeClock:
    """Manually advanced clock whose ``sleep`` moves time forward."""

    def __init__(self) -> None:
        self.now = 100.0
        self.sleeps: list[float] = []

    def __call__(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.sleeps.append(seconds)
        self.now += seconds


class FakeControl:
    """Stand-in for ``GameSession`` that returns recognisable frames.

    Every frame is filled with the number of steps taken so far, so a test
    can tell which call produced the frame it received.
    """

    def __init__(self, width: int = 8, height: int = 6) -> None:
        self.width = width
        self.height = height
        self.steps: list[list[bool]] = []
        self.resets: list[ResetMode] = []
        self.release_calls = 0
        self.pause_calls = 0
        self.fail_next_step = False

    @property
    def frame_size(self) -> tuple[int, int]:
        return self.width, self.height

    def _frame(self) -> np.ndarray:
        return np.full((self.height, self.width, 3), len(self.steps), dtype=np.uint8)

    def reset(self, mode: ResetMode) -> np.ndarray:
        self.resets.append(mode)
        self.steps.clear()
        return self._frame()

    def step(self, keys: Sequence[bool]) -> tuple[np.ndarray, float]:
        if self.fail_next_step:
            self.fail_next_step = False
            raise RuntimeError("game process is not running")
        self.steps.append(list(keys))
        return self._frame(), 1.5

    def release_all(self) -> None:
        self.release_calls += 1

    def pause(self) -> bool:
        self.pause_calls += 1
        return True
