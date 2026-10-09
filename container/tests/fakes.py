"""Test doubles shared by the sidecar tests."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from container.sidecar.screens import Screen

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


class FakeMenuGame:
    """A scripted front end: screens that react to key presses over fake time.

    It stands in for the capture, the keyboard and the classifier at once.
    Each frame is filled with the index of the screen it shows, which
    ``classify`` maps back. The menus appear after ``loading_s``; the right
    key on a menu leads, after ``transition_s`` of an unrecognisable
    transition, to the next one. Confirming "Random" plays the roll for
    ``roll_s`` and then leaves the menus for good.
    """

    _SCREENS = tuple(Screen)

    def __init__(
        self,
        menus: Sequence[Screen],
        *,
        loading_s: float = 2.0,
        transition_s: float = 0.5,
        roll_s: float = 3.0,
        dropped_presses: int = 0,
    ) -> None:
        self.clock = FakeClock()
        self.keyboard = self
        self.pressed: list[str] = []
        self._transition_s = transition_s
        self._roll_s = roll_s
        self._dropped_presses = dropped_presses
        # (time from which it shows, screen), in order of appearance.
        self._timeline: list[tuple[float, Screen]] = [(self.clock(), Screen.UNKNOWN)]
        self._upcoming = list(menus)
        self._show_next(self.clock() + loading_s)

    @property
    def screen(self) -> Screen:
        """The screen showing at the current fake time."""
        now = self.clock()
        return [screen for since, screen in self._timeline if since <= now][-1]

    def grab(self) -> np.ndarray:
        return np.full((4, 4, 3), self._SCREENS.index(self.screen), dtype=np.uint8)

    def classify(self, frame: np.ndarray) -> Screen:
        return self._SCREENS[int(frame[0, 0, 0])]

    def distances(self, frame: np.ndarray) -> dict[str, float]:  # noqa: ARG002
        return {"title": 0.0}

    def press(self, key: str) -> None:
        self.pressed.append(key)
        screen = self.screen
        expected = {
            Screen.CHARACTER_SELECT: "left",
            Screen.UNKNOWN: None,
        }.get(screen, "enter")
        if key != expected or self._timeline[-1][0] > self.clock():
            return
        if self._dropped_presses > 0:
            self._dropped_presses -= 1
            return
        now = self.clock()
        if screen is Screen.CHARACTER_SELECT:
            self._timeline.append((now, Screen.RANDOM_SELECTED))
        elif screen is Screen.RANDOM_SELECTED:
            # The roll flips through characters and flashes "Random" again.
            self._timeline += [
                (now, Screen.CHARACTER_SELECT),
                (now + self._roll_s / 2, Screen.RANDOM_SELECTED),
                (now + self._roll_s / 2 + 0.2, Screen.CHARACTER_SELECT),
                (now + self._roll_s, Screen.UNKNOWN),
            ]
        else:
            self._timeline.append((now, Screen.UNKNOWN))
            self._show_next(now + self._transition_s)

    def release(self, key: str) -> None:
        pass

    def flush(self) -> None:
        pass

    def _show_next(self, when: float) -> None:
        if self._upcoming:
            self._timeline.append((when, self._upcoming.pop(0)))
