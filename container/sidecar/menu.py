"""Key presses that take the game from launch to the start of a run.

Navigation is closed-loop: the navigator looks at the screen, presses the one
key that screen calls for and waits for the screen to change before pressing
anything else. How long the game takes to load therefore does not matter, and
neither does the screen navigation starts from.
"""

from __future__ import annotations

from dataclasses import dataclass, fields
import logging
import time
from typing import TYPE_CHECKING, Protocol

from container.sidecar.screens import Screen

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping

    import numpy as np

    from container.sidecar.keys import Keyboard

logger = logging.getLogger(__name__)

_ENV_PREFIX = "MENU_"

# The key that moves each screen one step closer to a run as "Random".
_KEY_FOR_SCREEN: dict[Screen, str] = {
    Screen.INTRO: "enter",
    Screen.TITLE: "enter",
    Screen.FILE_SELECT: "enter",
    Screen.MAIN_MENU: "enter",
    Screen.CHARACTER_SELECT: "left",
    Screen.RANDOM_SELECTED: "enter",
}


class ScreenReader(Protocol):
    """What navigation needs from a classifier; see ``ScreenClassifier``."""

    def classify(self, frame: np.ndarray) -> Screen:
        """Return the screen *frame* shows, or ``Screen.UNKNOWN``."""
        ...

    def distances(self, frame: np.ndarray) -> dict[str, float]:
        """Return the distance from *frame* to every screen reference."""
        ...


class NavigationError(RuntimeError):
    """Raised when the menus could not be navigated in time."""


@dataclass(frozen=True)
class MenuConfig:
    """Timings for menu navigation and pausing.

    Every field can be overridden with a ``MENU_<FIELD_NAME>`` environment
    variable, e.g. ``MENU_NAVIGATION_TIMEOUT_S=600``.

    Attributes:
        tap_hold_s: How long each key is held down. Long enough to register,
            short enough that menus do not auto-repeat the press.
        poll_s: How often the screen is looked at while waiting for a change.
        screen_settle_s: How long a screen must be showing before a key is
            pressed on it; menus ignore input while they slide in.
        press_retry_s: Press again when a screen has not changed this long
            after a key press (the game dropped the press).
        run_start_settle_s: How long the screen must stay off the menus after
            the character is confirmed before the run counts as loading.
        navigation_timeout_s: Give up when the menus have not led to a run
            after this long. Covers the game's loading times under load.
        restart_hold_s: How long R is held to quick-restart a run.
        restart_settle_s: Wait after a quick restart before checking that the
            new run accepts input.
        run_start_timeout_s: When a restarted run does not accept input
            after this long, fall back to navigating the menus.
        pause_timeout_s: How long the pause menu may take to open or close
            after the pause key before the press counts as ignored.
        pause_retry_s: Wait before pressing the pause key again while a
            starting run still ignores it.
        pause_settle_s: Seconds the pause menu needs to finish opening once
            it is recognisable. The game ignores the pause key until then.
        resume_settle_s: Seconds the game needs after the pause menu started
            closing before it accepts input again. Keys pressed earlier
            would still steer the menu, which can end the run.
    """

    tap_hold_s: float = 0.12
    poll_s: float = 0.1
    screen_settle_s: float = 0.3
    press_retry_s: float = 4.0
    run_start_settle_s: float = 1.0
    navigation_timeout_s: float = 180.0
    restart_hold_s: float = 2.0
    restart_settle_s: float = 1.0
    run_start_timeout_s: float = 90.0
    pause_timeout_s: float = 0.6
    pause_retry_s: float = 0.5
    pause_settle_s: float = 0.4
    resume_settle_s: float = 0.4

    @classmethod
    def from_env(cls, environ: Mapping[str, str]) -> MenuConfig:
        """Build a config from defaults plus ``MENU_*`` overrides.

        Raises:
            ValueError: If an override cannot be parsed as its field's type.
        """
        overrides: dict[str, float | int] = {}
        for field in fields(cls):
            raw = environ.get(_ENV_PREFIX + field.name.upper())
            if raw is None:
                continue
            parser = int if isinstance(field.default, int) else float
            overrides[field.name] = parser(raw)
        return cls(**overrides)  # type: ignore[arg-type]


def tap(
    keyboard: Keyboard,
    key: str,
    hold_s: float,
    sleep: Callable[[float], None] = time.sleep,
) -> None:
    """Hold *key* for *hold_s* seconds, then release it.

    Args:
        keyboard: Where key events are sent.
        key: Key name known to the keyboard.
        hold_s: Seconds between the press and the release.
        sleep: Function that blocks for a number of seconds.
    """
    keyboard.press(key)
    keyboard.flush()
    sleep(hold_s)
    keyboard.release(key)
    keyboard.flush()


class MenuNavigator:
    """Walks the front-end menus into a run with a random character."""

    def __init__(
        self,
        grab: Callable[[], np.ndarray],
        keyboard: Keyboard,
        classifier: ScreenReader,
        config: MenuConfig,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        """Create a navigator.

        Args:
            grab: Returns the game window's current ``(H, W, 3)`` RGB frame.
            keyboard: Where key events are sent.
            classifier: Recognises the screen in a frame.
            config: Navigation timings.
            clock: Monotonic time source in seconds.
            sleep: Function that blocks for a number of seconds.
        """
        self._grab = grab
        self._keyboard = keyboard
        self._classifier = classifier
        self._config = config
        self._clock = clock
        self._sleep = sleep

    def leave_menus(self, deadline: float) -> None:
        """Press through the menus until a run as "Random" starts loading.

        Works from any front-end screen, including the intro and screens it
        has already acted on, so it can be called again if the game turns up
        in a menu later. Returns once the character has been confirmed and
        the menus are gone; the run itself is still loading at that point.

        Args:
            deadline: ``clock()`` time at which to give up.

        Raises:
            NavigationError: If the deadline passes first. The message says
                what the screen looked like, which tells a slow machine
                (a known screen, or a loading screen) from a screen the
                references do not cover.
        """
        config = self._config
        showing = Screen.UNKNOWN
        showing_since = self._clock()
        pressed_on: Screen | None = None
        confirmed = False
        frame = self._grab()
        while self._clock() < deadline:
            frame = self._grab()
            screen = self._classifier.classify(frame)
            now = self._clock()
            if screen is not showing:
                logger.info("screen: %s -> %s", showing.value, screen.value)
                showing, showing_since = screen, now
            shown_for = now - showing_since
            if confirmed and screen is Screen.UNKNOWN:
                if shown_for >= config.run_start_settle_s:
                    return
            elif confirmed and screen is Screen.CHARACTER_SELECT:
                # The random roll flips through characters on this page;
                # pressing Left now would change the selection.
                pass
            elif screen is not Screen.UNKNOWN and shown_for >= (
                config.press_retry_s if screen is pressed_on else config.screen_settle_s
            ):
                # A screen that just appeared gets time to finish sliding in.
                # The screen the last press was made on gets longer, to react,
                # and has to show without a break for that long: the random
                # roll flashes "Random" again, which is not a dropped press.
                key = _KEY_FOR_SCREEN[screen]
                tap(self._keyboard, key, config.tap_hold_s, self._sleep)
                pressed_on = screen
                confirmed = screen is Screen.RANDOM_SELECTED
                showing_since = self._clock()
            self._sleep(config.poll_s)
        distances = ", ".join(
            f"{name} {distance:.1f}"
            for name, distance in self._classifier.distances(frame).items()
        )
        raise NavigationError(
            f"menus did not lead to a run in time; the screen is "
            f"{showing.value!r} (distance to each reference: {distances})"
        )
