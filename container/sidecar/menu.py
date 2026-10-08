"""Scripted key sequences that take the game from launch to the start of a run."""

from __future__ import annotations

from dataclasses import dataclass, fields
import time
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping

    from container.sidecar.keys import Keyboard

_ENV_PREFIX = "MENU_"


@dataclass(frozen=True)
class MenuConfig:
    """Timings and key counts for menu navigation.

    Every field can be overridden with a ``MENU_<FIELD_NAME>`` environment
    variable, e.g. ``MENU_BOOT_WAIT_S=30``.

    Attributes:
        boot_wait_s: Seconds to let the game settle after its window appears.
        intro_skip_presses: Enter presses that skip the intro cinematic.
        menu_enter_presses: Enter presses from the title screen to character
            select.
        random_left_presses: Left presses from the default character to
            "Random Character".
        tap_hold_s: How long each key is held down. Long enough to register,
            short enough that menus do not auto-repeat the press.
        tap_gap_s: Pause after each key press, to let menu animations finish.
        run_start_wait_s: Seconds to wait after confirming the character
            before control is handed to the agent; covers the random
            character roll and the run's loading screen.
        restart_hold_s: How long R is held to quick-restart a run.
        pause_settle_s: Seconds the pause menu needs to finish opening.
        pause_check_gap_s: Gap between the two frames compared to confirm
            that the game is frozen.
        resume_settle_s: Seconds the pause menu needs to close before the
            game accepts input again.
    """

    boot_wait_s: float = 20.0
    intro_skip_presses: int = 1
    menu_enter_presses: int = 3
    random_left_presses: int = 1
    tap_hold_s: float = 0.12
    tap_gap_s: float = 2.0
    run_start_wait_s: float = 10.0
    restart_hold_s: float = 2.0
    pause_settle_s: float = 0.5
    pause_check_gap_s: float = 0.15
    resume_settle_s: float = 0.5

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


@dataclass(frozen=True)
class Wait:
    """Script step: do nothing for ``seconds``."""

    seconds: float


@dataclass(frozen=True)
class Tap:
    """Script step: hold ``key`` for ``hold_s`` seconds, then release it."""

    key: str
    hold_s: float


Step = Wait | Tap


def _taps(key: str, count: int, config: MenuConfig) -> list[Step]:
    """Return *count* taps of *key*, each followed by the configured gap."""
    steps: list[Step] = []
    for _ in range(count):
        steps.extend((Tap(key, config.tap_hold_s), Wait(config.tap_gap_s)))
    return steps


def boot_script(config: MenuConfig) -> list[Step]:
    """Steps from a freshly opened game window to the first room of a run.

    Waits for the game to settle, skips the intro, enters character select,
    moves to "Random Character" and confirms.
    """
    return [
        Wait(config.boot_wait_s),
        *_taps("enter", config.intro_skip_presses, config),
        *_taps("enter", config.menu_enter_presses, config),
        *_taps("left", config.random_left_presses, config),
        *_taps("enter", 1, config),
        Wait(config.run_start_wait_s),
    ]


def restart_script(config: MenuConfig) -> list[Step]:
    """Steps that quick-restart the current run by holding R."""
    return [Tap("r", config.restart_hold_s), Wait(config.run_start_wait_s)]


def pause_toggle_script(config: MenuConfig, settle_s: float) -> list[Step]:
    """Steps that open or close the in-game pause menu with Escape."""
    return [Tap("escape", config.tap_hold_s), Wait(settle_s)]


def run_script(
    steps: list[Step],
    keyboard: Keyboard,
    sleep: Callable[[float], None] = time.sleep,
) -> None:
    """Execute *steps* in order against *keyboard*.

    Args:
        steps: Script to run.
        keyboard: Where key events are sent.
        sleep: Function that blocks for a number of seconds.
    """
    for step in steps:
        if isinstance(step, Wait):
            sleep(step.seconds)
            continue
        keyboard.press(step.key)
        keyboard.flush()
        sleep(step.hold_s)
        keyboard.release(step.key)
        keyboard.flush()
