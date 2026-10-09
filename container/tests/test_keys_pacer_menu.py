"""Tests for the sidecar's key tracking, tick pacing and menu navigation.

These are the pieces that decide *what* is pressed and *when*; they are
exercised with a recording keyboard and a manual clock, so no X server or
game is involved.
"""

from __future__ import annotations

import pytest

from constants import KEY_ORDER
from container.sidecar.keys import KEYSYM_NAMES, KeyTracker
from container.sidecar.menu import MenuConfig, MenuNavigator, NavigationError, tap
from container.sidecar.pacer import Pacer
from container.sidecar.screens import Screen
from container.tests.fakes import FakeClock, FakeKeyboard, FakeMenuGame

# --- KeyTracker -----------------------------------------------------------


def test_tracker_sends_only_changes_and_keeps_keys_held() -> None:
    keyboard = FakeKeyboard()
    tracker = KeyTracker(keyboard, ("w", "a", "s"))

    tracker.apply([True, False, True])
    tracker.apply([True, False, True])
    tracker.apply([False, True, True])

    assert keyboard.events == [
        ("press", "w"),
        ("press", "s"),
        ("release", "w"),
        ("press", "a"),
    ]
    assert tracker.held == frozenset({"a", "s"})
    assert keyboard.flushes == 3


def test_tracker_release_all_releases_exactly_the_held_keys() -> None:
    keyboard = FakeKeyboard()
    tracker = KeyTracker(keyboard, ("w", "a", "s"))
    tracker.apply([True, True, False])
    keyboard.events.clear()

    tracker.release_all()

    assert sorted(keyboard.events) == [("release", "a"), ("release", "w")]
    assert tracker.held == frozenset()


def test_tracker_rejects_wrong_length() -> None:
    tracker = KeyTracker(FakeKeyboard(), ("w", "a"))
    with pytest.raises(ValueError, match="expected 2 keys"):
        tracker.apply([True])


def test_every_action_and_menu_key_has_a_keysym() -> None:
    assert set(KEY_ORDER) <= set(KEYSYM_NAMES)
    assert {"enter", "left", "r"} <= set(KEYSYM_NAMES)


# --- Pacer ----------------------------------------------------------------


def test_pacer_ticks_at_fixed_rate_without_accumulating_drift() -> None:
    clock = FakeClock()
    pacer = Pacer(20, clock=clock, sleep=clock.sleep)
    start = clock.now

    for work_s in (0.01, 0.03, 0.0):
        clock.now += work_s
        assert pacer.wait_next() == 0.0

    # Deadlines are start+0.05, +0.10, +0.15 whatever the work took.
    assert clock.now == pytest.approx(start + 0.01 + 0.15)


def test_pacer_reports_lateness_and_restarts_schedule() -> None:
    clock = FakeClock()
    pacer = Pacer(20, clock=clock, sleep=clock.sleep)
    pacer.wait_next()

    clock.now += 0.25
    late_ms = pacer.wait_next()

    assert late_ms == pytest.approx(200.0)
    sleeps_before = len(clock.sleeps)
    # No burst of catch-up ticks: the next tick is a full period away.
    assert pacer.wait_next() == 0.0
    assert clock.sleeps[sleeps_before] == pytest.approx(0.05)


def test_pacer_reset_forgets_schedule() -> None:
    clock = FakeClock()
    pacer = Pacer(20, clock=clock, sleep=clock.sleep)
    pacer.wait_next()
    clock.now += 30.0

    pacer.reset()

    assert pacer.wait_next() == 0.0


def test_pacer_rejects_non_positive_rate() -> None:
    with pytest.raises(ValueError, match="tick_hz"):
        Pacer(0)


# --- Menu navigation ------------------------------------------------------

_MENUS_FROM_INTRO = [
    Screen.INTRO,
    Screen.TITLE,
    Screen.FILE_SELECT,
    Screen.MAIN_MENU,
    Screen.CHARACTER_SELECT,
]


def _navigator(game: FakeMenuGame, config: MenuConfig | None = None) -> MenuNavigator:
    return MenuNavigator(
        game.grab,
        game.keyboard,
        game,
        config if config is not None else MenuConfig(),
        clock=game.clock,
        sleep=game.clock.sleep,
    )


def test_tap_holds_the_key_before_releasing_it() -> None:
    keyboard = FakeKeyboard()
    clock = FakeClock()

    tap(keyboard, "enter", 0.1, clock.sleep)

    assert keyboard.events == [("press", "enter"), ("release", "enter")]
    assert clock.sleeps == [0.1]
    assert keyboard.flushes == 2


def test_navigator_walks_every_menu_from_the_intro_to_a_random_run() -> None:
    game = FakeMenuGame(_MENUS_FROM_INTRO)

    _navigator(game).leave_menus(game.clock() + 60.0)

    assert game.pressed == ["enter"] * 4 + ["left", "enter"]
    assert game.screen is Screen.UNKNOWN


def test_navigator_starts_from_whatever_screen_is_showing() -> None:
    # The intro ended by itself before the reset arrived.
    game = FakeMenuGame(_MENUS_FROM_INTRO[1:])

    _navigator(game).leave_menus(game.clock() + 60.0)

    assert game.pressed == ["enter"] * 3 + ["left", "enter"]


def test_navigator_waits_out_loading_instead_of_pressing_blind() -> None:
    game = FakeMenuGame(_MENUS_FROM_INTRO, loading_s=45.0, transition_s=8.0)

    _navigator(game).leave_menus(game.clock() + 300.0)

    assert game.pressed == ["enter"] * 4 + ["left", "enter"]


def test_navigator_repeats_a_press_the_game_dropped() -> None:
    game = FakeMenuGame(_MENUS_FROM_INTRO, dropped_presses=1)
    config = MenuConfig()

    started = game.clock()
    _navigator(game, config).leave_menus(started + 60.0)

    assert game.pressed == ["enter"] * 5 + ["left", "enter"]
    assert game.clock() - started >= config.press_retry_s


def test_navigator_presses_nothing_while_the_random_roll_plays() -> None:
    # The roll shows other characters and flashes "Random" again; a Left or
    # a second Enter during it would change or restart the selection.
    game = FakeMenuGame(_MENUS_FROM_INTRO, roll_s=6.0)

    _navigator(game).leave_menus(game.clock() + 60.0)

    assert game.pressed == ["enter"] * 4 + ["left", "enter"]


def test_navigator_fails_loudly_on_a_screen_it_cannot_leave() -> None:
    game = FakeMenuGame([Screen.TITLE], dropped_presses=10_000)

    with pytest.raises(NavigationError, match="'title'"):
        _navigator(game).leave_menus(game.clock() + 30.0)


def test_navigator_names_an_unrecognised_screen_when_it_gives_up() -> None:
    game = FakeMenuGame([], loading_s=1e9)

    with pytest.raises(NavigationError, match="'unknown'"):
        _navigator(game).leave_menus(game.clock() + 30.0)

    assert game.pressed == []


def test_menu_config_env_overrides_keep_field_types() -> None:
    config = MenuConfig.from_env({"MENU_NAVIGATION_TIMEOUT_S": "600", "OTHER": "x"})

    assert config.navigation_timeout_s == 600.0
    assert config.tap_hold_s == MenuConfig().tap_hold_s


def test_menu_config_rejects_unparseable_override() -> None:
    with pytest.raises(ValueError, match="could not convert"):
        MenuConfig.from_env({"MENU_PRESS_RETRY_S": "soon"})
