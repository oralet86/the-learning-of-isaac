"""Tests for the sidecar's key tracking, tick pacing and menu scripts.

These are the pieces that decide *what* is pressed and *when*; they are
exercised with a recording keyboard and a manual clock, so no X server or
game is involved.
"""

from __future__ import annotations

import pytest

from container.sidecar.keys import KEY_ORDER, KEYSYM_NAMES, KeyTracker
from container.sidecar.menu import (
    MenuConfig,
    Tap,
    Wait,
    boot_script,
    restart_script,
    run_script,
)
from container.sidecar.pacer import Pacer
from container.tests.fakes import FakeClock, FakeKeyboard

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


# --- Menu scripts ---------------------------------------------------------


def _tapped_keys(steps: list[Wait | Tap]) -> list[str]:
    return [step.key for step in steps if isinstance(step, Tap)]


def test_boot_script_key_sequence_follows_config() -> None:
    config = MenuConfig(
        intro_skip_presses=1, menu_enter_presses=3, random_left_presses=2
    )

    steps = boot_script(config)

    assert steps[0] == Wait(config.boot_wait_s)
    assert _tapped_keys(steps) == ["enter"] * 4 + ["left"] * 2 + ["enter"]
    assert steps[-1] == Wait(config.run_start_wait_s)


def test_restart_script_holds_r() -> None:
    config = MenuConfig(restart_hold_s=1.5)
    assert restart_script(config)[0] == Tap("r", 1.5)


def test_run_script_presses_holds_and_releases_in_order() -> None:
    keyboard = FakeKeyboard()
    clock = FakeClock()

    run_script([Wait(2.0), Tap("enter", 0.1), Wait(1.0)], keyboard, clock.sleep)

    assert keyboard.events == [("press", "enter"), ("release", "enter")]
    assert clock.sleeps == [2.0, 0.1, 1.0]
    assert keyboard.flushes == 2


def test_menu_config_env_overrides_keep_field_types() -> None:
    config = MenuConfig.from_env(
        {"MENU_BOOT_WAIT_S": "5", "MENU_RANDOM_LEFT_PRESSES": "3", "OTHER": "x"}
    )

    assert config.boot_wait_s == 5.0
    assert config.random_left_presses == 3
    assert config.tap_hold_s == MenuConfig().tap_hold_s


def test_menu_config_rejects_unparseable_override() -> None:
    with pytest.raises(ValueError, match="invalid literal"):
        MenuConfig.from_env({"MENU_RANDOM_LEFT_PRESSES": "two"})
