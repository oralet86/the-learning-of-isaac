"""Keyboard injection through the X server's XTEST extension.

The key-state bookkeeping (:class:`KeyTracker`) is separate from the X11 calls
(:class:`XKeyboard`) so the diffing logic can be tested without a display.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol

from Xlib import XK, X
from Xlib.ext import xtest

from constants import KEY_ORDER

if TYPE_CHECKING:
    from collections.abc import Sequence

    from Xlib.display import Display

# X keysym names for every key the sidecar can press: the action keys in
# `constants.KEY_ORDER` plus the extra keys that only the menu scripts use.
KEYSYM_NAMES: dict[str, str] = {
    "w": "w",
    "a": "a",
    "s": "s",
    "d": "d",
    "up": "Up",
    "down": "Down",
    "left": "Left",
    "right": "Right",
    "e": "e",
    "space": "space",
    "q": "q",
    "tab": "Tab",
    "ctrl": "Control_L",
    "enter": "Return",
    "escape": "Escape",
    "r": "r",
}


class Keyboard(Protocol):
    """Minimal key-injection interface used by the tracker and menu scripts."""

    def press(self, key: str) -> None:
        """Queue a key-down event for *key*."""
        ...

    def release(self, key: str) -> None:
        """Queue a key-up event for *key*."""
        ...

    def flush(self) -> None:
        """Deliver all queued events to the game."""
        ...


class XKeyboard:
    """Injects key events into an X display with XTEST."""

    def __init__(self, display: Display) -> None:
        """Resolve keycodes for every known key.

        Args:
            display: Open connection to the X server the game renders on.

        Raises:
            RuntimeError: If XTEST is unavailable or a key has no keycode.
        """
        if not display.query_extension("XTEST"):
            raise RuntimeError("X server does not provide the XTEST extension")
        self._display = display
        self._keycodes = {
            name: self._resolve_keycode(display, keysym_name)
            for name, keysym_name in KEYSYM_NAMES.items()
        }

    @staticmethod
    def _resolve_keycode(display: Display, keysym_name: str) -> int:
        """Map an X keysym name to a keycode on *display*."""
        keycode = display.keysym_to_keycode(XK.string_to_keysym(keysym_name))
        if keycode == 0:
            raise RuntimeError(f"no keycode for keysym {keysym_name!r}")
        return keycode

    def press(self, key: str) -> None:
        """Queue a key-down event for *key*."""
        xtest.fake_input(self._display, X.KeyPress, self._keycodes[key])

    def release(self, key: str) -> None:
        """Queue a key-up event for *key*."""
        xtest.fake_input(self._display, X.KeyRelease, self._keycodes[key])

    def flush(self) -> None:
        """Send queued events and wait until the server has processed them."""
        self._display.sync()


class KeyTracker:
    """Holds the action keys in a desired state, sending only the changes.

    Keys stay held between calls, so an action persists until the next one
    replaces it.
    """

    def __init__(
        self, keyboard: Keyboard, key_order: Sequence[str] = KEY_ORDER
    ) -> None:
        """Start with every key released.

        Args:
            keyboard: Where key events are sent.
            key_order: Key name for each position of the vectors passed to
                :meth:`apply`.
        """
        self._keyboard = keyboard
        self._key_order = tuple(key_order)
        self._held = [False] * len(self._key_order)

    @property
    def held(self) -> frozenset[str]:
        """Names of the keys currently held down."""
        return frozenset(
            name
            for name, is_held in zip(self._key_order, self._held, strict=True)
            if is_held
        )

    def apply(self, desired: Sequence[bool]) -> None:
        """Press and release keys so exactly the *desired* ones are held.

        Args:
            desired: One flag per key in ``key_order``.

        Raises:
            ValueError: If *desired* has the wrong length.
        """
        if len(desired) != len(self._key_order):
            raise ValueError(
                f"expected {len(self._key_order)} keys, got {len(desired)}"
            )
        for index, (name, want) in enumerate(
            zip(self._key_order, desired, strict=True)
        ):
            if want and not self._held[index]:
                self._keyboard.press(name)
            elif not want and self._held[index]:
                self._keyboard.release(name)
            self._held[index] = bool(want)
        self._keyboard.flush()

    def release_all(self) -> None:
        """Release every held key."""
        self.apply([False] * len(self._key_order))
