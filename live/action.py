#!/usr/bin/env python3
"""Send keyboard actions to The Binding of Isaac via Linux uinput.

Uses kernel-level event injection (evdev /dev/uinput) for the lowest
possible latency — no X11/Wayland round-trip.  Tracks currently-held keys
and only sends delta events: presses new keys, releases old ones, then
calls ``syn()`` once per frame.

Usage (RL training loop)::

    from action import ActionSender

    with ActionSender() as sender:
        sender.focus_game()
        while True:
            keys = agent.predict(frame)  # list[str] or list[int]
            sender.apply(keys)  # string-based
            #  OR
            sender.apply_vector(keys)  # index-based, fastest path
"""

from __future__ import annotations

import subprocess
import time
from typing import TYPE_CHECKING, Self

from evdev import UInput, ecodes

from constants import KEY_ORDER

if TYPE_CHECKING:
    from collections.abc import Iterable

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_ISAAC_WINDOW_NAME = "Binding of Isaac"

_NAME_TO_CODE: dict[str, int] = {
    "w": ecodes.KEY_W,
    "a": ecodes.KEY_A,
    "s": ecodes.KEY_S,
    "d": ecodes.KEY_D,
    "up": ecodes.KEY_UP,
    "down": ecodes.KEY_DOWN,
    "left": ecodes.KEY_LEFT,
    "right": ecodes.KEY_RIGHT,
    "e": ecodes.KEY_E,
    "space": ecodes.KEY_SPACE,
    "q": ecodes.KEY_Q,
    "tab": ecodes.KEY_TAB,
    "ctrl": ecodes.KEY_LEFTCTRL,
}

_NAME_TO_IDX: dict[str, int] = {name: i for i, name in enumerate(KEY_ORDER)}

_ALL_CODES: list[int] = [_NAME_TO_CODE[name] for name in KEY_ORDER]

_N_KEYS = len(KEY_ORDER)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _find_isaac_window() -> str | None:
    """Return the X11 window ID of the Isaac window, or None."""
    result = subprocess.run(
        ["xdotool", "search", "--name", _ISAAC_WINDOW_NAME],
        capture_output=True,
        text=True,
        timeout=3,
        check=False,
    )
    ids = result.stdout.strip().split()
    return ids[0] if ids else None


# ---------------------------------------------------------------------------
# ActionSender
# ---------------------------------------------------------------------------


class ActionSender:
    """Low-latency keyboard action injection for The Binding of Isaac.

    Opens a Linux uinput device and tracks the currently-held key state
    as a boolean mask over ``KEY_ORDER``.  Each call to ``apply()`` or
    ``apply_vector()`` diffs the desired state against the current state
    and sends only the key-up / key-down events that changed, followed by
    a single ``syn()``.

    The fastest path for the RL inference loop is ``apply_vector()``,
    which accepts a binary ``list[int]`` keyed by ``KEY_ORDER`` —
    zero string operations, zero dictionary lookups.
    """

    def __init__(self, device_name: str = "isaac-rl-keyboard") -> None:
        self._name_to_code = _NAME_TO_CODE
        self._name_to_idx = _NAME_TO_IDX
        self._all_codes = _ALL_CODES
        self._n_keys = _N_KEYS
        self._pressed: list[bool] = [False] * self._n_keys

        try:
            self._ui = UInput(
                events={ecodes.EV_KEY: self._all_codes},
                name=device_name,
            )
        except PermissionError:
            raise PermissionError(
                "Cannot open /dev/uinput. Add your user to the 'input' group "
                "or run as root."
            ) from None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def apply(self, keys: Iterable[str]) -> None:
        """Set the currently-held keys to *keys*.

        Only keys present in *keys* are held; any previously-held key
        not in *keys* is released.  Sends one event batch followed by
        a single ``syn()``.

        Args:
            keys: Iterable of key names from ``KEY_ORDER``.
        """
        desired = [False] * self._n_keys
        for key in keys:
            idx = self._name_to_idx[key]
            desired[idx] = True
        self._apply_mask(desired)

    def apply_vector(self, keys: list[int]) -> None:
        """Set the currently-held keys from a binary vector.

        The fastest action-application path.  *keys* is a ``list[int]``
        of length ``len(KEY_ORDER)`` where ``keys[i] == 1`` means
        "press key ``KEY_ORDER[i]``" and ``0`` means "release".

        Args:
            keys: Binary vector, same length as ``KEY_ORDER``.
        """
        if len(keys) != self._n_keys:
            raise ValueError(f"Expected {self._n_keys} keys, got {len(keys)}")
        desired = [bool(k) for k in keys]
        self._apply_mask(desired)

    def press(self, key: str) -> None:
        """Press and hold a single key without releasing others.

        Non-blocking — the key stays held until ``apply()`` or
        ``release_all()`` is called.
        """
        desired = list(self._pressed)
        desired[self._name_to_idx[key]] = True
        self._apply_mask(desired)

    def release(self, key: str) -> None:
        """Release a single key without affecting others."""
        desired = list(self._pressed)
        desired[self._name_to_idx[key]] = False
        self._apply_mask(desired)

    def release_all(self) -> None:
        """Release every currently-held key."""
        self._apply_mask([False] * self._n_keys)

    def tap(self, key: str, duration_s: float = 0.02) -> None:
        """Press a key, wait briefly, then release.

        Blocks for *duration_s* seconds.  Use ``apply()`` /
        ``apply_vector()`` with alternating frames instead for
        non-blocking taps in the RL training loop.

        Args:
            key: Key name from ``KEY_ORDER``.
            duration_s: Seconds to hold before releasing.
        """
        self.press(key)
        time.sleep(duration_s)
        self.release(key)

    def focus_game(self) -> bool:
        """Bring the Isaac window to the foreground and release stuck keys.

        Call once at the start of a session.  Returns ``True`` if the
        window was found and focused, ``False`` otherwise.
        """
        win_id = _find_isaac_window()
        if win_id is None:
            print("Isaac window not found. Is the game running?")
            return False

        subprocess.run(
            ["xdotool", "windowactivate", win_id],
            check=False,
            timeout=2,
        )
        self.release_all()
        return True

    def close(self) -> None:
        """Release all keys and destroy the uinput device."""
        self.release_all()
        self._ui.close()

    # ------------------------------------------------------------------
    # Context manager
    # ------------------------------------------------------------------

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *args: object) -> None:
        self.close()

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def pressed(self) -> frozenset[str]:
        """Frozen set of currently-held key names."""
        return frozenset(KEY_ORDER[i] for i in range(self._n_keys) if self._pressed[i])

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _apply_mask(self, desired: list[bool]) -> None:
        """Write key-up / key-down events for the diff, then syn."""
        codes = self._all_codes
        pressed = self._pressed
        for i in range(self._n_keys):
            if desired[i] and not pressed[i]:
                self._ui.write(ecodes.EV_KEY, codes[i], 1)
            elif not desired[i] and pressed[i]:
                self._ui.write(ecodes.EV_KEY, codes[i], 0)
        self._ui.syn()
        self._pressed = desired


# ---------------------------------------------------------------------------
# Main (smoke test)
# ---------------------------------------------------------------------------


def main() -> None:
    sender = ActionSender()
    try:
        if not sender.focus_game():
            print("Open Isaac and run this script again.")
            return

        print("Isaac focused. Testing all keys (tap each) ...\n")
        for _ in range(3):
            for key in KEY_ORDER:
                print(f"  {key}")
                sender.tap(key, duration_s=0.03)
    finally:
        sender.close()


if __name__ == "__main__":
    main()
