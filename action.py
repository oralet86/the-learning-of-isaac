#!/usr/bin/env python3
"""Send keyboard actions to the Isaac game window via uinput virtual keyboard."""

import logging
import subprocess
import time
from typing import TYPE_CHECKING

from evdev import UInput, ecodes

from constants import KEY_ORDER

if TYPE_CHECKING:
    from collections.abc import Iterable

logger = logging.getLogger(__name__)

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


def find_isaac_window() -> str | None:
    result = subprocess.run(
        ["xdotool", "search", "--name", _ISAAC_WINDOW_NAME],
        capture_output=True,
        text=True,
        timeout=3,
        check=False,
    )
    ids = result.stdout.strip().split()
    if not ids:
        return None
    return ids[0]


def focus_isaac(sender: "ActionSender | None" = None) -> bool:
    win_id = find_isaac_window()
    if win_id is None:
        print("Isaac window not found. Is the game running?")
        return False

    subprocess.run(
        ["xdotool", "windowactivate", win_id],
        check=False,
        timeout=3,
    )

    if sender is not None:
        sender.release_all()

    time.sleep(0.05)
    return True


class ActionSender:
    def __init__(self, device_name: str = "isaac-rl-keyboard") -> None:
        self._name_to_code = _NAME_TO_CODE
        self._pressed: set[str] = set()
        self._all_codes = [self._name_to_code[name] for name in KEY_ORDER]

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

    def hold_keys(self, keys: Iterable[str]) -> None:
        keys = set(keys)
        keys_to_press = keys - self._pressed
        keys_to_release = self._pressed - keys

        for name in keys_to_release:
            code = self._name_to_code[name]
            self._ui.write(ecodes.EV_KEY, code, 0)

        for name in keys_to_press:
            code = self._name_to_code[name]
            self._ui.write(ecodes.EV_KEY, code, 1)

        self._ui.syn()
        self._pressed = keys

    def tap(self, key: str, duration_ms: float = 50) -> None:
        self.hold_keys([key])
        time.sleep(duration_ms / 1000.0)
        self.release_all()

    def release_all(self) -> None:
        for name in list(self._pressed):
            code = self._name_to_code[name]
            self._ui.write(ecodes.EV_KEY, code, 0)
        self._ui.syn()
        self._pressed.clear()

    def close(self) -> None:
        self.release_all()
        self._ui.close()

    @property
    def pressed(self) -> frozenset[str]:
        return frozenset(self._pressed)


def main() -> None:
    sender = ActionSender()
    try:
        if not focus_isaac(sender):
            print("Please open Isaac and run the script again.")
            return

        print("Isaac focused. Testing all keys...\n")
        for _ in range(5):
            for key in KEY_ORDER:
                print(f"Pressing: {key}")
                sender.tap(key, duration_ms=60)
    finally:
        sender.close()


if __name__ == "__main__":
    main()
