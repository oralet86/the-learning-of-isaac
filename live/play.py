#!/usr/bin/env python3
"""Inference-mode entry point: the agent plays the real game on this machine.

Frames come from OBS and keys go out through uinput, so the game itself is
untouched. Start the run by hand, then hand control to the agent.

Press ``h`` to toggle agent control on/off (starts OFF).
Press ``Ctrl+C`` to exit.

Usage::

    python -m live.play
"""

from __future__ import annotations

import sys

import numpy as np
from pynput import keyboard

from constants import KEY_ORDER, STEP_HZ
from live.action import ActionSender
from live.env import IsaacEnv
from live.local import LocalGame
from live.model import RandomAgent
from live.preprocess import FramePreprocessor
from live.stream_game import StreamGame

_RELEASE_ALL = [0] * len(KEY_ORDER)
_REPORT_EVERY_STEPS = STEP_HZ * 5


def main() -> None:
    sender = ActionSender()
    stream = StreamGame()
    agent = RandomAgent()

    env = IsaacEnv(LocalGame(stream, sender), FramePreprocessor())

    if not sender.focus_game():
        sys.exit(1)

    # ------------------------------------------------------------------
    # Toggle listener (pynput, background thread)
    # ------------------------------------------------------------------

    toggle_active = False

    def _on_press(key: keyboard.Key | keyboard.KeyCode | None) -> None:
        nonlocal toggle_active
        if (
            isinstance(key, keyboard.KeyCode)
            and key.char is not None
            and key.char == "h"
        ):
            toggle_active = not toggle_active

    listener = keyboard.Listener(on_press=_on_press)
    listener.start()

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------

    stream.start()
    obs, _ = env.reset()
    print("Live loop running.  Press 'h' to toggle, Ctrl+C to stop.\n")

    prev_active = False
    steps = 0

    try:
        while True:
            # --- Toggle detection ---
            if toggle_active != prev_active:
                prev_active = toggle_active
                if toggle_active:
                    print(f"  Agent: ON  (step {steps})")
                else:
                    sender.release_all()
                    print(f"  Agent: OFF (step {steps})")

            action = agent.predict(obs) if toggle_active else _RELEASE_ALL
            obs, _, terminated, _, info = env.step(np.array(action))
            steps += 1

            if terminated:
                print(f"  Run ended (step {steps}); agent OFF")
                toggle_active = False

            if steps % _REPORT_EVERY_STEPS == 0:
                print(
                    f"  step {steps:>6}  capture fps {stream.effective_fps:5.1f}"
                    f"  late {info['late_ms']:5.1f} ms"
                )

    except KeyboardInterrupt:
        print("\nStopping...")
    finally:
        listener.stop()
        env.close()
        stream.stop()
        sender.close()


if __name__ == "__main__":
    main()
