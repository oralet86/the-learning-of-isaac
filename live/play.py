#!/usr/bin/env python3
"""End-to-end live gameplay loop.

Press ``h`` to toggle agent control on/off (starts OFF).
Press ``Ctrl+C`` to exit.

Usage::

    python -m live.play
"""

from __future__ import annotations

import sys

import numpy as np
from pynput import keyboard

from live.action import ActionSender
from live.env import IsaacEnv
from live.model import RandomAgent
from live.preprocess import FramePreprocessor
from live.stream_game import StreamGame

_RELEASE_ALL = [0] * 13


def main() -> None:
    sender = ActionSender()
    stream = StreamGame()
    preprocessor = FramePreprocessor()
    agent = RandomAgent()

    env = IsaacEnv(stream, sender, preprocessor)

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

    try:
        while True:
            # --- Toggle detection ---
            if toggle_active != prev_active:
                prev_active = toggle_active
                if toggle_active:
                    print(f"  Agent: ON  (frame {stream.frame_count})")
                else:
                    sender.release_all()
                    print(f"  Agent: OFF (frame {stream.frame_count})")

            if toggle_active:
                action = agent.predict(obs)
                obs, _, _, _, _ = env.step(np.array(action))
            else:
                obs, _, _, _, _ = env.step(np.array(_RELEASE_ALL))

            if stream.frame_count % 120 == 0:
                print(
                    f"  frame {stream.frame_count:>6}  fps {stream.effective_fps:5.1f}"
                )

    except KeyboardInterrupt:
        print("\nStopping...")
    else:
        print(
            f"\nERROR: stream ended after {stream.frame_count} frames",
            file=sys.stderr,
        )
        sys.exit(1)
    finally:
        listener.stop()
        stream.stop()
        sender.close()


if __name__ == "__main__":
    main()
