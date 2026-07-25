#!/usr/bin/env python3
"""End-to-end live gameplay loop.

Pipelines::

    input               preprocess        agent           output
    ─────────────────   ───────────────   ─────────────   ──────────────
    StreamGame          FramePreproces-   RandomAgent     ActionSender
    (obs virtual cam)   sor (no-op)       (random keys)   (uinput)

Each stage is a swappable component.  Insert filters, auxiliary models,
or PPO wrappers between stages without modifying the loop.

Press ``h`` to toggle pipeline processing on/off (starts OFF).
Press ``Ctrl+C`` to exit.

Usage::

    python -m live.play
"""

from __future__ import annotations

import sys

from pynput import keyboard

from live.action import ActionSender
from live.model import RandomAgent
from live.preprocess import FramePreprocessor
from live.stream_game import StreamGame


def main() -> None:
    sender = ActionSender()
    stream = StreamGame()
    preprocessor = FramePreprocessor()
    agent = RandomAgent()

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
    print("Live loop running.  Press 'h' to toggle, Ctrl+C to stop.\n")

    prev_active = False

    try:
        for frame in stream:
            # --- Toggle detection ---
            if toggle_active != prev_active:
                prev_active = toggle_active
                if toggle_active:
                    print(f"  Pipeline: ON  (frame {stream.frame_count})")
                else:
                    sender.release_all()
                    print(f"  Pipeline: OFF (frame {stream.frame_count})")

            if not toggle_active:
                continue

            # --- Preprocessing ---
            obs = preprocessor(frame)

            # --- Action determination ---
            keys = agent.predict(obs)

            # --- Output ---
            sender.apply_vector(keys)

            if stream.frame_count % 120 == 0:
                print(
                    f"  frame {stream.frame_count:>6}"
                    f"  fps {stream.effective_fps:5.1f}"
                    f"  keys {keys}"
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
