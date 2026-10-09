"""Save the game window's current frame, for debugging and screen references.

Run inside a running container, next to the sidecar::

    podman exec <container> python3 -m container.sidecar.screenshot /tmp/shot.npy
    podman cp <container>:/tmp/shot.npy title.npy

The ``.npy`` file holds one ``(H, W, 3)`` uint8 RGB frame, which
``python -m container.build_screens`` accepts as a reference screenshot.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np
from Xlib.display import Display

from constants import GAME_WINDOW_TITLE
from container.sidecar.capture import WindowCapture, find_window
from container.sidecar.config import DEFAULT_DISPLAY_NUM
from container.sidecar.display import display_name


def main() -> None:
    """Capture the game window once and write it to the given path.

    Raises:
        RuntimeError: If the game window is not on the display.
    """
    parser = argparse.ArgumentParser(description="Save the game window's frame.")
    parser.add_argument("output", type=Path, help="where to write the .npy frame")
    args = parser.parse_args()
    display_num = int(os.environ.get("DISPLAY_NUM", DEFAULT_DISPLAY_NUM))
    display = Display(display_name(display_num))
    window = find_window(display, GAME_WINDOW_TITLE)
    if window is None:
        raise RuntimeError(f"no window titled {GAME_WINDOW_TITLE!r} on the display")
    np.save(args.output, WindowCapture(display, window).grab())


if __name__ == "__main__":
    main()
