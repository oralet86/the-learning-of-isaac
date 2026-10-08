"""Starting the virtual X display the game renders on."""

from __future__ import annotations

import logging
from pathlib import Path
import subprocess
import time

logger = logging.getLogger(__name__)

# The game window must be fully on screen to be captured. Leave a margin for
# wherever the game places it, and never go below a floor in case the game
# refuses a small window and opens a larger one.
_SCREEN_MARGIN = 64
_MIN_SCREEN_SIZE = (1024, 640)
_COLOR_DEPTH = 24
_START_TIMEOUT_S = 5.0
_POLL_S = 0.1


def display_name(display_num: int) -> str:
    """Return the X display string for *display_num*, e.g. ``":99"``."""
    return f":{display_num}"


def screen_size(game_width: int, game_height: int) -> tuple[int, int]:
    """Return the virtual screen size needed for a game window of the given size."""
    return (
        max(game_width + _SCREEN_MARGIN, _MIN_SCREEN_SIZE[0]),
        max(game_height + _SCREEN_MARGIN, _MIN_SCREEN_SIZE[1]),
    )


def start_xvfb(
    display_num: int, game_width: int, game_height: int
) -> subprocess.Popen[bytes]:
    """Start Xvfb and wait until it accepts connections.

    Args:
        display_num: X display number to serve.
        game_width: Requested game window width.
        game_height: Requested game window height.

    Returns:
        The running Xvfb process; the caller terminates it on shutdown.

    Raises:
        RuntimeError: If Xvfb exits or does not create its socket in time.
    """
    # A lock left behind by a previous server makes Xvfb refuse to start
    # whenever the recorded PID happens to be alive again.
    Path(f"/tmp/.X{display_num}-lock").unlink(missing_ok=True)
    width, height = screen_size(game_width, game_height)
    process = subprocess.Popen(
        [
            "Xvfb",
            display_name(display_num),
            "-screen",
            "0",
            f"{width}x{height}x{_COLOR_DEPTH}",
            "-nolisten",
            "tcp",
        ]
    )
    socket_path = Path(f"/tmp/.X11-unix/X{display_num}")
    deadline = time.monotonic() + _START_TIMEOUT_S
    while not socket_path.is_socket():
        if process.poll() is not None:
            raise RuntimeError(f"Xvfb exited with code {process.returncode}")
        if time.monotonic() >= deadline:
            process.kill()
            raise RuntimeError(
                f"Xvfb did not come up on {display_name(display_num)} "
                f"within {_START_TIMEOUT_S:.0f}s"
            )
        time.sleep(_POLL_S)
    logger.info("Xvfb running on %s at %dx%d", display_name(display_num), width, height)
    return process
