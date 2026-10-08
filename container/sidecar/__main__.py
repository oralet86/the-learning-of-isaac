"""Sidecar entry point: launch the game, then serve it over TCP.

Run inside the container as ``python -m container.sidecar``.
"""

from __future__ import annotations

import logging
import os
import signal
import sys
from typing import TYPE_CHECKING

from Xlib.display import Display

from container.sidecar.config import SidecarConfig
from container.sidecar.server import serve_forever
from container.sidecar.session import GameSession

if TYPE_CHECKING:
    from types import FrameType

logger = logging.getLogger("container.sidecar")


def _raise_exit(signum: int, _frame: FrameType | None) -> None:
    """Turn a termination signal into a normal interpreter exit."""
    raise SystemExit(128 + signum)


def main() -> None:
    """Launch the game and serve clients until the container is stopped."""
    logging.basicConfig(
        level=os.environ.get("LOG_LEVEL", "INFO"),
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
        stream=sys.stderr,
    )
    # As PID 1 the process gets no default signal handlers, so SIGTERM from
    # `podman stop` would otherwise be ignored.
    signal.signal(signal.SIGTERM, _raise_exit)
    signal.signal(signal.SIGINT, _raise_exit)

    config = SidecarConfig.from_env(os.environ)
    session = GameSession(config, Display())
    try:
        session.launch()
        serve_forever(session, config.port, config.tick_hz, config.idle_pause_s)
    finally:
        session.shutdown()


if __name__ == "__main__":
    main()
