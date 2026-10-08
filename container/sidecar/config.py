"""Sidecar configuration, read from environment variables."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

from constants import GAME_WINDOW_TITLE, STEP_HZ
from container.sidecar.menu import MenuConfig
from container.sidecar.protocol import DEFAULT_PORT

if TYPE_CHECKING:
    from collections.abc import Mapping

# Smallest window size confirmed to work.
DEFAULT_GAME_WIDTH = 480
DEFAULT_GAME_HEIGHT = 270

# Where the image keeps the game and the pristine save. `container.build`
# passes these to the Containerfile, so they are defined only here.
DEFAULT_GAME_DIR = Path("/opt/isaac")
DEFAULT_SAVE_SOURCE_DIR = Path("/opt/save")

DEFAULT_DISPLAY_NUM = 99


@dataclass(frozen=True)
class SidecarConfig:
    """Everything the sidecar needs to launch and serve one game.

    Attributes:
        port: TCP port the sidecar listens on.
        game_width: Requested game window width in pixels.
        game_height: Requested game window height in pixels.
        tick_hz: Control rate; one STEP spans ``1 / tick_hz`` seconds.
        game_dir: Directory containing ``isaac-ng.exe``.
        save_source_dir: Directory holding the one save file installed into
            slot 1 before every launch.
        display_num: X display number of the virtual screen.
        window_title: Substring identifying the game window's title.
        window_timeout_s: How long to wait for the game window after launch.
        window_settle_s: How long the window size must stay unchanged before
            it is trusted.
        idle_pause_s: Pause the game when the client has sent nothing for
            this long, so a stalled trainer does not leave the game running
            unattended. ``0`` disables idle pausing.
        menu: Menu navigation timings.
    """

    port: int = DEFAULT_PORT
    game_width: int = DEFAULT_GAME_WIDTH
    game_height: int = DEFAULT_GAME_HEIGHT
    tick_hz: int = STEP_HZ
    game_dir: Path = DEFAULT_GAME_DIR
    save_source_dir: Path = DEFAULT_SAVE_SOURCE_DIR
    display_num: int = DEFAULT_DISPLAY_NUM
    window_title: str = GAME_WINDOW_TITLE
    window_timeout_s: float = 120.0
    window_settle_s: float = 3.0
    idle_pause_s: float = 0.25
    menu: MenuConfig = field(default_factory=MenuConfig)

    @classmethod
    def from_env(cls, environ: Mapping[str, str]) -> SidecarConfig:
        """Build a config from defaults plus environment overrides.

        Recognised variables: ``SIDECAR_PORT``, ``GAME_WIDTH``, ``GAME_HEIGHT``,
        ``GAME_DIR``, ``SAVE_SOURCE_DIR``, ``DISPLAY_NUM``, ``WINDOW_TIMEOUT_S``,
        ``IDLE_PAUSE_S`` and the ``MENU_*``
        variables described in :class:`~container.sidecar.menu.MenuConfig`.

        Raises:
            ValueError: If a variable cannot be parsed or is out of range.
        """
        config = cls(
            port=int(environ.get("SIDECAR_PORT", cls.port)),
            game_width=int(environ.get("GAME_WIDTH", cls.game_width)),
            game_height=int(environ.get("GAME_HEIGHT", cls.game_height)),
            game_dir=Path(environ.get("GAME_DIR", str(cls.game_dir))),
            save_source_dir=Path(
                environ.get("SAVE_SOURCE_DIR", str(cls.save_source_dir))
            ),
            display_num=int(environ.get("DISPLAY_NUM", cls.display_num)),
            window_timeout_s=float(
                environ.get("WINDOW_TIMEOUT_S", cls.window_timeout_s)
            ),
            idle_pause_s=float(environ.get("IDLE_PAUSE_S", cls.idle_pause_s)),
            menu=MenuConfig.from_env(environ),
        )
        if config.game_width <= 0 or config.game_height <= 0:
            raise ValueError(
                f"game size must be positive, got "
                f"{config.game_width}x{config.game_height}"
            )
        if config.idle_pause_s < 0:
            raise ValueError(
                f"IDLE_PAUSE_S must not be negative, got {config.idle_pause_s}"
            )
        return config
