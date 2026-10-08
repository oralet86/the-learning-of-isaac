"""Sidecar configuration, read from environment variables."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

from container.sidecar.menu import MenuConfig
from container.sidecar.protocol import DEFAULT_PORT

if TYPE_CHECKING:
    from collections.abc import Mapping

# Smallest window size confirmed to work; the entrypoint's defaults must match.
DEFAULT_GAME_WIDTH = 480
DEFAULT_GAME_HEIGHT = 270
DEFAULT_TICK_HZ = 20


@dataclass(frozen=True)
class SidecarConfig:
    """Everything the sidecar needs to launch and serve one game.

    Attributes:
        port: TCP port the sidecar listens on.
        game_width: Requested game window width in pixels.
        game_height: Requested game window height in pixels.
        tick_hz: Control rate; one STEP spans ``1 / tick_hz`` seconds.
        game_dir: Directory containing ``isaac-ng.exe``.
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
    tick_hz: int = DEFAULT_TICK_HZ
    game_dir: Path = Path("/opt/isaac")
    window_title: str = "Binding of Isaac"
    window_timeout_s: float = 120.0
    window_settle_s: float = 3.0
    idle_pause_s: float = 0.25
    menu: MenuConfig = field(default_factory=MenuConfig)

    @classmethod
    def from_env(cls, environ: Mapping[str, str]) -> SidecarConfig:
        """Build a config from defaults plus environment overrides.

        Recognised variables: ``SIDECAR_PORT``, ``GAME_WIDTH``, ``GAME_HEIGHT``,
        ``TICK_HZ``, ``GAME_DIR``, ``WINDOW_TIMEOUT_S``, ``IDLE_PAUSE_S`` and the ``MENU_*``
        variables described in :class:`~container.sidecar.menu.MenuConfig`.

        Raises:
            ValueError: If a variable cannot be parsed or is out of range.
        """
        config = cls(
            port=int(environ.get("SIDECAR_PORT", cls.port)),
            game_width=int(environ.get("GAME_WIDTH", cls.game_width)),
            game_height=int(environ.get("GAME_HEIGHT", cls.game_height)),
            tick_hz=int(environ.get("TICK_HZ", cls.tick_hz)),
            game_dir=Path(environ.get("GAME_DIR", str(cls.game_dir))),
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
        if config.tick_hz <= 0:
            raise ValueError(f"TICK_HZ must be positive, got {config.tick_hz}")
        if config.idle_pause_s < 0:
            raise ValueError(
                f"IDLE_PAUSE_S must not be negative, got {config.idle_pause_s}"
            )
        return config
