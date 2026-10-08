"""Launching and stopping the game process under Wine."""

from __future__ import annotations

import logging
import os
from pathlib import Path
import shutil
import signal
import subprocess

logger = logging.getLogger(__name__)

_EXECUTABLE = "isaac-ng.exe"
_SAVE_RELATIVE_DIR = Path("Documents", "My Games", "Binding of Isaac Repentance+")
_SAVE_SLOT_FILE = "persistentgamedata1.dat"
_STOP_TIMEOUT_S = 10.0

# Settings forced on every launch. Keys left out keep the game's defaults,
# which it writes back into the file on start.
_FIXED_OPTIONS: dict[str, str] = {
    "Fullscreen": "0",
    "UseExclusiveFullscreen": "0",
    "WindowPosX": "0",
    "WindowPosY": "0",
    "MusicEnabled": "0",
    "PauseOnFocusLost": "0",
    "EnableMods": "0",
    "PopUps": "0",
    "SteamCloud": "0",
    "VSync": "0",
    # Pre-accept the first-run beta notice, which would otherwise swallow the
    # menu script's key presses. The key name is tied to the game version.
    "AcceptedPublicBeta_v1.9.7.17": "1",
}


def save_dir(wine_prefix: Path, user: str) -> Path:
    """Return the game's save directory inside a Wine prefix.

    It holds ``options.ini``, the persistent save files and the state of any
    run in progress.
    """
    return wine_prefix / "drive_c" / "users" / user / _SAVE_RELATIVE_DIR


def options_path(wine_prefix: Path, user: str) -> Path:
    """Return where the game reads ``options.ini`` inside a Wine prefix."""
    return save_dir(wine_prefix, user) / "options.ini"


def install_save(source_dir: Path, directory: Path) -> None:
    """Copy the image's save file into slot 1 of the game's save directory.

    Done before every launch, so each game starts from the same unlocks no
    matter what earlier runs in this container changed.

    Args:
        source_dir: Directory holding exactly one ``.dat`` save file.
        directory: The game's save directory.

    Raises:
        FileNotFoundError: If *source_dir* does not hold exactly one ``.dat``.
    """
    candidates = sorted(source_dir.glob("*.dat"))
    if len(candidates) != 1:
        raise FileNotFoundError(
            f"expected exactly one .dat save in {source_dir}, found {len(candidates)}"
        )
    directory.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(candidates[0], directory / _SAVE_SLOT_FILE)


def clear_run_state(directory: Path) -> None:
    """Delete saved in-progress runs so every launch starts from a clean menu.

    Only ``gamestate*.dat`` is removed; the persistent save (unlocks) is kept.
    """
    for state_file in directory.glob("gamestate*.dat"):
        state_file.unlink()


def render_options(width: int, height: int) -> str:
    """Return ``options.ini`` contents requesting a *width* x *height* window."""
    options = {
        **_FIXED_OPTIONS,
        "WindowWidth": str(width),
        "WindowHeight": str(height),
    }
    lines = [f"{key}={value}" for key, value in options.items()]
    return "[Options]\n" + "\n".join(lines) + "\n"


def write_options(path: Path, width: int, height: int) -> None:
    """Write the game's ``options.ini`` so it opens at the requested size."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(render_options(width, height))


class GameProcess:
    """Owns the Wine process running the game."""

    def __init__(self, game_dir: Path) -> None:
        """Remember where the game lives; nothing is started yet.

        Args:
            game_dir: Directory containing ``isaac-ng.exe``.
        """
        self._game_dir = game_dir
        self._process: subprocess.Popen[bytes] | None = None

    @property
    def is_running(self) -> bool:
        """Whether the game process has been started and has not exited."""
        return self._process is not None and self._process.poll() is None

    def start(self) -> None:
        """Launch the game.

        Raises:
            RuntimeError: If the game is already running.
            FileNotFoundError: If the executable is missing.
        """
        if self.is_running:
            raise RuntimeError("game is already running")
        executable = self._game_dir / _EXECUTABLE
        if not executable.is_file():
            raise FileNotFoundError(executable)
        logger.info("launching %s", executable)
        # A new session gives the game its own process group, so stop() can
        # signal Wine and its children without touching the sidecar.
        self._process = subprocess.Popen(
            ["wine", _EXECUTABLE],
            cwd=self._game_dir,
            start_new_session=True,
        )

    def stop(self) -> None:
        """Terminate the game and the Wine server; safe to call repeatedly."""
        process = self._process
        self._process = None
        if process is None:
            return
        logger.info("stopping game (pid %d)", process.pid)
        # wineserver -k ends every Wine process in the prefix cleanly.
        subprocess.run(["wineserver", "-k"], check=False, timeout=_STOP_TIMEOUT_S)
        try:
            process.wait(timeout=_STOP_TIMEOUT_S)
        except subprocess.TimeoutExpired:
            logger.warning("game ignored wineserver -k; killing process group")
            os.killpg(process.pid, signal.SIGKILL)
            process.wait(timeout=_STOP_TIMEOUT_S)
