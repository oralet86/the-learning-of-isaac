"""Game-facing half of the sidecar: launch, reset, step and capture."""

from __future__ import annotations

import logging
import os
from pathlib import Path
import time
from typing import TYPE_CHECKING

from container.sidecar.capture import WindowCapture, find_window, focus_window
from container.sidecar.game import (
    GameProcess,
    clear_run_state,
    options_path,
    save_dir,
    write_options,
)
from container.sidecar.keys import KeyTracker, XKeyboard
from container.sidecar.menu import (
    boot_script,
    pause_toggle_script,
    restart_script,
    run_script,
)
from container.sidecar.pacer import Pacer
from container.sidecar.pausing import (
    MATCH_TOLERANCE,
    MIN_MENU_CONTRAST,
    crop_pause_region,
    matches_template,
    region_distance,
)
from container.sidecar.protocol import ResetMode

if TYPE_CHECKING:
    from collections.abc import Sequence

    import numpy as np
    from Xlib.display import Display
    from Xlib.xobject.drawable import Window

    from container.sidecar.config import SidecarConfig

logger = logging.getLogger(__name__)

_WINDOW_POLL_S = 0.5
_CALIBRATION_ATTEMPTS = 6
_CALIBRATION_RETRY_S = 1.0


class GameSession:
    """Runs one game instance and exposes it as reset/step/capture calls."""

    def __init__(self, config: SidecarConfig, display: Display) -> None:
        """Prepare a session; call :meth:`launch` before anything else.

        Args:
            config: Sidecar settings.
            display: Open connection to the X server the game renders on.
        """
        self._config = config
        self._display = display
        self._keyboard = XKeyboard(display)
        self._keys = KeyTracker(self._keyboard)
        self._pacer = Pacer(config.tick_hz)
        self._game = GameProcess(config.game_dir)
        self._window: Window | None = None
        self._capture: WindowCapture | None = None
        self._in_run = False
        self._paused = False
        self._pause_template: np.ndarray | None = None

    @property
    def frame_size(self) -> tuple[int, int]:
        """Current ``(width, height)`` of captured frames.

        The game may still resize its window until the first reset has
        finished, so every frame also carries its own size.

        Raises:
            RuntimeError: If the game has not been launched.
        """
        return self._require_capture().size

    def launch(self) -> None:
        """Start the game and wait for its window.

        Raises:
            TimeoutError: If no game window appears in time.
            RuntimeError: If the game exits before showing a window.
        """
        config = self._config
        prefix = Path(os.environ["WINEPREFIX"])
        user = os.environ.get("USER", "root")
        write_options(options_path(prefix, user), config.game_width, config.game_height)
        clear_run_state(save_dir(prefix, user))
        self._game.start()
        self._in_run = False
        self._paused = False
        self._pause_template = None
        self._window = self._wait_for_window()
        focus_window(self._display, self._window)
        self._capture = WindowCapture(self._display, self._window)
        logger.info("game window opened at %dx%d", *self._capture.size)

    def reset(self, mode: ResetMode) -> np.ndarray:
        """Bring the game to the start of a run and return the first frame.

        Args:
            mode: ``AUTO`` navigates the menus after a launch and
                quick-restarts otherwise; ``RELAUNCH`` restarts the game
                process first.
        """
        self._keys.release_all()
        self._resume()
        if mode is ResetMode.RELAUNCH or not self._game.is_running:
            self._game.stop()
            self.launch()
        if self._window is not None:
            focus_window(self._display, self._window)
        menu = self._config.menu
        script = restart_script(menu) if self._in_run else boot_script(menu)
        run_script(script, self._keyboard)
        if self._pause_template is None:
            self._calibrate_pause()
        self._in_run = True
        self._pacer.reset()
        frame = self._require_capture().grab()
        height, width, _ = frame.shape
        requested = (self._config.game_width, self._config.game_height)
        if (width, height) != requested:
            logger.warning(
                "game window is %dx%d, not the requested %dx%d",
                width,
                height,
                *requested,
            )
        return frame

    def step(self, keys: Sequence[bool]) -> tuple[np.ndarray, float]:
        """Hold *keys* for one tick and capture the result.

        Args:
            keys: One held/released flag per key in ``KEY_ORDER``.

        Returns:
            The frame at the end of the tick, and how many milliseconds late
            the request was relative to the tick schedule.

        Raises:
            RuntimeError: If the game process has exited.
        """
        if not self._game.is_running:
            raise RuntimeError("game process is not running")
        self._resume()
        self._keys.apply(keys)
        late_ms = self._pacer.wait_next()
        return self._require_capture().grab(), late_ms

    def release_all(self) -> None:
        """Release every held action key."""
        self._keys.release_all()

    def pause(self) -> bool:
        """Open the game's pause menu so the run waits for the next step.

        Keys are released first, because held keys would navigate the pause
        menu. The next :meth:`step` or :meth:`reset` resumes automatically.
        Does nothing outside a run or when already paused.

        Returns:
            Whether the game is now paused. ``False`` means the game ignored
            the request (e.g. mid-transition) and keeps running.
        """
        if self._paused or not self._in_run or not self._game.is_running:
            return self._paused
        self._keys.release_all()
        self._toggle_pause_menu(self._config.menu.pause_settle_s)
        self._paused = self._shows_pause_menu()
        if not self._paused:
            logger.warning("pause request did not open the pause menu")
        return self._paused

    def shutdown(self) -> None:
        """Release keys and stop the game."""
        self._keys.release_all()
        self._game.stop()

    def _resume(self) -> None:
        """Close the pause menu if this session opened it.

        Raises:
            RuntimeError: If the menu is still open after two attempts.
        """
        if not self._paused:
            return
        for _ in range(2):
            self._toggle_pause_menu(self._config.menu.resume_settle_s)
            if not self._shows_pause_menu():
                self._paused = False
                self._pacer.reset()
                return
        raise RuntimeError("pause menu did not close")

    def _calibrate_pause(self) -> None:
        """Capture what the pause menu looks like in this game.

        Runs in the first room of a run, where nothing can hurt the
        character: opens the menu, stores its text area as the template that
        later pauses are checked against, and closes it again. The game
        ignores Escape while a run is still fading in, so opening is retried
        until the picture actually changes.

        Raises:
            RuntimeError: If the menu never opened or did not close again.
        """
        menu = self._config.menu
        capture = self._require_capture()
        for attempt in range(1, _CALIBRATION_ATTEMPTS + 1):
            scene = crop_pause_region(capture.grab())
            self._toggle_pause_menu(menu.pause_settle_s)
            template = crop_pause_region(capture.grab())
            time.sleep(menu.pause_check_gap_s)
            steadiness = region_distance(template, crop_pause_region(capture.grab()))
            contrast = region_distance(scene, template)
            logger.info(
                "pause calibration attempt %d: contrast %.1f, steadiness %.1f",
                attempt,
                contrast,
                steadiness,
            )
            if contrast >= MIN_MENU_CONTRAST and steadiness < MATCH_TOLERANCE:
                self._pause_template = template
                self._paused = True
                self._resume()
                return
            if contrast >= MIN_MENU_CONTRAST:
                # Something opened but is not a steady menu; close it again
                # so the next attempt starts from the scene.
                self._toggle_pause_menu(menu.resume_settle_s)
            time.sleep(_CALIBRATION_RETRY_S)
        raise RuntimeError(
            f"could not calibrate the pause menu in {_CALIBRATION_ATTEMPTS} attempts"
        )

    def _toggle_pause_menu(self, settle_s: float) -> None:
        """Press Escape and wait for the menu animation to finish."""
        run_script(pause_toggle_script(self._config.menu, settle_s), self._keyboard)

    def _shows_pause_menu(self) -> bool:
        """Whether the current frame shows the calibrated pause menu."""
        if self._pause_template is None:
            return False
        frame = self._require_capture().grab()
        distance = region_distance(crop_pause_region(frame), self._pause_template)
        logger.debug("pause menu distance %.1f", distance)
        return matches_template(frame, self._pause_template)

    def _require_capture(self) -> WindowCapture:
        """Return the capture object, failing loudly before launch."""
        if self._capture is None:
            raise RuntimeError("game has not been launched")
        return self._capture

    def _wait_for_window(self) -> Window:
        """Poll until the game window is viewable and has stopped resizing.

        The game first opens a window at a default size and resizes it once
        it has read its options, so the first sighting is not trusted.
        """
        config = self._config
        deadline = time.monotonic() + config.window_timeout_s
        last_size: tuple[int, int] | None = None
        stable_since = 0.0
        while time.monotonic() < deadline:
            if not self._game.is_running:
                raise RuntimeError("game exited before creating its window")
            window = find_window(self._display, config.window_title)
            if window is not None:
                geometry = window.get_geometry()
                size = (geometry.width, geometry.height)
                now = time.monotonic()
                if size != last_size:
                    last_size, stable_since = size, now
                elif now - stable_since >= config.window_settle_s:
                    return window
            time.sleep(_WINDOW_POLL_S)
        raise TimeoutError(
            f"no stable window titled {config.window_title!r} after "
            f"{config.window_timeout_s:.0f}s"
        )
