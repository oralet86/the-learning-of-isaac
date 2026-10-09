"""Game-facing half of the sidecar: launch, reset, step and capture."""

from __future__ import annotations

import logging
import os
from pathlib import Path
import time
from typing import TYPE_CHECKING

from container.sidecar.capture import WindowCapture, find_window, focus_window
from container.sidecar.game import (
    GAME_PROCESS_NAME,
    GameProcess,
    clear_run_state,
    install_save,
    options_path,
    save_dir,
    write_options,
)
from container.sidecar.keys import KeyTracker, XKeyboard
from container.sidecar.menu import MenuNavigator, NavigationError, tap
from container.sidecar.pacer import Pacer
from container.sidecar.protocol import ResetMode
from container.sidecar.screens import Screen, ScreenClassifier
from container.sidecar.spinners import demote_spinning_threads

if TYPE_CHECKING:
    from collections.abc import Sequence

    import numpy as np
    from Xlib.display import Display
    from Xlib.xobject.drawable import Window

    from container.sidecar.config import SidecarConfig

logger = logging.getLogger(__name__)

_WINDOW_POLL_S = 0.5
# How often the frame is checked while waiting for the pause menu to move.
_PAUSE_POLL_S = 0.05
# Opens and closes the pause menu in a run. Unlike Escape, which does the
# same there, it does nothing anywhere else: Escape leaves the run from the
# death screen (and while the character is still dying) and walks backwards
# through the front-end menus.
_PAUSE_KEY = "p"


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
        self._game = GameProcess(config.game_dir, config.gpu_device)
        self._window: Window | None = None
        self._capture: WindowCapture | None = None
        self._classifier = ScreenClassifier.load()
        self._in_run = False
        self._paused = False
        self._spinners_demoted = False

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
        install_save(config.save_source_dir, save_dir(prefix, user))
        clear_run_state(save_dir(prefix, user))
        self._game.start()
        self._in_run = False
        self._paused = False
        self._spinners_demoted = False
        self._window = self._wait_for_window()
        focus_window(self._display, self._window)
        self._capture = WindowCapture(self._display, self._window)
        logger.info("game window opened at %dx%d", *self._capture.size)
        # Without this the busy-wait thread takes a whole core all through
        # the intro and the menus, which is what starves a large fleet while
        # every game is loading at once.
        demote_spinning_threads(GAME_PROCESS_NAME)

    def reset(self, mode: ResetMode) -> np.ndarray:
        """Bring the game to the start of a run and return the first frame.

        The run is verified to be a run (its pause menu opens) before this
        returns, so a game stuck in a menu is an error, not a silent success.

        Args:
            mode: ``AUTO`` navigates the menus after a launch and
                quick-restarts otherwise; ``RELAUNCH`` restarts the game
                process first.

        Raises:
            NavigationError: If the menus did not lead to a run in time.
        """
        self._keys.release_all()
        self._resume()
        if mode is ResetMode.RELAUNCH or not self._game.is_running:
            self._game.stop()
            self.launch()
        if self._window is not None:
            focus_window(self._display, self._window)
        if not (self._in_run and self._quick_restart()):
            self._start_run_from_menus()
        self._in_run = True
        if not self._spinners_demoted:
            # Again in the starting room, where the game is idle and nothing
            # legitimate needs a core: catches threads started after launch.
            demote_spinning_threads(GAME_PROCESS_NAME)
            self._spinners_demoted = True
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
            RuntimeError: If the game process has exited, or the game is not
                in a run (before the first reset, or after it left the run
                for a front-end menu).
        """
        if not self._game.is_running:
            raise RuntimeError("game process is not running")
        if not self._in_run:
            raise RuntimeError("the game is not in a run; reset to start one")
        self._resume()
        self._keys.apply(keys)
        late_ms = self._pacer.wait_next()
        frame = self._require_capture().grab()
        if self._classifier.shows_pause_menu(frame):
            # An earlier pause request took effect after it was given up on.
            # Remembering it makes the next step close the menu instead of
            # steering through it.
            logger.warning("pause menu is open during a step; resuming next step")
            self._paused = True
        return frame, late_ms

    def release_all(self) -> None:
        """Release every held action key."""
        self._keys.release_all()

    def pause(self) -> bool:
        """Open the game's pause menu so the run waits for the next step.

        Keys are released first, because held keys would navigate the pause
        menu. The next :meth:`step` or :meth:`reset` resumes automatically.
        Does nothing outside a run, on the death screen or when already
        paused.

        Returns:
            Whether the game is now paused. ``False`` means the game ignored
            the request (e.g. mid-transition) and keeps running.
        """
        if self._paused or not self._in_run or not self._game.is_running:
            return self._paused
        self._keys.release_all()
        frame = self._require_capture().grab()
        if self._classifier.shows_death_screen(frame):
            # Nothing moves here and there is no pause menu to wait for.
            logger.debug("not pausing on the death screen")
            return False
        screen = self._classifier.classify(frame)
        if screen is not Screen.UNKNOWN:
            # There is no pause menu here, and waiting for one on every
            # idle moment stalls the client; the next step reports it instead.
            logger.error("the game left the run and shows %s", screen.value)
            self._in_run = False
            return False
        self._paused = self._toggle_pause_until(pause_menu_showing=True)
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
            if self._toggle_pause_until(pause_menu_showing=False):
                self._paused = False
                self._pacer.reset()
                return
        raise RuntimeError("pause menu did not close")

    def _start_run_from_menus(self) -> None:
        """Navigate the front-end menus and wait until the run accepts input.

        If the game shows a menu again while the run is awaited (a key press
        landed somewhere unexpected), navigation simply continues from there.

        Raises:
            NavigationError: If no run started before the navigation timeout.
        """
        menu = self._config.menu
        navigator = MenuNavigator(
            self._require_capture().grab, self._keyboard, self._classifier, menu
        )
        started = time.monotonic()
        deadline = started + menu.navigation_timeout_s
        while True:
            navigator.leave_menus(deadline)
            if self._await_run(deadline):
                logger.info(
                    "run started %.1fs after navigation began",
                    time.monotonic() - started,
                )
                return
            if time.monotonic() >= deadline:
                raise NavigationError(
                    "the character was confirmed but no run accepted input in time"
                )

    def _quick_restart(self) -> bool:
        """Restart the current run by holding R and wait until it accepts input.

        Returns:
            ``False`` if no run answered in time or the game turned out to be
            in a front-end menu; the caller then navigates the menus, which
            starts a run from wherever the game is or fails loudly.
        """
        menu = self._config.menu
        tap(self._keyboard, "r", menu.restart_hold_s)
        time.sleep(menu.restart_settle_s)
        if self._await_run(time.monotonic() + menu.run_start_timeout_s):
            return True
        logger.warning("quick restart did not lead to a run; navigating the menus")
        return False

    def _await_run(self, deadline: float) -> bool:
        """Wait until a starting run accepts input, proving the game is in one.

        A run ignores the pause key while it loads and fades in, so the
        pause menu is requested until it opens, then closed again. Only a
        run has that menu.

        Args:
            deadline: ``time.monotonic()`` value at which to give up.

        Returns:
            ``True`` once the run answered; ``False`` if the deadline passed
            or the game is showing a front-end menu.
        """
        capture = self._require_capture()
        while time.monotonic() < deadline:
            frame = capture.grab()
            screen = self._classifier.classify(frame)
            if screen is not Screen.UNKNOWN:
                logger.warning("expected a run, but the game shows %s", screen.value)
                return False
            if self._classifier.shows_death_screen(frame):
                # No pause menu here; wait for the restart to happen.
                time.sleep(self._config.menu.pause_retry_s)
                continue
            if self._toggle_pause_until(pause_menu_showing=True):
                self._paused = True
                self._resume()
                return True
            time.sleep(self._config.menu.pause_retry_s)
        return False

    def _toggle_pause_until(self, *, pause_menu_showing: bool) -> bool:
        """Press the pause key and wait for the pause menu to open or close.

        Args:
            pause_menu_showing: The state to wait for.

        Returns:
            Whether the pause menu reached that state within
            ``pause_timeout_s`` and was still in it once its animation had
            finished. Returning before the animation ends would let the next
            key press land in a menu that is neither open nor closed.
        """
        menu = self._config.menu
        capture = self._require_capture()
        tap(self._keyboard, _PAUSE_KEY, menu.tap_hold_s)
        deadline = time.monotonic() + menu.pause_timeout_s
        while (
            self._classifier.shows_pause_menu(capture.grab()) is not pause_menu_showing
        ):
            if time.monotonic() >= deadline:
                return False
            time.sleep(_PAUSE_POLL_S)
        time.sleep(menu.pause_settle_s if pause_menu_showing else menu.resume_settle_s)
        return self._classifier.shows_pause_menu(capture.grab()) is pause_menu_showing

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
