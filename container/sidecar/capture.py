"""Screen capture of the game window through the X server."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from Xlib import X
from Xlib.error import BadWindow, XError

if TYPE_CHECKING:
    from Xlib.display import Display
    from Xlib.xobject.drawable import Window

_ALL_PLANES = 0xFFFFFFFF
_BYTES_PER_PIXEL = 4


def find_window(display: Display, title_fragment: str) -> Window | None:
    """Return the largest viewable window whose title contains *title_fragment*.

    The game also creates tiny helper windows; picking the largest match
    selects the one it actually renders into.

    Args:
        display: Open X connection.
        title_fragment: Substring to look for in window titles.

    Returns:
        The matching window, or ``None`` if there is none yet.
    """
    best: Window | None = None
    best_area = 0
    pending = [display.screen().root]
    while pending:
        window = pending.pop()
        try:
            pending.extend(window.query_tree().children)
            name = window.get_wm_name()
            if not isinstance(name, str) or title_fragment not in name:
                continue
            if window.get_attributes().map_state != X.IsViewable:
                continue
            geometry = window.get_geometry()
        except BadWindow:
            # Windows can disappear while the tree is being walked.
            continue
        area = geometry.width * geometry.height
        if area > best_area:
            best, best_area = window, area
    return best


def focus_window(display: Display, window: Window) -> None:
    """Give *window* keyboard focus.

    Without a window manager nothing focuses the game on its own, and
    injected key events are only delivered to the focused window.
    """
    window.set_input_focus(X.RevertToPointerRoot, X.CurrentTime)
    display.sync()


class WindowCapture:
    """Grabs RGB frames from one X window."""

    def __init__(self, display: Display, window: Window) -> None:
        """Bind to a window.

        Args:
            display: Open X connection.
            window: Game window to capture. It must be fully on screen.

        Raises:
            RuntimeError: If the display does not use 32-bit pixels.
        """
        self._display = display
        self._window = window
        depth = window.get_geometry().depth
        bits_per_pixel = {
            pixmap_format.depth: pixmap_format.bits_per_pixel
            for pixmap_format in display.display.info.pixmap_formats
        }.get(depth)
        if bits_per_pixel != _BYTES_PER_PIXEL * 8:
            raise RuntimeError(
                f"unsupported X pixel format: depth {depth}, "
                f"{bits_per_pixel} bits per pixel (need 32)"
            )

    @property
    def size(self) -> tuple[int, int]:
        """Current ``(width, height)`` of the window in pixels."""
        geometry = self._window.get_geometry()
        return geometry.width, geometry.height

    def grab(self) -> np.ndarray:
        """Capture the window's current contents.

        The size is read on every call because the game resizes its window
        while it starts up.

        Returns:
            ``(height, width, 3)`` uint8 RGB array.

        Raises:
            RuntimeError: If the window is gone, off screen, or was resized
                between reading its size and grabbing it.
        """
        try:
            width, height = self.size
            image = self._window.get_image(0, 0, width, height, X.ZPixmap, _ALL_PLANES)
        except XError as error:
            raise RuntimeError(f"cannot capture the game window: {error}") from error
        expected = width * height * _BYTES_PER_PIXEL
        if len(image.data) != expected:
            raise RuntimeError(
                f"captured {len(image.data)} bytes, expected {expected} "
                f"for a {width}x{height} window"
            )
        bgrx = np.frombuffer(image.data, dtype=np.uint8).reshape(
            height, width, _BYTES_PER_PIXEL
        )
        return np.ascontiguousarray(bgrx[:, :, 2::-1])
