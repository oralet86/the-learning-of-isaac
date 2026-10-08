"""Recognising the game's pause menu in a frame.

Pausing is a toggle (Escape opens and closes the menu), so the sidecar must
know whether a key press really took effect or it would drift out of sync and
"resume" into a paused game. Whole-frame comparisons are unreliable because
some floors keep animating behind the menu, so only the part of the frame
covered by the menu's text is compared against a template captured from this
very game at the start of a run.
"""

from __future__ import annotations

import numpy as np

# Part of the frame covered by the pause menu's "OPTIONS / MY STUFF /
# RESUME GAME" text, as fractions of the frame size: (left, right, top, bottom).
PAUSE_REGION = (0.42, 0.62, 0.68, 0.87)

# Mean absolute pixel difference (0-255) below which two crops count as the
# same picture. A paused frame differs from the template by sensor-free
# rendering noise only; gameplay under the menu's position differs by far more.
MATCH_TOLERANCE = 10.0

# Minimum difference between the scene and the opened menu for a calibration
# to be trusted; guards against calibrating on a frame where nothing changed.
MIN_MENU_CONTRAST = 25.0


def crop_pause_region(frame: np.ndarray) -> np.ndarray:
    """Return the part of *frame* that the pause menu's text covers.

    Args:
        frame: ``(H, W, 3)`` uint8 RGB frame.
    """
    height, width = frame.shape[:2]
    left, right, top, bottom = PAUSE_REGION
    return frame[
        int(height * top) : int(height * bottom),
        int(width * left) : int(width * right),
    ]


def region_distance(first: np.ndarray, second: np.ndarray) -> float:
    """Return the mean absolute pixel difference between two crops.

    Crops of different shapes (the window was resized) are infinitely far
    apart, so they never match.
    """
    if first.shape != second.shape:
        return float("inf")
    return float(np.mean(np.abs(first.astype(np.int16) - second.astype(np.int16))))


def matches_template(frame: np.ndarray, template: np.ndarray) -> bool:
    """Return whether *frame* shows the pause menu captured in *template*."""
    return region_distance(crop_pause_region(frame), template) < MATCH_TOLERANCE
