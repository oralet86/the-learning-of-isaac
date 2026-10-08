"""Tests for `container.sidecar.pausing`.

A wrong answer here desynchronises the Escape toggle (a "resume" would pause
the game instead), so the template comparison is checked for each case it
must tell apart.
"""

from __future__ import annotations

import numpy as np

from container.sidecar.pausing import (
    MATCH_TOLERANCE,
    PAUSE_REGION,
    crop_pause_region,
    matches_template,
    region_distance,
)

_HEIGHT, _WIDTH = 100, 200


def _scene(value: int = 30) -> np.ndarray:
    return np.full((_HEIGHT, _WIDTH, 3), value, dtype=np.uint8)


def _with_menu(frame: np.ndarray, value: int = 210) -> np.ndarray:
    """Paint a bright block over the pause region, like the menu's paper."""
    painted = frame.copy()
    crop_pause_region(painted)[:] = value
    return painted


def test_crop_covers_the_configured_fraction_of_the_frame() -> None:
    left, right, top, bottom = PAUSE_REGION
    crop = crop_pause_region(_scene())

    assert crop.shape == (
        int(_HEIGHT * bottom) - int(_HEIGHT * top),
        int(_WIDTH * right) - int(_WIDTH * left),
        3,
    )


def test_crop_is_a_view_scaled_with_the_frame_size() -> None:
    small = crop_pause_region(np.zeros((100, 100, 3), dtype=np.uint8))
    large = crop_pause_region(np.zeros((200, 200, 3), dtype=np.uint8))

    assert large.shape[0] == 2 * small.shape[0]
    assert large.shape[1] == 2 * small.shape[1]


def test_distance_is_mean_absolute_difference_without_uint8_wraparound() -> None:
    dark = np.full((4, 4, 3), 10, dtype=np.uint8)
    bright = np.full((4, 4, 3), 250, dtype=np.uint8)

    assert region_distance(dark, bright) == 240.0
    assert region_distance(bright, dark) == 240.0
    assert region_distance(dark, dark) == 0.0


def test_distance_between_different_shapes_is_infinite() -> None:
    assert region_distance(np.zeros((4, 4, 3)), np.zeros((2, 4, 3))) == float("inf")


def test_menu_matches_its_template_even_when_the_background_animates() -> None:
    template = crop_pause_region(_with_menu(_scene())).copy()
    other_background = _with_menu(_scene(value=90))

    assert matches_template(other_background, template)


def test_menu_with_small_rendering_differences_still_matches() -> None:
    template = crop_pause_region(_with_menu(_scene())).copy()
    slightly_off = _with_menu(_scene(), value=210 + int(MATCH_TOLERANCE) - 2)

    assert matches_template(slightly_off, template)


def test_gameplay_frame_does_not_match_the_template() -> None:
    template = crop_pause_region(_with_menu(_scene())).copy()

    assert not matches_template(_scene(), template)


def test_resized_window_never_matches_an_old_template() -> None:
    template = crop_pause_region(_with_menu(_scene())).copy()
    resized = _with_menu(np.full((50, 100, 3), 30, dtype=np.uint8))

    assert not matches_template(resized, template)
