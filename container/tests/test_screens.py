"""Tests for `container.sidecar.screens`.

A wrong answer here presses the wrong key in a menu, or desynchronises the
pause toggle (a "resume" would pause the game instead), so each case the
classifier must tell apart is checked, on synthetic screens.
"""

from __future__ import annotations

import numpy as np
import pytest

from container.sidecar.screens import (
    CHARACTER_SELECT_TOLERANCE,
    DEATH_REGION,
    DEATH_SCREEN_REFERENCE,
    GRID_SHAPE,
    PAUSE_MENU_REFERENCE,
    PAUSE_REGION,
    REFERENCE_NAMES,
    REFERENCES_PATH,
    SAME_SCREEN_TOLERANCE,
    Screen,
    ScreenClassifier,
    region_cells,
    signature,
)

_SCALE = 5
_HEIGHT, _WIDTH = GRID_SHAPE[0] * _SCALE, GRID_SHAPE[1] * _SCALE


def _references() -> dict[str, np.ndarray]:
    """One distinct random picture per reference, far apart from each other."""
    rng = np.random.default_rng(0)
    references = {
        name: rng.integers(0, 256, GRID_SHAPE).astype(np.float32)
        for name in REFERENCE_NAMES
    }
    # Like in the game, the two character pages are near each other.
    references[Screen.RANDOM_SELECTED.value] = np.clip(
        references[Screen.CHARACTER_SELECT.value] + 8.0, 0, 255
    )
    return references


def _frame(grid: np.ndarray, scale: int = _SCALE) -> np.ndarray:
    """Render a signature back into a frame that has exactly that signature."""
    pixels = np.repeat(np.repeat(grid, scale, axis=0), scale, axis=1)
    return np.repeat(pixels[:, :, np.newaxis], 3, axis=2).astype(np.uint8)


def _with_overlay(
    frame: np.ndarray,
    references: dict[str, np.ndarray],
    name: str,
    region: tuple[float, float, float, float],
) -> np.ndarray:
    """Draw reference *name* over its *region* of *frame*."""
    row_cells, column_cells = region_cells(region)
    rows = slice(row_cells.start * _SCALE, row_cells.stop * _SCALE)
    columns = slice(column_cells.start * _SCALE, column_cells.stop * _SCALE)
    painted = frame.copy()
    painted[rows, columns] = _frame(references[name])[rows, columns]
    return painted


def test_signature_is_the_mean_brightness_of_each_cell() -> None:
    grid = np.arange(GRID_SHAPE[0] * GRID_SHAPE[1]).reshape(GRID_SHAPE) % 256

    assert np.allclose(signature(_frame(grid.astype(np.float32))), grid)


def test_signature_does_not_depend_on_the_window_size() -> None:
    grid = np.random.default_rng(1).integers(0, 256, GRID_SHAPE).astype(np.float32)

    assert np.allclose(signature(_frame(grid, 2)), signature(_frame(grid, 5)))


def test_signature_of_a_region_matches_that_part_of_the_whole() -> None:
    frame = np.random.default_rng(2).integers(0, 256, (301, 517, 3), dtype=np.uint8)
    cells = (slice(36, 46), slice(40, 59))

    assert np.allclose(signature(frame, cells), signature(frame)[cells], atol=1e-3)


def test_signature_rejects_a_frame_smaller_than_the_grid() -> None:
    with pytest.raises(ValueError, match="smaller"):
        signature(np.zeros((10, 10, 3), dtype=np.uint8))


@pytest.mark.parametrize("screen", [s for s in Screen if s is not Screen.UNKNOWN])
def test_each_reference_is_classified_as_its_own_screen(screen: Screen) -> None:
    references = _references()
    classifier = ScreenClassifier(references)

    assert classifier.classify(_frame(references[screen.value])) is screen


def test_small_rendering_differences_do_not_change_the_screen() -> None:
    references = _references()
    title = np.clip(references["title"] + (SAME_SCREEN_TOLERANCE - 1), 0, 255)

    assert ScreenClassifier(references).classify(_frame(title)) is Screen.TITLE


def test_another_character_is_not_mistaken_for_random() -> None:
    references = _references()
    other = np.clip(
        references["character_select"] - (CHARACTER_SELECT_TOLERANCE - 2), 0, 255
    )

    assert (
        ScreenClassifier(references).classify(_frame(other)) is Screen.CHARACTER_SELECT
    )


def test_unfamiliar_picture_is_unknown() -> None:
    gameplay = np.random.default_rng(3).integers(0, 256, GRID_SHAPE).astype(np.float32)

    assert ScreenClassifier(_references()).classify(_frame(gameplay)) is Screen.UNKNOWN


def test_pause_menu_and_death_screen_are_recognised_over_any_background() -> None:
    references = _references()
    classifier = ScreenClassifier(references)
    rng = np.random.default_rng(4)

    for _ in range(3):
        room = _frame(rng.integers(0, 256, GRID_SHAPE).astype(np.float32))
        paused = _with_overlay(room, references, PAUSE_MENU_REFERENCE, PAUSE_REGION)
        dead = _with_overlay(room, references, DEATH_SCREEN_REFERENCE, DEATH_REGION)
        assert not classifier.shows_pause_menu(room)
        assert not classifier.shows_death_screen(room)
        assert classifier.shows_pause_menu(paused)
        assert not classifier.shows_death_screen(paused)
        assert classifier.shows_death_screen(dead)
        assert not classifier.shows_pause_menu(dead)


def test_front_end_menus_do_not_look_like_run_overlays() -> None:
    references = _references()
    classifier = ScreenClassifier(references)

    for screen in Screen:
        if screen is not Screen.UNKNOWN:
            frame = _frame(references[screen.value])
            assert not classifier.shows_pause_menu(frame)
            assert not classifier.shows_death_screen(frame)


def test_classifier_rejects_missing_or_misshapen_references() -> None:
    references = _references()

    with pytest.raises(ValueError, match="missing"):
        ScreenClassifier({"title": references["title"]})
    with pytest.raises(ValueError, match="shape"):
        ScreenClassifier({**references, "title": np.zeros((2, 2))})


def test_shipped_references_load_and_are_mutually_distinct() -> None:
    classifier = ScreenClassifier.load(REFERENCES_PATH)

    with np.load(REFERENCES_PATH) as archive:
        for screen in Screen:
            if screen is not Screen.UNKNOWN:
                frame = _frame(archive[screen.value].astype(np.float32))
                assert classifier.classify(frame) is screen
                assert not classifier.shows_pause_menu(frame)
                assert not classifier.shows_death_screen(frame)
        assert classifier.shows_death_screen(
            _frame(archive[DEATH_SCREEN_REFERENCE].astype(np.float32))
        )
        assert classifier.shows_pause_menu(
            _frame(archive[PAUSE_MENU_REFERENCE].astype(np.float32))
        )
