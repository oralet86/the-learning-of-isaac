"""Recognising the game's menu screens and its pause menu in a frame.

Menu navigation and pausing both press keys whose effect depends on what the
game is currently showing, so the sidecar has to know the screen instead of
assuming it. The menus are static artwork: a frame is reduced to a coarse
grid of brightness values (its *signature*) and compared against reference
signatures recorded from the game (``screens.npz``, rebuilt with
``python -m container.build_screens``).

The references were recorded from a 16:9 window. Signatures are independent
of the window's size, but not of its aspect ratio or of the game's artwork:
after a game update that redraws a menu, navigation fails loudly and the
references must be recorded again.
"""

from __future__ import annotations

from enum import Enum
from pathlib import Path
from typing import Self

import numpy as np

# Signature grid as (rows, columns); 16:9 like the window, 5 px per cell at
# the default 480x270.
GRID_SHAPE = (54, 96)

REFERENCES_PATH = Path(__file__).with_name("screens.npz")

# Mean absolute brightness difference (0-255) between signatures.
#
# A static menu redrawn by the game differs from its reference by under 2;
# different menus differ by more than 20.
SAME_SCREEN_TOLERANCE = 5.0
# The character-select page with any character on it. Characters differ from
# each other by up to 9; the nearest other screen is more than 20 away.
CHARACTER_SELECT_TOLERANCE = 13.0
# The intro cinematic keeps its desk-and-pencil frame while the drawing in the
# middle changes; scenes differ by up to 9, other screens by more than 28.
INTRO_TOLERANCE = 15.0
# The pause menu's paper covers its whole region, so runs differ only by
# rendering noise there; gameplay under it differs by more than 25.
PAUSE_MENU_TOLERANCE = 10.0

# Part of the frame covered by the pause menu's "OPTIONS / MY STUFF /
# RESUME GAME / EXIT GAME" text, as fractions of the frame size:
# (left, right, top, bottom).
PAUSE_REGION = (0.42, 0.62, 0.68, 0.87)

# The "EXIT" tag in the bottom-left corner of the death screen, in the same
# form. It is drawn identically after every death (distance 0); the nearest
# other picture is the file-select screen's "BACK" tag, 17 away.
DEATH_REGION = (0.0, 0.125, 0.80, 0.98)
DEATH_SCREEN_TOLERANCE = 8.0


class Screen(Enum):
    """What the game is showing, as far as navigation needs to know."""

    INTRO = "intro"
    TITLE = "title"
    FILE_SELECT = "file_select"
    MAIN_MENU = "main_menu"
    # Character select with "Random" under the cursor.
    RANDOM_SELECTED = "random_selected"
    # Character select with any other character under the cursor.
    CHARACTER_SELECT = "character_select"
    # Anything else: a transition, a loading screen, or a run.
    UNKNOWN = "unknown"


# References compared on one region only, next to one per recognisable
# `Screen`: these overlays sit on top of a run, whose picture is arbitrary.
PAUSE_MENU_REFERENCE = "pause_menu"
DEATH_SCREEN_REFERENCE = "death_screen"
REFERENCE_NAMES: tuple[str, ...] = (
    *(screen.value for screen in Screen if screen is not Screen.UNKNOWN),
    PAUSE_MENU_REFERENCE,
    DEATH_SCREEN_REFERENCE,
)

# Screens that match their reference pixel for pixel, apart from noise.
_STATIC_SCREENS = (
    Screen.TITLE,
    Screen.FILE_SELECT,
    Screen.MAIN_MENU,
    Screen.RANDOM_SELECTED,
)


def region_cells(region: tuple[float, float, float, float]) -> tuple[slice, slice]:
    """Return the grid rows and columns covered by a frame region.

    Args:
        region: ``(left, right, top, bottom)`` as fractions of the frame size.
    """
    rows, columns = GRID_SHAPE
    left, right, top, bottom = region
    return (
        slice(int(rows * top), int(rows * bottom)),
        slice(int(columns * left), int(columns * right)),
    )


def signature(
    frame: np.ndarray, cells: tuple[slice, slice] | None = None
) -> np.ndarray:
    """Reduce *frame* to a ``GRID_SHAPE`` grid of mean brightness values.

    Args:
        frame: ``(H, W, 3)`` uint8 RGB frame, at least as large as the grid.
        cells: Grid rows and columns to compute; the rest of the frame is not
            read, which is much cheaper for a small region. ``None`` computes
            the whole grid.

    Returns:
        float32 array in ``[0, 255]``, of ``GRID_SHAPE`` or the shape of
        *cells* within it.

    Raises:
        ValueError: If *frame* is smaller than the grid.
    """
    height, width = frame.shape[:2]
    rows, columns = GRID_SHAPE
    if height < rows or width < columns:
        raise ValueError(
            f"frame of {width}x{height} is smaller than the {columns}x{rows} grid"
        )
    row_cells, column_cells = cells if cells is not None else (slice(None), slice(None))
    row_edges = np.linspace(0, height, rows + 1).astype(np.intp)
    column_edges = np.linspace(0, width, columns + 1).astype(np.intp)
    row_starts = row_edges[:-1][row_cells]
    column_starts = column_edges[:-1][column_cells]
    row_ends = row_edges[1:][row_cells]
    column_ends = column_edges[1:][column_cells]
    top, left = row_starts[0], column_starts[0]
    brightness = frame[top : row_ends[-1], left : column_ends[-1]].mean(
        axis=2, dtype=np.float32
    )
    sums = np.add.reduceat(
        np.add.reduceat(brightness, row_starts - top, axis=0),
        column_starts - left,
        axis=1,
    )
    cell_sizes = np.outer(row_ends - row_starts, column_ends - column_starts)
    return (sums / cell_sizes).astype(np.float32)


def signature_distance(first: np.ndarray, second: np.ndarray) -> float:
    """Return the mean absolute difference between two signatures (or parts)."""
    return float(np.mean(np.abs(first - second)))


class ScreenClassifier:
    """Tells the game's screens apart by comparing against references."""

    def __init__(self, references: dict[str, np.ndarray]) -> None:
        """Create a classifier.

        Args:
            references: Signature per name in ``REFERENCE_NAMES``.

        Raises:
            ValueError: If a reference is missing or has the wrong shape.
        """
        missing = sorted(set(REFERENCE_NAMES) - set(references))
        if missing:
            raise ValueError(f"missing screen references: {missing}")
        for name in REFERENCE_NAMES:
            if references[name].shape != GRID_SHAPE:
                raise ValueError(
                    f"screen reference {name!r} has shape {references[name].shape}, "
                    f"expected {GRID_SHAPE}"
                )
        self._references = {
            name: references[name].astype(np.float32) for name in REFERENCE_NAMES
        }
        self._pause_cells = region_cells(PAUSE_REGION)
        self._death_cells = region_cells(DEATH_REGION)

    @classmethod
    def load(cls, path: Path = REFERENCES_PATH) -> Self:
        """Load the references recorded from the game.

        Raises:
            FileNotFoundError: If *path* does not exist.
            ValueError: If the file lacks a reference or one is malformed.
        """
        with np.load(path) as archive:
            return cls({name: archive[name] for name in archive.files})

    def distances(self, frame: np.ndarray) -> dict[str, float]:
        """Return the distance from *frame* to every screen reference.

        The pause menu and the death screen are compared on their own region
        only, the other screens on the whole frame. Meant for logs and for
        choosing tolerances.
        """
        grid = signature(frame)
        distances = {
            screen.value: signature_distance(grid, self._references[screen.value])
            for screen in Screen
            if screen is not Screen.UNKNOWN
        }
        distances[PAUSE_MENU_REFERENCE] = self._region_distance(
            frame, PAUSE_MENU_REFERENCE, self._pause_cells
        )
        distances[DEATH_SCREEN_REFERENCE] = self._region_distance(
            frame, DEATH_SCREEN_REFERENCE, self._death_cells
        )
        return distances

    def classify(self, frame: np.ndarray) -> Screen:
        """Return the screen *frame* shows, or ``Screen.UNKNOWN``."""
        grid = signature(frame)

        def distance_to(screen: Screen) -> float:
            return signature_distance(grid, self._references[screen.value])

        nearest = min(_STATIC_SCREENS, key=distance_to)
        if distance_to(nearest) < SAME_SCREEN_TOLERANCE:
            return nearest
        on_character_page = min(
            distance_to(Screen.CHARACTER_SELECT), distance_to(Screen.RANDOM_SELECTED)
        )
        if on_character_page < CHARACTER_SELECT_TOLERANCE:
            return Screen.CHARACTER_SELECT
        if distance_to(Screen.INTRO) < INTRO_TOLERANCE:
            return Screen.INTRO
        return Screen.UNKNOWN

    def shows_pause_menu(self, frame: np.ndarray) -> bool:
        """Return whether *frame* shows a run with its pause menu open.

        Only a run has this menu, so a match also proves the game is in a
        run rather than in one of the front-end menus. Cheap enough to call
        on every frame: only the menu's region is read.
        """
        distance = self._region_distance(frame, PAUSE_MENU_REFERENCE, self._pause_cells)
        return distance < PAUSE_MENU_TOLERANCE

    def shows_death_screen(self, frame: np.ndarray) -> bool:
        """Return whether *frame* shows the screen that follows a death.

        A run cannot be paused there, and nothing moves anyway.
        """
        distance = self._region_distance(
            frame, DEATH_SCREEN_REFERENCE, self._death_cells
        )
        return distance < DEATH_SCREEN_TOLERANCE

    def _region_distance(
        self, frame: np.ndarray, name: str, cells: tuple[slice, slice]
    ) -> float:
        """Distance between *frame* and reference *name* inside *cells*."""
        return signature_distance(
            signature(frame, cells), self._references[name][cells]
        )
