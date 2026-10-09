#!/usr/bin/env python3
"""Rebuild the sidecar's screen references from screenshots of the game.

Menu navigation recognises screens by comparing frames against
``container/sidecar/screens.npz``. Record the references again whenever a
game update redraws a menu (navigation then fails with "unrecognised
screen").

Put one screenshot per reference into a directory, named after the reference
(``title.png``, ``main_menu.npy``, ...), and run::

    python -m container.build_screens <directory>

Screenshots are full game-window frames at a 16:9 size: either ``.npy`` frames
the sidecar dumped (see ``SCREEN_DUMP_DIR``) or lossless ``.png`` images. The
names are those in ``container.sidecar.screens.REFERENCE_NAMES``:

- ``intro``: any scene of the intro cinematic.
- ``title``: the "press start" screen.
- ``file_select``: save-file selection.
- ``main_menu``: the page with "New Run" under the cursor.
- ``character_select``: character selection with Isaac under the cursor.
- ``random_selected``: character selection with "Random" under the cursor.
- ``pause_menu``: a run with the pause menu open.
- ``death_screen``: the "Dear Diary" screen that follows a death.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import cv2
import numpy as np

from container.sidecar.screens import REFERENCE_NAMES, REFERENCES_PATH, signature

logger = logging.getLogger("container.build_screens")


def load_screenshot(directory: Path, name: str) -> np.ndarray:
    """Load the screenshot for reference *name* as an RGB frame.

    Args:
        directory: Where the screenshots are.
        name: Reference name; the file is ``<name>.npy`` or ``<name>.png``.

    Raises:
        FileNotFoundError: If neither file exists or the image is unreadable.
    """
    array_path = directory / f"{name}.npy"
    if array_path.is_file():
        return np.load(array_path)
    image_path = directory / f"{name}.png"
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"no readable {array_path.name} or {image_path.name}")
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def build_references(directory: Path) -> dict[str, np.ndarray]:
    """Return the signature of every reference screenshot in *directory*.

    Raises:
        FileNotFoundError: If a screenshot is missing.
    """
    # Whole brightness levels keep the file small; the rounding is far below
    # every matching tolerance.
    return {
        name: np.rint(signature(load_screenshot(directory, name))).astype(np.uint8)
        for name in REFERENCE_NAMES
    }


def main() -> None:
    """Write the reference file from a directory of screenshots."""
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    parser = argparse.ArgumentParser(
        description="Rebuild the sidecar's screen references from screenshots."
    )
    parser.add_argument("directory", type=Path, help="directory of screenshots")
    parser.add_argument("--output", type=Path, default=REFERENCES_PATH)
    args = parser.parse_args()
    references = build_references(args.directory)
    np.savez_compressed(args.output, **references)  # type: ignore[arg-type]
    logger.info("wrote %d references to %s", len(references), args.output)


if __name__ == "__main__":
    main()
