"""Screenshot capture for the data generator, reusing ``live.stream_game``.

The mod never writes image files itself — it only poses scenes and signals
ready via the file-exchange protocol in :mod:`datagen.exchange`. This module
owns turning the live OBS/ffmpeg frame stream into a saved PNG once the mod
says a scene is settled.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np
from PIL import Image

if TYPE_CHECKING:
    from pathlib import Path

    from live.stream_game import StreamGame

logger = logging.getLogger(__name__)


def capture_frame(
    stream: StreamGame, output_path: Path, timeout_s: float = 5.0
) -> None:
    """Grab the next available frame from `stream` and save it as a PNG.

    Args:
        stream: An already-started ``StreamGame`` instance.
        output_path: Destination PNG path; parent directories are created.
        timeout_s: Max time to wait for a frame.

    Raises:
        TimeoutError: If no frame arrives within `timeout_s`.
    """
    frame = stream.get_frame(timeout=timeout_s)
    if frame is None:
        raise TimeoutError(f"no frame received from stream within {timeout_s}s")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    frame_uint8: np.ndarray = frame.astype(np.uint8)
    Image.fromarray(frame_uint8, mode="RGB").save(output_path)
    logger.info("saved frame to %s", output_path)
