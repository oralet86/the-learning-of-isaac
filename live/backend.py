"""Common interface to a running game, shared by training and inference.

``RemoteGame`` (a containerised game behind a sidecar) and ``LocalGame`` (the
real game on this machine) both implement :class:`GameBackend`, so the
environment, preprocessing and agent code is identical in both modes.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    from collections.abc import Sequence

    import numpy as np


class GameBackend(Protocol):
    """A game that can be reset, stepped with key vectors and observed."""

    # Milliseconds by which the most recent step missed its tick deadline;
    # 0.0 while the caller keeps up with the control rate.
    last_late_ms: float

    def reset(self) -> np.ndarray:
        """Bring the game to the start of a run.

        Returns:
            First frame of the run, ``(H, W, 3)`` uint8 RGB.
        """
        ...

    def step(self, keys: Sequence[int]) -> np.ndarray:
        """Hold *keys* for one control tick.

        Args:
            keys: Binary vector over ``constants.KEY_ORDER``; 1 = held.

        Returns:
            Frame at the end of the tick, ``(H, W, 3)`` uint8 RGB.
        """
        ...

    def release_all(self) -> None:
        """Release every held key."""
        ...

    def pause(self) -> None:
        """Release keys and, where supported, pause the run.

        Call this before the caller stops stepping for a while (for example
        during a policy update). The next ``step`` or ``reset`` resumes.
        """
        ...

    def close(self) -> None:
        """Release keys and free the backend's resources."""
        ...
