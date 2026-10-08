"""Inference-mode backend: the real game on this machine.

Frames come from OBS through ``StreamGame`` and keys go out through uinput
with ``ActionSender``. Nothing is injected into the game.
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING, Protocol

import numpy as np

from constants import STEP_HZ
from container.sidecar.pacer import Pacer

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

# How long to wait for a frame before deciding the capture stream has died.
FRAME_TIMEOUT_S = 2.0


class FrameSource(Protocol):
    """The part of ``StreamGame`` this backend uses."""

    def get_frame(self, timeout: float | None = None) -> np.ndarray | None:
        """Return the next queued frame, or ``None`` on timeout/end."""
        ...


class KeySink(Protocol):
    """The part of ``ActionSender`` this backend uses."""

    def apply_vector(self, keys: list[int]) -> None:
        """Hold exactly the keys flagged in *keys*."""
        ...

    def release_all(self) -> None:
        """Release every held key."""
        ...


class LocalGame:
    """Paces the real game at the control rate; implements ``GameBackend``.

    The capture stream delivers frames faster than the control rate and
    queues them, so each step discards the backlog and returns the newest
    frame. The stream and sender are owned by the caller, which starts and
    stops them.
    """

    def __init__(
        self,
        stream: FrameSource,
        sender: KeySink,
        step_hz: float = STEP_HZ,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        """Wrap a started capture stream and a key sender.

        Args:
            stream: Source of ``(H, W, 3)`` frames holding 0-255 values.
            sender: Destination for key vectors.
            step_hz: Control rate.
            clock: Monotonic time source in seconds.
            sleep: Function that blocks for a number of seconds.
        """
        self._stream = stream
        self._sender = sender
        self._pacer = Pacer(step_hz, clock=clock, sleep=sleep)
        self.last_late_ms = 0.0

    def reset(self) -> np.ndarray:
        """Release keys and return the current frame.

        The run itself is started by hand (daily runs cannot be scripted
        safely), so this does not touch the game's menus.
        """
        self._sender.release_all()
        self._pacer.reset()
        return self._newest_frame()

    def step(self, keys: Sequence[int]) -> np.ndarray:
        """Hold *keys* until the next control tick and return the newest frame."""
        self._sender.apply_vector(list(keys))
        self.last_late_ms = self._pacer.wait_next()
        return self._newest_frame()

    def release_all(self) -> None:
        """Release every held key."""
        self._sender.release_all()

    def pause(self) -> None:
        """Release keys; the real game is never paused by the agent."""
        self._sender.release_all()

    def close(self) -> None:
        """Release keys; the stream and sender stay open for their owner."""
        self._sender.release_all()

    def _newest_frame(self) -> np.ndarray:
        """Drain queued frames and return the most recent one as uint8.

        Raises:
            RuntimeError: If the capture stream delivers no frame in time.
        """
        newest = self._stream.get_frame(timeout=FRAME_TIMEOUT_S)
        if newest is None:
            raise RuntimeError("capture stream ended or timed out")
        while (queued := self._stream.get_frame(timeout=0.0)) is not None:
            newest = queued
        return newest.astype(np.uint8)
