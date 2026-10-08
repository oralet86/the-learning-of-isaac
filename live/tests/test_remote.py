"""Tests for `live.remote.RemoteGame`.

The client is run against the real sidecar request handler serving a fake
game on a loopback socket, so both halves of the protocol are exercised
together without a container.
"""

from __future__ import annotations

import socket
import threading
from typing import TYPE_CHECKING

import numpy as np
import pytest

from constants import KEY_ORDER, STEP_HZ
from container.sidecar.protocol import ProtocolError, ResetMode
from container.sidecar.server import ClientHandler
from container.tests.fakes import FakeControl
from live.remote import RemoteGame, SidecarError

if TYPE_CHECKING:
    from collections.abc import Iterator

_NO_KEYS = [0] * len(KEY_ORDER)


class _FakeSidecar:
    """Accepts one client on an ephemeral loopback port and serves it."""

    def __init__(self, tick_hz: int = STEP_HZ) -> None:
        self.control = FakeControl()
        self._listener = socket.create_server(("127.0.0.1", 0))
        self.port: int = self._listener.getsockname()[1]
        self._tick_hz = tick_hz
        self._thread = threading.Thread(target=self._serve, daemon=True)
        self._thread.start()

    def _serve(self) -> None:
        connection, _ = self._listener.accept()
        with connection:
            ClientHandler(connection, self.control, self._tick_hz).serve()

    def join(self) -> None:
        self._thread.join(timeout=5)
        self._listener.close()
        assert not self._thread.is_alive()


@pytest.fixture
def sidecar() -> Iterator[_FakeSidecar]:
    instance = _FakeSidecar()
    yield instance
    instance.join()


def test_reset_then_steps_return_frames_in_order(sidecar: _FakeSidecar) -> None:
    keys = list(_NO_KEYS)
    keys[KEY_ORDER.index("d")] = 1

    with RemoteGame("127.0.0.1", sidecar.port) as game:
        first = game.reset()
        second = game.step(keys)
        third = game.step(_NO_KEYS)

    assert first.shape == (6, 8, 3)
    assert first.dtype == np.uint8
    assert [int(frame[0, 0, 0]) for frame in (first, second, third)] == [0, 1, 2]
    assert game.last_late_ms == pytest.approx(1.5)
    assert sidecar.control.resets == [ResetMode.AUTO]
    assert sidecar.control.steps[0][KEY_ORDER.index("d")] is True
    assert sum(sidecar.control.steps[0]) == 1


def test_error_reply_raises_and_next_reset_relaunches(sidecar: _FakeSidecar) -> None:
    with RemoteGame("127.0.0.1", sidecar.port) as game:
        game.reset()
        sidecar.control.fail_next_step = True
        with pytest.raises(SidecarError, match="not running"):
            game.step(_NO_KEYS)

        game.reset()
        game.reset()

    assert sidecar.control.resets == [
        ResetMode.AUTO,
        ResetMode.RELAUNCH,
        ResetMode.AUTO,
    ]


def test_step_rejects_wrong_key_count_before_sending(sidecar: _FakeSidecar) -> None:
    with RemoteGame("127.0.0.1", sidecar.port) as game:
        with pytest.raises(ValueError, match="expected 13 keys"):
            game.step([1, 0])
        assert game.step(_NO_KEYS).shape == (6, 8, 3)


def test_release_all_and_close_release_keys(sidecar: _FakeSidecar) -> None:
    with RemoteGame("127.0.0.1", sidecar.port) as game:
        game.release_all()

    sidecar.join()
    # Once explicitly, once when the handler saw the client leave.
    assert sidecar.control.release_calls == 2


def test_control_rate_mismatch_is_rejected_at_connect() -> None:
    sidecar = _FakeSidecar(tick_hz=STEP_HZ + 10)
    with pytest.raises(ProtocolError, match="Hz"):
        RemoteGame("127.0.0.1", sidecar.port)
    sidecar.join()


def test_pause_reaches_the_sidecar(sidecar: _FakeSidecar) -> None:
    with RemoteGame("127.0.0.1", sidecar.port) as game:
        game.pause()
        assert sidecar.control.pause_calls == 1
        assert game.step(_NO_KEYS).shape == (6, 8, 3)
