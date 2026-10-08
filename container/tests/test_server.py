"""Tests for `container.sidecar.server.ClientHandler`.

The handler is driven over a real socket pair against a fake game, covering
the request/reply contract the trainer depends on and the guarantee that
keys are released whenever a client goes away.
"""

from __future__ import annotations

import socket
import threading
from typing import TYPE_CHECKING

import numpy as np
import pytest

from constants import KEY_ORDER
from container.sidecar import protocol
from container.sidecar.protocol import (
    FrameHeader,
    HelloReply,
    MessageType,
    ResetMode,
)
from container.sidecar.server import ClientHandler
from container.tests.fakes import FakeControl

if TYPE_CHECKING:
    from collections.abc import Iterator

_TICK_HZ = 20
_NO_KEYS = protocol.pack_keys([0] * len(KEY_ORDER))


class _Served:
    """A handler running in a thread, plus the client end of its socket."""

    def __init__(self) -> None:
        self.control = FakeControl()
        self.client, server_side = socket.socketpair()
        self._server_side = server_side
        handler = ClientHandler(server_side, self.control, _TICK_HZ)
        self.thread = threading.Thread(target=handler.serve, daemon=True)
        self.thread.start()

    def request(
        self, message_type: MessageType, payload: bytes = b""
    ) -> tuple[MessageType, bytes]:
        protocol.send_message(self.client, message_type, payload)
        return protocol.recv_message(self.client)

    def finish(self) -> None:
        self.client.close()
        self.thread.join(timeout=5)
        self._server_side.close()
        assert not self.thread.is_alive()


@pytest.fixture
def served() -> Iterator[_Served]:
    instance = _Served()
    yield instance
    instance.finish()


def test_hello_announces_size_rate_and_key_order(served: _Served) -> None:
    message_type, payload = served.request(MessageType.HELLO, protocol.pack_hello())

    assert message_type is MessageType.HELLO_REPLY
    assert HelloReply.unpack(payload) == HelloReply(
        protocol.PROTOCOL_VERSION, 8, 6, _TICK_HZ, tuple(KEY_ORDER)
    )


def test_step_applies_keys_and_returns_counted_frame(served: _Served) -> None:
    keys = [0] * len(KEY_ORDER)
    keys[3] = 1

    served.request(MessageType.STEP, _NO_KEYS)
    message_type, payload = served.request(MessageType.STEP, protocol.pack_keys(keys))

    assert message_type is MessageType.FRAME
    header, pixels = FrameHeader.unpack(payload)
    assert (header.width, header.height, header.step) == (8, 6, 2)
    assert header.late_ms == pytest.approx(1.5)
    frame = np.frombuffer(pixels, dtype=np.uint8).reshape(6, 8, 3)
    assert np.all(frame == 2)
    assert served.control.steps[-1] == [bool(key) for key in keys]


def test_reset_passes_mode_and_returns_frame(served: _Served) -> None:
    message_type, _ = served.request(
        MessageType.RESET, protocol.pack_reset(ResetMode.RELAUNCH)
    )

    assert message_type is MessageType.FRAME
    assert served.control.resets == [ResetMode.RELAUNCH]


def test_release_all_is_acknowledged(served: _Served) -> None:
    assert served.request(MessageType.RELEASE_ALL) == (MessageType.OK, b"")
    assert served.control.release_calls == 1


def test_game_failure_is_reported_and_connection_survives(served: _Served) -> None:
    served.control.fail_next_step = True

    message_type, payload = served.request(MessageType.STEP, _NO_KEYS)

    assert message_type is MessageType.ERROR
    assert "not running" in payload.decode()
    assert served.request(MessageType.STEP, _NO_KEYS)[0] is MessageType.FRAME


def test_wrong_key_count_is_a_protocol_error_that_drops_the_client(
    served: _Served,
) -> None:
    message_type, payload = served.request(MessageType.STEP, b"\x01")

    assert message_type is MessageType.ERROR
    assert "expected 13" in payload.decode()
    served.thread.join(timeout=5)
    assert not served.thread.is_alive()


def test_version_mismatch_is_rejected(served: _Served) -> None:
    message_type, payload = served.request(
        MessageType.HELLO, protocol.pack_hello(protocol.PROTOCOL_VERSION + 1)
    )

    assert message_type is MessageType.ERROR
    assert "protocol" in payload.decode()


@pytest.mark.parametrize("polite", [True, False])
def test_keys_are_released_when_the_client_leaves(polite: bool) -> None:
    served = _Served()
    served.request(MessageType.STEP, _NO_KEYS)
    if polite:
        protocol.send_message(served.client, MessageType.CLOSE)

    served.finish()

    assert served.control.release_calls == 1


def test_pause_request_pauses_and_is_acknowledged(served: _Served) -> None:
    assert served.request(MessageType.PAUSE) == (MessageType.OK, b"")
    assert served.control.pause_calls == 1


def test_game_is_paused_when_the_client_leaves(served: _Served) -> None:
    served.request(MessageType.STEP, _NO_KEYS)
    protocol.send_message(served.client, MessageType.CLOSE)
    served.thread.join(timeout=5)

    assert served.control.pause_calls == 1


def test_idle_client_triggers_one_pause_per_silence() -> None:
    control = FakeControl()
    client, server_side = socket.socketpair()
    handler = ClientHandler(server_side, control, _TICK_HZ, idle_pause_s=0.05)
    thread = threading.Thread(target=handler.serve, daemon=True)
    thread.start()
    try:
        protocol.send_message(client, MessageType.STEP, _NO_KEYS)
        protocol.recv_message(client)
        # Stay silent well past the idle limit; the handler must pause once
        # and then wait, not pause again on every poll.
        deadline = threading.Event()
        deadline.wait(0.3)
        assert control.pause_calls == 1

        protocol.send_message(client, MessageType.STEP, _NO_KEYS)
        assert protocol.recv_message(client)[0] is MessageType.FRAME
    finally:
        client.close()
        thread.join(timeout=5)
        server_side.close()
    assert not thread.is_alive()
