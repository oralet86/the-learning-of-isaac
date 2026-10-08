"""Tests for `container.sidecar.protocol`.

Both ends of the TCP link import this module, so these cover what they rely
on: every message survives a round-trip, and corrupt or truncated input
fails loudly instead of being misread as a frame.
"""

from __future__ import annotations

import socket
import struct

import numpy as np
import pytest

from container.sidecar import protocol
from container.sidecar.protocol import (
    FrameHeader,
    HelloReply,
    MessageType,
    ProtocolError,
    ResetMode,
)


class _Buffer:
    """In-memory ``ByteStream`` that hands data back in small chunks."""

    def __init__(self, data: bytes = b"", chunk: int = 3) -> None:
        self.data = bytearray(data)
        self.chunk = chunk

    def recv(self, bufsize: int, /) -> bytes:
        size = min(bufsize, self.chunk)
        out, self.data = bytes(self.data[:size]), self.data[size:]
        return out

    def sendall(self, data: bytes, /) -> None:
        self.data.extend(data)


def test_message_round_trip_across_partial_reads() -> None:
    stream = _Buffer()
    protocol.send_message(stream, MessageType.STEP, b"\x01\x00\x01")
    protocol.send_message(stream, MessageType.CLOSE)

    assert protocol.recv_message(stream) == (MessageType.STEP, b"\x01\x00\x01")
    assert protocol.recv_message(stream) == (MessageType.CLOSE, b"")


def test_recv_message_rejects_truncated_stream() -> None:
    whole = protocol.encode_message(MessageType.STEP, b"\x01\x00")
    with pytest.raises(ConnectionError):
        protocol.recv_message(_Buffer(whole[:-1]))


def test_recv_message_rejects_unknown_type_and_bad_length() -> None:
    with pytest.raises(ProtocolError, match="unknown message type"):
        protocol.recv_message(_Buffer(struct.pack("!I", 1) + b"\xfa"))
    with pytest.raises(ProtocolError, match="invalid message body length"):
        protocol.recv_message(_Buffer(struct.pack("!I", 0)))
    with pytest.raises(ProtocolError, match="invalid message body length"):
        protocol.recv_message(_Buffer(struct.pack("!I", protocol.MAX_BODY_BYTES + 1)))


def test_hello_reply_round_trip() -> None:
    reply = HelloReply(1, 960, 540, 20, ("w", "a", "ctrl"))
    assert HelloReply.unpack(reply.pack()) == reply
    with pytest.raises(ProtocolError):
        HelloReply.unpack(b"\x00")


def test_hello_round_trip_and_size_check() -> None:
    assert protocol.unpack_hello(protocol.pack_hello(7)) == 7
    with pytest.raises(ProtocolError):
        protocol.unpack_hello(b"\x00\x00\x00")


def test_frame_round_trip_preserves_pixels() -> None:
    frame = np.arange(4 * 5 * 3, dtype=np.uint8).reshape(4, 5, 3)
    header = FrameHeader(width=5, height=4, step=42, late_ms=2.5)

    decoded, pixels = FrameHeader.unpack(header.pack() + frame.tobytes())

    assert decoded == header
    restored = np.frombuffer(pixels, dtype=np.uint8).reshape(4, 5, 3)
    np.testing.assert_array_equal(restored, frame)


def test_frame_rejects_pixel_count_mismatch() -> None:
    header = FrameHeader(width=5, height=4, step=0, late_ms=0.0)
    with pytest.raises(ProtocolError, match="pixel bytes"):
        FrameHeader.unpack(header.pack() + bytes(10))


def test_keys_round_trip_and_length_check() -> None:
    payload = protocol.pack_keys([1, 0, 0, 1])
    assert protocol.unpack_keys(payload, 4) == [True, False, False, True]
    with pytest.raises(ProtocolError, match="expected 13"):
        protocol.unpack_keys(payload, 13)


def test_reset_round_trip_and_unknown_mode() -> None:
    for mode in ResetMode:
        assert protocol.unpack_reset(protocol.pack_reset(mode)) is mode
    with pytest.raises(ProtocolError):
        protocol.unpack_reset(b"\x09")


def test_framing_works_over_a_real_socket() -> None:
    left, right = socket.socketpair()
    with left, right:
        payload = bytes(range(256)) * 40
        protocol.send_message(left, MessageType.FRAME, payload)
        assert protocol.recv_message(right) == (MessageType.FRAME, payload)
