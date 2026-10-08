"""Wire format shared by the in-container sidecar and the trainer-side client.

Every message is ``[u32 body length][u8 message type][payload]`` in network byte
order. The exchange is strictly request/reply: the client sends one request and
reads exactly one reply (``CLOSE`` is the only request without a reply).

This module is the single source of truth for the format; it is imported by both
``container.sidecar.server`` and ``live.remote`` so the two sides cannot drift.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
import struct
from typing import Protocol

PROTOCOL_VERSION = 2
DEFAULT_PORT = 5599

# Upper bound on a single message body, to fail fast on a corrupt length prefix.
MAX_BODY_BYTES = 64 * 1024 * 1024

_LENGTH = struct.Struct("!I")
_HELLO = struct.Struct("!H")
_HELLO_REPLY = struct.Struct("!HHHH")
_FRAME_HEADER = struct.Struct("!HHIf")
_RESET = struct.Struct("!B")

RGB_CHANNELS = 3


class ProtocolError(RuntimeError):
    """Raised when a peer sends a malformed or unexpected message."""


class MessageType(IntEnum):
    """Message type tags; requests are below 100, replies at or above."""

    HELLO = 1
    STEP = 2
    RESET = 3
    RELEASE_ALL = 4
    CLOSE = 5
    PAUSE = 6

    HELLO_REPLY = 101
    FRAME = 102
    OK = 103
    ERROR = 104


class ResetMode(IntEnum):
    """How the sidecar should bring the game back to the start of a run."""

    # Navigate the menus if the game was just launched, otherwise quick-restart.
    AUTO = 0
    # Kill the game process, relaunch it and navigate the menus again.
    RELAUNCH = 1


class ByteStream(Protocol):
    """The subset of ``socket.socket`` the framing helpers rely on."""

    def recv(self, bufsize: int, /) -> bytes:
        """Return up to *bufsize* bytes, or ``b""`` at end of stream."""
        ...

    def sendall(self, data: bytes, /) -> None:
        """Send all of *data*."""
        ...


@dataclass(frozen=True)
class HelloReply:
    """Sidecar capabilities announced in response to ``HELLO``.

    Attributes:
        version: Protocol version the sidecar speaks.
        width: Current game window width. Informational: the window can
            still change size until the first reset, and every frame carries
            its own size.
        height: Current game window height; see ``width``.
        tick_hz: Control rate; one ``STEP`` advances the game by ``1 / tick_hz`` s.
        key_order: Key names, in the order ``STEP`` key vectors are interpreted.
    """

    version: int
    width: int
    height: int
    tick_hz: int
    key_order: tuple[str, ...]

    def pack(self) -> bytes:
        """Serialise to a ``HELLO_REPLY`` payload."""
        names = ",".join(self.key_order).encode("ascii")
        return (
            _HELLO_REPLY.pack(self.version, self.width, self.height, self.tick_hz)
            + names
        )

    @classmethod
    def unpack(cls, payload: bytes) -> HelloReply:
        """Parse a ``HELLO_REPLY`` payload.

        Raises:
            ProtocolError: If the payload is too short.
        """
        if len(payload) < _HELLO_REPLY.size:
            raise ProtocolError(f"HELLO_REPLY payload too short: {len(payload)} bytes")
        version, width, height, tick_hz = _HELLO_REPLY.unpack_from(payload)
        names = payload[_HELLO_REPLY.size :].decode("ascii")
        key_order = tuple(names.split(",")) if names else ()
        return cls(version, width, height, tick_hz, key_order)


@dataclass(frozen=True)
class FrameHeader:
    """Metadata preceding the raw RGB bytes of a ``FRAME`` reply.

    Attributes:
        width: Frame width in pixels.
        height: Frame height in pixels.
        step: Number of ticks the sidecar has served since it started.
        late_ms: How far past its tick deadline the request arrived; ``0.0``
            when the client is keeping up with the control rate.
    """

    width: int
    height: int
    step: int
    late_ms: float

    @property
    def pixel_bytes(self) -> int:
        """Number of RGB bytes that must follow this header."""
        return self.width * self.height * RGB_CHANNELS

    def pack(self) -> bytes:
        """Serialise to the fixed-size header."""
        return _FRAME_HEADER.pack(self.width, self.height, self.step, self.late_ms)

    @classmethod
    def unpack(cls, payload: bytes) -> tuple[FrameHeader, memoryview]:
        """Split a ``FRAME`` payload into its header and pixel bytes.

        Raises:
            ProtocolError: If the payload length does not match the header.
        """
        if len(payload) < _FRAME_HEADER.size:
            raise ProtocolError(f"FRAME payload too short: {len(payload)} bytes")
        header = cls(*_FRAME_HEADER.unpack_from(payload))
        pixels = memoryview(payload)[_FRAME_HEADER.size :]
        if len(pixels) != header.pixel_bytes:
            raise ProtocolError(
                f"FRAME carries {len(pixels)} pixel bytes, header "
                f"{header.width}x{header.height} needs {header.pixel_bytes}"
            )
        return header, pixels


def pack_hello(version: int = PROTOCOL_VERSION) -> bytes:
    """Serialise a ``HELLO`` payload carrying the client's protocol version."""
    return _HELLO.pack(version)


def unpack_hello(payload: bytes) -> int:
    """Return the protocol version from a ``HELLO`` payload.

    Raises:
        ProtocolError: If the payload has the wrong size.
    """
    if len(payload) != _HELLO.size:
        raise ProtocolError(f"HELLO payload must be {_HELLO.size} bytes")
    return _HELLO.unpack(payload)[0]


def pack_keys(keys: list[int] | tuple[int, ...]) -> bytes:
    """Serialise a binary key vector to a ``STEP`` payload (one byte per key)."""
    return bytes(1 if key else 0 for key in keys)


def unpack_keys(payload: bytes, expected: int) -> list[bool]:
    """Parse a ``STEP`` payload into a held/released flag per key.

    Raises:
        ProtocolError: If the payload does not carry exactly *expected* keys.
    """
    if len(payload) != expected:
        raise ProtocolError(f"STEP carries {len(payload)} keys, expected {expected}")
    return [byte != 0 for byte in payload]


def pack_reset(mode: ResetMode) -> bytes:
    """Serialise a ``RESET`` payload."""
    return _RESET.pack(int(mode))


def unpack_reset(payload: bytes) -> ResetMode:
    """Parse a ``RESET`` payload.

    Raises:
        ProtocolError: If the payload is malformed or names an unknown mode.
    """
    if len(payload) != _RESET.size:
        raise ProtocolError(f"RESET payload must be {_RESET.size} byte")
    try:
        return ResetMode(_RESET.unpack(payload)[0])
    except ValueError as error:
        raise ProtocolError(str(error)) from error


def encode_message(message_type: MessageType, payload: bytes = b"") -> bytes:
    """Frame *payload* as a complete message ready to be sent."""
    return _LENGTH.pack(len(payload) + 1) + bytes([message_type]) + payload


def send_message(
    stream: ByteStream, message_type: MessageType, payload: bytes = b""
) -> None:
    """Frame and send one message."""
    stream.sendall(encode_message(message_type, payload))


def recv_exact(stream: ByteStream, count: int) -> bytes:
    """Read exactly *count* bytes.

    Raises:
        ConnectionError: If the peer closes the stream first.
    """
    buffer = bytearray()
    while len(buffer) < count:
        chunk = stream.recv(count - len(buffer))
        if not chunk:
            raise ConnectionError(
                f"peer closed the connection after {len(buffer)}/{count} bytes"
            )
        buffer.extend(chunk)
    return bytes(buffer)


def recv_message(stream: ByteStream) -> tuple[MessageType, bytes]:
    """Read one framed message.

    Returns:
        The message type and its payload.

    Raises:
        ConnectionError: If the peer closes the stream mid-message.
        ProtocolError: If the length prefix or type tag is invalid.
    """
    (body_length,) = _LENGTH.unpack(recv_exact(stream, _LENGTH.size))
    if not 1 <= body_length <= MAX_BODY_BYTES:
        raise ProtocolError(f"invalid message body length {body_length}")
    body = recv_exact(stream, body_length)
    try:
        message_type = MessageType(body[0])
    except ValueError as error:
        raise ProtocolError(f"unknown message type {body[0]}") from error
    return message_type, body[1:]
