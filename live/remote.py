"""Training-mode backend: a containerised game driven through its sidecar."""

from __future__ import annotations

import logging
import socket
from typing import TYPE_CHECKING, Self

import numpy as np

from constants import KEY_ORDER
from container.sidecar import protocol
from container.sidecar.protocol import (
    FrameHeader,
    HelloReply,
    MessageType,
    ProtocolError,
    ResetMode,
)
from live.backend import STEP_HZ

if TYPE_CHECKING:
    from collections.abc import Sequence

logger = logging.getLogger(__name__)

DEFAULT_CONNECT_TIMEOUT_S = 10.0
# A reset navigates menus (or relaunches the game), so replies take a while.
DEFAULT_REPLY_TIMEOUT_S = 180.0


class SidecarError(RuntimeError):
    """Raised when the sidecar reports that a request failed."""


class RemoteGame:
    """TCP client for one sidecar; implements ``GameBackend``."""

    def __init__(
        self,
        host: str,
        port: int = protocol.DEFAULT_PORT,
        connect_timeout_s: float = DEFAULT_CONNECT_TIMEOUT_S,
        reply_timeout_s: float = DEFAULT_REPLY_TIMEOUT_S,
    ) -> None:
        """Connect to a sidecar and check that both sides agree on the protocol.

        Args:
            host: Address the sidecar's port is published on.
            port: Published sidecar port.
            connect_timeout_s: Seconds to wait for the TCP connection.
            reply_timeout_s: Seconds to wait for any single reply.

        Raises:
            OSError: If the connection cannot be established.
            ProtocolError: If the sidecar's protocol version, key order or
                control rate differ from this side's.
        """
        self._socket = socket.create_connection((host, port), connect_timeout_s)
        self._socket.settimeout(reply_timeout_s)
        self._socket.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        self._needs_relaunch = False
        self.last_late_ms = 0.0
        try:
            self.hello = self._handshake()
        except BaseException:
            self._socket.close()
            raise

    def reset(self) -> np.ndarray:
        """Start a new run; relaunches the game if the last request failed."""
        mode = ResetMode.RELAUNCH if self._needs_relaunch else ResetMode.AUTO
        protocol.send_message(
            self._socket, MessageType.RESET, protocol.pack_reset(mode)
        )
        frame = self._read_frame()
        self._needs_relaunch = False
        return frame

    def step(self, keys: Sequence[int]) -> np.ndarray:
        """Hold *keys* for one tick and return the resulting frame.

        Raises:
            ValueError: If *keys* has the wrong length.
            SidecarError: If the sidecar could not step the game.
        """
        if len(keys) != len(KEY_ORDER):
            raise ValueError(f"expected {len(KEY_ORDER)} keys, got {len(keys)}")
        protocol.send_message(
            self._socket, MessageType.STEP, protocol.pack_keys(list(keys))
        )
        return self._read_frame()

    def release_all(self) -> None:
        """Release every held key in the container."""
        protocol.send_message(self._socket, MessageType.RELEASE_ALL)
        self._expect(MessageType.OK)

    def pause(self) -> None:
        """Pause the containerised game until the next step or reset.

        The sidecar also does this by itself when no request arrives for a
        short while; calling it explicitly avoids even that short gap.
        """
        protocol.send_message(self._socket, MessageType.PAUSE)
        self._expect(MessageType.OK)

    def close(self) -> None:
        """Tell the sidecar this client is done and close the connection.

        The sidecar releases all keys and pauses when a client goes away.
        """
        try:
            protocol.send_message(self._socket, MessageType.CLOSE)
        except OSError:
            logger.debug("sidecar connection already closed", exc_info=True)
        finally:
            self._socket.close()

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *args: object) -> None:
        self.close()

    def _handshake(self) -> HelloReply:
        """Exchange HELLO and verify the sidecar is compatible."""
        protocol.send_message(self._socket, MessageType.HELLO, protocol.pack_hello())
        hello = HelloReply.unpack(self._expect(MessageType.HELLO_REPLY))
        if hello.version != protocol.PROTOCOL_VERSION:
            raise ProtocolError(
                f"sidecar speaks protocol {hello.version}, "
                f"client speaks {protocol.PROTOCOL_VERSION}"
            )
        if list(hello.key_order) != KEY_ORDER:
            raise ProtocolError(
                f"sidecar key order {hello.key_order} differs from "
                f"constants.KEY_ORDER {KEY_ORDER}"
            )
        if hello.tick_hz != STEP_HZ:
            raise ProtocolError(
                f"sidecar runs at {hello.tick_hz} Hz, expected {STEP_HZ} Hz"
            )
        return hello

    def _expect(self, expected: MessageType) -> bytes:
        """Read one reply and return its payload.

        Raises:
            SidecarError: If the sidecar replied with ERROR.
            ProtocolError: If the reply has an unexpected type.
        """
        message_type, payload = protocol.recv_message(self._socket)
        if message_type is MessageType.ERROR:
            self._needs_relaunch = True
            raise SidecarError(payload.decode("utf-8", errors="replace"))
        if message_type is not expected:
            raise ProtocolError(
                f"expected {expected.name}, sidecar sent {message_type.name}"
            )
        return payload

    def _read_frame(self) -> np.ndarray:
        """Read a FRAME reply into an ``(H, W, 3)`` uint8 array."""
        header, pixels = FrameHeader.unpack(self._expect(MessageType.FRAME))
        self.last_late_ms = header.late_ms
        return np.frombuffer(pixels, dtype=np.uint8).reshape(
            header.height, header.width, protocol.RGB_CHANNELS
        )
