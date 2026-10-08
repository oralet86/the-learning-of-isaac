"""TCP front end of the sidecar: turns protocol messages into game calls."""

from __future__ import annotations

import logging
import select
import socket
from typing import TYPE_CHECKING, Protocol

from constants import KEY_ORDER
from container.sidecar import protocol
from container.sidecar.protocol import (
    FrameHeader,
    HelloReply,
    MessageType,
    ProtocolError,
    ResetMode,
)

if TYPE_CHECKING:
    from collections.abc import Sequence

    import numpy as np

logger = logging.getLogger(__name__)


class GameControl(Protocol):
    """What the server needs from the game; implemented by ``GameSession``."""

    @property
    def frame_size(self) -> tuple[int, int]:
        """``(width, height)`` of captured frames."""
        ...

    def reset(self, mode: ResetMode) -> np.ndarray:
        """Start a new run and return its first frame."""
        ...

    def step(self, keys: Sequence[bool]) -> tuple[np.ndarray, float]:
        """Advance one tick; return the frame and request lateness in ms."""
        ...

    def release_all(self) -> None:
        """Release every held key."""
        ...

    def pause(self) -> bool:
        """Pause the run until the next step; return whether it is paused."""
        ...


class ClientHandler:
    """Serves one client connection until it closes."""

    def __init__(
        self,
        connection: socket.socket,
        control: GameControl,
        tick_hz: int,
        idle_pause_s: float = 0.0,
    ) -> None:
        """Bind a connection to the game.

        Args:
            connection: Accepted client socket.
            control: Game to drive.
            tick_hz: Control rate announced to the client.
            idle_pause_s: Pause the game when no request has arrived for this
                long; ``0`` disables idle pausing.
        """
        self._connection = connection
        self._control = control
        self._tick_hz = tick_hz
        self._idle_pause_s = idle_pause_s
        self._step_count = 0

    def serve(self) -> None:
        """Process requests until the client sends CLOSE or disconnects.

        The game is always paused (which also releases all keys) when the
        client goes away, so a crashed trainer cannot leave a run playing.
        """
        try:
            while self._handle_one():
                pass
        except ConnectionError as error:
            logger.info("client disconnected: %s", error)
        except ProtocolError as error:
            logger.error("protocol error, dropping client: %s", error)
            self._send_error(str(error))
        finally:
            self._leave_game_safe()

    def _leave_game_safe(self) -> None:
        """Release keys and pause once the client is gone.

        A failure here must not take the sidecar down with it: the next
        client can still recover the game with a relaunching reset.
        """
        try:
            self._control.release_all()
            self._control.pause()
        except RuntimeError:
            logger.exception("could not pause the game after the client left")

    def _pause_if_idle(self) -> None:
        """Pause the game if the client stays silent past the idle limit.

        The game runs in real time, so whenever the trainer stalls (a policy
        update, another environment resetting) the run would otherwise play
        on with the last keys held.
        """
        if self._idle_pause_s <= 0:
            return
        readable, _, _ = select.select([self._connection], [], [], self._idle_pause_s)
        if not readable:
            self._control.pause()

    def _handle_one(self) -> bool:
        """Handle one request; return ``False`` when the client is done."""
        self._pause_if_idle()
        message_type, payload = protocol.recv_message(self._connection)
        if message_type is MessageType.CLOSE:
            return False
        try:
            self._dispatch(message_type, payload)
        except ProtocolError:
            raise
        except (RuntimeError, TimeoutError, ValueError, OSError) as error:
            # Game-side failures are reported to the client, which decides
            # whether to retry (e.g. with a RELAUNCH reset) or give up.
            logger.exception("request %s failed", message_type.name)
            self._send_error(f"{type(error).__name__}: {error}")
        return True

    def _dispatch(self, message_type: MessageType, payload: bytes) -> None:
        """Run the game call for one request and send its reply."""
        if message_type is MessageType.HELLO:
            self._hello(payload)
        elif message_type is MessageType.STEP:
            keys = protocol.unpack_keys(payload, len(KEY_ORDER))
            frame, late_ms = self._control.step(keys)
            self._step_count += 1
            self._send_frame(frame, late_ms)
        elif message_type is MessageType.RESET:
            frame = self._control.reset(protocol.unpack_reset(payload))
            self._send_frame(frame, 0.0)
        elif message_type is MessageType.RELEASE_ALL:
            self._control.release_all()
            protocol.send_message(self._connection, MessageType.OK)
        elif message_type is MessageType.PAUSE:
            self._control.pause()
            protocol.send_message(self._connection, MessageType.OK)
        else:
            raise ProtocolError(f"unexpected request {message_type.name}")

    def _hello(self, payload: bytes) -> None:
        """Check the client's protocol version and announce capabilities."""
        client_version = protocol.unpack_hello(payload)
        if client_version != protocol.PROTOCOL_VERSION:
            raise ProtocolError(
                f"client speaks protocol {client_version}, "
                f"sidecar speaks {protocol.PROTOCOL_VERSION}"
            )
        width, height = self._control.frame_size
        reply = HelloReply(
            protocol.PROTOCOL_VERSION, width, height, self._tick_hz, tuple(KEY_ORDER)
        )
        protocol.send_message(self._connection, MessageType.HELLO_REPLY, reply.pack())

    def _send_frame(self, frame: np.ndarray, late_ms: float) -> None:
        """Send *frame* as a FRAME reply."""
        height, width, _ = frame.shape
        header = FrameHeader(width, height, self._step_count, late_ms)
        protocol.send_message(
            self._connection, MessageType.FRAME, header.pack() + frame.tobytes()
        )

    def _send_error(self, message: str) -> None:
        """Best-effort ERROR reply; the client may already be gone."""
        try:
            protocol.send_message(
                self._connection, MessageType.ERROR, message.encode("utf-8")
            )
        except OSError:
            logger.debug("could not deliver error reply", exc_info=True)


def serve_forever(
    control: GameControl, port: int, tick_hz: int, idle_pause_s: float = 0.0
) -> None:
    """Accept clients one at a time and serve each until it disconnects.

    Args:
        control: Game to drive.
        port: TCP port to listen on (all interfaces; the container's port
            mapping decides who can reach it).
        tick_hz: Control rate announced to clients.
        idle_pause_s: Pause the game when a client is silent for this long;
            ``0`` disables idle pausing.
    """
    with socket.create_server(("0.0.0.0", port)) as listener:
        logger.info("sidecar listening on port %d", port)
        while True:
            connection, address = listener.accept()
            logger.info("client connected from %s", address)
            with connection:
                connection.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
                ClientHandler(connection, control, tick_hz, idle_pause_s).serve()
