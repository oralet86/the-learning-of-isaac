"""Starts and stops the game containers used for training."""

from __future__ import annotations

from dataclasses import dataclass
import logging
import subprocess
import time
from typing import Self

from container.build import DEFAULT_IMAGE, DEFAULT_RUNTIME
from container.sidecar.config import DEFAULT_GAME_HEIGHT, DEFAULT_GAME_WIDTH
from container.sidecar.protocol import DEFAULT_PORT as SIDECAR_PORT
from live.remote import RemoteGame

logger = logging.getLogger(__name__)

DEFAULT_BASE_PORT = 5600
_LOOPBACK = "127.0.0.1"
_READY_POLL_S = 1.0
_STOP_TIMEOUT_S = 5


@dataclass(frozen=True)
class FleetConfig:
    """How to run the training containers.

    Attributes:
        num_envs: Number of containers (one game each).
        image: Container image to run.
        base_port: Host port of the first container; container ``i`` is
            published on ``base_port + i``.
        game_width: Game window width requested in every container.
        game_height: Game window height requested in every container.
        runtime: Container CLI to invoke.
        name_prefix: Container names are ``<name_prefix>-<index>``.
        ready_timeout_s: How long to wait for every sidecar to accept
            connections.
    """

    num_envs: int
    image: str = DEFAULT_IMAGE
    base_port: int = DEFAULT_BASE_PORT
    game_width: int = DEFAULT_GAME_WIDTH
    game_height: int = DEFAULT_GAME_HEIGHT
    runtime: str = DEFAULT_RUNTIME
    name_prefix: str = "tloi-env"
    ready_timeout_s: float = 180.0


class ContainerFleet:
    """A set of running game containers, removed again on exit.

    Use as a context manager so containers are cleaned up even when the
    trainer fails::

        with ContainerFleet(FleetConfig(num_envs=4)) as fleet:
            for host, port in fleet.endpoints:
                ...
    """

    def __init__(self, config: FleetConfig) -> None:
        """Prepare a fleet; nothing is started until :meth:`start`.

        Raises:
            ValueError: If ``num_envs`` is not positive.
        """
        if config.num_envs <= 0:
            raise ValueError(f"num_envs must be positive, got {config.num_envs}")
        self._config = config
        self._started: list[str] = []

    @property
    def endpoints(self) -> list[tuple[str, int]]:
        """``(host, port)`` of every container's sidecar, in index order."""
        return [
            (_LOOPBACK, self._config.base_port + index)
            for index in range(self._config.num_envs)
        ]

    def start(self) -> None:
        """Start every container and wait until all sidecars answer.

        Raises:
            subprocess.CalledProcessError: If a container fails to start
                (for example because its name or port is already in use).
            TimeoutError: If a sidecar does not become ready in time.
        """
        config = self._config
        for index, (_, port) in enumerate(self.endpoints):
            name = f"{config.name_prefix}-{index}"
            command = [
                config.runtime,
                "run",
                "--detach",
                "--rm",
                "--name",
                name,
                # Loopback only: the sidecar has no authentication.
                "--publish",
                f"{_LOOPBACK}:{port}:{SIDECAR_PORT}",
                "--env",
                f"GAME_WIDTH={config.game_width}",
                "--env",
                f"GAME_HEIGHT={config.game_height}",
                config.image,
            ]
            subprocess.run(command, check=True, capture_output=True, text=True)
            self._started.append(name)
            logger.info("started container %s on port %d", name, port)
        self._wait_ready()

    def stop(self) -> None:
        """Stop and remove every container this fleet started."""
        if not self._started:
            return
        command = [
            self._config.runtime,
            "rm",
            "--force",
            "--time",
            str(_STOP_TIMEOUT_S),
            *self._started,
        ]
        result = subprocess.run(command, check=False, capture_output=True, text=True)
        if result.returncode != 0:
            logger.error("failed to remove containers: %s", result.stderr.strip())
        self._started.clear()

    def __enter__(self) -> Self:
        try:
            self.start()
        except BaseException:
            self.stop()
            raise
        return self

    def __exit__(self, *args: object) -> None:
        self.stop()

    def _wait_ready(self) -> None:
        """Block until every sidecar completes a handshake."""
        deadline = time.monotonic() + self._config.ready_timeout_s
        for host, port in self.endpoints:
            while True:
                try:
                    RemoteGame(host, port).close()
                    break
                except OSError as error:
                    if time.monotonic() >= deadline:
                        raise TimeoutError(
                            f"sidecar on port {port} not ready after "
                            f"{self._config.ready_timeout_s:.0f}s"
                        ) from error
                    time.sleep(_READY_POLL_S)
            logger.info("sidecar on port %d is ready", port)
