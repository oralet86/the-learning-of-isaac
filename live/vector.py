"""Vectorised training environment: one worker process per game container."""

from __future__ import annotations

from functools import partial

from gymnasium.vector import AsyncVectorEnv

from live.env import IsaacEnv
from live.preprocess import FramePreprocessor
from live.remote import RemoteGame


def make_remote_env(
    host: str, port: int, max_episode_steps: int | None = None
) -> IsaacEnv:
    """Build one environment connected to the sidecar at *host*:*port*.

    Runs inside a worker process, so the connection and the preprocessing
    belong to that worker.

    Args:
        host: Address the sidecar's port is published on.
        port: Published sidecar port.
        max_episode_steps: Step limit per episode; ``None`` for no limit.
    """
    return IsaacEnv(
        RemoteGame(host, port),
        FramePreprocessor(),
        max_episode_steps=max_episode_steps,
    )


def pause_all(envs: AsyncVectorEnv) -> None:
    """Pause every game until the next vector step.

    Call this before the trainer stops stepping (for example for a policy
    update), so no run plays on unattended in the meantime.
    """
    envs.call("pause")


def make_vector_env(
    endpoints: list[tuple[str, int]], max_episode_steps: int | None = None
) -> AsyncVectorEnv:
    """Build a vector environment with one worker process per sidecar.

    The games run in real time, so all workers step concurrently and a
    vector step takes one control tick regardless of the number of games.

    Args:
        endpoints: ``(host, port)`` of each sidecar.
        max_episode_steps: Step limit per episode; ``None`` for no limit.
    """
    env_fns = [
        partial(make_remote_env, host, port, max_episode_steps)
        for host, port in endpoints
    ]
    # float16 observations have no ctypes equivalent, which the shared-memory
    # transport requires, so observations are sent through pipes instead.
    return AsyncVectorEnv(env_fns, shared_memory=False)
