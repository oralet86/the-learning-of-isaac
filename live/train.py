#!/usr/bin/env python3
"""Training-mode entry point: a fleet of game containers behind a vector env.

Starts one container per environment, connects a worker process to each
sidecar and runs a rollout loop. The policy is still the random placeholder;
this proves the plumbing the learner will sit on.

Usage::

    python -m live.train --total-steps 2000
"""

from __future__ import annotations

import argparse
import logging
import time

import numpy as np

from constants import STEP_HZ
from container.build import DEFAULT_IMAGE
from container.sidecar.config import DEFAULT_GAME_HEIGHT, DEFAULT_GAME_WIDTH
from live.fleet import DEFAULT_BASE_PORT, ContainerFleet, FleetConfig
from live.model import RandomAgent
from live.vector import make_vector_env

logger = logging.getLogger("live.train")

_LOG_EVERY_STEPS = 100
# Each game needs about two CPU cores under software rendering. Four leaves
# headroom on an 8-core machine for busy rooms and for the trainer itself.
DEFAULT_NUM_ENVS = 4


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Run a rollout loop over a fleet of game containers."
    )
    parser.add_argument("--num-envs", type=int, default=DEFAULT_NUM_ENVS)
    parser.add_argument("--total-steps", type=int, default=1000)
    parser.add_argument("--max-episode-steps", type=int, default=600)
    parser.add_argument("--image", default=DEFAULT_IMAGE)
    parser.add_argument("--base-port", type=int, default=DEFAULT_BASE_PORT)
    parser.add_argument("--game-width", type=int, default=DEFAULT_GAME_WIDTH)
    parser.add_argument("--game-height", type=int, default=DEFAULT_GAME_HEIGHT)
    return parser.parse_args()


def run_rollout(
    num_envs: int, endpoints: list[tuple[str, int]], args: argparse.Namespace
) -> None:
    """Step every environment with the placeholder agent and log throughput.

    Args:
        num_envs: Number of environments.
        endpoints: ``(host, port)`` of each sidecar.
        args: Parsed command-line arguments.
    """
    agents = [RandomAgent() for _ in range(num_envs)]
    envs = make_vector_env(endpoints, args.max_episode_steps)
    try:
        logger.info("resetting %d environments (menu navigation)...", num_envs)
        observations, _ = envs.reset()
        episodes = 0
        worst_late_ms = 0.0
        window_start = time.monotonic()
        for step in range(1, args.total_steps + 1):
            actions = np.array(
                [
                    agent.predict(obs)
                    for agent, obs in zip(agents, observations, strict=True)
                ]
            )
            observations, _, terminated, truncated, infos = envs.step(actions)
            episodes += int(np.sum(terminated | truncated))
            # Steps on which every environment auto-reset carry no step info.
            if "late_ms" in infos:
                worst_late_ms = max(worst_late_ms, float(np.max(infos["late_ms"])))
            if step % _LOG_EVERY_STEPS == 0:
                elapsed = time.monotonic() - window_start
                logger.info(
                    "step %d: %.1f vector steps/s (target %d), "
                    "worst lateness %.0f ms, %d episodes finished",
                    step,
                    _LOG_EVERY_STEPS / elapsed,
                    STEP_HZ,
                    worst_late_ms,
                    episodes,
                )
                worst_late_ms = 0.0
                window_start = time.monotonic()
    finally:
        envs.close()


def main() -> None:
    """Start the fleet, run the rollout, and always remove the containers."""
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s"
    )
    args = parse_args()
    config = FleetConfig(
        num_envs=args.num_envs,
        image=args.image,
        base_port=args.base_port,
        game_width=args.game_width,
        game_height=args.game_height,
    )
    with ContainerFleet(config) as fleet:
        run_rollout(args.num_envs, fleet.endpoints, args)


if __name__ == "__main__":
    main()
