"""Gymnasium environment for The Binding of Isaac.

Wraps a :class:`~live.backend.GameBackend` (a containerised game for
training, the real game for inference), a frame preprocessor and an event
detector into the standard RL interface. One step is one control tick
(50 ms at 20 Hz); keys stay held between steps.

Observation:
    Single-channel grayscale ``(TARGET_HEIGHT, TARGET_WIDTH)`` fp16
    array in ``[0, 1]``.

Action:
    ``MultiDiscrete([2] * 13)`` — 13 independent binary key presses,
    indexed by ``KEY_ORDER``.  0 = release, 1 = hold.

Reward and termination:
    Derived from the events the detector reads off the raw frame (floor
    transition, win, death); see :mod:`live.events`.

Truncation:
    After ``max_episode_steps`` steps, if set.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import gymnasium as gym
from gymnasium.spaces import Box, MultiDiscrete
import numpy as np

from constants import KEY_ORDER, TARGET_HEIGHT, TARGET_WIDTH
from live.events import NullDetector, RewardConfig, compute_reward

if TYPE_CHECKING:
    from collections.abc import Callable

    from live.backend import GameBackend
    from live.events import EventDetector


class IsaacEnv(gym.Env):
    """Isaac environment for reinforcement learning, in either mode."""

    def __init__(
        self,
        backend: GameBackend,
        preprocessor: Callable[[np.ndarray], np.ndarray],
        detector: EventDetector | None = None,
        reward_config: RewardConfig | None = None,
        max_episode_steps: int | None = None,
    ) -> None:
        """Assemble an environment around a game backend.

        Args:
            backend: The game to control.
            preprocessor: Maps a raw ``(H, W, 3)`` uint8 frame to an
                observation matching ``observation_space``.
            detector: Reads game events from raw frames. Defaults to
                ``NullDetector`` (no rewards, no termination).
            reward_config: Reward per event. Defaults to ``RewardConfig()``.
            max_episode_steps: Truncate episodes after this many steps;
                ``None`` for no limit.
        """
        super().__init__()

        self.backend = backend
        self.preprocessor = preprocessor
        self.detector = detector if detector is not None else NullDetector()
        self.reward_config = (
            reward_config if reward_config is not None else RewardConfig()
        )
        self.max_episode_steps = max_episode_steps

        self.observation_space = Box(
            low=0.0,
            high=1.0,
            shape=(TARGET_HEIGHT, TARGET_WIDTH),
            dtype=np.float16,
        )

        self.action_space = MultiDiscrete([2] * len(KEY_ORDER))

        self._episode_steps = 0

    # ------------------------------------------------------------------
    # Gymnasium API
    # ------------------------------------------------------------------

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict | None = None,  # noqa: ARG002
    ) -> tuple[np.ndarray, dict]:
        """Start a new episode.

        Args:
            seed: Passed through to ``gym.Env.reset()``. It seeds the action
                space only; the game's own seed is not controlled.
            options: Ignored.

        Returns:
            ``(observation, info)`` tuple.
        """
        super().reset(seed=seed)
        frame = self.backend.reset()
        self.detector.reset()
        self._episode_steps = 0
        return self.preprocessor(frame), {}

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, bool, dict]:
        """Hold the action's keys for one tick and return the outcome.

        Args:
            action: 13-element integer array (0 = release, 1 = hold)
                    over ``KEY_ORDER``.

        Returns:
            ``(observation, reward, terminated, truncated, info)`` tuple.
        """
        frame = self.backend.step(action.tolist())
        self._episode_steps += 1

        events = self.detector.detect(frame)
        reward = compute_reward(events, self.reward_config)
        terminated = events.episode_over
        truncated = (
            self.max_episode_steps is not None
            and self._episode_steps >= self.max_episode_steps
        )

        info: dict = {
            "episode_steps": self._episode_steps,
            "late_ms": self.backend.last_late_ms,
            "death": events.death,
            "win": events.win,
            "floor_transition": events.floor_transition,
        }

        return self.preprocessor(frame), reward, terminated, truncated, info

    def pause(self) -> None:
        """Pause the run until the next step; see ``GameBackend.pause``."""
        self.backend.pause()

    def close(self) -> None:
        """Release keys and close the backend."""
        self.backend.close()
