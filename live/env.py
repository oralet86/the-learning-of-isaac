"""Gymnasium environment for The Binding of Isaac: Rebirth.

Wraps :class:`StreamGame`, :class:`FramePreprocessor`, and
:class:`ActionSender` into a standard RL environment interface.
Each step corresponds to one game frame (~16.7 ms at 60 fps).

Observation:
    Single-channel grayscale ``(TARGET_HEIGHT, TARGET_WIDTH)`` fp16
    array in ``[0, 1]``.

Action:
    ``MultiDiscrete([2] * 13)`` — 13 independent binary key presses,
    indexed by ``KEY_ORDER``.  0 = release, 1 = hold.

Reward:
    Stub ``0.0`` (will be replaced by HUD reading).

Termination:
    Always ``False`` for now (no death detection yet).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import gymnasium as gym
from gymnasium.spaces import Box, MultiDiscrete
import numpy as np

from constants import KEY_ORDER, TARGET_HEIGHT, TARGET_WIDTH

if TYPE_CHECKING:
    from live.action import ActionSender
    from live.preprocess import FramePreprocessor
    from live.stream_game import StreamGame


class IsaacEnv(gym.Env):
    """Live Isaac environment for reinforcement learning."""

    def __init__(
        self,
        stream: StreamGame,
        sender: ActionSender,
        preprocessor: FramePreprocessor,
    ) -> None:
        super().__init__()

        self.stream = stream
        self.sender = sender
        self.preprocessor = preprocessor

        self.observation_space = Box(
            low=0.0,
            high=1.0,
            shape=(TARGET_HEIGHT, TARGET_WIDTH),
            dtype=np.float16,
        )

        self.action_space = MultiDiscrete([2] * len(KEY_ORDER))

        self._prev_frame_count = 0

    # ------------------------------------------------------------------
    # Gymnasium API
    # ------------------------------------------------------------------

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict | None = None,  # noqa: ARG002
    ) -> tuple[np.ndarray, dict]:
        """Start a new episode.  Reads one frame from the stream.

        Args:
            seed: Passed through to ``gym.Env.reset()``.
            options: Ignored.

        Returns:
            ``(observation, info)`` tuple.
        """
        super().reset(seed=seed)
        self._prev_frame_count = self.stream.frame_count
        obs = self._get_obs()
        return obs, {}

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, bool, dict]:
        """Apply an action and return the next observation.

        Args:
            action: 13-element integer array (0 = release, 1 = hold)
                    over ``KEY_ORDER``.

        Returns:
            ``(observation, reward, terminated, truncated, info)`` tuple.
        """
        self.sender.apply_vector(action.tolist())

        obs = self._get_obs()

        reward = 0.0
        terminated = False
        truncated = False

        info: dict = {
            "frame_count": self.stream.frame_count,
            "fps": self.stream.effective_fps,
            "steps_since_reset": (self.stream.frame_count - self._prev_frame_count),
        }

        return obs, reward, terminated, truncated, info

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _get_obs(self) -> np.ndarray:
        """Read the latest frame from the stream, preprocess it."""
        frame = self.stream.get_frame()
        if frame is None:
            raise RuntimeError("Stream ended unexpectedly")
        return self.preprocessor(frame)
