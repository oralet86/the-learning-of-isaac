"""Game events read from the screen, and the rewards derived from them.

Nothing is embedded in the game, so death, victory and floor transitions
have to be recognised from frames. This module fixes the interface; the
trained detector plugs in behind :class:`EventDetector`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    import numpy as np


@dataclass(frozen=True)
class GameEvents:
    """Events that happened on the current step.

    Attributes:
        death: The character died; the episode is over.
        win: The run was won; the episode is over.
        floor_transition: A new floor was entered on this step. Reported once
            per transition, not for every frame of the transition screen.
    """

    death: bool = False
    win: bool = False
    floor_transition: bool = False

    @property
    def episode_over(self) -> bool:
        """Whether the run ended on this step."""
        return self.death or self.win


class EventDetector(Protocol):
    """Recognises game events from raw frames."""

    def reset(self) -> None:
        """Clear any per-episode state; called at the start of each run."""
        ...

    def detect(self, frame: np.ndarray) -> GameEvents:
        """Return the events visible in *frame*.

        Args:
            frame: Raw ``(H, W, 3)`` uint8 RGB frame from the backend.
        """
        ...


class NullDetector:
    """Detector that never reports an event.

    Placeholder until the trained detector exists; with it, episodes only end
    through the environment's step limit.
    """

    def reset(self) -> None:
        """Nothing to clear."""

    def detect(self, frame: np.ndarray) -> GameEvents:  # noqa: ARG002
        """Report no events."""
        return GameEvents()


@dataclass(frozen=True)
class RewardConfig:
    """Reward assigned to each event.

    Attributes:
        floor_transition: Reward for reaching a new floor.
        win: Reward for winning the run.
        death: Reward (normally negative) for dying.
        step: Reward added on every step, e.g. a small time penalty.
    """

    floor_transition: float = 1.0
    win: float = 10.0
    death: float = -1.0
    step: float = 0.0


def compute_reward(events: GameEvents, config: RewardConfig) -> float:
    """Return the reward for one step's *events* under *config*."""
    reward = config.step
    if events.floor_transition:
        reward += config.floor_transition
    if events.win:
        reward += config.win
    if events.death:
        reward += config.death
    return reward
