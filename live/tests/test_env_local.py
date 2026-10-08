"""Tests for `live.env.IsaacEnv`, `live.events` and `live.local.LocalGame`.

Uses scripted fake backends, detectors, streams and clocks: the point is the
episode bookkeeping (rewards, termination, truncation) and the local
backend's pacing and frame selection, none of which need a running game.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest

from constants import KEY_ORDER, TARGET_HEIGHT, TARGET_WIDTH
from live.env import IsaacEnv
from live.events import GameEvents, NullDetector, RewardConfig, compute_reward
from live.local import LocalGame
from live.preprocess import FramePreprocessor

if TYPE_CHECKING:
    from collections.abc import Sequence

_NO_KEYS = np.zeros(len(KEY_ORDER), dtype=np.int64)


class _FakeBackend:
    """Backend whose frames encode how many steps have been taken."""

    def __init__(self) -> None:
        self.last_late_ms = 0.0
        self.resets = 0
        self.steps: list[list[int]] = []
        self.closed = False
        self.pauses = 0

    def _frame(self) -> np.ndarray:
        return np.full((54, 96, 3), len(self.steps), dtype=np.uint8)

    def reset(self) -> np.ndarray:
        self.resets += 1
        self.steps.clear()
        return self._frame()

    def step(self, keys: Sequence[int]) -> np.ndarray:
        self.steps.append(list(keys))
        self.last_late_ms = 3.0
        return self._frame()

    def release_all(self) -> None:
        pass

    def pause(self) -> None:
        self.pauses += 1

    def close(self) -> None:
        self.closed = True


class _ScriptedDetector:
    """Reports a fixed sequence of events, one per step."""

    def __init__(self, script: list[GameEvents]) -> None:
        self._script = script
        self._index = 0
        self.resets = 0

    def reset(self) -> None:
        self.resets += 1
        self._index = 0

    def detect(self, frame: np.ndarray) -> GameEvents:  # noqa: ARG002
        events = self._script[self._index]
        self._index += 1
        return events


# --- events ---------------------------------------------------------------


def test_compute_reward_sums_event_rewards() -> None:
    config = RewardConfig(floor_transition=1.0, win=10.0, death=-2.0, step=-0.01)

    assert compute_reward(GameEvents(), config) == pytest.approx(-0.01)
    assert compute_reward(GameEvents(floor_transition=True), config) == pytest.approx(
        0.99
    )
    assert compute_reward(
        GameEvents(floor_transition=True, win=True), config
    ) == pytest.approx(10.99)
    assert compute_reward(GameEvents(death=True), config) == pytest.approx(-2.01)


def test_null_detector_never_ends_an_episode() -> None:
    events = NullDetector().detect(np.zeros((2, 2, 3), dtype=np.uint8))
    assert events == GameEvents()
    assert not events.episode_over


# --- IsaacEnv -------------------------------------------------------------


def test_env_observation_matches_declared_space_with_real_preprocessor() -> None:
    env = IsaacEnv(_FakeBackend(), FramePreprocessor())

    observation, _ = env.reset()
    stepped, *_ = env.step(_NO_KEYS)

    for array in (observation, stepped):
        assert array.shape == (TARGET_HEIGHT, TARGET_WIDTH)
        assert array.dtype == np.float16
        assert env.observation_space.contains(array)


def test_env_passes_keys_and_reports_rewards_and_termination() -> None:
    backend = _FakeBackend()
    detector = _ScriptedDetector(
        [GameEvents(), GameEvents(floor_transition=True), GameEvents(death=True)]
    )
    env = IsaacEnv(backend, lambda frame: frame, detector)
    action = _NO_KEYS.copy()
    action[KEY_ORDER.index("space")] = 1

    env.reset()
    outcomes = [env.step(action) for _ in range(3)]

    assert backend.steps[0] == action.tolist()
    assert [reward for _, reward, _, _, _ in outcomes] == [0.0, 1.0, -1.0]
    assert [terminated for _, _, terminated, _, _ in outcomes] == [False, False, True]
    assert outcomes[1][4]["floor_transition"] is True
    assert outcomes[2][4]["late_ms"] == 3.0
    assert outcomes[2][4]["episode_steps"] == 3


def test_env_truncates_at_step_limit_and_reset_starts_over() -> None:
    backend = _FakeBackend()
    detector = _ScriptedDetector([GameEvents()] * 4)
    env = IsaacEnv(backend, lambda frame: frame, detector, max_episode_steps=2)

    env.reset()
    truncated_flags = [env.step(_NO_KEYS)[3] for _ in range(2)]
    env.reset()
    after_reset = env.step(_NO_KEYS)[3]

    assert truncated_flags == [False, True]
    assert after_reset is False
    assert backend.resets == 2
    assert detector.resets == 2


def test_env_pause_is_forwarded_to_backend() -> None:
    backend = _FakeBackend()
    IsaacEnv(backend, lambda frame: frame).pause()
    assert backend.pauses == 1


def test_env_close_closes_backend() -> None:
    backend = _FakeBackend()
    IsaacEnv(backend, lambda frame: frame).close()
    assert backend.closed


# --- LocalGame ------------------------------------------------------------


class _FakeStream:
    """Hands out queued frames, then ``None`` like an empty queue."""

    def __init__(self) -> None:
        self.queue: list[np.ndarray] = []

    def push(self, *values: int) -> None:
        self.queue.extend(
            np.full((2, 2, 3), value, dtype=np.float16) for value in values
        )

    def get_frame(self, timeout: float | None = None) -> np.ndarray | None:  # noqa: ARG002
        return self.queue.pop(0) if self.queue else None


class _FakeSender:
    def __init__(self) -> None:
        self.vectors: list[list[int]] = []
        self.releases = 0

    def apply_vector(self, keys: list[int]) -> None:
        self.vectors.append(keys)

    def release_all(self) -> None:
        self.releases += 1


class _Clock:
    def __init__(self) -> None:
        self.now = 50.0
        self.sleeps: list[float] = []

    def __call__(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.sleeps.append(seconds)
        self.now += seconds


def _local_game() -> tuple[LocalGame, _FakeStream, _FakeSender, _Clock]:
    stream, sender, clock = _FakeStream(), _FakeSender(), _Clock()
    game = LocalGame(stream, sender, step_hz=20, clock=clock, sleep=clock.sleep)
    return game, stream, sender, clock


def test_local_step_returns_newest_queued_frame_as_uint8() -> None:
    game, stream, sender, _ = _local_game()
    stream.push(1, 2, 3)

    frame = game.step([1] + [0] * 12)

    assert frame.dtype == np.uint8
    assert int(frame[0, 0, 0]) == 3
    assert stream.queue == []
    assert sender.vectors == [[1] + [0] * 12]


def test_local_step_paces_at_control_rate_and_reports_lateness() -> None:
    game, stream, _, clock = _local_game()

    stream.push(1)
    game.step([0] * 13)
    clock.now += 0.02
    stream.push(2)
    game.step([0] * 13)
    assert clock.sleeps == [pytest.approx(0.05), pytest.approx(0.03)]
    assert game.last_late_ms == 0.0

    clock.now += 0.2
    stream.push(3)
    game.step([0] * 13)
    assert game.last_late_ms == pytest.approx(150.0)


def test_local_reset_releases_keys_without_touching_the_game() -> None:
    game, stream, sender, _ = _local_game()
    stream.push(7)

    frame = game.reset()

    assert int(frame[0, 0, 0]) == 7
    assert sender.releases == 1
    assert sender.vectors == []


def test_local_pause_only_releases_keys() -> None:
    game, _, sender, _ = _local_game()

    game.pause()

    assert sender.releases == 1
    assert sender.vectors == []


def test_local_raises_when_stream_has_no_frame() -> None:
    game, _, _, _ = _local_game()
    with pytest.raises(RuntimeError, match="capture stream"):
        game.step([0] * 13)
