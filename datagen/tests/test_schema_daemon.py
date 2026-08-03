"""Unit tests for `datagen.daemon`'s raw-JSON -> `GroundTruthRecord` wiring.

Fixtures mirror the ground-truth contract the mod is expected to emit once
`tloi_datagen/MOD_CHANGES_NEEDED.md` items 1, 2, 4, 5, 6 are implemented --
same contract `visual_extraction/labels.py` is already written against.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any, cast

import numpy as np

from datagen.daemon import _process_one
from datagen.run_config import RunConfig

if TYPE_CHECKING:
    from pathlib import Path

    from live.stream_game import StreamGame


class _FakeStream:
    """Stands in for `live.stream_game.StreamGame` -- returns one blank RGB frame."""

    def get_frame(self, timeout: float = 5.0) -> np.ndarray:
        _ = timeout
        return np.zeros((4, 4, 3), dtype=np.uint8)


def _raw_ground_truth(**overrides: Any) -> dict[str, Any]:  # noqa: ANN401
    raw: dict[str, Any] = {
        "scenario_id": "session_123",
        "seed": 42,
        "stage_bucket": 2,
        "player": {"frame_position": [0.5, 0.5]},
        "item_delta": {"action": 1},
        "active_item": {"held": False, "item_id": None, "charge": None},
        "currency": {"coins": 5, "bombs": 2, "keys": 1},
        "consumable": {
            "kind": 2,
            "card_id": None,
            "pill_color": 3,
            "pill_known": True,
        },
        "stats": {
            "damage": 3.5,
            "fire_rate": 4.2,
            "speed": 1.0,
            "range": 6.5,
            "shot_speed": 1.0,
            "luck": 0.0,
        },
        "health": {
            "unknown": False,
            "hearts": 4,
            "max_hearts": 6,
            "soul_hearts": 0,
            "black_hearts": 0,
            "bone_hearts": 0,
            "rotten_hearts": 0,
            "eternal_hearts": 0,
            "golden_hearts": 0,
            "broken_hearts": 0,
        },
        "enemies": [{"frame_position": [0.3, 0.2]}],
        "doors": [],
    }
    raw.update(overrides)
    return raw


def _run(tmp_path: Path, raw: dict[str, Any], capture_index: int = 0) -> Any:  # noqa: ANN401
    config = RunConfig(
        exchange_dir=tmp_path / "exchange", output_dir=tmp_path / "output"
    )
    config.exchange_dir.mkdir(parents=True, exist_ok=True)
    pending_path = config.exchange_dir / f"{raw['scenario_id']}.json"
    pending_path.write_text(json.dumps(raw))
    stream = cast("StreamGame", _FakeStream())
    return _process_one(config, stream, pending_path, capture_index)


def test_capture_index_makes_image_id_unique_per_call(tmp_path: Path) -> None:
    record_a = _run(tmp_path, _raw_ground_truth(scenario_id="s1"), capture_index=0)
    record_b = _run(tmp_path, _raw_ground_truth(scenario_id="s1"), capture_index=1)

    assert record_a.image_id != record_b.image_id
    assert record_a.scenario_id == record_b.scenario_id == "s1"


def test_stage_bucket_and_frame_positions_pass_through(tmp_path: Path) -> None:
    record = _run(tmp_path, _raw_ground_truth())

    assert record.stage_bucket == 2
    assert record.player.frame_position == (0.5, 0.5)
    assert record.enemies[0].frame_position == (0.3, 0.2)


def test_player2_frame_position_wired_when_present(tmp_path: Path) -> None:
    raw = _raw_ground_truth(player2={"frame_position": [0.6, 0.4]})
    record = _run(tmp_path, raw)

    assert record.player2 is not None
    assert record.player2.frame_position == (0.6, 0.4)


def test_player2_absent_stays_none(tmp_path: Path) -> None:
    record = _run(tmp_path, _raw_ground_truth())

    assert record.player2 is None


def test_item_delta_action_is_int(tmp_path: Path) -> None:
    record = _run(tmp_path, _raw_ground_truth(item_delta={"action": 2}))

    assert record.item_delta.action == 2
    assert isinstance(record.item_delta.action, int)


def test_consumable_kind_is_int_with_pill_known(tmp_path: Path) -> None:
    record = _run(tmp_path, _raw_ground_truth())

    assert record.consumable.kind == 2
    assert record.consumable.pill_color == 3
    assert record.consumable.pill_known is True


def test_consumable_pill_known_absent_defaults_to_none(tmp_path: Path) -> None:
    raw = _raw_ground_truth(consumable={"kind": 2, "card_id": None, "pill_color": 3})
    record = _run(tmp_path, raw)

    assert record.consumable.pill_known is None


def test_consumable_none_kind(tmp_path: Path) -> None:
    raw = _raw_ground_truth(
        consumable={
            "kind": 0,
            "card_id": None,
            "pill_color": None,
            "pill_known": None,
        }
    )
    record = _run(tmp_path, raw)

    assert record.consumable.kind == 0
    assert record.consumable.pill_known is None
