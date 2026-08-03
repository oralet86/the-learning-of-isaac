"""Unit tests for `visual_extraction.labels` -- pure marshaling/masking logic.

Fixtures are hand-built dicts matching the target JSON contract documented
in `labels.py`'s module docstring, not real captured data (none exists
yet -- see `tloi_datagen/MOD_CHANGES_NEEDED.md`).
"""

from __future__ import annotations

from typing import Any

import pytest

from visual_extraction import labels


def _base_record(**overrides: Any) -> dict[str, Any]:  # noqa: ANN401
    record: dict[str, Any] = {
        "stage_bucket": 3,
        "player": {"frame_position": [0.25, 0.75]},
        "player2": None,
        "item_delta": {"action": 1},
        "active_item": {"held": False, "item_id": None, "charge": None},
        "currency": {"coins": 5, "bombs": 2, "keys": 1},
        "consumable": {
            "kind": 0,
            "card_id": None,
            "pill_color": None,
            "pill_known": None,
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
        "doors": [],
        "enemies": [],
    }
    record.update(overrides)
    return record


def test_stage_reads_bucket_straight_through() -> None:
    target, mask = labels.encode_stage(_base_record(stage_bucket=7))
    assert target.item() == 7
    assert mask.item() == 1.0


def test_item_delta_reads_action_straight_through() -> None:
    target, mask = labels.encode_item_delta(_base_record(item_delta={"action": 2}))
    assert target.item() == 2
    assert mask.item() == 1.0


def test_active_item_not_held_masks_id_and_charge() -> None:
    heads = labels.encode_active_item(
        _base_record(active_item={"held": False, "item_id": None, "charge": None})
    )
    assert heads["held"][0].item() == 0.0
    assert heads["id"][1].item() == 0.0
    assert heads["charge"][1].item() == 0.0


def test_active_item_held_unmasks_id_and_charge() -> None:
    heads = labels.encode_active_item(
        _base_record(active_item={"held": True, "item_id": 105, "charge": 0.5})
    )
    assert heads["held"][0].item() == 1.0
    assert heads["id"][0].item() == 105
    assert heads["id"][1].item() == 1.0
    assert heads["charge"][0].item() == pytest.approx(0.5)
    assert heads["charge"][1].item() == 1.0


def test_consumable_none_masks_card_and_pill_fields() -> None:
    heads = labels.encode_consumable(
        _base_record(
            consumable={
                "kind": 0,
                "card_id": None,
                "pill_color": None,
                "pill_known": None,
            }
        )
    )
    assert heads["card_id"][1].item() == 0.0
    assert heads["pill_color"][1].item() == 0.0
    assert heads["pill_known"][1].item() == 0.0


def test_consumable_card_unmasks_card_id_only() -> None:
    heads = labels.encode_consumable(
        _base_record(
            consumable={
                "kind": labels.KIND_CARD,
                "card_id": 12,
                "pill_color": None,
                "pill_known": None,
            }
        )
    )
    assert heads["card_id"][0].item() == 12
    assert heads["card_id"][1].item() == 1.0
    assert heads["pill_color"][1].item() == 0.0


def test_consumable_pill_without_known_field_stays_masked() -> None:
    """Simulates today's data, where the mod doesn't emit `pill_known` yet."""
    heads = labels.encode_consumable(
        _base_record(
            consumable={
                "kind": labels.KIND_PILL,
                "card_id": None,
                "pill_color": 3,
                "pill_known": None,
            }
        )
    )
    assert heads["pill_color"][0].item() == 3
    assert heads["pill_color"][1].item() == 1.0
    assert heads["pill_known"][1].item() == 0.0


def test_consumable_pill_known_unmasks_once_mod_reports_it() -> None:
    heads = labels.encode_consumable(
        _base_record(
            consumable={
                "kind": labels.KIND_PILL,
                "card_id": None,
                "pill_color": 3,
                "pill_known": True,
            }
        )
    )
    assert heads["pill_known"][0].item() == 1.0
    assert heads["pill_known"][1].item() == 1.0


def test_health_unknown_masks_all_counts() -> None:
    heads = labels.encode_health(_base_record(health={"unknown": True}))
    assert heads["unknown"][0].item() == 1.0
    assert heads["counts"][1].item() == 0.0
    assert heads["counts"][0].tolist() == [0.0] * labels.NUM_HEALTH_COUNTS


def test_health_known_unmasks_counts() -> None:
    record = _base_record()
    heads = labels.encode_health(record)
    assert heads["unknown"][0].item() == 0.0
    assert heads["counts"][1].item() == 1.0
    assert heads["counts"][0][0].item() == record["health"]["hearts"]


def test_doors_empty_list_leaves_all_slots_absent() -> None:
    heads = labels.encode_doors(_base_record(doors=[]))
    assert heads["presence"][0].tolist() == [0.0] * labels.NUM_DOOR_SLOTS
    assert heads["presence"][1].tolist() == [1.0] * labels.NUM_DOOR_SLOTS
    assert heads["state"][1].sum().item() == 0.0


def test_doors_packs_by_slot_index_and_masks_state_by_presence() -> None:
    heads = labels.encode_doors(
        _base_record(
            doors=[
                {"slot": 2, "is_open": True, "is_locked": False, "is_busted": False},
                {"slot": 5, "is_open": False, "is_locked": True, "is_busted": False},
            ]
        )
    )
    presence, presence_mask = heads["presence"]
    state, state_mask = heads["state"]

    assert presence[2].item() == 1.0
    assert presence[5].item() == 1.0
    assert presence.sum().item() == 2.0
    assert presence_mask.tolist() == [1.0] * labels.NUM_DOOR_SLOTS

    assert state[2].tolist() == [1.0, 0.0, 0.0]
    assert state[5].tolist() == [0.0, 1.0, 0.0]
    assert state_mask[2].item() == 1.0
    assert state_mask[0].item() == 0.0


def test_encode_record_produces_a_full_bundle() -> None:
    bundle = labels.encode_record(_base_record())
    assert isinstance(bundle, labels.LabelBundle)
    assert bundle.stage.item() == 3
    assert bundle.doors_presence.shape == (labels.NUM_DOOR_SLOTS,)
    assert bundle.doors_state.shape == (labels.NUM_DOOR_SLOTS, 3)
    assert bundle.detection_class.shape == (
        labels.DETECTION_GRID_ROWS,
        labels.DETECTION_GRID_COLS,
    )
    assert bundle.detection_offset.shape == (
        labels.DETECTION_GRID_ROWS,
        labels.DETECTION_GRID_COLS,
        2,
    )


def test_detections_places_player_as_player_class() -> None:
    # Grid is 12 cols x 7 rows; player's default position [0.25, 0.75] falls
    # in column floor(0.25*12)=3, row floor(0.75*7)=5.
    heads = labels.encode_detections(_base_record())
    detection_class, class_mask = heads["class"]
    _, offset_mask = heads["offset"]

    assert detection_class[5, 3].item() == labels.DETECTION_CLASS_PLAYER
    assert (detection_class != labels.DETECTION_CLASS_NONE).sum().item() == 1
    assert (
        class_mask.sum().item()
        == labels.DETECTION_GRID_ROWS * labels.DETECTION_GRID_COLS
    )
    assert offset_mask[5, 3].item() == 1.0
    assert offset_mask[0, 0].item() == 0.0


def test_detections_places_player2_as_player_class_too() -> None:
    # player2 at [0.9, 0.1] -> column floor(0.9*12)=10, row floor(0.1*7)=0,
    # distinct from the default player's cell (5, 3).
    heads = labels.encode_detections(
        _base_record(player2={"frame_position": [0.9, 0.1]})
    )
    detection_class, _ = heads["class"]

    assert detection_class[5, 3].item() == labels.DETECTION_CLASS_PLAYER
    assert detection_class[0, 10].item() == labels.DETECTION_CLASS_PLAYER
    assert (detection_class == labels.DETECTION_CLASS_PLAYER).sum().item() == 2


def test_detections_places_enemies_as_enemy_class() -> None:
    heads = labels.encode_detections(
        _base_record(enemies=[{"frame_position": [0.3, 0.2]}])
    )
    detection_class, _ = heads["class"]
    offset, offset_mask = heads["offset"]

    # x=0.3 -> column floor(0.3*12)=3, offset 0.3*12-3=0.6.
    # y=0.2 -> row floor(0.2*7)=1, offset 0.2*7-1=0.4.
    assert detection_class[1, 3].item() == labels.DETECTION_CLASS_ENEMY
    assert offset[1, 3].tolist() == pytest.approx([0.6, 0.4])
    assert offset_mask[1, 3].item() == 1.0
    # Default player still occupies its own separate cell.
    assert detection_class[5, 3].item() == labels.DETECTION_CLASS_PLAYER


def test_detections_at_the_frame_edge_clamps_into_the_last_cell() -> None:
    # x=1.0/y=1.0 would floor to column 12/row 7, one past the last valid
    # index (11/6).
    heads = labels.encode_detections(
        _base_record(enemies=[{"frame_position": [1.0, 1.0]}])
    )
    detection_class, _ = heads["class"]
    last_row = labels.DETECTION_GRID_ROWS - 1
    last_col = labels.DETECTION_GRID_COLS - 1
    assert detection_class[last_row, last_col].item() == labels.DETECTION_CLASS_ENEMY


def test_detections_player_wins_over_enemy_in_the_same_cell() -> None:
    # Both land in the default player's cell (5, 3): x=0.28 -> col 3,
    # y=0.78 -> row 5.
    heads = labels.encode_detections(
        _base_record(enemies=[{"frame_position": [0.28, 0.78]}])
    )
    detection_class, _ = heads["class"]
    assert detection_class[5, 3].item() == labels.DETECTION_CLASS_PLAYER
    assert (detection_class == labels.DETECTION_CLASS_ENEMY).sum().item() == 0


def test_detections_player2_wins_over_player_in_the_same_cell() -> None:
    # player2 placed at the exact same cell as the default player -- it's
    # processed last, so its offset should be what survives.
    heads = labels.encode_detections(
        _base_record(player2={"frame_position": [0.26, 0.76]})
    )
    detection_class, _ = heads["class"]
    offset, _ = heads["offset"]
    assert detection_class[5, 3].item() == labels.DETECTION_CLASS_PLAYER
    assert (detection_class == labels.DETECTION_CLASS_PLAYER).sum().item() == 1
    assert offset[5, 3].tolist() == pytest.approx(
        [0.26 * 12 - 3, 0.76 * 7 - 5], abs=1e-5
    )
