"""Standalone validator for the ``tloi_datagen`` mod's ground-truth JSON contract.

Checks that a captured ground-truth file (or a directory of them) matches
exactly what ``tloi_datagen/query/ground_truth.lua``'s ``GroundTruthQuery.CollectState``
is expected to produce -- required keys present, correct types, values in
range, and (this is the part a schema library wouldn't catch) *no* fields
that were deliberately removed because no training head reads them (see
``visual_extraction/labels.py``). Point it at the mod's raw exchange files
or at already-processed ``datagen/output/labels/*.json`` files (pass
``--daemon-labels`` for the latter, since the daemon adds `image_id`/
`timestamp`).

Stdlib only, no project imports -- runnable from a bare checkout with no
virtualenv, exactly when you'd most want to sanity-check the mod's raw
output before wiring up the rest of the pipeline::

    python datagen/validate_ground_truth.py path/to/scenario.json
    python datagen/validate_ground_truth.py datagen/runtime/ --glob "*.json"
    python datagen/validate_ground_truth.py datagen/output/labels/ --daemon-labels
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

# -- Fields that were deliberately removed from the contract (see prior
# cleanup pass) -- called out by name if still present, since that's almost
# always a stale mod build rather than a real new field. --
REMOVED_TOP_LEVEL_FIELDS = frozenset(
    {
        "stage",
        "room_shape",
        "room_type",
        "items_held",
        "camera_viewport_position",
        "capture_mode",
    }
)
REMOVED_PLAYER_FIELDS = frozenset({"character", "position"})
REMOVED_ENEMY_FIELDS = frozenset({"type", "variant", "subtype", "position"})
REMOVED_ITEM_DELTA_FIELDS = frozenset({"item_id"})

REQUIRED_TOP_LEVEL_FIELDS = frozenset(
    {
        "scenario_id",
        "seed",
        "stage_bucket",
        "player",
        "item_delta",
        "active_item",
        "currency",
        "consumable",
        "stats",
        "health",
        "enemies",
        "doors",
    }
)
OPTIONAL_TOP_LEVEL_FIELDS = frozenset({"player2"})
DAEMON_ADDED_FIELDS = frozenset({"image_id", "timestamp"})

_HEALTH_COUNT_FIELDS = (
    "hearts",
    "max_hearts",
    "soul_hearts",
    "black_hearts",
    "bone_hearts",
    "rotten_hearts",
    "eternal_hearts",
    "golden_hearts",
    "broken_hearts",
)
_STATS_FIELDS = ("damage", "fire_rate", "speed", "range", "shot_speed", "luck")


def _is_number(value: Any) -> bool:  # noqa: ANN401
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _is_int(value: Any) -> bool:  # noqa: ANN401
    return isinstance(value, int) and not isinstance(value, bool)


def _is_bool(value: Any) -> bool:  # noqa: ANN401
    return isinstance(value, bool)


def _check_type(value: Any, predicate: Any, path: str, expected: str) -> list[str]:  # noqa: ANN401
    if not predicate(value):
        return [f"{path}: expected {expected}, got {value!r} ({type(value).__name__})"]
    return []


def _check_extra_keys(
    obj: dict[str, Any], allowed: frozenset[str], removed: frozenset[str], path: str
) -> list[str]:
    errors = []
    for key in obj:
        if key in allowed:
            continue
        if key in removed:
            errors.append(
                f"{path}.{key}: field was deliberately removed from the contract "
                "(no training head reads it) -- stale mod build?"
            )
        else:
            errors.append(f"{path}.{key}: unexpected field, not part of the contract")
    return errors


def _check_missing_keys(
    obj: dict[str, Any], required: frozenset[str], path: str
) -> list[str]:
    return [f"{path}.{key}: missing required field" for key in required if key not in obj]


def validate_frame_position(value: Any, path: str) -> list[str]:  # noqa: ANN401
    if not isinstance(value, list) or len(value) != 2:
        return [f"{path}: expected a [x, y] pair, got {value!r}"]
    errors = []
    for i, axis in enumerate(("x", "y")):
        component = value[i]
        if not _is_number(component):
            errors.append(f"{path}[{i}] ({axis}): expected a number, got {component!r}")
        elif not (0.0 <= component <= 1.0):
            errors.append(f"{path}[{i}] ({axis}): expected a value in [0, 1], got {component}")
    return errors


def _validate_player(obj: Any, path: str) -> list[str]:  # noqa: ANN401
    if not isinstance(obj, dict):
        return [f"{path}: expected an object, got {obj!r}"]
    errors = _check_missing_keys(obj, frozenset({"frame_position"}), path)
    errors += _check_extra_keys(obj, frozenset({"frame_position"}), REMOVED_PLAYER_FIELDS, path)
    if "frame_position" in obj:
        errors += validate_frame_position(obj["frame_position"], f"{path}.frame_position")
    return errors


def _validate_item_delta(obj: Any, path: str) -> list[str]:  # noqa: ANN401
    if not isinstance(obj, dict):
        return [f"{path}: expected an object, got {obj!r}"]
    errors = _check_missing_keys(obj, frozenset({"action"}), path)
    errors += _check_extra_keys(obj, frozenset({"action"}), REMOVED_ITEM_DELTA_FIELDS, path)
    action = obj.get("action")
    if "action" in obj and (not _is_int(action) or action not in (0, 1, 2)):
        errors.append(f"{path}.action: expected 0 (none), 1 (add), or 2 (remove), got {action!r}")
    return errors


def _validate_active_item(obj: Any, path: str) -> list[str]:  # noqa: ANN401
    if not isinstance(obj, dict):
        return [f"{path}: expected an object, got {obj!r}"]
    allowed = frozenset({"held", "item_id", "charge"})
    errors = _check_missing_keys(obj, allowed, path)
    errors += _check_extra_keys(obj, allowed, frozenset(), path)
    if "held" not in obj:
        return errors
    held = obj["held"]
    errors += _check_type(held, _is_bool, f"{path}.held", "a bool")
    if held is True:
        if obj.get("item_id") is None or not _is_int(obj["item_id"]):
            errors.append(f"{path}.item_id: expected an int when held is true, got {obj.get('item_id')!r}")
        charge = obj.get("charge")
        if charge is None or not _is_number(charge) or not (0.0 <= charge <= 1.0):
            errors.append(f"{path}.charge: expected a number in [0, 1] when held is true, got {charge!r}")
    elif held is False:
        if obj.get("item_id") is not None:
            errors.append(f"{path}.item_id: expected null when held is false, got {obj['item_id']!r}")
        if obj.get("charge") is not None:
            errors.append(f"{path}.charge: expected null when held is false, got {obj['charge']!r}")
    return errors


def _validate_currency(obj: Any, path: str) -> list[str]:  # noqa: ANN401
    if not isinstance(obj, dict):
        return [f"{path}: expected an object, got {obj!r}"]
    allowed = frozenset({"coins", "bombs", "keys"})
    errors = _check_missing_keys(obj, allowed, path)
    errors += _check_extra_keys(obj, allowed, frozenset(), path)
    for field in allowed & obj.keys():
        value = obj[field]
        if not _is_int(value) or value < 0:
            errors.append(f"{path}.{field}: expected a non-negative int, got {value!r}")
    return errors


def _validate_consumable(obj: Any, path: str) -> list[str]:  # noqa: ANN401
    if not isinstance(obj, dict):
        return [f"{path}: expected an object, got {obj!r}"]
    allowed = frozenset({"kind", "card_id", "pill_color", "pill_known"})
    errors = _check_missing_keys(obj, allowed, path)
    errors += _check_extra_keys(obj, allowed, frozenset(), path)
    kind = obj.get("kind")
    if "kind" in obj and (not _is_int(kind) or kind not in (0, 1, 2)):
        errors.append(f"{path}.kind: expected 0 (none), 1 (card), or 2 (pill), got {kind!r}")

    card_id, pill_color, pill_known = obj.get("card_id"), obj.get("pill_color"), obj.get("pill_known")
    if kind == 1:
        if card_id is None or not _is_int(card_id):
            errors.append(f"{path}.card_id: expected an int when kind is 1 (card), got {card_id!r}")
        if pill_color is not None:
            errors.append(f"{path}.pill_color: expected null when kind is 1 (card), got {pill_color!r}")
    elif kind == 2:
        if pill_color is None or not _is_int(pill_color):
            errors.append(f"{path}.pill_color: expected an int when kind is 2 (pill), got {pill_color!r}")
        if card_id is not None:
            errors.append(f"{path}.card_id: expected null when kind is 2 (pill), got {card_id!r}")
        if pill_known is not None and not _is_bool(pill_known):
            errors.append(f"{path}.pill_known: expected a bool or null, got {pill_known!r}")
    elif kind == 0:
        if card_id is not None:
            errors.append(f"{path}.card_id: expected null when kind is 0 (none), got {card_id!r}")
        if pill_color is not None:
            errors.append(f"{path}.pill_color: expected null when kind is 0 (none), got {pill_color!r}")
    return errors


def _validate_stats(obj: Any, path: str) -> list[str]:  # noqa: ANN401
    if not isinstance(obj, dict):
        return [f"{path}: expected an object, got {obj!r}"]
    allowed = frozenset(_STATS_FIELDS)
    errors = _check_missing_keys(obj, allowed, path)
    errors += _check_extra_keys(obj, allowed, frozenset(), path)
    for field in allowed & obj.keys():
        errors += _check_type(obj[field], _is_number, f"{path}.{field}", "a number")
    return errors


def _validate_health(obj: Any, path: str) -> list[str]:  # noqa: ANN401
    if not isinstance(obj, dict):
        return [f"{path}: expected an object, got {obj!r}"]
    allowed = frozenset({"unknown", *_HEALTH_COUNT_FIELDS})
    errors = _check_missing_keys(obj, frozenset({"unknown"}), path)
    errors += _check_extra_keys(obj, allowed, frozenset(), path)
    unknown = obj.get("unknown")
    errors += _check_type(unknown, _is_bool, f"{path}.unknown", "a bool")
    if unknown is True:
        for field in _HEALTH_COUNT_FIELDS:
            if obj.get(field) is not None:
                errors.append(f"{path}.{field}: expected null under Curse of the Unknown, got {obj.get(field)!r}")
    elif unknown is False:
        errors += _check_missing_keys(obj, frozenset(_HEALTH_COUNT_FIELDS), path)
        for field in set(_HEALTH_COUNT_FIELDS) & obj.keys():
            value = obj[field]
            if not _is_int(value) or value < 0:
                errors.append(f"{path}.{field}: expected a non-negative int, got {value!r}")
    return errors


def _validate_enemy(obj: Any, path: str) -> list[str]:  # noqa: ANN401
    if not isinstance(obj, dict):
        return [f"{path}: expected an object, got {obj!r}"]
    errors = _check_missing_keys(obj, frozenset({"frame_position"}), path)
    errors += _check_extra_keys(obj, frozenset({"frame_position"}), REMOVED_ENEMY_FIELDS, path)
    if "frame_position" in obj:
        errors += validate_frame_position(obj["frame_position"], f"{path}.frame_position")
    return errors


def _validate_door(obj: Any, path: str) -> list[str]:  # noqa: ANN401
    if not isinstance(obj, dict):
        return [f"{path}: expected an object, got {obj!r}"]
    allowed = frozenset({"slot", "is_open", "is_locked", "is_busted"})
    errors = _check_missing_keys(obj, allowed, path)
    errors += _check_extra_keys(obj, allowed, frozenset(), path)
    slot = obj.get("slot")
    if "slot" in obj and (
        isinstance(slot, bool) or not isinstance(slot, int) or not (0 <= slot <= 7)
    ):
        errors.append(f"{path}.slot: expected an int in [0, 7], got {slot!r}")
    for field in ("is_open", "is_locked", "is_busted"):
        if field in obj:
            errors += _check_type(obj[field], _is_bool, f"{path}.{field}", "a bool")
    return errors


def validate_record(record: Any, daemon_labels: bool = False) -> list[str]:  # noqa: ANN401
    """Validate one ground-truth dict against the mod's current output contract.

    Args:
        record: A parsed ground-truth JSON value -- typed `Any` rather than
            `dict` since it comes straight from `json.loads` on untrusted
            mod output and may not actually be an object at all.
        daemon_labels: Whether `record` is a post-daemon label file (which
            legitimately adds `image_id`/`timestamp`) rather than the mod's
            raw exchange-file output.

    Returns:
        A list of human-readable error strings; empty if `record` is valid.
    """
    if not isinstance(record, dict):
        return [f"<root>: expected a JSON object, got {type(record).__name__}"]

    allowed = REQUIRED_TOP_LEVEL_FIELDS | OPTIONAL_TOP_LEVEL_FIELDS
    if daemon_labels:
        allowed = allowed | DAEMON_ADDED_FIELDS

    errors = _check_missing_keys(record, REQUIRED_TOP_LEVEL_FIELDS, "<root>")
    errors += _check_extra_keys(record, allowed, REMOVED_TOP_LEVEL_FIELDS, "<root>")

    if "scenario_id" in record:
        errors += _check_type(record["scenario_id"], lambda v: isinstance(v, str), "scenario_id", "a string")
    if "seed" in record:
        errors += _check_type(record["seed"], _is_int, "seed", "an int")
    if "stage_bucket" in record:
        bucket = record["stage_bucket"]
        if not _is_int(bucket) or not (0 <= bucket <= 9):
            errors.append(f"stage_bucket: expected an int in [0, 9], got {bucket!r}")

    if "player" in record:
        errors += _validate_player(record["player"], "player")
    if "player2" in record:
        errors += _validate_player(record["player2"], "player2")
    if "item_delta" in record:
        errors += _validate_item_delta(record["item_delta"], "item_delta")
    if "active_item" in record:
        errors += _validate_active_item(record["active_item"], "active_item")
    if "currency" in record:
        errors += _validate_currency(record["currency"], "currency")
    if "consumable" in record:
        errors += _validate_consumable(record["consumable"], "consumable")
    if "stats" in record:
        errors += _validate_stats(record["stats"], "stats")
    if "health" in record:
        errors += _validate_health(record["health"], "health")

    if "enemies" in record:
        enemies = record["enemies"]
        if not isinstance(enemies, list):
            errors.append(f"enemies: expected a list, got {enemies!r}")
        else:
            for i, enemy in enumerate(enemies):
                errors += _validate_enemy(enemy, f"enemies[{i}]")

    if "doors" in record:
        doors = record["doors"]
        if not isinstance(doors, list):
            errors.append(f"doors: expected a list, got {doors!r}")
        else:
            for i, door in enumerate(doors):
                errors += _validate_door(door, f"doors[{i}]")

    if daemon_labels:
        errors += _check_missing_keys(record, DAEMON_ADDED_FIELDS, "<root>")

    return errors


def _iter_json_files(path: Path, pattern: str) -> list[Path]:
    if path.is_file():
        return [path]
    if path.is_dir():
        return sorted(path.glob(pattern))
    raise FileNotFoundError(f"no such file or directory: {path}")


def main() -> int:
    """CLI entry point. Returns a process exit code (0 = all valid)."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paths", nargs="+", type=Path, help="JSON file(s) or directory(ies) to validate")
    parser.add_argument("--glob", default="*.json", help="Glob pattern used when a path is a directory")
    parser.add_argument(
        "--daemon-labels",
        action="store_true",
        help="Validate post-daemon datagen/output/labels/*.json files instead of the mod's raw exchange output",
    )
    args = parser.parse_args()

    files: list[Path] = []
    for path in args.paths:
        files.extend(_iter_json_files(path, args.glob))

    if not files:
        print("no JSON files found")
        return 1

    total_errors = 0
    for file_path in files:
        try:
            record = json.loads(file_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            print(f"FAIL {file_path}: invalid JSON ({exc})")
            total_errors += 1
            continue

        errors = validate_record(record, daemon_labels=args.daemon_labels)
        if errors:
            print(f"FAIL {file_path}")
            for error in errors:
                print(f"  - {error}")
            total_errors += len(errors)
        else:
            print(f"OK   {file_path}")

    print()
    print(f"{len(files)} file(s) checked, {total_errors} error(s)")
    return 1 if total_errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
