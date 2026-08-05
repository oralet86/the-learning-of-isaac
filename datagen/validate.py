"""Validator for the ``tloi_datagen`` mod's ground-truth JSON, checked against
what ``visual_extraction/labels.py`` actually consumes -- not a hand-copied
mirror of it.

`validate_record` runs the real `visual_extraction.labels.encode_record`
against the JSON (wrapped to track exactly which keys it reads) and reports:

- Any exception `encode_record` raises -- means this record cannot be fed
  to the model as-is (missing/malformed required field).
- Every key present in the JSON that `encode_record` never read -- a field
  the mod emits that no training head consumes, whether that's a stale
  field left over from a contract change or something new that hasn't been
  wired into a head yet. Nothing here is a hardcoded "removed fields" list;
  it falls out of actually running the consumer's code.
- Range checks for values `encode_record` accepts but doesn't itself bound
  (e.g. it happily builds a tensor from `stage_bucket = 47`) -- these pull
  their bounds directly from `labels.py`'s own constants
  (`NUM_STAGE_BUCKETS`, `KIND_CARD`/`KIND_PILL`, etc.), not reinvented ones.

`scenario_id`/`seed` (and `image_id`/`timestamp` for daemon-produced label
files) are the one explicit exception: `labels.py` never reads them at all
(they're pipeline plumbing -- `dataset.py`'s train/val split key and
image/label filename, not model input) but are legitimately supposed to be
there, so they're allowlisted rather than flagged as unused.

Run with no arguments and it validates every label/image pair under
``datagen/output`` automatically: each ``labels/*.json`` is checked against
the contract above (auto-detecting the extra `image_id`/`timestamp` fields
the capture daemon adds, since anything under a ``labels/`` directory is
daemon output, not the mod's raw exchange file) *and* cross-checked against
``images/`` -- a label with no matching image, or an image with no
matching label, is reported as an error. Point it at a specific file or
directory to validate just that instead (e.g. the mod's raw exchange
files, which skip the pairing check since there's no `images/` counterpart
to check against).

Requires this project's virtualenv (imports `visual_extraction.labels`,
which pulls in `torch`), and must be run as a module from the repo root so
`visual_extraction` is importable (a plain `python datagen/validate.py`
won't put the repo root on `sys.path`)::

    python -m datagen.validate                          # datagen/output, paired
    python -m datagen.validate path/to/scenario.json     # one raw exchange file
    python -m datagen.validate datagen/runtime/          # a directory of raw files
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from visual_extraction import labels

DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent / "output"

# `labels.py` never reads these -- they're pipeline plumbing consumed
# elsewhere (`visual_extraction/dataset.py`'s train/val split key, and the
# capture daemon's own image/label filenames), not model input. Everything
# else `encode_record` doesn't read is genuinely unexpected/unused.
ALWAYS_ALLOWED_METADATA = frozenset({"scenario_id", "seed"})
DAEMON_ADDED_FIELDS = frozenset({"image_id", "timestamp"})


def _is_number(value: Any) -> bool:  # noqa: ANN401
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _is_int(value: Any) -> bool:  # noqa: ANN401
    return isinstance(value, int) and not isinstance(value, bool)


def _is_bool(value: Any) -> bool:  # noqa: ANN401
    return isinstance(value, bool)


class _Tracked(dict[str, Any]):
    """A dict that remembers every key read via `[key]` or `.get(key)`.

    Wrapping the record in this (recursively, through nested dicts/lists)
    lets `_find_unused_keys` tell exactly which JSON keys
    `labels.encode_record` actually consumed, instead of hand-maintaining a
    parallel list of "fields the model needs."
    """

    def __init__(self, data: dict[str, Any]) -> None:
        super().__init__({key: _wrap(value) for key, value in data.items()})
        self.accessed: set[str] = set()

    def __getitem__(self, key: str) -> Any:  # noqa: ANN401
        self.accessed.add(key)
        return super().__getitem__(key)

    def get(self, key: str, default: Any = None) -> Any:  # noqa: ANN401
        self.accessed.add(key)
        return super().get(key, default)


def _wrap(value: Any) -> Any:  # noqa: ANN401
    if isinstance(value, dict):
        return _Tracked(value)
    if isinstance(value, list):
        return [_wrap(item) for item in value]
    return value


def _find_unused_keys(
    node: Any,  # noqa: ANN401
    path: str,
    allowed: frozenset[str] = frozenset(),
) -> list[str]:
    """Recursively report keys `encode_record` never touched, skipping `allowed` at this level only.

    A key that's `null` and never read isn't flagged -- daemon-produced
    label files always carry every dataclass field, even ones a
    discriminator (e.g. `consumable.kind`) makes irrelevant this record,
    serialized as `null` rather than omitted (unlike the mod's raw
    exchange JSON, which omits nil keys entirely -- see
    ``tloi_datagen/query/ground_truth.lua``). Only a *non-null* value that
    nothing reads is a real problem worth surfacing.
    """
    if isinstance(node, _Tracked):
        errors = []
        for key, value in node.items():
            if key not in node.accessed:
                if key not in allowed and value is not None:
                    errors.append(
                        f"{path}.{key}: present with a non-null value but never read by "
                        "visual_extraction.labels.encode_record"
                    )
                continue
            errors += _find_unused_keys(value, f"{path}.{key}")
        return errors
    if isinstance(node, list):
        errors = []
        for i, item in enumerate(node):
            errors += _find_unused_keys(item, f"{path}[{i}]")
        return errors
    return []


def _frame_position_range_errors(value: Any, path: str) -> list[str]:  # noqa: ANN401
    # Shape (must be a 2-element list) is already covered by
    # `encode_record`'s own `x, y = frame_position` unpacking, which raises
    # on anything else -- this only supplements the numeric range check
    # `encode_record` doesn't itself perform.
    if not isinstance(value, list) or len(value) != 2:
        return []
    errors = []
    for i, axis in enumerate(("x", "y")):
        component = value[i]
        if _is_number(component) and not (0.0 <= component <= 1.0):
            errors.append(
                f"{path}[{i}] ({axis}): expected a value in [0, 1], got {component}"
            )
    return errors


def _validate_ranges(record: dict[str, Any]) -> list[str]:
    """Bound checks for values `encode_record` accepts but doesn't itself range-check.

    Sourced directly from `labels.py`'s own class-count constants (not
    reinvented magic numbers) -- see this module's docstring.
    """
    errors: list[str] = []

    def _bounded(value: Any, path: str, high: int) -> None:  # noqa: ANN401
        if _is_int(value) and not (0 <= value < high):
            errors.append(f"{path}: expected an int in [0, {high}), got {value!r}")

    stage_bucket = record.get("stage_bucket")
    if stage_bucket is not None:
        _bounded(stage_bucket, "stage_bucket", labels.NUM_STAGE_BUCKETS)

    item_delta = record.get("item_delta")
    if isinstance(item_delta, dict):
        _bounded(
            item_delta.get("action"), "item_delta.action", labels.NUM_ITEM_DELTA_ACTIONS
        )

    consumable = record.get("consumable")
    if isinstance(consumable, dict):
        kind = consumable.get("kind")
        _bounded(kind, "consumable.kind", labels.NUM_CONSUMABLE_KINDS)
        if kind == labels.KIND_CARD:
            _bounded(consumable.get("card_id"), "consumable.card_id", labels.NUM_CARDS)
        elif kind == labels.KIND_PILL:
            _bounded(
                consumable.get("pill_color"),
                "consumable.pill_color",
                labels.NUM_PILL_COLORS,
            )
        pill_known = consumable.get("pill_known")
        if pill_known is not None and not _is_bool(pill_known):
            errors.append(
                f"consumable.pill_known: expected a bool or null, got {pill_known!r}"
            )

    active_item = record.get("active_item")
    if isinstance(active_item, dict):
        held = active_item.get("held")
        if held is not None and not _is_bool(held):
            errors.append(f"active_item.held: expected a bool, got {held!r}")

    health = record.get("health")
    if isinstance(health, dict):
        unknown = health.get("unknown")
        if unknown is not None and not _is_bool(unknown):
            errors.append(f"health.unknown: expected a bool, got {unknown!r}")

    doors = record.get("doors")
    if isinstance(doors, list):
        for i, door in enumerate(doors):
            if isinstance(door, dict):
                _bounded(door.get("slot"), f"doors[{i}].slot", labels.NUM_DOOR_SLOTS)

    for who in ("player", "player2"):
        entity = record.get(who)
        if isinstance(entity, dict):
            errors += _frame_position_range_errors(
                entity.get("frame_position"), f"{who}.frame_position"
            )
    enemies = record.get("enemies")
    if isinstance(enemies, list):
        for i, enemy in enumerate(enemies):
            if isinstance(enemy, dict):
                errors += _frame_position_range_errors(
                    enemy.get("frame_position"), f"enemies[{i}].frame_position"
                )

    return errors


def validate_record(record: Any, daemon_labels: bool = False) -> list[str]:  # noqa: ANN401
    """Validate one ground-truth dict against what `visual_extraction.labels` actually consumes.

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

    wrapped = _Tracked(record)
    try:
        labels.encode_record(wrapped)
    except Exception as exc:
        # Deliberately broad: any failure means this record can't be fed to
        # the model, which is exactly what's being checked. Stop here
        # rather than also running the unused-key check below --
        # encode_record calls each field's encoder in sequence, so a
        # failure partway through means later fields were never touched at
        # all, which would show up as false-positive "unused" noise on top
        # of the real, actionable error above.
        return [
            f"<root>: visual_extraction.labels.encode_record failed -- "
            f"{type(exc).__name__}: {exc}"
        ]

    allowed = ALWAYS_ALLOWED_METADATA | (
        DAEMON_ADDED_FIELDS if daemon_labels else frozenset()
    )
    errors = _find_unused_keys(wrapped, "<root>", allowed)
    errors += _validate_ranges(record)

    if daemon_labels:
        errors += [
            f"<root>.{key}: missing required field"
            for key in DAEMON_ADDED_FIELDS
            if key not in record
        ]

    return errors


def _iter_json_files(path: Path, pattern: str) -> list[Path]:
    if path.is_file():
        return [path]
    if path.is_dir():
        return sorted(path.glob(pattern))
    raise FileNotFoundError(f"no such file or directory: {path}")


def _is_daemon_labels_path(file_path: Path) -> bool:
    """Whether `file_path` lives under a `labels/` directory (daemon output, not raw mod output).

    Checked by directory name rather than by peeking at the JSON's
    contents -- a malformed record that happens to be missing `image_id`
    shouldn't silently get validated against the wrong (more lenient)
    contract.
    """
    return file_path.parent.name == "labels"


def _validate_one_file(file_path: Path, *, daemon_labels: bool) -> list[str]:
    """Load and validate one JSON file. Returns error strings, `["invalid JSON: ..."]` on a parse failure."""
    try:
        record = json.loads(file_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        return [f"invalid JSON ({exc})"]
    return validate_record(record, daemon_labels=daemon_labels)


def _report(file_path: Path, errors: list[str]) -> int:
    """Print one file's result. Returns the number of errors (0 if valid)."""
    if errors:
        print(f"FAIL {file_path}")
        for error in errors:
            print(f"  - {error}")
    else:
        print(f"OK   {file_path}")
    return len(errors)


def _validate_output_dir(output_dir: Path) -> tuple[int, int]:
    """Validate every `labels/*.json` against the contract, paired against `images/`.

    Returns:
        `(files_checked, total_errors)`, counting each label file plus one
        extra "file" per orphaned image (an image with no matching label).
    """
    labels_dir = output_dir / "labels"
    images_dir = output_dir / "images"
    label_paths = sorted(labels_dir.glob("*.json")) if labels_dir.is_dir() else []
    image_stems = (
        {p.stem for p in images_dir.glob("*.png")} if images_dir.is_dir() else set()
    )

    files_checked = 0
    total_errors = 0
    paired_stems: set[str] = set()
    for label_path in label_paths:
        try:
            record = json.loads(label_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            files_checked += 1
            total_errors += _report(label_path, [f"invalid JSON ({exc})"])
            continue

        errors = validate_record(record, daemon_labels=True)
        image_id = record.get("image_id") if isinstance(record, dict) else None
        if image_id is not None:
            paired_stems.add(image_id)
            if image_id not in image_stems:
                errors.append(f"no matching image: images/{image_id}.png")
        files_checked += 1
        total_errors += _report(label_path, errors)

    for orphan_stem in sorted(image_stems - paired_stems):
        orphan_path = images_dir / f"{orphan_stem}.png"
        files_checked += 1
        total_errors += _report(
            orphan_path, [f"no matching label: labels/{orphan_stem}.json"]
        )

    return files_checked, total_errors


def main() -> int:
    """CLI entry point. Returns a process exit code (0 = all valid)."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "paths",
        nargs="*",
        type=Path,
        help=f"JSON file(s)/directory(ies) to validate. Defaults to {DEFAULT_OUTPUT_DIR}, "
        "validated as paired images/labels.",
    )
    parser.add_argument(
        "--glob", default="*.json", help="Glob pattern used when a path is a directory"
    )
    args = parser.parse_args()
    paths: list[Path] = args.paths or [DEFAULT_OUTPUT_DIR]

    files_checked = 0
    total_errors = 0
    for path in paths:
        if not path.exists():
            print(f"FAIL {path}: no such file or directory")
            total_errors += 1
            continue

        if path.is_dir() and (path / "labels").is_dir():
            checked, errors = _validate_output_dir(path)
            files_checked += checked
            total_errors += errors
            continue

        for file_path in _iter_json_files(path, args.glob):
            errors = _validate_one_file(
                file_path, daemon_labels=_is_daemon_labels_path(file_path)
            )
            files_checked += 1
            total_errors += _report(file_path, errors)

    if files_checked == 0:
        print("no JSON files found")
        return 1

    print()
    print(f"{files_checked} file(s) checked, {total_errors} error(s)")
    return 1 if total_errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
