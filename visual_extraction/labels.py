"""Ground-truth JSON -> per-head training tensors.

Deliberately thin: every value consumed here is already the exact label
the model should learn (already bucketed, already normalized, already an
integer class id) -- computed by the ``tloi_datagen`` mod, not derived
here. If a bug requires adding real logic to this file (a lookup table, a
normalization formula, an enum remap), the fix almost always belongs in
the mod instead; see ``tloi_datagen/MOD_CHANGES_NEEDED.md``.

This module is written against a **target** ground-truth JSON contract
that the mod does not produce yet (``tloi_datagen/MOD_CHANGES_NEEDED.md``
items 1, 2, 4, 5, 6 are still open). The expected shape, once those land::

    {
        "stage_bucket": int,  # 0-9, see MOD_CHANGES_NEEDED.md #1
        "player": {"frame_position": [float, float]},  # each in [0, 1]
        "player2": {"frame_position": [float, float]} | None,
        "item_delta": {"action": int},  # 0=none, 1=add, 2=remove
        "active_item": {"held": bool, "charge": float | None},
        "currency": {"coins": int, "bombs": int, "keys": int},
        "consumable": {
            "kind": int,  # 0=none, 1=card, 2=pill
            "card_id": int | None,
            "pill_color": int | None,
            "pill_known": bool | None,  # not tracked yet, see MOD_CHANGES_NEEDED.md #4
        },
        "stats": {
            "damage": float,
            "fire_rate": float,
            "speed": float,
            "range": float,
            "shot_speed": float,
            "luck": float,
        },
        "health": {
            "unknown": bool,
            "hearts": int | None,
            "max_hearts": int | None,
            "soul_hearts": int | None,
            "black_hearts": int | None,
            "bone_hearts": int | None,
            "rotten_hearts": int | None,
            "eternal_hearts": int | None,
            "golden_hearts": int | None,
            "broken_hearts": int | None,
        },
        "doors": [
            {"slot": int, "is_open": bool, "is_locked": bool, "is_busted": bool},
            ...,
        ],
        "enemies": [
            {"frame_position": [float, float]},  # see MOD_CHANGES_NEEDED.md #6
            ...,
        ],
    }
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

# -- Class-count constants, verified against IsaacDocs (https://wofsauge.github.io/IsaacDocs/rep/) --
NUM_STAGE_BUCKETS = 10
NUM_ITEM_DELTA_ACTIONS = 3  # 0=none, 1=add, 2=remove -- MOD_CHANGES_NEEDED.md #5
NUM_CONSUMABLE_KINDS = 3  # 0=none, 1=card, 2=pill -- MOD_CHANGES_NEEDED.md #5
NUM_CARDS = 98  # enums/Card.html -- NUM_CARDS=98 (tarot cards, runes, soul stones share this space)
NUM_PILL_COLORS = 16  # enums/PillColor.html -- 0-13 standard + PILL_GOLD=15
PILL_COLOR_MASK = 2047  # enums/PillColor.html -- mask off PILL_GIANT_FLAG (1 << 11) before classifying
NUM_DOOR_SLOTS = 8  # enums/DoorSlot.html -- NUM_DOOR_SLOTS=8
NUM_HEALTH_COUNTS = 9
NUM_STATS = 6
NUM_CURRENCY = 3
# Coarse spatial detection grid, see MOD_CHANGES_NEEDED.md #2 and #6. Not
# square -- the captured frame is widescreen (~16:9-ish; actual resolution
# comes from OBS at capture time, not fixed in code), so the grid tracks
# that aspect rather than distorting it into a square. Shared by player,
# player2, and enemies -- see `encode_detections`.
DETECTION_GRID_COLS = 12
DETECTION_GRID_ROWS = 7

DETECTION_CLASS_NONE = 0
DETECTION_CLASS_PLAYER = 1
DETECTION_CLASS_ENEMY = 2

KIND_CARD = 1
KIND_PILL = 2

_STATS_ORDER = ("damage", "fire_rate", "speed", "range", "shot_speed", "luck")
_HEALTH_COUNT_ORDER = (
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


@dataclass
class LabelBundle:
    """One captured frame's training targets, one (target, mask) pair per head.

    Every mask is a float tensor: ``1.0`` where the target is valid and
    should contribute to the loss, ``0.0`` where it should be ignored
    (e.g. `detection_offset` at grid cells with no player or enemy, or
    `health_counts` under Curse of the Unknown). Heads that are always
    valid still carry a mask of all-ones, so `losses.py`/`network.py` can
    treat every head identically rather than special-casing which ones
    need masking.
    """

    stage: torch.Tensor
    stage_mask: torch.Tensor

    item_delta_action: torch.Tensor
    item_delta_action_mask: torch.Tensor

    active_item_held: torch.Tensor
    active_item_held_mask: torch.Tensor
    active_item_charge: torch.Tensor
    active_item_charge_mask: torch.Tensor

    currency: torch.Tensor
    currency_mask: torch.Tensor

    consumable_kind: torch.Tensor
    consumable_kind_mask: torch.Tensor
    consumable_card_id: torch.Tensor
    consumable_card_id_mask: torch.Tensor
    consumable_pill_color: torch.Tensor
    consumable_pill_color_mask: torch.Tensor
    consumable_pill_known: torch.Tensor
    consumable_pill_known_mask: torch.Tensor

    stats: torch.Tensor
    stats_mask: torch.Tensor

    health_unknown: torch.Tensor
    health_unknown_mask: torch.Tensor
    health_counts: torch.Tensor
    health_counts_mask: torch.Tensor

    doors_presence: torch.Tensor
    doors_presence_mask: torch.Tensor
    doors_state: torch.Tensor
    doors_state_mask: torch.Tensor

    detection_class: torch.Tensor
    detection_class_mask: torch.Tensor
    detection_offset: torch.Tensor
    detection_offset_mask: torch.Tensor


def _ones() -> torch.Tensor:
    return torch.ones(())


def _zeros() -> torch.Tensor:
    return torch.zeros(())


def _flag(value: bool) -> torch.Tensor:
    return torch.tensor(1.0 if value else 0.0)


def encode_stage(record: dict[str, Any]) -> tuple[torch.Tensor, torch.Tensor]:
    """Reads `stage_bucket` straight into a class-index tensor."""
    return torch.tensor(record["stage_bucket"], dtype=torch.long), _ones()


def encode_item_delta(record: dict[str, Any]) -> tuple[torch.Tensor, torch.Tensor]:
    """Reads `item_delta.action` (already an int code) straight into a class-index tensor."""
    action = record["item_delta"]["action"]
    return torch.tensor(action, dtype=torch.long), _ones()


def encode_active_item(
    record: dict[str, Any],
) -> dict[str, tuple[torch.Tensor, torch.Tensor]]:
    """`held`/`charge`, with `charge` masked by `held`.

    The ground-truth contract no longer carries which specific collectible
    is active -- classifying that from a HUD icon isn't a head this model
    predicts; `held`/`charge` are.
    """
    active_item = record["active_item"]
    held = active_item["held"]
    held_mask = _flag(held)
    charge = active_item["charge"] if held else 0.0
    return {
        "held": (_flag(held), _ones()),
        "charge": (torch.tensor(charge, dtype=torch.float32), held_mask),
    }


def encode_currency(record: dict[str, Any]) -> tuple[torch.Tensor, torch.Tensor]:
    """`coins`/`bombs`/`keys` straight into a regression tensor."""
    currency = record["currency"]
    target = torch.tensor(
        [currency["coins"], currency["bombs"], currency["keys"]], dtype=torch.float32
    )
    return target, _ones()


def encode_consumable(
    record: dict[str, Any],
) -> dict[str, tuple[torch.Tensor, torch.Tensor]]:
    """`kind` straight into a class-index tensor; sub-fields masked by `kind`.

    `pill_known`'s mask is `0.0` whenever the mod hasn't reported it yet
    (`pill_known is None`), which today is always -- see
    ``tloi_datagen/MOD_CHANGES_NEEDED.md`` item 4. Once the mod starts
    populating it, this head starts training with zero code changes here.
    """
    consumable = record["consumable"]
    kind = consumable["kind"]
    is_card = _flag(kind == KIND_CARD)
    is_pill = _flag(kind == KIND_PILL)

    card_id = consumable["card_id"] if kind == KIND_CARD else 0
    pill_color = consumable["pill_color"] if kind == KIND_PILL else 0
    pill_known = consumable.get("pill_known")
    pill_known_mask = is_pill if pill_known is not None else _zeros()

    return {
        "kind": (torch.tensor(kind, dtype=torch.long), _ones()),
        "card_id": (torch.tensor(card_id, dtype=torch.long), is_card),
        "pill_color": (torch.tensor(pill_color, dtype=torch.long), is_pill),
        "pill_known": (_flag(bool(pill_known)), pill_known_mask),
    }


def encode_stats(record: dict[str, Any]) -> tuple[torch.Tensor, torch.Tensor]:
    """The 6 combat/movement stats straight into a regression tensor."""
    stats = record["stats"]
    target = torch.tensor([stats[name] for name in _STATS_ORDER], dtype=torch.float32)
    return target, _ones()


def encode_health(
    record: dict[str, Any],
) -> dict[str, tuple[torch.Tensor, torch.Tensor]]:
    """`unknown` plus 9 heart-type counts, each masked by `not unknown`.

    Mirrors ``PlayerHealth``'s existing semantics: under Curse of the
    Unknown every count field is already `None` in the ground truth, so
    the mask just reflects that, rather than deriving anything new.
    """
    health = record["health"]
    unknown = health["unknown"]
    counts_mask = _flag(not unknown)
    counts = [health[name] if not unknown else 0 for name in _HEALTH_COUNT_ORDER]
    return {
        "unknown": (_flag(unknown), _ones()),
        "counts": (torch.tensor(counts, dtype=torch.float32), counts_mask),
    }


def encode_doors(
    record: dict[str, Any],
) -> dict[str, tuple[torch.Tensor, torch.Tensor]]:
    """Scatters the (already frame-filtered) `doors` list into fixed `NUM_DOOR_SLOTS` arrays.

    `presence` is a valid label at every slot regardless of whether a door
    is there (absence is itself the label), so its mask is all-ones.
    `state` (open/locked/busted) is only meaningful where a door actually
    exists, so it's masked by `presence`.
    """
    presence = torch.zeros(NUM_DOOR_SLOTS)
    state = torch.zeros(NUM_DOOR_SLOTS, 3)
    for door in record["doors"]:
        slot = door["slot"]
        presence[slot] = 1.0
        state[slot, 0] = float(door["is_open"])
        state[slot, 1] = float(door["is_locked"])
        state[slot, 2] = float(door["is_busted"])
    return {
        "presence": (presence, torch.ones(NUM_DOOR_SLOTS)),
        "state": (state, presence),
    }


def encode_detections(
    record: dict[str, Any],
) -> dict[str, tuple[torch.Tensor, torch.Tensor]]:
    """Scatters player/player2/enemies into one shared coarse detection grid.

    A variable number of enemies doesn't fit the fixed-size heads every
    other field uses, so this is a small YOLO-style grid instead: each of
    `DETECTION_GRID_COLS x DETECTION_GRID_ROWS` cells holds at most one
    detection, classified 3-way (`DETECTION_CLASS_NONE`/`_PLAYER`/`_ENEMY`).
    `offset` is where within the cell the occupying entity sits (`[0, 1]`
    fraction of the cell's width/height) -- the cell index itself is
    coarse, `offset` recovers sub-cell precision. Player and player2 both
    map to `DETECTION_CLASS_PLAYER`; the grid doesn't distinguish which
    twin is which, only that a player is there.

    Only position is used for enemies, not `type`/`variant`/`subtype` (out
    of scope for now). Collisions -- two entities landing in the same
    cell -- are resolved by a fixed placement order (enemies, then player,
    then player2), so a later entity in that order silently overwrites an
    earlier one occupying the same cell. Known limitation of
    one-detection-per-cell, acceptable at this resolution for typical room
    enemy counts and how rarely two players occupy the exact same cell.
    """
    detection_class = torch.zeros(
        DETECTION_GRID_ROWS, DETECTION_GRID_COLS, dtype=torch.long
    )
    offset = torch.zeros(DETECTION_GRID_ROWS, DETECTION_GRID_COLS, 2)

    def place(frame_position: list[float], class_id: int) -> None:
        x, y = frame_position
        col = min(int(x * DETECTION_GRID_COLS), DETECTION_GRID_COLS - 1)
        row = min(int(y * DETECTION_GRID_ROWS), DETECTION_GRID_ROWS - 1)
        detection_class[row, col] = class_id
        offset[row, col, 0] = x * DETECTION_GRID_COLS - col
        offset[row, col, 1] = y * DETECTION_GRID_ROWS - row

    for enemy in record["enemies"]:
        place(enemy["frame_position"], DETECTION_CLASS_ENEMY)
    place(record["player"]["frame_position"], DETECTION_CLASS_PLAYER)
    player2 = record.get("player2")
    if player2 is not None:
        place(player2["frame_position"], DETECTION_CLASS_PLAYER)

    occupied = (detection_class != DETECTION_CLASS_NONE).float()
    return {
        "class": (
            detection_class,
            torch.ones(DETECTION_GRID_ROWS, DETECTION_GRID_COLS),
        ),
        "offset": (offset, occupied),
    }


@dataclass(frozen=True)
class HeadSpec:
    """Describes one output head: its target field name, size, and kind.

    Shared between `network.py` (sizing each head's output layer) and
    `losses.py` (picking each head's criterion) so the two never drift
    out of sync with each other or with `LabelBundle`'s field names.

    Attributes:
        name: Matches a `LabelBundle` field name exactly (e.g. `"stage"`,
            `"doors_state"`).
        out_dim: Number of output units. For `"classification"` heads,
            the number of classes; for regression/binary heads, the
            dimensionality of the target vector.
        kind: `"classification"` (softmax/cross-entropy over `out_dim`
            classes), `"regression"` (unbounded, `out_dim`-vector),
            `"bounded_regression"` (sigmoid-activated, target already in
            `[0, 1]`), or `"binary"` (single sigmoid logit per unit,
            `out_dim` independent binary targets).
        pred_shape: The shape (excluding batch dim) `network.py` reshapes
            this head's flat `out_dim` output into before returning it --
            e.g. `(NUM_DOOR_SLOTS, 3)` for `doors_state`. Always a concrete
            tuple, never `None` -- built via `_head()` below, which
            defaults it to the trivial flat `(out_dim,)` shape for heads
            that don't need reshaping. Keeping this field mandatory (not
            optional) means `network.py`'s forward loop applies it
            unconditionally, with no per-head `is None` branching to worry
            about under `torch.compile`.
    """

    name: str
    out_dim: int
    kind: str
    pred_shape: tuple[int, ...]


def _head(
    name: str, out_dim: int, kind: str, pred_shape: tuple[int, ...] | None = None
) -> HeadSpec:
    """Builds a `HeadSpec`, defaulting `pred_shape` to the flat `(out_dim,)` shape.

    Runs once at import time while building `HEAD_SPECS` -- this is where
    the "default to flat shape" choice lives, not inside any per-forward-call
    code path, so it has no bearing on `torch.compile`/graph-capture
    behavior either way.
    """
    return HeadSpec(name, out_dim, kind, pred_shape or (out_dim,))


HEAD_SPECS: tuple[HeadSpec, ...] = (
    _head("stage", NUM_STAGE_BUCKETS, "classification"),
    _head("item_delta_action", NUM_ITEM_DELTA_ACTIONS, "classification"),
    _head("active_item_held", 1, "binary"),
    _head("active_item_charge", 1, "bounded_regression"),
    _head("currency", NUM_CURRENCY, "regression"),
    _head("consumable_kind", NUM_CONSUMABLE_KINDS, "classification"),
    _head("consumable_card_id", NUM_CARDS, "classification"),
    _head("consumable_pill_color", NUM_PILL_COLORS, "classification"),
    _head("consumable_pill_known", 1, "binary"),
    _head("stats", NUM_STATS, "regression"),
    _head("health_unknown", 1, "binary"),
    _head("health_counts", NUM_HEALTH_COUNTS, "regression"),
    _head("doors_presence", NUM_DOOR_SLOTS, "binary"),
    _head("doors_state", NUM_DOOR_SLOTS * 3, "binary", pred_shape=(NUM_DOOR_SLOTS, 3)),
    # detection_class is a *spatial* classification head (3-way per grid
    # cell, not once per sample) -- reshaped channel-first so
    # `F.cross_entropy`'s K-dimensional form (channel dim second) applies
    # unchanged: (N, 3, ROWS, COLS) against an (N, ROWS, COLS) target.
    _head(
        "detection_class",
        3 * DETECTION_GRID_ROWS * DETECTION_GRID_COLS,
        "classification",
        pred_shape=(3, DETECTION_GRID_ROWS, DETECTION_GRID_COLS),
    ),
    _head(
        "detection_offset",
        DETECTION_GRID_ROWS * DETECTION_GRID_COLS * 2,
        "bounded_regression",
        pred_shape=(DETECTION_GRID_ROWS, DETECTION_GRID_COLS, 2),
    ),
)


def encode_record(record: dict[str, Any]) -> LabelBundle:
    """Encodes one ground-truth JSON dict into a full `LabelBundle`."""
    stage_target, stage_mask = encode_stage(record)
    action_target, action_mask = encode_item_delta(record)
    active_item = encode_active_item(record)
    currency_target, currency_mask = encode_currency(record)
    consumable = encode_consumable(record)
    stats_target, stats_mask = encode_stats(record)
    health = encode_health(record)
    doors = encode_doors(record)
    detections = encode_detections(record)

    return LabelBundle(
        stage=stage_target,
        stage_mask=stage_mask,
        item_delta_action=action_target,
        item_delta_action_mask=action_mask,
        active_item_held=active_item["held"][0],
        active_item_held_mask=active_item["held"][1],
        active_item_charge=active_item["charge"][0],
        active_item_charge_mask=active_item["charge"][1],
        currency=currency_target,
        currency_mask=currency_mask,
        consumable_kind=consumable["kind"][0],
        consumable_kind_mask=consumable["kind"][1],
        consumable_card_id=consumable["card_id"][0],
        consumable_card_id_mask=consumable["card_id"][1],
        consumable_pill_color=consumable["pill_color"][0],
        consumable_pill_color_mask=consumable["pill_color"][1],
        consumable_pill_known=consumable["pill_known"][0],
        consumable_pill_known_mask=consumable["pill_known"][1],
        stats=stats_target,
        stats_mask=stats_mask,
        health_unknown=health["unknown"][0],
        health_unknown_mask=health["unknown"][1],
        health_counts=health["counts"][0],
        health_counts_mask=health["counts"][1],
        doors_presence=doors["presence"][0],
        doors_presence_mask=doors["presence"][1],
        doors_state=doors["state"][0],
        doors_state_mask=doors["state"][1],
        detection_class=detections["class"][0],
        detection_class_mask=detections["class"][1],
        detection_offset=detections["offset"][0],
        detection_offset_mask=detections["offset"][1],
    )
