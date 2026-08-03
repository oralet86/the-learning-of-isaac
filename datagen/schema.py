"""Ground-truth record dataclasses for a single captured frame."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class PlayerStateM1:
    """Minimal player state read back from the mod.

    Attributes:
        character: Numeric ``PlayerType`` id, as reported by the mod's
            ``EntityPlayer:GetPlayerType()`` call.
        position: World-space ``(x, y)`` position.
    """

    character: int
    position: tuple[float, float]


@dataclass
class CurrencyState:
    """Consumables held, as reported by the mod's ``EntityPlayer`` getters.

    Attributes:
        coins: ``EntityPlayer:GetNumCoins()``.
        bombs: ``EntityPlayer:GetNumBombs()``.
        keys: ``EntityPlayer:GetNumKeys()``.
    """

    coins: int
    bombs: int
    keys: int


@dataclass
class Consumable:
    """The player's pocket item slot (card, rune, or pill), read live each room visit.

    A card and a pill can never both be held at once -- they share a single
    pocket item slot, same as real gameplay. Cards, playing cards, runes,
    and soul stones all share one id space in the game's ``Card`` enum.

    Attributes:
        kind: ``"card"``, ``"pill"``, or ``"none"``.
        card_id: The ``Card`` enum id, or ``None`` unless `kind` is ``"card"``.
        pill_color: The ``PillColor`` enum id, or ``None`` unless `kind` is
            ``"pill"``.
    """

    kind: str
    card_id: int | None
    pill_color: int | None


@dataclass
class PlayerStats:
    """The 6 stats the in-game stat UI shows, each rounded to 2 decimal places.

    Attributes:
        damage: ``EntityPlayer.Damage``.
        fire_rate: The in-game "Tears" stat (tears per second), derived from
            ``EntityPlayer.MaxFireDelay`` as ``30 / (fire_delay + 1)`` -- the
            raw ``MaxFireDelay`` value itself isn't kept, only this
            human-facing form.
        speed: ``EntityPlayer.MoveSpeed``.
        range: ``EntityPlayer.TearRange``.
        shot_speed: ``EntityPlayer.ShotSpeed``.
        luck: ``EntityPlayer.Luck``.
    """

    damage: float
    fire_rate: float
    speed: float
    range: float
    shot_speed: float
    luck: float


@dataclass
class PlayerHealth:
    """Precise health breakdown, as reported by the mod's ``EntityPlayer`` getters.

    Units vary by heart type (matching the underlying API): `hearts`,
    `max_hearts`, `soul_hearts`, `black_hearts`, `rotten_hearts`, and
    `eternal_hearts` are in half-heart units; `bone_hearts`, `golden_hearts`,
    and `broken_hearts` are in whole-heart units.

    Attributes:
        unknown: ``True`` under Curse of the Unknown, which replaces the
            entire heart HUD with a single "?" icon -- the real counts
            aren't visually recoverable from a captured frame under this
            curse, so every count field below is ``None`` when this is
            ``True``, deliberately not the omniscient values.
        hearts: Red hearts (``GetHearts``).
        max_hearts: Heart containers (``GetMaxHearts``).
        soul_hearts: Soul hearts, including black hearts (``GetSoulHearts``).
        black_hearts: Black hearts -- the mod pop-counts these from
            ``GetBlackHearts``'s bit mask return value into an actual count.
        bone_hearts: ``GetBoneHearts``.
        rotten_hearts: ``GetRottenHearts``.
        eternal_hearts: ``GetEternalHearts``.
        golden_hearts: ``GetGoldenHearts``.
        broken_hearts: ``GetBrokenHearts``.
    """

    unknown: bool
    hearts: int | None
    max_hearts: int | None
    soul_hearts: int | None
    black_hearts: int | None
    bone_hearts: int | None
    rotten_hearts: int | None
    eternal_hearts: int | None
    golden_hearts: int | None
    broken_hearts: int | None


@dataclass
class ItemDelta:
    """The item pickup/drop action actually taken for a single room visit.

    Logged explicitly as ground truth (not left to be inferred later by
    diffing ``items_held`` across records), for training a pickup/drop
    detector as an RL reward signal.

    Attributes:
        action: ``"add"``, ``"remove"``, or ``"none"``.
        item_id: The ``CollectibleType`` id involved, or ``None`` when
            `action` is ``"none"``.
    """

    action: str
    item_id: int | None


@dataclass
class ActiveItem:
    """The active item held (if any) for a single room visit, re-rolled every visit.

    Attributes:
        held: Whether an active item was rolled this visit. Kept as an
            explicit field (rather than inferring "not held" from `item_id`
            being ``None``) because the mod's vendored JSON encoder
            serializes a Lua table with every field nil as ``[]`` rather
            than ``{}`` -- without a field that's always present, "no
            active item" would come through as an empty list, not an
            object.
        item_id: The ``CollectibleType`` id held, or ``None`` if `held` is
            ``False``.
        charge: Current charge normalized to ``[0, 1]`` (current charge /
            that item's own ``MaxCharges``, with items that have no charge
            bar at all treated as fully charged), or ``None`` if `held` is
            ``False``. The raw charge/max-charge pair isn't logged --
            downstream consumers should treat every active item as
            effectively "1 charge bar" regardless of its real max charge.
    """

    held: bool
    item_id: int | None
    charge: float | None


@dataclass
class Door:
    """A single door present in the room.

    Attributes:
        slot: Numeric ``DoorSlot`` enum value (which side/position).
        is_open: ``GridEntityDoor:IsOpen()``.
        is_locked: ``GridEntityDoor:IsLocked()``.
        is_busted: ``GridEntityDoor:IsBusted()``.
    """

    slot: int
    is_open: bool
    is_locked: bool
    is_busted: bool


@dataclass
class EnemyState:
    """A single enemy's identity and position.

    Attributes:
        type: ``Entity.Type`` (``EntityType`` enum value).
        variant: ``Entity.Variant``.
        subtype: ``Entity.SubType``.
        position: World-space ``(x, y)`` position.
    """

    type: int
    variant: int
    subtype: int
    position: tuple[float, float]


@dataclass
class GroundTruthRecord:
    """Ground truth for a single captured frame.

    Attributes:
        image_id: Filename stem shared by the image and this record.
        scenario_id: Identifier assigned by the mod, echoed in its
            exchange-file names (see ``datagen.exchange``).
        seed: RNG seed used to build this scenario.
        stage: Stage identifier string passed to the mod.
        room_shape: Numeric ``RoomShape`` enum value (``Room:GetRoomShape()``).
        room_type: Numeric ``RoomType`` enum value (``Room:GetType()``).
        camera_viewport_position: Named viewport position within the room
            visit (e.g. ``"center"``, ``"corner_tl"``, ``"edge_top"``,
            ``"end_a"``) -- how many and which names appear depends on the
            room's shape (see ``tloi_datagen/scenario/viewports.lua``).
        player: Minimal player state.
        player2: The second twin's state (Jacob & Esau / Tainted
            Forgotten), or ``None`` for every other character.
        items_held: Player 1's total collectible count
            (``EntityPlayer:GetCollectibleCount()``). Duo characters'
            player2 gets its own independent passive-item baseline, but it
            isn't tracked here -- only player1's.
        item_delta: The item pickup/drop action taken for this room visit.
        active_item: The active item held (if any) for this room visit.
            player2's active item is not tracked or logged at all.
        currency: Coins/bombs/keys held. Shared team-wide even for duo
            characters -- not duplicated per player.
        consumable: Player 1's pocket item slot (card/pill), re-rolled every
            room visit. Duo characters' player2 gets its own independent
            pocket-slot roll, but it isn't tracked here -- only player1's.
        stats: Core combat/movement stats.
        health: Precise per-heart-type health breakdown.
        enemies: Enemies within the current camera viewport's frame (a
            single 1x1 room's worth of view around the viewport position),
            not every enemy in the whole room for shapes larger than 1x1.
        doors: Doors present in the room, one entry per side that has one.
        capture_mode: ``"idle"`` or ``"animated"``.
        timestamp: Capture time, ISO 8601, set by the capture daemon.
    """

    image_id: str
    scenario_id: str
    seed: int
    stage: str
    room_shape: int
    room_type: int
    camera_viewport_position: str
    player: PlayerStateM1
    player2: PlayerStateM1 | None
    items_held: int
    item_delta: ItemDelta
    active_item: ActiveItem
    currency: CurrencyState
    consumable: Consumable
    stats: PlayerStats
    health: PlayerHealth
    enemies: list[EnemyState]
    doors: list[Door]
    capture_mode: str
    timestamp: str
