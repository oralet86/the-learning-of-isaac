---@class TLOIConfig
local Config = {}

-- Ticks to wait after posing a scene before treating it as "settled" for
-- idle-mode capture.
Config.SETTLE_TICKS = 30

-- Ticks to wait after moving the player to the next camera-viewport
-- position within the same room visit, before capturing. Shorter than
-- SETTLE_TICKS since only the player/camera position changed -- doors,
-- items, and health were already posed and settled for this room visit.
Config.VIEWPORT_SETTLE_TICKS = 10

-- Extra settle ticks added on top of SETTLE_TICKS/SETTLE_TICKS_ANIMATED for
-- RoomType.ROOM_BOSS. Room:SetClear(true) (see pose_and_settle) suppresses
-- the boss-intro cutscene lock, but scripted multi-phase bosses (e.g. Mom's
-- Heart/It Lives) still need extra real time to finish their own internal
-- setup before it's safe to Game():ChangeRoom() away again -- without this,
-- teleporting out of such a room too soon can crash the game with a native
-- access violation that Lua's pcall cannot catch.
Config.BOSS_SETTLE_TICKS_BONUS = 150

-- "animated" mode captures deliberately mid-animation (walk cycles, damage
-- flashes, particle effects still playing) via a much shorter settle
-- window, instead of waiting for a fully settled frame like "idle" mode --
-- valuable for matching the in-motion distribution real captured footage
-- will have, alongside "idle" mode's clean/static labels. Rolled once per
-- room visit (same granularity as item_delta/active_item), with
-- CAPTURE_MODE_ANIMATED_PCT chance of "animated", so both modes appear
-- within a single run.
Config.CAPTURE_MODE_ANIMATED_PCT = 0.3
Config.SETTLE_TICKS_ANIMATED = 5
Config.VIEWPORT_SETTLE_TICKS_ANIMATED = 2

-- Number of camera-viewport positions to sample for a 1x1 room visit. The
-- whole room is always in-frame regardless of player position for this
-- shape, so unlike the other shape buckets this is purely for
-- player-position/pose diversity across captures, not framing coverage.
-- Position 1 is always the room center; the rest are additional valid
-- random positions (see scenario/viewports.lua's single_room_positions).
Config.VIEWPORTS_PER_1X1_ROOM = 3

-- Ticks to wait for the Python daemon to ack a posed scenario before giving
-- up and retrying -- guards against silently hanging forever if the daemon
-- isn't running. 300 ticks ~= 5s at the game's fixed 60 tick/s rate.
Config.CAPTURE_TIMEOUT_TICKS = 300

-- Ticks to wait for MC_POST_NEW_ROOM after requesting a room/stage change
-- before giving up and retrying -- guards against silently hanging forever
-- if a room change is rejected/ignored (this happens for boss rooms
-- specifically: the intro cutscene has no auto-timeout and blocks the
-- *next* scenario's room change while it's waiting for a skip input this
-- pipeline never sends). 300 ticks ~= 5s.
Config.ROOM_WAIT_TIMEOUT_TICKS = 300

-- Number of room visits per stage, randomized within this range per stage.
Config.ROOMS_PER_STAGE_MIN = 5
Config.ROOMS_PER_STAGE_MAX = 10

-- Baseline passive-item count applied once per run (right after the
-- character restart), randomized within this range.
Config.ITEM_BASELINE_MIN = 10
Config.ITEM_BASELINE_MAX = 30

-- Tainted Isaac cannot hold more than 8 passive items at once -- picking up
-- a 9th drops his currently selected item on a pedestal instead. Both the
-- baseline and the per-visit item-delta "add" roll are capped at this for
-- him instead of ITEM_BASELINE_MIN/MAX (see spawning/items.lua).
Config.TAINTED_ISAAC_ITEM_CAP = 8

-- Per-room-visit item-delta roll: ADD_PCT chance to add one random item,
-- REMOVE_PCT chance to remove one currently-held item, remainder ("none")
-- is left implicit so the two can't drift out of sync with a third
-- hardcoded fraction.
Config.ITEM_DELTA_ADD_PCT = 0.30
Config.ITEM_DELTA_REMOVE_PCT = 0.30

-- Random-position sampling for repositioning the player each room visit:
-- how far to inset the sampled point from the room's grid edges, and how
-- many sample+validate attempts before falling back to Room:GetCenterPos().
Config.ROOM_POSITION_MARGIN = 1
Config.ROOM_POSITION_MAX_ATTEMPTS = 10

-- Duo character (Jacob & Esau, Tainted Forgotten) second-player positioning
-- each room visit: a random point within this radius range (world units --
-- a grid tile is ~40 units) of player1's position, via
-- Positioning.SampleNearbyPosition, rather than anywhere valid in the room.
-- 30-120 is roughly "less than 1 tile away" to "3 tiles away" -- close
-- enough to read as a pair, far enough to not just overlap player1.
Config.PLAYER2_NEARBY_MIN_RADIUS = 30
Config.PLAYER2_NEARBY_MAX_RADIUS = 120

-- Realistic in-game ranges for the per-room-visit currency randomization.
Config.COINS_MAX = 99
Config.BOMBS_MAX = 9
Config.KEYS_MAX = 9

-- Keeper/Tainted Keeper coin heart container caps -- Keeper cannot have
-- more than 3 Coin Hearts (even before Hush is defeated), Tainted Keeper
-- never more than 2. Used by spawning/hearts.lua's character-specific
-- health constraints.
Config.KEEPER_COIN_HEART_CAP = 3
Config.TAINTED_KEEPER_COIN_HEART_CAP = 2

-- Chance of holding a (randomly picked) active item each room visit, with
-- a random charge from 0 to that item's own max charge. Re-rolled every
-- visit (unlike passive items, which persist/drift) -- if not rolled, any
-- active item from a previous visit is cleared.
Config.ACTIVE_ITEM_CHANCE = 0.5

-- Per-room-visit pocket-item-slot roll: CARD_CHANCE chance to hold a random
-- card (tarot/playing card/rune/soul stone -- Card is a single unified id
-- space covering all of these), PILL_CHANCE chance to hold a random pill,
-- remainder ("none") is left implicit, same pattern as ITEM_DELTA_ADD/
-- REMOVE_PCT. Re-rolled every visit, like the active item.
Config.CONSUMABLE_CARD_CHANCE = 0.35
Config.CONSUMABLE_PILL_CHANCE = 0.35

-- Per-door-slot state distribution, applied once per room visit to every
-- door slot the current room actually has (see spawning/doors.lua). Must
-- sum to 1.0.
Config.DOOR_STATE_WEIGHTS = {
    open = 0.35,
    closed = 0.35,
    locked = 0.20,
    barred = 0.05,
    broken = 0.05,
}

-- Secret/super-secret/ultra-secret-room-targeting door slots are handled
-- separately from DOOR_STATE_WEIGHTS above (see spawning/doors.lua) --
-- unlike every other door, they render as plain solid wall until blown open
-- by a bomb, so "open"/"closed"/"locked"/"barred" don't correspond to any
-- real visual state for them. This is the per-room-visit chance such a slot
-- is rolled as already-blown (visible, `broken` in ground truth); the
-- remainder is left completely untouched (still a solid wall, correctly
-- absent from ground truth -- see query/ground_truth.lua's
-- is_visually_present).
Config.SECRET_DOOR_BLOWN_PCT = 0.3

-- Shared file-exchange directory the mod and the Python daemon
-- (datagen/exchange.py) both read/write. The mod runs as a Windows process
-- under Proton/Wine, so this must be the Windows-style path Wine's default
-- Z: drive mapping (host / -> Z:\) resolves to the same folder
-- datagen/run_config.py's DEFAULT_EXCHANGE_DIR points to on the Linux side.
-- If you move to a different machine/install, update this together with
-- datagen/run_config.py.
Config.EXCHANGE_DIR = "Z:\\home\\mert\\Code\\the-learning-of-isaac\\datagen\\runtime\\"

return Config
