---@class TLOIConfig
local Config = {}

-- Ticks to wait after posing a scene before treating it as "settled" for
-- idle-mode capture. Was 120 (2s) while chasing the ChangeRoom crash;
-- cut back to 60 now that L-shape rooms are excluded from teleports.
Config.SETTLE_TICKS = 60

-- Ticks to wait after moving the player to the next camera-viewport
-- position within the same room visit, before capturing. Shorter than
-- SETTLE_TICKS since only the player/camera position changed -- doors,
-- items, and health were already posed and settled for this room visit.
-- Was 40 while chasing the ChangeRoom crash; cut back to 25 now that
-- L-shape rooms are excluded from teleports.
Config.VIEWPORT_SETTLE_TICKS = 25

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
Config.SETTLE_TICKS_ANIMATED = 30
Config.VIEWPORT_SETTLE_TICKS_ANIMATED = 25

-- Base number of camera-viewport positions to sample per "1x1 unit" of a
-- room's area (see scenario/viewports.lua's ROOM_UNIT_AREA) -- e.g. a 1x1
-- room (1 unit) samples this many positions, a long-axis room (2 units)
-- samples twice this many, a 2x2/L-shape room (4 units) samples four times
-- this many. Position 1 is always the room center, for a deterministic
-- baseline capture; the rest are additional valid random positions (see
-- scenario/viewports.lua's random_room_positions).
Config.VIEWPORTS_PER_ROOM_UNIT = 3

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

-- Proactive workaround for the ChangeRoom native access-violation crash
-- seen on the Mines floor (see scenario/build_m1.lua's
-- TeleportToRandomRoom): when on a minecart floor (Mines/Ashpit = stages
-- 3-4 with a Repentance-alt StageType 4 or 5), exclude 1x2-shaped rooms
-- from the teleport candidate pool. The 1x2 minecart track room is the
-- suspected crash trigger: the crash log died inside Game():ChangeRoom()
-- into an unidentified room shortly after the Mines start room, and the
-- minecart room is the one known 1x2 special room there (see
-- https://bindingofisaacrebirth.wiki.gg/wiki/Rooms#Mine_Cart_Room).
Config.EXCLUDE_1X2_ON_MINECART_FLOORS = true

-- Collectible ids that must never be handed out by spawning/items.lua,
-- as a set keyed by collectible id. Consulted by both the
-- passive/familiar pool and the active pool (see get_item_pool /
-- get_active_pool). Story/quest items that trigger special rooms or
-- scripted events are excluded because they destabilize automated runs:
--   IBS (725) randomly poops, corrupting scene captures.
--   Knife Piece 1 (626) spawns the Knife Piece Room (the minecart quest
--     room), which hard-crashes the game on the ChangeRoom into it; Knife
--     Piece 2 (627) goes with it.
--   Broken Shovel 1/2 (550/551) summon Mom's Foot stomping every few
--     seconds; Mom's Shovel (552) is the completed story active.
--   Dad's Note (668) is the Ascent trigger item; Polaroid (327) and
--     Negative (328) are the end-game story items.
-- Key Pieces 1/2 (238/239) are deliberately kept: they only open the Mega
-- Satan door and cause no special-room or scripting side effects.
Config.EXCLUDED_ITEM_IDS = {
    [327] = true, -- Polaroid
    [328] = true, -- Negative
    [550] = true, -- Broken Shovel 1
    [551] = true, -- Broken Shovel 2
    [552] = true, -- Mom's Shovel
    [626] = true, -- Knife Piece 1
    [627] = true, -- Knife Piece 2
    [668] = true, -- Dad's Note
    [725] = true, -- IBS
}

-- Proactive workaround for the ChangeRoom native access-violation crash
-- (see scenario/build_m1.lua's TeleportToRandomRoom): exclude every
-- L-shape room (LTL/LTR/LBL/LBR) from the teleport candidate pool. Two
-- crashes with full logging both died on the ChangeRoom into an LTL room
-- (variant 728 on the Womb, variant 1063 on the Catacombs) with zero
-- successful LTL entries, while LBL/LBR each entered fine exactly once --
-- too small a sample to trust the non-LTL orientations, so all four are
-- excluded; narrow to LTL only (or re-enable entirely) once enough
-- crash-free data accumulates.
Config.EXCLUDE_L_SHAPE_ROOMS = true

-- Rooms that must never be teleported into, keyed by RoomConfig_Room.Name
-- (see scenario/build_m1.lua's TeleportToRandomRoom). The Knife Piece Room
-- (variant 10019 -- the minecart/yellow-button quest room of the
-- Mines/Ashpit chapter) hard-crashes the game natively on the ChangeRoom
-- into it: four of seven logged crashes are teleports into it, including
-- grid 61 on s.4.4 twice and grid 162 on s.4.5 twice -- the same grid
-- positions across different random layouts, so the room sits at a fixed
-- position per floor skin. The room only exists when the player holds
-- Knife Piece 1 (a possible baseline item), and it is 1x1-shaped, so
-- neither shape-based exclusion catches it -- name is the reliable key
-- here (its variant id on other floor skins is unverified).
Config.EXCLUDED_ROOM_NAMES = { ["Knife Piece Room"] = true }

-- Shared file-exchange directory the mod and the Python daemon
-- (datagen/exchange.py) both read/write. The mod runs as a Windows process
-- under Proton/Wine, so this must be the Windows-style path Wine's default
-- Z: drive mapping (host / -> Z:\) resolves to the same folder
-- datagen/run_config.py's DEFAULT_EXCHANGE_DIR points to on the Linux side.
-- If you move to a different machine/install, update this together with
-- datagen/run_config.py.
Config.EXCHANGE_DIR = "Z:\\home\\mert\\Code\\the-learning-of-isaac\\datagen\\runtime\\"

return Config
