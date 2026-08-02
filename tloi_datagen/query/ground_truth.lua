---@class GroundTruthQuery
local GroundTruthQuery = {}

---@param player EntityPlayer
---@return table
local function collect_currency(player)
    return {
        coins = player:GetNumCoins(),
        bombs = player:GetNumBombs(),
        keys = player:GetNumKeys(),
    }
end

-- The player's pocket item slot: a card (unified id space covering
-- tarot/playing cards, runes, and soul stones) or a pill, never both at
-- once. Read live via GetCard()/GetPill() rather than trusting
-- Consumables.RollConsumables's intent, per this project's "log actual
-- queried state" principle -- protects against a partial/clamped SetCard
-- call the same way every other collect_* helper here does.
---@param player EntityPlayer
---@return table
local function collect_consumable(player)
    local card_id = player:GetCard(0)
    local pill_color = player:GetPill(0)
    if card_id ~= 0 then
        return { kind = "card", card_id = card_id, pill_color = nil }
    elseif pill_color ~= 0 then
        return { kind = "pill", card_id = nil, pill_color = pill_color }
    end
    return { kind = "none", card_id = nil, pill_color = nil }
end

---@param value number
---@return number
local function round2(value)
    return math.floor(value * 100 + 0.5) / 100
end

-- The 6 stats the in-game stat UI shows (damage, tears/firerate, range,
-- speed, shot speed, luck). `fire_rate` is the human-facing "Tears" stat
-- (tears per second), not the raw MaxFireDelay it's derived from -- only
-- one fire-rate representation is kept, not both.
---@param player EntityPlayer
---@return table
local function collect_stats(player)
    return {
        damage = round2(player.Damage),
        fire_rate = round2(30 / (player.MaxFireDelay + 1)),
        speed = round2(player.MoveSpeed),
        range = round2(player.TearRange),
        shot_speed = round2(player.ShotSpeed),
        luck = round2(player.Luck),
    }
end

-- EntityPlayer:GetBlackHearts() does not return a count -- its docs say it
-- returns a bit mask of which soul-heart slots are black hearts. Counts the
-- set bits (32 is a safe upper bound; real heart-slot counts never come
-- close) to get an actual number.
---@param mask integer
---@return integer
local function count_black_hearts(mask)
    local count = 0
    for i = 0, 31 do
        if (mask >> i) & 1 == 1 then
            count = count + 1
        end
    end
    return count
end

-- Curse of the Unknown replaces the entire heart HUD with a single "?"
-- icon -- the real container/current counts aren't visually recoverable
-- from a captured frame under this curse, so this is a deliberate
-- exception to the project's "log actual queried state" principle: logging
-- the omniscient counts here would mislabel a frame that, visually, shows
-- nothing but "?". `unknown = true` with every count field omitted is
-- returned instead; downstream consumers must treat `unknown` as the
-- authoritative signal for whether the numeric fields are meaningful.
---@param player EntityPlayer
---@return table
local function collect_health(player)
    local level = Game():GetLevel()
    if level:GetCurses() & LevelCurse.CURSE_OF_THE_UNKNOWN > 0 then
        return { unknown = true }
    end
    return {
        unknown = false,
        hearts = player:GetHearts(),
        max_hearts = player:GetMaxHearts(),
        soul_hearts = player:GetSoulHearts(),
        black_hearts = count_black_hearts(player:GetBlackHearts()),
        bone_hearts = player:GetBoneHearts(),
        rotten_hearts = player:GetRottenHearts(),
        eternal_hearts = player:GetEternalHearts(),
        golden_hearts = player:GetGoldenHearts(),
        broken_hearts = player:GetBrokenHearts(),
    }
end

-- Only enemies actually within the current camera viewport's frame are
-- logged -- filtered to a bounding box of `viewport_half_extent` around
-- `viewport_center`, matching the in-game camera's real coverage (always a
-- single 1x1 room's worth of view, following the player). Both are
-- computed by scenario.viewports from the room's actual shape/bounds.
---@param viewport_center Vector
---@param viewport_half_extent Vector
---@return table[]
local function collect_enemies(viewport_center, viewport_half_extent)
    local entities = Game():GetRoom():GetEntities()
    local enemies = {}
    for i = 0, #entities - 1 do
        local entity = entities:Get(i)
        if entity:IsActiveEnemy(false) then
            local dx = math.abs(entity.Position.X - viewport_center.X)
            local dy = math.abs(entity.Position.Y - viewport_center.Y)
            if dx <= viewport_half_extent.X and dy <= viewport_half_extent.Y then
                enemies[#enemies + 1] = {
                    type = entity.Type,
                    variant = entity.Variant,
                    subtype = entity.SubType,
                    position = { entity.Position.X, entity.Position.Y },
                }
            end
        end
    end
    return enemies
end

-- Doors leading to secret/super-secret/ultra-secret rooms render as an
-- ordinary solid wall until that wall is blown open (bombed) -- unlike
-- every other door state (open/closed/locked/barred/broken), there is no
-- doorframe sprite at all to see beforehand, so nothing is visually present
-- in a captured frame to detect until Doors.RandomizeDoors has actually
-- blown it open (GridEntityDoor:TryBlowOpen sets Busted true on success).
-- GridEntityDoor still exists at the slot either way (Room:GetDoor still
-- returns it), so this can't be filtered by nil-ness -- it has to be
-- checked explicitly per door.
local SECRET_ROOM_TYPES = {
    [RoomType.ROOM_SECRET] = true,
    [RoomType.ROOM_SUPERSECRET] = true,
    [RoomType.ROOM_ULTRASECRET] = true,
}

---@param door GridEntityDoor
---@return boolean
local function is_visually_present(door)
    if SECRET_ROOM_TYPES[door.TargetRoomType] then
        return door:IsBusted()
    end
    return true
end

-- Iterates every DoorSlot (0-7 -- LEFT0/UP0/RIGHT0/DOWN0 for a 1x1 room;
-- the 1-suffixed slots only apply to larger room shapes, but are checked
-- regardless since Room:GetDoor() just returns nil for a slot that doesn't
-- apply to the current room's shape) and records the ones that actually
-- have a door, are visually present (see is_visually_present), AND are
-- actually within the current camera viewport's frame -- same
-- bounding-box check collect_enemies already does, and for the same
-- reason: for any room shape bigger than 1x1, a door on the far side of
-- the room from the current viewport position isn't in the captured image
-- at all, so logging it unconditionally would claim a door is present in
-- frames that visually show nothing but wall/off-screen space. GridEntity's
-- `Position` field (world-space) is what's compared against, same as
-- entity.Position for enemies.
---@param viewport_center Vector
---@param viewport_half_extent Vector
---@return table[]
local function collect_doors(viewport_center, viewport_half_extent)
    local room = Game():GetRoom()
    local doors = {}
    for slot = 0, DoorSlot.NUM_DOOR_SLOTS - 1 do
        local door = room:GetDoor(slot)
        if door ~= nil and is_visually_present(door) then
            local dx = math.abs(door.Position.X - viewport_center.X)
            local dy = math.abs(door.Position.Y - viewport_center.Y)
            if dx <= viewport_half_extent.X and dy <= viewport_half_extent.Y then
                doors[#doors + 1] = {
                    slot = slot,
                    is_open = door:IsOpen(),
                    is_locked = door:IsLocked(),
                    is_busted = door:IsBusted(),
                }
            end
        end
    end
    return doors
end

---@param player EntityPlayer
---@return table
local function collect_player(player)
    local pos = player.Position
    return {
        character = player:GetPlayerType(),
        position = { pos.X, pos.Y },
    }
end

-- Reads the active item slot fresh via GetActiveItem/GetActiveCharge at
-- collection time, rather than trusting ItemSpawning.RollActiveItem's
-- return value from earlier in the room visit -- the player is free to use
-- the active item (changing its charge) or, in principle, have it changed
-- by other effects during the settle window between posing and this
-- collection call, so re-querying here is what keeps this a "log actual
-- queried state" value like every other field in this file rather than a
-- stale intent snapshot. `charge` is normalized to [0, 1] the same way
-- ItemSpawning.RollActiveItem does (see its docstring for the MaxCharges
-- == 0 special case).
---@param player EntityPlayer
---@return table
local function collect_active_item(player)
    local item_id = player:GetActiveItem(ActiveSlot.SLOT_PRIMARY)
    if item_id == CollectibleType.COLLECTIBLE_NULL then
        return { held = false, item_id = nil, charge = nil }
    end

    local max_charge = Isaac.GetItemConfig():GetCollectible(item_id).MaxCharges
    local charge = player:GetActiveCharge(ActiveSlot.SLOT_PRIMARY)
    local normalized_charge = max_charge == 0 and 1.0 or charge / max_charge

    return { held = true, item_id = item_id, charge = normalized_charge }
end

-- Builds ground truth by querying live player/game state (never
-- re-deriving it from what the scenario builder intended to spawn), so a
-- partial/clamped spawn is reflected accurately.
---@param player EntityPlayer
---@param scenario_id string
---@param seed integer
---@param stage string
---@param camera_viewport_position string
---@param viewport_half_extent Vector Half-extent of the current viewport's frame (a single 1x1 room's worth of view), for filtering `enemies` to what's actually in-frame. From scenario.viewports.
---@param item_delta {action: string, item_id: integer?} The action taken this room visit (see ItemSpawning.RollDelta) -- kept as-is rather than re-queried, since it's a logged event, not a live-queryable snapshot value.
---@param player2 EntityPlayer? The second twin (Jacob & Esau / Tainted Forgotten), or nil for every other character.
---@param capture_mode string "idle" or "animated" -- rolled once per room visit by main.lua, per Config.CAPTURE_MODE_ANIMATED_PCT.
---@return table
function GroundTruthQuery.CollectState(
    player,
    scenario_id,
    seed,
    stage,
    camera_viewport_position,
    viewport_half_extent,
    item_delta,
    player2,
    capture_mode
)
    local room = Game():GetRoom()
    return {
        scenario_id = scenario_id,
        seed = seed,
        stage = stage,
        room_shape = room:GetRoomShape(),
        room_type = room:GetType(),
        camera_viewport_position = camera_viewport_position,
        player = collect_player(player),
        -- player2 is only present in the returned table for duo characters
        -- (Jacob & Esau, Tainted Forgotten) -- the vendored JSON encoder
        -- omits nil-valued keys entirely, so downstream consumers see the
        -- key missing rather than null for every other character, and must
        -- treat "missing" as "not a duo character". player2's active item
        -- is deliberately not tracked/logged at all -- unneeded complexity.
        player2 = player2 ~= nil and collect_player(player2) or nil,
        items_held = player:GetCollectibleCount(),
        item_delta = item_delta,
        active_item = collect_active_item(player),
        currency = collect_currency(player),
        consumable = collect_consumable(player),
        stats = collect_stats(player),
        health = collect_health(player),
        -- The player's own live position is the viewport center -- it's
        -- exactly where pose_and_settle/advance_viewport just placed it
        -- for this capture, and reading it back (rather than re-passing the
        -- intended target vector) keeps this a queried-state value like
        -- everything else here.
        enemies = collect_enemies(player.Position, viewport_half_extent),
        doors = collect_doors(player.Position, viewport_half_extent),
        capture_mode = capture_mode,
    }
end

return GroundTruthQuery
