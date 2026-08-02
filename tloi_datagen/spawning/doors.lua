local Config = include("config")
local Log = include("log")

---@class Doors
local Doors = {}

-- Cumulative-weight roll order -- must match the keys in
-- Config.DOOR_STATE_WEIGHTS.
local STATE_ORDER = { "open", "closed", "locked", "barred", "broken" }

-- Secret/super-secret/ultra-secret room doors render as plain solid wall
-- until blown open by a bomb -- none of STATE_ORDER's states correspond to
-- a real visual state for them (see query/ground_truth.lua's
-- is_visually_present and Config.SECRET_DOOR_BLOWN_PCT), so they're routed
-- through roll_secret_door_state below instead of the general weighted roll.
local SECRET_ROOM_TYPES = {
    [RoomType.ROOM_SECRET] = true,
    [RoomType.ROOM_SUPERSECRET] = true,
    [RoomType.ROOM_ULTRASECRET] = true,
}

---@param rng RNG
---@return string
local function roll_state(rng)
    local roll = rng:RandomFloat()
    local cumulative = 0
    for _, state in ipairs(STATE_ORDER) do
        cumulative = cumulative + Config.DOOR_STATE_WEIGHTS[state]
        if roll < cumulative then
            return state
        end
    end
    return STATE_ORDER[#STATE_ORDER]
end

-- Applies `state` to `door` using only methods/fields that are writable in
-- this REPENTOGON build: GridEntityDoor:Open, :Close, :SetLocked, :Bar, and
-- the `Busted` field. Direct `door.Variant = DoorVariant.*` assignment --
-- the usual technique for a distinct barred/hidden *sprite*, as opposed to
-- the SetLocked/Bar *behavior* -- throws "no member named 'Variant'"
-- (GridEntityDoor exposes it as read-only), so "barred"'s sprite falls back
-- to whatever :Bar() alone produces.
---@param door GridEntityDoor
---@param state string
local function apply_state(door, state)
    if state == "open" then
        door:Open()
    elseif state == "closed" then
        door:Close(true)
    elseif state == "locked" then
        door:SetLocked(true)
    elseif state == "barred" then
        door:SetLocked(true)
        door:Bar()
    elseif state == "broken" then
        door.Busted = true
    end
end

-- Config.SECRET_DOOR_BLOWN_PCT chance of blowing the wall open (visible,
-- logged as "broken"); otherwise the door is left completely untouched --
-- no API call at all -- so it stays a plain solid wall, matching real
-- gameplay's binary undiscovered/discovered state for these rooms.
-- TryBlowOpen (rather than directly setting Busted) is the documented
-- "blow this door open by explosive force" API.
---@param door GridEntityDoor
---@param rng RNG
---@return string state "broken" if blown open, "hidden" if left untouched.
local function roll_secret_door_state(door, rng)
    if rng:RandomFloat() < Config.SECRET_DOOR_BLOWN_PCT then
        door:TryBlowOpen(true, nil)
        return "broken"
    end
    return "hidden"
end

-- Rolls and applies a random door state to every door slot the current room
-- actually has. Secret/super-secret/ultra-secret-room-targeting slots go
-- through roll_secret_door_state (blown-open-or-untouched); every other
-- slot gets the general weighted roll (open/closed/locked/barred/broken,
-- Config.DOOR_STATE_WEIGHTS). No separate room-shape/neighbor validity
-- check is needed -- Room:GetDoor(slot) already returns nil for a slot the
-- current room shape doesn't have, so this just skips nils, same pattern as
-- query/ground_truth.lua's collect_doors.
---@param room Room
---@param rng RNG
---@return table[] applied Per-door {slot, state} actually requested (query/ground_truth.lua's collect_doors reads back the resulting live state for the ground-truth record).
function Doors.RandomizeDoors(room, rng)
    local applied = {}
    for slot = 0, DoorSlot.NUM_DOOR_SLOTS - 1 do
        local door = room:GetDoor(slot)
        if door ~= nil then
            local state
            if SECRET_ROOM_TYPES[door.TargetRoomType] then
                state = roll_secret_door_state(door, rng)
            else
                state = roll_state(rng)
                apply_state(door, state)
            end
            applied[#applied + 1] = { slot = slot, state = state }
        end
    end
    Log.Info("RandomizeDoors: applied " .. #applied .. " door states")
    return applied
end

return Doors
