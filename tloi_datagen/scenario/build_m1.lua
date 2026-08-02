local Log = include("log")

---@class ScenarioBuilderM1
local ScenarioBuilderM1 = {}

-- "goto" does NOT change floors at all -- it navigates to specific *rooms*
-- within the current floor ("goto d.10", "goto s.error.21"), silently
-- never firing MC_POST_NEW_ROOM if given a stage id instead. The real
-- floor-change command is "stage <N><letter>" (e.g. "stage 1a" for Cellar).
-- `stage_command` keeps the "s.<num>.<type>" internal representation
-- (matches Level:GetStage()/GetStageType(), used elsewhere in this mod)
-- and is translated to the "stage" command's <N><letter> syntax here.
local STAGE_TYPE_LETTERS = {
    [0] = "", -- StageType.STAGETYPE_ORIGINAL, e.g. Basement
    [1] = "a", -- StageType.STAGETYPE_WOTL, e.g. Cellar
    [2] = "b", -- StageType.STAGETYPE_AFTERBIRTH, e.g. Burning Basement
    [4] = "c", -- StageType.STAGETYPE_REPENTANCE, e.g. Downpour
    [5] = "d", -- StageType.STAGETYPE_REPENTANCE_B, e.g. Dross
}

---@param stage_command string
function ScenarioBuilderM1.GoToStage(stage_command)
    local stage_num, stage_type = stage_command:match("^s%.(%d+)%.(%d+)$")
    local letter = STAGE_TYPE_LETTERS[tonumber(stage_type)] or ""
    local console_command = "stage " .. stage_num .. letter
    Log.Info("GoToStage: issuing \"" .. console_command .. "\" (from stage_command=" .. stage_command .. ")")
    Isaac.ExecuteCommand(console_command)
end

-- "stage"-ing to the floor the player is already on is presumed to also be
-- a no-op that never fires MC_POST_NEW_ROOM (not yet separately confirmed,
-- but the same risk applies as "goto" did) -- callers must check this first
-- and skip waiting for a room load when it's already true (see main.lua).
---@param stage_command string
---@return boolean
function ScenarioBuilderM1.IsOnStage(stage_command)
    local stage_num, stage_type = stage_command:match("^s%.(%d+)%.(%d+)$")
    local level = Game():GetLevel()
    local current_stage, current_type = level:GetStage(), level:GetStageType()
    local on_stage = current_stage == tonumber(stage_num) and current_type == tonumber(stage_type)
    Log.Info(
        "IsOnStage: target="
            .. stage_command
            .. " current=(stage="
            .. tostring(current_stage)
            .. ", type="
            .. tostring(current_type)
            .. ") -> "
            .. tostring(on_stage)
    )
    return on_stage
end

-- Teleports to a random room on the current floor, other than the one
-- currently occupied, so successive scenarios actually see different rooms
-- instead of repeatedly re-posing the same one. All room *types* and
-- *shapes* are valid targets (normal, boss, shop, secret, treasure, etc.;
-- 1x1, long rooms, L-shapes, 2x2) -- per-shape camera-viewport sampling
-- (scenario.viewports) is what makes non-1x1 shapes produce correctly
-- framed captures. Game():ChangeRoom() (not Level:ChangeRoom(),
-- whose own docs say it doesn't update fx layers properly) fires
-- MC_POST_NEW_ROOM, which the caller must wait for before posing the new
-- room.
--
-- Wrapped defensively: RoomDescriptor/RoomConfig_Room field access here
-- (GetRooms, .Data.Shape, .Data.Type) is less battle-tested than the
-- EntityPlayer calls elsewhere in this mod, so a documentation mismatch
-- degrades to "stay in the current room" (logged) rather than crashing the
-- whole generation loop.
---@param rng RNG
---@return boolean teleported True if a different room was found and Game:ChangeRoom() was called (caller must wait for MC_POST_NEW_ROOM); false if no alternative room existed, or the room list couldn't be read, and the current room should be posed directly.
function ScenarioBuilderM1.TeleportToRandomRoom(rng)
    local ok, result = pcall(function()
        local level = Game():GetLevel()
        local rooms = level:GetRooms()
        local current_idx = level:GetCurrentRoomIndex()

        local candidates = {}
        for i = 0, rooms.Size - 1 do
            local descriptor = rooms:Get(i)
            local data = descriptor.Data
            if data ~= nil and descriptor.GridIndex ~= current_idx then
                candidates[#candidates + 1] = descriptor.GridIndex
            end
        end

        Log.Info(
            "TeleportToRandomRoom: current_idx="
                .. tostring(current_idx)
                .. " candidates="
                .. #candidates
                .. "/"
                .. rooms.Size
        )

        if #candidates == 0 then
            return false
        end

        local pick = candidates[rng:RandomInt(#candidates) + 1]
        Log.Info("TeleportToRandomRoom: picked room index " .. tostring(pick) .. ", calling Game():ChangeRoom()")
        Game():ChangeRoom(pick)
        return true
    end)

    if not ok then
        Log.Info("TeleportToRandomRoom: ERROR -- " .. tostring(result))
        return false
    end
    return result
end

return ScenarioBuilderM1
