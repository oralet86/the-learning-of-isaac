local Config = include("config")
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

-- Every L-shape orientation -- excluded from teleport targets when
-- Config.EXCLUDE_L_SHAPE_ROOMS is set (two logged crashes both died on the
-- ChangeRoom into an LTL room; see config.lua).
local L_SHAPE_SHAPES = {
    [RoomShape.ROOMSHAPE_LTL] = true,
    [RoomShape.ROOMSHAPE_LTR] = true,
    [RoomShape.ROOMSHAPE_LBL] = true,
    [RoomShape.ROOMSHAPE_LBR] = true,
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
-- framed captures -- except for two proactive workarounds, both gated on
-- Config flags (see config.lua): on the minecart floors (Mines/Ashpit),
-- 1x2-shaped rooms are excluded when Config.EXCLUDE_1X2_ON_MINECART_FLOORS
-- is set (the 1x2 minecart track room is a suspected ChangeRoom crash
-- trigger), and every L-shape room is excluded when
-- Config.EXCLUDE_L_SHAPE_ROOMS is set (two logged crashes both died on
-- the ChangeRoom into an LTL room). Game():ChangeRoom() (not
-- Level:ChangeRoom(), whose own docs say it doesn't update fx layers
-- properly) fires MC_POST_NEW_ROOM, which the caller must wait for before
-- posing the new room.
--
-- Wrapped defensively: RoomDescriptor/RoomConfig_Room field access here
-- (GetRooms, .Data.Shape, .Data.Type) is less battle-tested than the
-- EntityPlayer calls elsewhere in this mod, so a documentation mismatch
-- degrades to "stay in the current room" (logged) rather than crashing the
-- whole generation loop. Note this pcall cannot catch the native
-- access-violation crash itself -- that is why every candidate and the
-- picked room are logged in full (type/variant/shape/name) *before* the
-- ChangeRoom call, so a crash's log tail names the exact room the game
-- died entering.
---@param rng RNG
---@return boolean teleported True if a different room was found and Game:ChangeRoom() was called (caller must wait for MC_POST_NEW_ROOM); false if no alternative room existed, or the room list couldn't be read, and the current room should be posed directly.
function ScenarioBuilderM1.TeleportToRandomRoom(rng)
    local ok, result = pcall(function()
        local level = Game():GetLevel()
        local rooms = level:GetRooms()
        local current_idx = level:GetCurrentRoomIndex()

        -- Floor context. Stages 3-4 with a Repentance-alt StageType (4/5)
        -- are the Mines/Ashpit chapter (see data/stage_route.lua's
        -- stage-index comments); the minecart track rooms exist only there.
        local stage = level:GetStage()
        local stage_type = level:GetStageType()
        local exclude_1x2 = Config.EXCLUDE_1X2_ON_MINECART_FLOORS
            and stage >= 3
            and stage <= 4
            and (stage_type == 4 or stage_type == 5)
        Log.Info(
            "TeleportToRandomRoom: stage="
                .. tostring(stage)
                .. " stage_type="
                .. tostring(stage_type)
                .. " exclude_1x2="
                .. tostring(exclude_1x2)
                .. " exclude_l_shapes="
                .. tostring(Config.EXCLUDE_L_SHAPE_ROOMS)
        )

        -- Current room descriptor, for correlating crashes with the room
        -- the mod was teleporting away from.
        local current_desc = level:GetCurrentRoomDesc()
        if current_desc ~= nil and current_desc.Data ~= nil then
            Log.Info(
                "TeleportToRandomRoom: current room idx="
                    .. tostring(current_idx)
                    .. " type="
                    .. tostring(current_desc.Data.Type)
                    .. " variant="
                    .. tostring(current_desc.Data.Variant)
                    .. " shape="
                    .. tostring(current_desc.Data.Shape)
                    .. " name=\""
                    .. tostring(current_desc.Data.Name)
                    .. "\""
            )
        else
            Log.Info(
                "TeleportToRandomRoom: current room idx="
                    .. tostring(current_idx)
                    .. " (descriptor unavailable)"
            )
        end

        local candidates = {}
        local excluded = {}
        for i = 0, rooms.Size - 1 do
            local descriptor = rooms:Get(i)
            local data = descriptor.Data
            if data ~= nil and descriptor.GridIndex ~= current_idx then
                local is_1x2 = data.Shape == RoomShape.ROOMSHAPE_1x2
                local is_l_shape = L_SHAPE_SHAPES[data.Shape]
                local exclude_reason
                if exclude_1x2 and is_1x2 then
                    exclude_reason = "1x2 on minecart floor"
                elseif Config.EXCLUDE_L_SHAPE_ROOMS and is_l_shape then
                    exclude_reason = "L-shape room"
                elseif Config.EXCLUDED_ROOM_NAMES[data.Name] then
                    exclude_reason = "excluded room name: " .. tostring(data.Name)
                end
                Log.Info(
                    "TeleportToRandomRoom: room idx="
                        .. tostring(descriptor.GridIndex)
                        .. " type="
                        .. tostring(data.Type)
                        .. " variant="
                        .. tostring(data.Variant)
                        .. " shape="
                        .. tostring(data.Shape)
                        .. " name=\""
                        .. tostring(data.Name)
                        .. "\""
                        .. (exclude_reason ~= nil and (" (EXCLUDED: " .. exclude_reason .. ")") or " (candidate)")
                )
                if exclude_reason ~= nil then
                    excluded[#excluded + 1] = descriptor.GridIndex
                else
                    candidates[#candidates + 1] = descriptor.GridIndex
                end
            end
        end

        Log.Info(
            "TeleportToRandomRoom: current_idx="
                .. tostring(current_idx)
                .. " candidates="
                .. #candidates
                .. "/"
                .. rooms.Size
                .. " excluded="
                .. #excluded
        )
        if #excluded > 0 then
            Log.Info(
                "TeleportToRandomRoom: excluded room indices: "
                    .. table.concat(excluded, ",")
            )
        end

        if #candidates == 0 then
            Log.Info("TeleportToRandomRoom: no candidates remain, staying in current room")
            return false
        end

        local pick = candidates[rng:RandomInt(#candidates) + 1]
        -- Last log line before ChangeRoom: after a crash this names the
        -- exact room the game died entering (the game's own "Room X.Y"
        -- line only prints once the room has actually loaded).
        for i = 0, rooms.Size - 1 do
            local descriptor = rooms:Get(i)
            if descriptor.GridIndex == pick and descriptor.Data ~= nil then
                Log.Info(
                    "TeleportToRandomRoom: PICKED room index "
                        .. tostring(pick)
                        .. " (type="
                        .. tostring(descriptor.Data.Type)
                        .. " variant="
                        .. tostring(descriptor.Data.Variant)
                        .. " shape="
                        .. tostring(descriptor.Data.Shape)
                        .. " name=\""
                        .. tostring(descriptor.Data.Name)
                        .. "\"), calling Game():ChangeRoom()"
                )
                break
            end
        end
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
