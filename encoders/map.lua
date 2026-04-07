local C               = {}

-- 13x13 level grid in Isaac (169 cells)
local GRID_W          = 13
local GRID_H          = 13
local NUM_CELLS       = GRID_W * GRID_H

-- Per-room features (in this order):
-- 1: known (0/1). If 0, everything else is 0.
-- 2: room_type (label float). -1 if empty (no room descriptor), otherwise RoomType enum value.
-- 3: visited (0/1)
-- 4: current_room (0/1)
-- 5: has_unopened_door (0/1)  -- "exits that lead to an unvisited room"
-- 6: distance_to_current (Manhattan distance in grid cells)
local FEATS_PER_CELL  = 6

-- DisplayFlags bitwise (docs mention these bits)
local DISPLAY_VISIBLE = (1 << 0)
local DISPLAY_SHADOW  = (1 << 1)

-- RoomConfigRoom.Doors uses bitmask of DoorSlotFlag enum:
-- LEFT0 = 1<<DoorSlot.LEFT0, UP0 = 1<<DoorSlot.UP0, RIGHT0 = 1<<DoorSlot.RIGHT0, DOWN0 = 1<<DoorSlot.DOWN0,
-- LEFT1 = 1<<DoorSlot.LEFT1, ... (secondary doors for large rooms)
-- We only need direction; treat LEFT0/LEFT1 the same, etc.
-- We avoid referencing DoorSlot values directly (they exist as enums), and just test bits from 0..7.
---@param bit_index integer
---@return integer
local function door_mask(bit_index)
    return (1 << bit_index)
end

---@param idx integer
---@return integer, integer
local function grid_xy(idx)
    local x = idx % GRID_W
    local y = idx // GRID_W
    return x, y
end

---@param a integer
---@param b integer
---@return integer
local function manhattan(a, b)
    local ax, ay = grid_xy(a)
    local bx, by = grid_xy(b)
    local dx = ax - bx
    if dx < 0 then dx = -dx end
    local dy = ay - by
    if dy < 0 then dy = -dy end
    return dx + dy
end

local out = {}
for i = 1, NUM_CELLS * FEATS_PER_CELL do
    out[i] = 0.0
end

local function clear_out()
    for i = 1, #out do
        out[i] = 0.0
    end
end

---@param cell_idx integer
---@param known number
---@param room_type number
---@param visited number
---@param current_room number
---@param has_unopened_door number
---@param dist number
local function set_cell(cell_idx, known, room_type, visited, current_room, has_unopened_door, dist)
    local base = (cell_idx * FEATS_PER_CELL) + 1
    if known == 0 then
        -- if known == 0, we have yet not know the room exists: everything else must be 0
        out[base]     = 0.0
        out[base + 1] = 0.0
        out[base + 2] = 0.0
        out[base + 3] = 0.0
        out[base + 4] = 0.0
        out[base + 5] = 0.0
        return
    end

    out[base]     = known
    out[base + 1] = room_type
    out[base + 2] = visited
    out[base + 3] = current_room
    out[base + 4] = has_unopened_door
    out[base + 5] = dist
end

---@param level Level
---@param grid_index integer
---@return RoomDescriptor|nil
local function get_room_desc(level, grid_index)
    local ok, desc = pcall(function()
        return level:GetRoomByIdx(grid_index)
    end)
    if not ok then
        return nil
    end
    -- Some invalid descriptors may exist with Data == nil.
    if desc == nil or desc.Data == nil then
        return nil
    end
    return desc
end

---@param desc RoomDescriptor
---@return boolean
local function is_known(desc)
    -- "Known" if it appears on minimap as visible OR shadow, or if visited.
    -- DisplayFlags: Visible (1<<0), Room Shadow (1<<1) :contentReference[oaicite:3]{index=3}
    if desc.VisitedCount ~= nil and desc.VisitedCount > 0 then
        return true
    end
    local flags = desc.DisplayFlags or 0
    return ((flags & DISPLAY_VISIBLE) ~= 0) or ((flags & DISPLAY_SHADOW) ~= 0)
end

---@param desc RoomDescriptor
---@return boolean
local function is_visited(desc)
    return (desc.VisitedCount ~= nil and desc.VisitedCount > 0)
end

---@param desc RoomDescriptor
---@return integer
local function safe_index(desc)
    -- SafeGridIndex is recommended for stable indexing across shapes :contentReference[oaicite:4]{index=4}
    return desc.SafeGridIndex or desc.GridIndex
end

---@param level Level
---@param desc RoomDescriptor
---@return boolean
local function has_unopened_exit(level, desc)
    -- We define "has_unopened_door" as: this room has a valid door position
    -- that leads to a room that exists but has not been visited yet.
    --
    -- We only compute this for known rooms; caller ensures that.
    local data = desc.Data
    if data == nil then
        return false
    end

    local doors_mask = data.Doors or 0
    local idx = safe_index(desc)

    -- Adjacent offsets for LEFT/UP/RIGHT/DOWN in the 13x13 grid
    -- (DoorSlotFlag includes LEFT0..DOWN1; map both 0/4 to LEFT, 1/5 to UP, 2/6 to RIGHT, 3/7 to DOWN)
    local function check_neighbor(nei_idx)
        if nei_idx < 0 or nei_idx >= NUM_CELLS then
            return false
        end
        local nei_desc = get_room_desc(level, nei_idx)
        if nei_desc == nil then
            return false
        end
        return not is_visited(nei_desc)
    end

    -- LEFT0 (bit 0) or LEFT1 (bit 4)
    if ((doors_mask & door_mask(0)) ~= 0) or ((doors_mask & door_mask(4)) ~= 0) then
        if check_neighbor(idx - 1) then return true end
    end
    -- UP0 (bit 1) or UP1 (bit 5)
    if ((doors_mask & door_mask(1)) ~= 0) or ((doors_mask & door_mask(5)) ~= 0) then
        if check_neighbor(idx - GRID_W) then return true end
    end
    -- RIGHT0 (bit 2) or RIGHT1 (bit 6)
    if ((doors_mask & door_mask(2)) ~= 0) or ((doors_mask & door_mask(6)) ~= 0) then
        if check_neighbor(idx + 1) then return true end
    end
    -- DOWN0 (bit 3) or DOWN1 (bit 7)
    if ((doors_mask & door_mask(3)) ~= 0) or ((doors_mask & door_mask(7)) ~= 0) then
        if check_neighbor(idx + GRID_W) then return true end
    end

    return false
end

---@param _player EntityPlayer
---@return number[]
function C.Encode(_player)
    clear_out()

    local level = Game():GetLevel()
    local cur_desc = level:GetCurrentRoomDesc()
    local cur_idx = safe_index(cur_desc)

    -- Iterate over all possible grid cells (0..168)
    for grid_idx = 0, (NUM_CELLS - 1) do
        local desc = get_room_desc(level, grid_idx)

        if desc == nil then
            -- No room descriptor here -> treat as "unknown/empty" until revealed.
            -- Since known=0 forces all zeros, room_type (-1) is not emitted unless known.
            set_cell(grid_idx, 0, 0, 0, 0, 0, 0)
        else
            local known = is_known(desc) and 1.0 or 0.0
            if known == 0.0 then
                set_cell(grid_idx, 0, 0, 0, 0, 0, 0)
            else
                local visited = is_visited(desc) and 1.0 or 0.0
                local this_idx = safe_index(desc)
                local current = (this_idx == cur_idx) and 1.0 or 0.0

                -- room_type label encoding: use the RoomType enum integer from the room config
                -- (stored in RoomDescriptor.Data.Type) :contentReference[oaicite:5]{index=5}
                local room_type = -1.0
                if desc.Data ~= nil and desc.Data.Type ~= nil then
                    room_type = desc.Data.Type
                end

                local unopened = has_unopened_exit(level, desc) and 1.0 or 0.0
                local dist = manhattan(this_idx, cur_idx)

                set_cell(grid_idx, 1.0, room_type, visited, current, unopened, dist)
            end
        end
    end

    return out
end

return C
