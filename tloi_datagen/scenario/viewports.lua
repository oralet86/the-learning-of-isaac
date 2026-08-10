local Config = include("config")
local Positioning = include("scenario.positioning")

-- The in-game camera always follows the player and is clamped to room
-- bounds by the engine itself -- there is no separate camera object to
-- move. "Camera viewport position" here just means "where to put the
-- player before capturing", computed generically from room bounds rather
-- than per-shape hand-rolled geometry.
---@class Viewports
local Viewports = {}

-- Room shapes that are a single elongated axis (either orientation) rather
-- than a full quadrant grid. IH/IV/IIV/IIH are lower-usage variants of the
-- same "long room" concept as 1x2/2x1 -- included defensively since
-- TeleportToRandomRoom no longer filters by shape, but whether all of
-- these actually spawn as normal traversable rooms (vs. internal-only) is
-- unverified; confirm empirically and prune this list if some never
-- appear.
local LONG_AXIS_SHAPES = {
    [RoomShape.ROOMSHAPE_IH] = true,
    [RoomShape.ROOMSHAPE_IV] = true,
    [RoomShape.ROOMSHAPE_1x2] = true,
    [RoomShape.ROOMSHAPE_IIV] = true,
    [RoomShape.ROOMSHAPE_2x1] = true,
    [RoomShape.ROOMSHAPE_IIH] = true,
}

-- 2x2 and every L-shape rotation share the same "4 unit-area" bounding box
-- -- an L-shape is missing one quadrant, but SampleValidPosition only ever
-- lands in the room's actual valid area regardless, so no separate
-- treatment is needed for the missing quadrant here.
local QUADRANT_GRID_SHAPES = {
    [RoomShape.ROOMSHAPE_2x2] = true,
    [RoomShape.ROOMSHAPE_LTL] = true,
    [RoomShape.ROOMSHAPE_LTR] = true,
    [RoomShape.ROOMSHAPE_LBL] = true,
    [RoomShape.ROOMSHAPE_LBR] = true,
}

-- L-shape rotations specifically: the bounding-box center is the inner
-- corner, which sits inside the missing quadrant's wall -- Room:GetCenterPos()
-- is not a valid floor position for these, so they get a sampled valid
-- position for the baseline "center" viewport instead (see
-- random_room_positions).
local L_SHAPE_SHAPES = {
    [RoomShape.ROOMSHAPE_LTL] = true,
    [RoomShape.ROOMSHAPE_LTR] = true,
    [RoomShape.ROOMSHAPE_LBL] = true,
    [RoomShape.ROOMSHAPE_LBR] = true,
}

-- How many "1x1 units" of area a room shape covers -- used to scale how
-- many viewport positions get sampled (see random_room_positions) so
-- bigger rooms get proportionally more position diversity than a 1x1.
-- Unrecognized/future shapes default to 1 unit.
---@param shape integer
---@return integer
local function room_unit_area(shape)
    if LONG_AXIS_SHAPES[shape] then
        return 2
    end
    if QUADRANT_GRID_SHAPES[shape] then
        return 4
    end
    return 1
end

-- Single random-sampling hot path for every room shape: framing itself
-- doesn't change across positions for a 1x1 room, but for larger shapes it
-- does -- either way this is purely about player-position/pose diversity
-- within an otherwise-identical capture, and letting Room:GetRandomPosition
-- pick anywhere valid in the room covers that better than a fixed lerp
-- grid. First position is always the room center for a deterministic
-- baseline capture -- except for L-shape rooms, where the geometric center
-- is inside the missing quadrant's wall, so a sampled valid position is
-- used instead (see L_SHAPE_SHAPES).
---@param room Room
---@param unit_area integer
---@param shape integer
---@return {name: string, position: Vector}[]
local function random_room_positions(room, unit_area, shape)
    local count = Config.VIEWPORTS_PER_ROOM_UNIT * unit_area
    local first_position = room:GetCenterPos()
    if L_SHAPE_SHAPES[shape] then
        first_position = Positioning.SampleValidPosition(room)
    end
    local positions = { { name = "center", position = first_position } }
    for i = 2, count do
        positions[#positions + 1] = { name = "pos_" .. tostring(i), position = Positioning.SampleValidPosition(room) }
    end
    return positions
end

-- Half-extent (in world units) of a single 1x1 room's worth of view --
-- i.e. half the bounding box, divided by how many "1x1 units" wide/tall
-- the current room's bounding box spans. 1x1 itself is 1 unit in each
-- axis; every other bucket handled here (long-axis, quadrant-grid) spans
-- 2 units along its extended axis/axes. Used by ground_truth.lua to
-- filter enemies to what's actually in-frame for a given viewport.
---@param room Room
---@param shape integer
---@param top_left Vector
---@param bottom_right Vector
---@return Vector
local function unit_half_extent(room, shape, top_left, bottom_right)
    local full_width = bottom_right.X - top_left.X
    local full_height = bottom_right.Y - top_left.Y

    if shape == RoomShape.ROOMSHAPE_1x1 then
        return Vector(full_width / 2, full_height / 2)
    end

    if LONG_AXIS_SHAPES[shape] then
        local horizontal = room:GetGridWidth() >= room:GetGridHeight()
        if horizontal then
            return Vector(full_width / 4, full_height / 2)
        end
        return Vector(full_width / 2, full_height / 4)
    end

    -- Quadrant-grid shapes (2x2, L-shapes): 2 units wide and tall.
    return Vector(full_width / 4, full_height / 4)
end

-- Returns the ordered list of representative camera-viewport positions for
-- the room's actual shape, plus the half-extent of a single 1x1-room's
-- worth of view around any of those positions (for frame-filtering
-- ground-truth queries against).
---@param room Room
---@return {name: string, position: Vector}[]
---@return Vector half_extent
function Viewports.GetPositions(room)
    local shape = room:GetRoomShape()
    local top_left = room:GetTopLeftPos()
    local bottom_right = room:GetBottomRightPos()
    local half_extent = unit_half_extent(room, shape, top_left, bottom_right)
    local positions = random_room_positions(room, room_unit_area(shape), shape)

    return positions, half_extent
end

return Viewports
