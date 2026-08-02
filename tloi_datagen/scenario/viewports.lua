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

-- 2x2 and every L-shape rotation share the same "3x3 lerp grid, drop
-- invalid points" treatment -- for a 2x2 room every point is valid; for an
-- L-shape, IsPositionInRoom filters out the missing quadrant's corner and
-- its two adjacent edge points, leaving points that sit in the two valid
-- arms plus one near the elbow/junction.
local QUADRANT_GRID_SHAPES = {
    [RoomShape.ROOMSHAPE_2x2] = true,
    [RoomShape.ROOMSHAPE_LTL] = true,
    [RoomShape.ROOMSHAPE_LTR] = true,
    [RoomShape.ROOMSHAPE_LBL] = true,
    [RoomShape.ROOMSHAPE_LBR] = true,
}

---@param a number
---@param b number
---@param t number
---@return number
local function lerp(a, b, t)
    return a + (b - a) * t
end

-- Extreme lerp fraction used for corners/edges/ends, inset slightly from
-- the exact 0/1 boundary -- t=0/1 lands exactly on GetTopLeftPos()/
-- GetBottomRightPos(), which is right at the edge of what
-- Room:IsPositionInRoom()'s margin check will accept (confirmed in-engine:
-- every quadrant-grid candidate can fail this check for some room
-- layouts). A small inset keeps these points meaningfully "near the
-- corner/edge/end" while giving the margin check room to pass.
local EXTREME_T = 0.1

-- Named 3x3 grid of (tx, ty) lerp fractions between top-left and
-- bottom-right, used for both the 2x2 and L-shape buckets. Order matters
-- only for readability of logs/ground-truth, not correctness.
local QUADRANT_POINTS = {
    { name = "corner_tl", tx = EXTREME_T, ty = EXTREME_T },
    { name = "edge_top", tx = 0.5, ty = EXTREME_T },
    { name = "corner_tr", tx = 1 - EXTREME_T, ty = EXTREME_T },
    { name = "edge_left", tx = EXTREME_T, ty = 0.5 },
    { name = "center", tx = 0.5, ty = 0.5 },
    { name = "edge_right", tx = 1 - EXTREME_T, ty = 0.5 },
    { name = "corner_bl", tx = EXTREME_T, ty = 1 - EXTREME_T },
    { name = "edge_bottom", tx = 0.5, ty = 1 - EXTREME_T },
    { name = "corner_br", tx = 1 - EXTREME_T, ty = 1 - EXTREME_T },
}

---@param room Room
---@param top_left Vector
---@param bottom_right Vector
---@return {name: string, position: Vector}[]
local function quadrant_grid_positions(room, top_left, bottom_right)
    local positions = {}
    for _, point in ipairs(QUADRANT_POINTS) do
        local candidate = Vector(lerp(top_left.X, bottom_right.X, point.tx), lerp(top_left.Y, bottom_right.Y, point.ty))
        if room:IsPositionInRoom(candidate, Config.ROOM_POSITION_MARGIN) then
            positions[#positions + 1] = {
                name = point.name,
                position = room:GetClampedPosition(candidate, Config.ROOM_POSITION_MARGIN),
            }
        end
    end
    return positions
end

---@param room Room
---@param top_left Vector
---@param bottom_right Vector
---@return {name: string, position: Vector}[]
local function long_axis_positions(room, top_left, bottom_right)
    local horizontal = room:GetGridWidth() >= room:GetGridHeight()
    local names = { "end_a", "mid", "end_b" }
    local positions = {}
    for i, t in ipairs({ EXTREME_T, 0.5, 1 - EXTREME_T }) do
        local candidate
        if horizontal then
            candidate = Vector(lerp(top_left.X, bottom_right.X, t), lerp(top_left.Y, bottom_right.Y, 0.5))
        else
            candidate = Vector(lerp(top_left.X, bottom_right.X, 0.5), lerp(top_left.Y, bottom_right.Y, t))
        end
        positions[#positions + 1] = {
            name = names[i],
            position = room:GetClampedPosition(candidate, Config.ROOM_POSITION_MARGIN),
        }
    end
    return positions
end

-- 1x1 rooms always fit entirely in-frame regardless of where the player
-- stands, so framing itself never changes across these positions -- unlike
-- the long-axis/quadrant-grid buckets above, this is purely for
-- player-position/pose diversity within an otherwise-identical capture.
-- First position is always the room center for a deterministic baseline
-- capture; the rest are valid random positions via
-- Positioning.SampleValidPosition, one call per extra position.
---@param room Room
---@return {name: string, position: Vector}[]
local function single_room_positions(room)
    local positions = { { name = "center", position = room:GetCenterPos() } }
    for i = 2, Config.VIEWPORTS_PER_1X1_ROOM do
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

    local positions
    if shape == RoomShape.ROOMSHAPE_1x1 then
        positions = single_room_positions(room)
    elseif LONG_AXIS_SHAPES[shape] then
        positions = long_axis_positions(room, top_left, bottom_right)
    elseif QUADRANT_GRID_SHAPES[shape] then
        positions = quadrant_grid_positions(room, top_left, bottom_right)
    end

    -- Any unrecognized/future shape, or a quadrant-grid room where every
    -- candidate got filtered out by IsPositionInRoom (confirmed happens
    -- in-engine -- corner/edge lerp points sit right at the wall boundary
    -- and can fail the margin check for some room layouts): fall back to a
    -- single guaranteed-valid center position rather than returning an
    -- empty list.
    if positions == nil or #positions == 0 then
        positions = { { name = "center", position = room:GetCenterPos() } }
    end

    return positions, half_extent
end

return Viewports
