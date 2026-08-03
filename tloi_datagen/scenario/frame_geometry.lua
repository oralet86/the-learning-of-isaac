-- Pure crop-geometry math, no Isaac-engine dependency (only reads .X/.Y
-- fields off whatever position/center/half-extent tables are passed in, so
-- a plain {X=, Y=} table works exactly like a real Vector). Shared by every
-- ground_truth.lua caller that needs to know where an entity sits within
-- the *captured frame* rather than the room -- see
-- tloi_datagen/MOD_CHANGES_NEEDED.md items 2 and 6: player, player2, and
-- each enemy all use this same formula against the same per-room-visit
-- viewport_center/viewport_half_extent (from scenario.viewports).
---@class FrameGeometry
local FrameGeometry = {}

---@param value number
---@return number
local function clamp01(value)
    if value < 0 then
        return 0
    end
    if value > 1 then
        return 1
    end
    return value
end

-- Maps a world-space position to its normalized [0, 1] location within the
-- current camera viewport's frame (0,0 = top-left of frame, 1,1 =
-- bottom-right), clamped so an entity outside the frame's bounding box
-- still lands at the nearest edge rather than going negative/>1.
---@param position {X: number, Y: number}
---@param viewport_center {X: number, Y: number}
---@param viewport_half_extent {X: number, Y: number}
---@return {x: number, y: number}
function FrameGeometry.NormalizedPosition(position, viewport_center, viewport_half_extent)
    local x = (position.X - (viewport_center.X - viewport_half_extent.X)) / (2 * viewport_half_extent.X)
    local y = (position.Y - (viewport_center.Y - viewport_half_extent.Y)) / (2 * viewport_half_extent.Y)
    return { x = clamp01(x), y = clamp01(y) }
end

return FrameGeometry
