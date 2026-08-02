local Config = include("config")

-- Room:GetRandomPosition(Margin) samples a random position with a radius of
-- Margin units away from any obstacles directly -- no need for manual grid
-- sampling + FindFreeTilePosition retry logic. It has no RNG parameter, so
-- unlike the rest of this mod's randomization, this one draw isn't seeded
-- from the per-visit RNG chain; accepted as a pragmatic tradeoff for using
-- the simpler, more robust built-in over hand-rolled grid math.
---@class Positioning
local Positioning = {}

---@param room Room
---@return Vector
function Positioning.SampleValidPosition(room)
    return room:GetRandomPosition(Config.ROOM_POSITION_MARGIN)
end

-- Samples a random position at a random angle, [min_radius, max_radius]
-- world units from `center`, then clamps to the room's actual bounds (same
-- margin-based clamping scenario.viewports uses for its lerp-grid points)
-- -- used to keep a duo character's second player (Jacob & Esau, Tainted
-- Forgotten) positioned close to the first each room visit, rather than
-- anywhere valid in the room the way SampleValidPosition samples. Not
-- collision-aware, unlike SampleValidPosition -- Room:GetRandomPosition's
-- built-in obstacle avoidance has no equivalent "avoid this rock, but stay
-- near this point" variant, so the result can in principle land on/inside
-- an obstacle tile. Accepted as a minor visual imperfection: this only sets
-- a teleport target, not a physics-validated move.
---@param room Room
---@param center Vector
---@param min_radius number
---@param max_radius number
---@param rng RNG
---@return Vector
function Positioning.SampleNearbyPosition(room, center, min_radius, max_radius, rng)
    local angle = rng:RandomFloat() * 2 * math.pi
    local radius = min_radius + rng:RandomFloat() * (max_radius - min_radius)
    local candidate = center + Vector(math.cos(angle) * radius, math.sin(angle) * radius)
    return room:GetClampedPosition(candidate, Config.ROOM_POSITION_MARGIN)
end

return Positioning
