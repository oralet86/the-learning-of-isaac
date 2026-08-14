local Log = include("log")

---@class Pose
local Pose = {}

-- The player's body sprite (001.000_player.anm2, or a character-specific
-- body anm2 like Azazel's) only defines two animation families relevant to
-- posing -- verified against the game's own anm2 files: WalkUp/Down/Left/
-- Right (looping walk cycles, also used as the idle animation) and
-- HeadUp/Down/Left/Right (short aim animations -- the idle body with the
-- head turned toward the aim direction, which is exactly the in-game
-- "shooting" look). There are no Idle* or Shoot* animations.
local WALK_ANIMS = { up = "WalkUp", down = "WalkDown", left = "WalkLeft", right = "WalkRight" }
local AIM_ANIMS = { up = "HeadUp", down = "HeadDown", left = "HeadLeft", right = "HeadRight" }

-- The 8 pose keys rolled per room visit: 4-way walking + 4-way shooting.
local POSE_KEYS = {
    "walk_up", "walk_down", "walk_left", "walk_right",
    "shoot_up", "shoot_down", "shoot_left", "shoot_right",
}

-- Direction enum values for the Repentogon SetHeadDirection head lock
-- (same values as the vanilla Direction enum: LEFT=0, UP=1, RIGHT=2,
-- DOWN=3). Only used when Repentogon is loaded; plain vanilla play is
-- unaffected.
local HEAD_DIRECTIONS = { up = 1, down = 3, left = 0, right = 2 }

-- Per-sprite resolution cache: sprite anm2 path + pose key -> resolved
-- {name=..., frames=...}, or false when the sprite has no usable animation
-- for that pose. Characters load different body anm2s (e.g. the Soul has
-- no walk animation at all), so resolutions are probed once and cached per
-- sprite for the session.
local resolved_cache = {}

-- Probing an animation's existence and frame count: the sprite API offers
-- no direct "does animation X exist / how long is it" query, so this plays
-- the animation and then round-trips SetFrame/GetFrame to count its valid
-- frames -- clamping or an error on an out-of-range frame both read back
-- as a frame mismatch. Returns the frame count (at least 1), or nil when
-- the animation doesn't exist on the sprite.
---@param sprite Sprite
---@param anim_name string
---@return integer?
local function probe_animation(sprite, anim_name)
    local play_ok, _ = pcall(sprite.Play, sprite, anim_name, true)
    if not play_ok or not sprite:IsPlaying(anim_name) then
        return nil
    end
    for frame = 0, 127 do
        local set_ok, _ = pcall(sprite.SetFrame, sprite, frame)
        if not set_ok or sprite:GetFrame() ~= frame then
            return math.max(1, frame)
        end
    end
    return 128
end

-- Resolves a pose key to the concrete animation name and frame count for a
-- given sprite, probing and caching per (sprite, pose). Walk poses fall
-- back to the matching aim animation for characters without a walk
-- animation (e.g. the Soul); unresolvable poses resolve to nil, telling
-- the caller to leave the sprite alone that tick.
---@param sprite Sprite
---@param pose_key string
---@return {name: string, frames: integer}?
local function resolve(sprite, pose_key)
    local cache_key = sprite:GetFilename() .. "|" .. pose_key
    local cached = resolved_cache[cache_key]
    if cached ~= nil then
        if cached == false then
            return nil
        end
        return cached
    end

    local kind, direction = pose_key:match("^(%w+)_(%w+)$")
    local primary = kind == "walk" and WALK_ANIMS[direction] or AIM_ANIMS[direction]
    local primary_frames = probe_animation(sprite, primary)
    if primary_frames ~= nil then
        local entry = { name = primary, frames = primary_frames }
        resolved_cache[cache_key] = entry
        return entry
    end

    if kind == "walk" then
        local fallback = AIM_ANIMS[direction]
        local fallback_frames = probe_animation(sprite, fallback)
        if fallback_frames ~= nil then
            Log.Info(
                "pose: " .. primary .. " missing on " .. sprite:GetFilename() .. ", falling back to " .. fallback
            )
            local entry = { name = fallback, frames = fallback_frames }
            resolved_cache[cache_key] = entry
            return entry
        end
    end

    Log.Info("pose: no usable animation for " .. pose_key .. " on " .. sprite:GetFilename() .. ", leaving sprite untouched")
    resolved_cache[cache_key] = false
    return nil
end

-- Rolls one of the 8 pose keys uniformly.
---@param rng RNG
---@return string
function Pose.RollPose(rng)
    return POSE_KEYS[rng:RandomInt(#POSE_KEYS) + 1]
end

-- Forces the player's body sprite onto the given pose's animation and
-- holds a specific cycle frame. Must be reasserted every tick by the
-- caller, because the engine recomputes the player's animation each frame;
-- the player never actually moves or fires, only the rendered sprite
-- changes, so nothing about the scene or ground truth is affected. When
-- Repentogon is loaded, the engine's separate character-head overlay
-- layers are locked to the same direction so they stay consistent with the
-- forced pose.
---@param player EntityPlayer
---@param pose_key string
---@param frame integer Frame index into the pose's animation cycle; wrapped modulo the animation's actual frame count.
function Pose.Apply(player, pose_key, frame)
    local sprite = player:GetSprite()
    local resolved = resolve(sprite, pose_key)
    if resolved == nil then
        return
    end
    sprite:SetFrame(resolved.name, frame % resolved.frames)

    local direction = pose_key:match("^%w+_(%w+)$")
    if player.SetHeadDirection ~= nil then
        player:SetHeadDirection(HEAD_DIRECTIONS[direction], 10, true)
    end
end

return Pose
