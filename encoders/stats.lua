local S = {}

---@param player EntityPlayer
---@return number[]
function S.Encode(player)
    local stats = {}

    stats[1] = player.MoveSpeed
    local max_fire_delay = player.MaxFireDelay
    stats[2] = 30.0 / (max_fire_delay + 1.0)
    stats[3] = player.Damage
    stats[4] = player.TearRange
    stats[5] = player.ShotSpeed
    stats[6] = player.Luck

    return stats
end

return S
