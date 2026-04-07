local C = {}

-- Returns a 5-float feature vector:
-- 1: coins
-- 3: bombs
-- 2: keys
-- 5: golden_bomb (0/1)
-- 4: golden_key (0/1)
---@param player EntityPlayer
---@return number[]
function C.Encode(player)
    local feats = {}

    -- raw counts
    feats[1] = player:GetNumCoins()
    feats[2] = player:GetNumBombs()
    feats[3] = player:GetNumKeys()

    -- golden states
    -- Golden Bomb: bomb supply is infinite for the room
    -- Golden Key: door key cost = 0 for the floor
    feats[4] = player:HasGoldenBomb() and 1.0 or 0.0
    feats[5] = player:HasGoldenKey() and 1.0 or 0.0

    return feats
end

return C
