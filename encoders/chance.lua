local C = {}
local game = Game()

---@param collectible CollectibleType
---@return boolean
local function anyPlayerHasCollectible(collectible)
    for i = 0, game:GetNumPlayers() - 1 do
        local player = game:GetPlayer(i)
        if player:HasCollectible(collectible, false) then
            return true
        end
    end
    return false
end

---@param trinket TrinketType
---@return boolean
local function anyPlayerHasTrinket(trinket)
    for i = 0, game:GetNumPlayers() - 1 do
        local player = game:GetPlayer(i)
        if player:HasTrinket(trinket, false) then
            return true
        end
    end
    return false
end

---@param level Level
---@return boolean
local function canDealSpawn(level)
    local stage = level:GetStage()

    if stage == LevelStage.STAGE1_1 and game.Difficulty ~= Difficulty.DIFFICULTY_GREED and game.Difficulty ~= Difficulty.DIFFICULTY_GREEDIER then
        return false
    end
    if stage == LevelStage.STAGE4_3 then return false end
    if stage >= LevelStage.STAGE5 then return false end
    if stage == LevelStage.STAGE7_GREED then return false end

    return true
end

---@param player EntityPlayer
---@return number[]
function C.Encode(player)
    local level = game:GetLevel()
    local room = level:GetCurrentRoom()

    if not canDealSpawn(level) then
        return { 0.0, 0.0 }
    end

    local totalChance = math.min(room:GetDevilRoomChance(), 1.0)

    if totalChance <= 0.0001 then
        return { 0.0, 0.0 }
    end

    if anyPlayerHasCollectible(CollectibleType.COLLECTIBLE_DUALITY) then
        return { totalChance, totalChance }
    end

    local angelRoomSpawned = game:GetStateFlag(GameStateFlag.STATE_FAMINE_SPAWNED)
    local devilRoomSpawned = game:GetStateFlag(GameStateFlag.STATE_DEVILROOM_SPAWNED)
    local devilRoomVisited = game:GetStateFlag(GameStateFlag.STATE_DEVILROOM_VISITED)

    local devilSplit = 1.0

    if anyPlayerHasCollectible(CollectibleType.COLLECTIBLE_EUCHARIST) then
        devilSplit = 0.0
    elseif devilRoomSpawned and devilRoomVisited and game:GetDevilRoomDeals() > 0 then
        if anyPlayerHasCollectible(CollectibleType.COLLECTIBLE_BOOK_OF_VIRTUES) or
            anyPlayerHasCollectible(CollectibleType.COLLECTIBLE_ACT_OF_CONTRITION) or
            level:GetAngelRoomChance() > 0.0 then
            devilSplit = 0.5
        end
    elseif devilRoomSpawned or anyPlayerHasCollectible(CollectibleType.COLLECTIBLE_BOOK_OF_VIRTUES) or level:GetAngelRoomChance() > 0.0 then
        if not (devilRoomVisited or angelRoomSpawned) then
            devilSplit = 0.0
        else
            devilSplit = 0.5
        end
    end

    if devilSplit == 0.5 then
        if anyPlayerHasTrinket(TrinketType.TRINKET_ROSARY_BEAD) then
            devilSplit = devilSplit * 0.5
        end
        if game:GetDonationModAngel() >= 10 then
            devilSplit = devilSplit * 0.5
        end
        if anyPlayerHasCollectible(CollectibleType.COLLECTIBLE_KEY_PIECE_1) then
            devilSplit = devilSplit * 0.75
        end
        if anyPlayerHasCollectible(CollectibleType.COLLECTIBLE_KEY_PIECE_2) then
            devilSplit = devilSplit * 0.75
        end
        if level:GetStateFlag(LevelStateFlag.STATE_EVIL_BUM_KILLED) then
            devilSplit = devilSplit * 0.75
        end
        if level:GetStateFlag(LevelStateFlag.STATE_BUM_LEFT) and not level:GetStateFlag(LevelStateFlag.STATE_EVIL_BUM_LEFT) then
            devilSplit = devilSplit * 0.9
        end
        if level:GetStateFlag(LevelStateFlag.STATE_EVIL_BUM_LEFT) and not level:GetStateFlag(LevelStateFlag.STATE_BUM_LEFT) then
            devilSplit = devilSplit * 1.1
        end

        local angelChanceVal = level:GetAngelRoomChance()
        if angelChanceVal > 0.0 or (angelChanceVal < 0.0 and (anyPlayerHasCollectible(CollectibleType.COLLECTIBLE_BOOK_OF_VIRTUES) or anyPlayerHasCollectible(CollectibleType.COLLECTIBLE_ACT_OF_CONTRITION))) then
            devilSplit = devilSplit * (1.0 - angelChanceVal)
        end

        if anyPlayerHasCollectible(CollectibleType.COLLECTIBLE_BOOK_OF_VIRTUES) then
            devilSplit = devilSplit * 0.75
        end

        devilSplit = math.max(0.0, math.min(devilSplit, 1.0))
    end

    return {
        totalChance * devilSplit,
        totalChance * (1.0 - devilSplit)
    }
end

return C
