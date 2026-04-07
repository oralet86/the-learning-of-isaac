local C = {}
local game = Game()

---@param player EntityPlayer
---@return integer
local function countLocustsForPlayer(player)
    -- Abyss locusts are familiars with variant ABYSS_LOCUST.
    -- We filter by owning player via GetPtrHash.
    local entities = Isaac.FindByType(
        EntityType.ENTITY_FAMILIAR,
        FamiliarVariant.ABYSS_LOCUST,
        -1, -- any subtype
        false,
        false
    )

    local count = 0
    local player_hash = GetPtrHash(player)

    for _, ent in ipairs(entities) do
        local fam = ent:ToFamiliar()
        if fam and fam.Player and GetPtrHash(fam.Player) == player_hash then
            count = count + 1
        end
    end

    return count
end

---@return number[]
local function initFeatures()
    local f = {}
    for i = 1, 3 do
        f[i] = 0.0
    end
    return f
end

---@param player EntityPlayer
---@return number[]
function C.Encode(player)
    -- Layout:
    -- 1: num_blue_flies
    -- 2: num_blue_spiders
    -- 3: num_locusts (Abyss locust familiars belonging to this player)
    local feat = initFeatures()

    feat[1] = player:GetNumBlueFlies()
    feat[2] = player:GetNumBlueSpiders()
    feat[3] = countLocustsForPlayer(player)

    return feat
end

return C
