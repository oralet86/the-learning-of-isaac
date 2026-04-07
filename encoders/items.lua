local C = {}

-- Fixed collectible multi-hot size
local MAX_COLLECTIBLE_ID = 750
local COLLECTIBLE_VEC_LEN = MAX_COLLECTIBLE_ID

-- Transformations (PlayerForm)
local NUM_FORMS = PlayerForm.NUM_PLAYER_FORMS or 15

-- Cached state (collectibles)
local owned_collectibles = {}
for i = 1, COLLECTIBLE_VEC_LEN do
    owned_collectibles[i] = 0.0
end

local last_collectible_count = -1
local last_queue_empty = true
local last_queued_collectible_id = 0

---@param player EntityPlayer
local function full_resync(player)
    -- Clear cache
    for i = 1, COLLECTIBLE_VEC_LEN do
        owned_collectibles[i] = 0.0
    end

    -- Resync by scanning 1..750 only (fixed, stable)
    for id = 1, MAX_COLLECTIBLE_ID do
        local n = player:GetCollectibleNum(id)
        if n ~= nil and n > 0 then
            owned_collectibles[id] = 1.0
        end
    end
end

---@param player EntityPlayer
local function update_collectible_cache(player)
    local count = player:GetCollectibleCount()
    local queue_empty = player:IsItemQueueEmpty()

    -- Track queued collectible while item queue is non-empty
    if not queue_empty then
        local q = player.QueuedItem
        if q ~= nil and q.Item ~= nil and q.Item:IsCollectible() then
            last_queued_collectible_id = q.Item.ID
        end
    end

    -- Queue just finished: pickup completed
    if (not last_queue_empty) and queue_empty then
        local id = last_queued_collectible_id
        if id ~= nil and id >= 1 and id <= MAX_COLLECTIBLE_ID then
            owned_collectibles[id] = 1.0
        end
        last_queued_collectible_id = 0
    end

    -- If the count changed without a queue transition, inventory likely changed (drop/reroll/remove).
    -- Do a rare full resync.
    if last_collectible_count == -1 then
        full_resync(player)
    else
        if count ~= last_collectible_count and (queue_empty and last_queue_empty) then
            full_resync(player)
        end
    end

    last_collectible_count = count
    last_queue_empty = queue_empty
end

-- Output buffer (reused)
-- Layout:
-- [1 .. 750]                         collectibles multi-hot
-- [751]                              active_item_id
-- [752]                              active_charge_norm
-- [753]                              trinket_id (slot 0)
-- [754 .. 753 + NUM_FORMS]           transformations multi-hot
local OUT_LEN = COLLECTIBLE_VEC_LEN + 3 + NUM_FORMS
local out = {}
for i = 1, OUT_LEN do
    out[i] = 0.0
end

---@param x number
---@return number
local function clamp01(x)
    if x < 0 then return 0.0 end
    if x > 1 then return 1.0 end
    return x
end

---@param player EntityPlayer
---@return number[]
function C.Encode(player)
    update_collectible_cache(player)

    -- 1) collectibles multi-hot
    for i = 1, COLLECTIBLE_VEC_LEN do
        out[i] = owned_collectibles[i]
    end

    -- 2) active item id
    local active_id = player:GetActiveItem()
    if active_id == nil then active_id = 0 end
    out[COLLECTIBLE_VEC_LEN + 1] = active_id

    -- 3) active charge normalized by MaxCharges (if known)
    local charge = player:GetActiveCharge()
    if charge == nil then charge = 0 end

    local max_charges = 0
    if active_id ~= nil and active_id > 0 then
        local cfg = Isaac.GetItemConfig():GetCollectible(active_id)
        if cfg ~= nil and cfg.MaxCharges ~= nil then
            max_charges = cfg.MaxCharges
        end
    end

    local charge_norm = 0.0
    if max_charges ~= nil and max_charges > 0 then
        charge_norm = clamp01(charge / max_charges)
    end
    out[COLLECTIBLE_VEC_LEN + 2] = charge_norm

    -- 4) trinket id (slot 0)
    local trinket_id = player:GetTrinket(0)
    if trinket_id == nil then trinket_id = 0 end
    out[COLLECTIBLE_VEC_LEN + 3] = trinket_id

    -- 5) transformations multi-hot
    local base = COLLECTIBLE_VEC_LEN + 3
    for form = 0, (NUM_FORMS - 1) do
        out[base + 1 + form] = player:HasPlayerForm(form) and 1.0 or 0.0
    end

    return out
end

return C
