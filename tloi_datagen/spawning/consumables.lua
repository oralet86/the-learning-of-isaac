local Config = include("config")
local Log = include("log")

---@class Consumables
local Consumables = {}

-- Built lazily and cached -- Card is a single unified id space covering
-- tarot cards, playing cards, runes, and soul stones, so one pool covers
-- all of "pills/cards/runes" from original.md rather than needing separate
-- rune handling. 0 is CARD_NULL and -1 is CARD_RANDOM; both excluded.
---@type integer[]?
local card_pool_cache = nil

---@return integer[]
local function get_card_pool()
    if card_pool_cache ~= nil then
        return card_pool_cache
    end

    local pool = {}
    for id = 1, Card.NUM_CARDS - 1 do
        pool[#pool + 1] = id
    end

    card_pool_cache = pool
    Log.Info("get_card_pool: built pool of " .. #pool .. " cards")
    return pool
end

-- Built lazily and cached, same pattern as get_card_pool. 0 is PILL_NULL.
-- PILL_GIANT_FLAG/PILL_COLOR_MASK are bitmask helper constants for
-- horsepills, not real pill ids, so they're excluded.
---@type integer[]?
local pill_pool_cache = nil

---@return integer[]
local function get_pill_pool()
    if pill_pool_cache ~= nil then
        return pill_pool_cache
    end

    local pool = {}
    for id = 1, PillColor.NUM_PILLS - 1 do
        if id ~= PillColor.PILL_GIANT_FLAG and id ~= PillColor.PILL_COLOR_MASK then
            pool[#pool + 1] = id
        end
    end

    pill_pool_cache = pool
    Log.Info("get_pill_pool: built pool of " .. #pool .. " pills")
    return pool
end

-- Rolls this room visit's pocket-item-slot content: Config.CONSUMABLE_CARD_CHANCE
-- chance to hold a random card, Config.CONSUMABLE_PILL_CHANCE chance to hold a
-- random pill, remainder left empty. The slot is always cleared first
-- (both SetCard(0) and SetPill(0), regardless of what was previously held),
-- then re-rolled -- same "clear then reroll every visit" treatment as
-- ItemSpawning.RollActiveItem, since a stale card/pill from a prior room
-- visit shouldn't carry over. No return value: ground_truth.lua reads the
-- result back live via GetCard()/GetPill() rather than trusting this roll's
-- intent, per the project's "log actual queried state" principle.
---@param player EntityPlayer
---@param rng RNG
function Consumables.RollConsumables(player, rng)
    player:SetCard(0, 0)
    player:SetPill(0, 0)

    local roll = rng:RandomFloat()

    if roll < Config.CONSUMABLE_CARD_CHANCE then
        local pool = get_card_pool()
        local card_id = pool[rng:RandomInt(#pool) + 1]
        player:SetCard(0, card_id)
        return
    end

    if roll < Config.CONSUMABLE_CARD_CHANCE + Config.CONSUMABLE_PILL_CHANCE then
        local pool = get_pill_pool()
        local pill_color = pool[rng:RandomInt(#pool) + 1]
        player:SetPill(0, pill_color)
        return
    end
end

return Consumables
