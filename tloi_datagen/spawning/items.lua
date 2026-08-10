local Config = include("config")
local Log = include("log")

---@class ItemSpawning
local ItemSpawning = {}

-- Built lazily on first use and cached -- every real passive/familiar
-- collectible in the currently-running game, queried dynamically via
-- ItemConfig rather than a hand-typed list, so there's no risk of a typo'd
-- constant name silently breaking a pool entry and no need to keep this
-- list in sync as the game's item roster changes. Excludes actives and
-- trinkets (out of scope -- see original.md), plus Config.EXCLUDED_ITEM_IDS
-- (e.g. IBS, whose random pooping would corrupt scene captures).
--
-- NOTE: ItemConfig:GetCollectibles():Get(i) is documented as returning
-- unusable userdata, so this iterates ids directly via GetCollectible(id)
-- instead, up to GetCollectibles().Size - 1 (the documented max valid id).
---@type integer[]?
local item_pool_cache = nil

---@return integer[]
local function get_item_pool()
    if item_pool_cache ~= nil then
        return item_pool_cache
    end

    local item_config = Isaac.GetItemConfig()
    local max_id = item_config:GetCollectibles().Size - 1
    local pool = {}
    local excluded_count = 0
    for id = 1, max_id do
        local item = item_config:GetCollectible(id)
        if item ~= nil
            and (item.Type == ItemType.ITEM_PASSIVE or item.Type == ItemType.ITEM_FAMILIAR)
            and not Config.EXCLUDED_ITEM_IDS[id] then
            pool[#pool + 1] = id
        else
            excluded_count = excluded_count + 1
        end
    end

    item_pool_cache = pool
    Log.Info(
        "get_item_pool: built pool of "
            .. #pool
            .. " passive/familiar collectibles ("
            .. excluded_count
            .. " ids skipped: actives, trinkets, Config.EXCLUDED_ITEM_IDS)"
    )
    return pool
end

-- Applies a random baseline passive-item count (Config.ITEM_BASELINE_MIN to
-- MAX, or 1-Config.TAINTED_ISAAC_ITEM_CAP for Tainted Isaac -- see below) to
-- `player`, sampling *with* replacement -- duplicates/stacked passives are
-- normal real gameplay, and EntityPlayer:GetCollectibleCount() counts total
-- copies, not distinct types. Called once per run, right after the
-- character restart. Returns the ids actually applied, read back from the
-- player afterward rather than just echoing intent, per the project's "log
-- actual queried state" principle.
---@param player EntityPlayer
---@param rng RNG
---@return integer[] applied_item_ids
function ItemSpawning.ApplyBaseline(player, rng)
    local pool = get_item_pool()
    local count
    if player:GetPlayerType() == PlayerType.PLAYER_ISAAC_B then
        -- Tainted Isaac cannot hold more than Config.TAINTED_ISAAC_ITEM_CAP
        -- passive items at once, which is below Config.ITEM_BASELINE_MIN,
        -- so the normal range can't apply to him at all.
        count = 1 + rng:RandomInt(Config.TAINTED_ISAAC_ITEM_CAP)
    else
        count = Config.ITEM_BASELINE_MIN
            + rng:RandomInt(Config.ITEM_BASELINE_MAX - Config.ITEM_BASELINE_MIN + 1)
    end

    local applied = {}
    for _ = 1, count do
        local item_id = pool[rng:RandomInt(#pool) + 1]
        player:AddCollectible(item_id, 0, false)
        applied[#applied + 1] = item_id
    end
    return applied
end

---@param player EntityPlayer
---@param pool integer[]
---@return integer[]
local function currently_held_from_pool(player, pool)
    local held = {}
    for _, item_id in ipairs(pool) do
        if player:GetCollectibleNum(item_id) > 0 then
            held[#held + 1] = item_id
        end
    end
    return held
end

-- action encoding (MOD_CHANGES_NEEDED.md item 5, matches
-- visual_extraction/labels.py's already-shipped "0=none, 1=add, 2=remove"
-- doc comment -- not a free choice, must stay in sync with that consumer).
local ITEM_DELTA_ACTION_NONE = 0
local ITEM_DELTA_ACTION_ADD = 1
local ITEM_DELTA_ACTION_REMOVE = 2

-- Rolls one of three item-delta actions for a single room visit --
-- Config.ITEM_DELTA_ADD_PCT chance to add one random pool item,
-- Config.ITEM_DELTA_REMOVE_PCT chance to remove one currently-held pool
-- item, remainder "none" -- and applies it. "Currently held" is discovered
-- live via GetCollectibleNum per pool id, not mod-side bookkeeping, so it
-- stays correct even though items now persist and drift across a whole run
-- instead of resetting every visit. If "remove" is rolled but nothing from
-- the pool is currently held (e.g. very early in a run, or an unlucky
-- remove streak), degrades to "none" rather than erroring. For Tainted
-- Isaac, "add" also degrades to "none" once he's already at
-- Config.TAINTED_ISAAC_ITEM_CAP items -- see ApplyBaseline's docstring.
---@param player EntityPlayer
---@param rng RNG
---@return {action: integer, item_id: integer?}
function ItemSpawning.RollDelta(player, rng)
    local roll = rng:RandomFloat()
    local at_tainted_isaac_cap = player:GetPlayerType() == PlayerType.PLAYER_ISAAC_B
        and player:GetCollectibleCount() >= Config.TAINTED_ISAAC_ITEM_CAP

    if roll < Config.ITEM_DELTA_ADD_PCT and not at_tainted_isaac_cap then
        local pool = get_item_pool()
        local item_id = pool[rng:RandomInt(#pool) + 1]
        player:AddCollectible(item_id, 0, false)
        return { action = ITEM_DELTA_ACTION_ADD, item_id = item_id }
    end

    if roll < Config.ITEM_DELTA_ADD_PCT + Config.ITEM_DELTA_REMOVE_PCT then
        local held = currently_held_from_pool(player, get_item_pool())
        if #held > 0 then
            local item_id = held[rng:RandomInt(#held) + 1]
            player:RemoveCollectible(item_id)
            return { action = ITEM_DELTA_ACTION_REMOVE, item_id = item_id }
        end
    end

    return { action = ITEM_DELTA_ACTION_NONE, item_id = nil }
end

-- Built lazily, same pattern/rationale as get_item_pool -- every real
-- active collectible, queried dynamically rather than hand-typed.
---@type integer[]?
local active_pool_cache = nil

---@return integer[]
local function get_active_pool()
    if active_pool_cache ~= nil then
        return active_pool_cache
    end

    local item_config = Isaac.GetItemConfig()
    local max_id = item_config:GetCollectibles().Size - 1
    local pool = {}
    for id = 1, max_id do
        local item = item_config:GetCollectible(id)
        if item ~= nil and item.Type == ItemType.ITEM_ACTIVE and not Config.EXCLUDED_ITEM_IDS[id] then
            pool[#pool + 1] = id
        end
    end

    active_pool_cache = pool
    Log.Info("get_active_pool: built pool of " .. #pool .. " active collectibles")
    return pool
end

-- Rolls whether to hold a random active item this room visit
-- (Config.ACTIVE_ITEM_CHANCE), with a random charge from 0 to that item's
-- own max charge. Re-rolled every visit (unlike passive items, which
-- persist/drift): any active item held from a previous visit is always
-- cleared first via RemoveCollectible (works for actives, not just
-- passives), regardless of this visit's roll outcome, so a stale item is
-- never left behind in the active slot.
-- `charge` is normalized to [0, 1] (current charge / that item's own max
-- charge) rather than logging raw charge/max_charge pair -- downstream
-- consumers are meant to treat every active item as effectively "1 charge
-- bar" regardless of its real MaxCharges, so the absolute charge count
-- isn't logged at all. Items with MaxCharges == 0 (no charge bar at all,
-- e.g. single-use/instant actives) are always normalized to 1.0 (ready to
-- use) rather than dividing by zero.
-- Returns a table with `held` always present (never nil) even when no
-- active item was rolled -- a table with *every* field nil (item_id,
-- charge both nil at once, unlike item_delta which always has a non-nil
-- `action` string) is indistinguishable from an empty table to the
-- vendored JSON encoder, which serializes an empty table as `[]` rather
-- than `{}` -- without a field that's always present, the daemon's
-- `active_item_raw.get(...)` call would see a Python list instead of a
-- dict and crash. `held` guarantees the table is never fully empty.
---@param player EntityPlayer
---@param rng RNG
---@return {held: boolean, item_id: integer?, charge: number?}
function ItemSpawning.RollActiveItem(player, rng)
    local current_active = player:GetActiveItem(ActiveSlot.SLOT_PRIMARY)
    if current_active ~= CollectibleType.COLLECTIBLE_NULL then
        player:RemoveCollectible(current_active)
    end

    if rng:RandomFloat() >= Config.ACTIVE_ITEM_CHANCE then
        return { held = false, item_id = nil, charge = nil }
    end

    local pool = get_active_pool()
    local item_id = pool[rng:RandomInt(#pool) + 1]
    local max_charge = Isaac.GetItemConfig():GetCollectible(item_id).MaxCharges
    local charge = rng:RandomInt(max_charge + 1)
    player:AddCollectible(item_id, charge, false, ActiveSlot.SLOT_PRIMARY)

    local live_max_charge = Isaac.GetItemConfig():GetCollectible(player:GetActiveItem(ActiveSlot.SLOT_PRIMARY)).MaxCharges
    local normalized_charge = live_max_charge == 0 and 1.0
        or player:GetActiveCharge(ActiveSlot.SLOT_PRIMARY) / live_max_charge

    return {
        held = true,
        item_id = player:GetActiveItem(ActiveSlot.SLOT_PRIMARY),
        charge = normalized_charge,
    }
end

return ItemSpawning
