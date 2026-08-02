local Config = include("config")
local Log = include("log")

---@class HeartState
local HeartState = {}

-- Mirrors query/ground_truth.lua's count_black_hearts: EntityPlayer:GetBlackHearts()
-- does not return a count, it returns a bit mask of which soul-heart slots
-- are black hearts. Needed here (not just in ground_truth.lua) because
-- removing the *current* black hearts before applying a new random count
-- requires knowing how many there actually are.
---@param mask integer
---@return integer
local function count_black_hearts(mask)
    local count = 0
    for i = 0, 31 do
        if (mask >> i) & 1 == 1 then
            count = count + 1
        end
    end
    return count
end

-- Character-specific health constraints, applied to the freshly-rolled
-- random targets before they're written to the player -- letting every
-- character roll the fully generic range produces combinations that can't
-- exist in real gameplay (e.g. Keeper holding black hearts, which the game
-- instantly converts to blue flies on pickup). Mutates `new` in place;
-- every field is expected to already be present with its generic roll.
-- Player types not handled by any branch here keep the fully generic roll
-- untouched.
---@param new table
---@param player_type integer
---@param rng RNG
local function apply_character_constraints(new, player_type, rng)
    if player_type == PlayerType.PLAYER_KEEPER or player_type == PlayerType.PLAYER_KEEPER_B then
        -- Keeper/Tainted Keeper: coin hearts use the same underlying
        -- max_hearts/hearts fields as red hearts (just re-skinned), capped
        -- at Config.KEEPER_COIN_HEART_CAP/TAINTED_KEEPER_COIN_HEART_CAP
        -- containers, and always whole-heart amounts -- half coin hearts
        -- don't exist in real gameplay (Keeper always takes one full Coin
        -- Heart of damage). Every other heart type is impossible for these
        -- two -- all non-coin heart pickups convert to blue flies (gold
        -- hearts convert to coins instead, but the net effect is still
        -- "never actually held as a heart").
        local cap_containers = player_type == PlayerType.PLAYER_KEEPER and Config.KEEPER_COIN_HEART_CAP
            or Config.TAINTED_KEEPER_COIN_HEART_CAP
        new.max_hearts = 2 * (1 + rng:RandomInt(cap_containers))
        new.hearts = 2 * (1 + rng:RandomInt(new.max_hearts / 2))
        new.soul_hearts = 0
        new.black_hearts = 0
        new.bone_hearts = 0
        new.rotten_hearts = 0
        new.eternal_hearts = 0
        new.golden_hearts = 0
        new.broken_hearts = 0
    elseif player_type == PlayerType.PLAYER_THELOST or player_type == PlayerType.PLAYER_THELOST_B then
        -- The Lost / Tainted Lost: no health of any kind. The Lost starts
        -- with no health and cannot gain health by any means, dying from
        -- any damage taken; Tainted Lost's health is simply "no health".
        new.max_hearts = 0
        new.hearts = 0
        new.soul_hearts = 0
        new.black_hearts = 0
        new.bone_hearts = 0
        new.rotten_hearts = 0
        new.eternal_hearts = 0
        new.golden_hearts = 0
        new.broken_hearts = 0
    elseif
        player_type == PlayerType.PLAYER_BLUEBABY
        or player_type == PlayerType.PLAYER_BLUEBABY_B
        or player_type == PlayerType.PLAYER_BETHANY_B
    then
        -- ???, Tainted ???, Tainted Bethany: cannot have red heart
        -- containers at all. Every other heart type is left at its normal
        -- random roll.
        new.max_hearts = 0
        new.hearts = 0
    elseif player_type == PlayerType.PLAYER_BETHANY then
        -- Bethany: the inverse of the above -- can only have red hearts.
        -- Soul/black hearts are converted to her separate "soul charge"
        -- resource on pickup rather than held as hearts, so those and
        -- every other non-red type are zeroed here.
        new.soul_hearts = 0
        new.black_hearts = 0
        new.bone_hearts = 0
        new.rotten_hearts = 0
        new.eternal_hearts = 0
        new.golden_hearts = 0
        new.broken_hearts = 0
    end
end

-- Order matters here: every *new* random amount is added before any *old*
-- amount is removed, so total health only ever goes up during this
-- function, never down to (or through) zero -- if old and new amounts were
-- applied type-by-type instead, a moment where every heart type is
-- simultaneously near-empty could trigger player death mid-randomization,
-- which would derail the run this room visit belongs to. Uses each type's
-- own Add*Hearts setter with a negative amount to remove (documented as the
-- correct way to do so for all of these), never Remove* singular functions.
---@param player EntityPlayer
---@param rng RNG
---@return table applied_health
function HeartState.RandomizeHealth(player, rng)
    local old_hearts = player:GetHearts()
    local old_max_hearts = player:GetMaxHearts()
    local old_soul_hearts = player:GetSoulHearts()
    local old_black_hearts = count_black_hearts(player:GetBlackHearts())
    local old_bone_hearts = player:GetBoneHearts()
    local old_rotten_hearts = player:GetRottenHearts()
    local old_eternal_hearts = player:GetEternalHearts()
    local old_golden_hearts = player:GetGoldenHearts()
    local old_broken_hearts = player:GetBrokenHearts()

    -- New random targets. max_hearts and hearts are floored above zero (1-6
    -- containers, at least 1 full red heart) so total current health is
    -- never zero even before the old amounts below are removed -- unless
    -- overridden below for a character that can't have red hearts at all.
    local new = {
        max_hearts = 2 * (1 + rng:RandomInt(6)),
        soul_hearts = rng:RandomInt(13),
        black_hearts = rng:RandomInt(5),
        bone_hearts = rng:RandomInt(4),
        rotten_hearts = rng:RandomInt(3),
        eternal_hearts = rng:RandomInt(2),
        golden_hearts = rng:RandomInt(3),
        broken_hearts = rng:RandomInt(3),
    }
    new.hearts = 2 + rng:RandomInt(new.max_hearts - 1)

    apply_character_constraints(new, player:GetPlayerType(), rng)

    -- Eternal hearts are the one exception to the "add new before removing
    -- old" ordering above: 2+ eternal hearts present at the same instant
    -- auto-combine into a red heart (a real game mechanic, not a bug), so
    -- old must be removed *before* new is added here specifically --
    -- new.eternal_hearts is always 0 or 1, so removing old first guarantees
    -- the player is never holding more than 1 at once. No death risk from
    -- this ordering: dropping eternal hearts to 0 never kills the player on
    -- its own, unlike the other heart types below.
    player:AddEternalHearts(-old_eternal_hearts)
    player:AddEternalHearts(new.eternal_hearts)

    player:AddMaxHearts(new.max_hearts, false)
    player:AddHearts(new.hearts)
    player:AddSoulHearts(new.soul_hearts)
    player:AddBlackHearts(new.black_hearts)
    player:AddBoneHearts(new.bone_hearts)
    player:AddRottenHearts(new.rotten_hearts)
    player:AddGoldenHearts(new.golden_hearts)
    player:AddBrokenHearts(new.broken_hearts)

    player:AddSoulHearts(-old_soul_hearts)
    player:AddBlackHearts(-old_black_hearts)
    player:AddBoneHearts(-old_bone_hearts)
    player:AddRottenHearts(-old_rotten_hearts)
    player:AddGoldenHearts(-old_golden_hearts)
    player:AddBrokenHearts(-old_broken_hearts)
    player:AddHearts(-old_hearts)
    player:AddMaxHearts(-old_max_hearts, false)

    local applied = {
        hearts = player:GetHearts(),
        max_hearts = player:GetMaxHearts(),
        soul_hearts = player:GetSoulHearts(),
        black_hearts = count_black_hearts(player:GetBlackHearts()),
        bone_hearts = player:GetBoneHearts(),
        rotten_hearts = player:GetRottenHearts(),
        eternal_hearts = player:GetEternalHearts(),
        golden_hearts = player:GetGoldenHearts(),
        broken_hearts = player:GetBrokenHearts(),
    }
    Log.Info(
        "RandomizeHealth: hearts="
            .. applied.hearts
            .. " max="
            .. applied.max_hearts
            .. " soul="
            .. applied.soul_hearts
            .. " black="
            .. applied.black_hearts
            .. " bone="
            .. applied.bone_hearts
            .. " rotten="
            .. applied.rotten_hearts
            .. " eternal="
            .. applied.eternal_hearts
            .. " golden="
            .. applied.golden_hearts
            .. " broken="
            .. applied.broken_hearts
    )
    return applied
end

-- Randomizes coins/bombs/keys within Config's configured ranges. No
-- death-risk ordering concerns here (unlike hearts), so old amounts are
-- just removed before the new ones are applied.
---@param player EntityPlayer
---@param rng RNG
---@return table applied_currency
function HeartState.RandomizeCurrency(player, rng)
    player:AddCoins(-player:GetNumCoins())
    player:AddBombs(-player:GetNumBombs())
    player:AddKeys(-player:GetNumKeys())

    player:AddCoins(rng:RandomInt(Config.COINS_MAX + 1))
    player:AddBombs(rng:RandomInt(Config.BOMBS_MAX + 1))
    player:AddKeys(rng:RandomInt(Config.KEYS_MAX + 1))

    local applied = {
        coins = player:GetNumCoins(),
        bombs = player:GetNumBombs(),
        keys = player:GetNumKeys(),
    }
    Log.Info("RandomizeCurrency: coins=" .. applied.coins .. " bombs=" .. applied.bombs .. " keys=" .. applied.keys)
    return applied
end

return HeartState
