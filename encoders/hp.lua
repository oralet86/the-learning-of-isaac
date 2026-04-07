local C = {}

C.HP_NONE = 0
C.HP_RED_FULL = 1
C.HP_RED_HALF = 2
C.HP_RED_EMPTY_CONTAINER = 3
C.HP_SOUL_FULL = 4
C.HP_SOUL_HALF = 5
C.HP_BLACK_FULL = 6
C.HP_BLACK_HALF = 7
C.HP_BONE_EMPTY = 8
C.HP_ROTTEN_RED = 9

C.HP_SLOTS = 12

local holy_mantle_available = false

local function initHPFeatures()
    local hp = {}
    for i = 1, 16 do hp[i] = 0.0 end
    return hp
end

local function fillSlots(hp, idx, count, code)
    local used = 0
    while count > 0 and idx <= C.HP_SLOTS do
        hp[idx] = code
        idx = idx + 1
        used = used + 1
        count = count - 1
    end
    return used
end

function C.OnNewRoom(player)
    if player:HasCollectible(CollectibleType.COLLECTIBLE_HOLY_MANTLE) or
        player:GetEffects():GetCollectibleEffectNum(CollectibleType.COLLECTIBLE_HOLY_MANTLE) > 0 then
        holy_mantle_available = true
    else
        holy_mantle_available = false
    end
end

function C.OnDamage()
    if holy_mantle_available then
        holy_mantle_available = false
    end
end

function C.Encode(player)
    local hp_feat = initHPFeatures()

    local hearts_half = player:GetHearts()
    local max_hearts_half = player:GetMaxHearts()
    local soul_black_halves = player:GetSoulHearts()
    local black_bits = player:GetBlackHearts()
    local bone_hearts = player:GetBoneHearts()
    local rotten_hearts = player:GetRottenHearts()
    local eternal_hearts = player:GetEternalHearts()
    local golden_hearts = player:GetGoldenHearts()

    -- Split red HP between red containers and bone hearts
    local red_in_containers_half = math.min(hearts_half, max_hearts_half)
    local remaining_half = hearts_half - red_in_containers_half
    if remaining_half < 0 then remaining_half = 0 end

    local bone_capacity_half = bone_hearts * 2
    local bone_fill_half = math.min(remaining_half, bone_capacity_half)

    local max_red_containers = max_hearts_half // 2
    local red_full = red_in_containers_half // 2
    local red_half = red_in_containers_half % 2

    local slot = 1

    -- Rotten hearts
    if rotten_hearts > 0 then
        local used = fillSlots(hp_feat, slot, rotten_hearts, C.HP_ROTTEN_RED)
        slot = slot + used
    end

    -- Red containers
    local red_containers = max_red_containers

    local full_to_place = math.min(red_full, red_containers)
    if full_to_place > 0 then
        local used = fillSlots(hp_feat, slot, full_to_place, C.HP_RED_FULL)
        slot = slot + used
        red_containers = red_containers - used
    end

    if red_half == 1 and red_containers > 0 and slot <= C.HP_SLOTS then
        hp_feat[slot] = C.HP_RED_HALF
        slot = slot + 1
        red_containers = red_containers - 1
    end

    if red_containers > 0 then
        local used = fillSlots(hp_feat, slot, red_containers, C.HP_RED_EMPTY_CONTAINER)
        slot = slot + used
    end

    -- Bone hearts
    if bone_hearts > 0 and slot <= C.HP_SLOTS then
        local full_bone = bone_fill_half // 2
        local half_bone = bone_fill_half % 2

        local used_full = fillSlots(hp_feat, slot, math.min(full_bone, bone_hearts), C.HP_RED_FULL)
        slot = slot + used_full

        local remaining_bone = bone_hearts - used_full

        if half_bone == 1 and remaining_bone > 0 and slot <= C.HP_SLOTS then
            hp_feat[slot] = C.HP_RED_HALF
            slot = slot + 1
            remaining_bone = remaining_bone - 1
        end

        if remaining_bone > 0 and slot <= C.HP_SLOTS then
            local used_empty = fillSlots(hp_feat, slot, remaining_bone, C.HP_BONE_EMPTY)
            slot = slot + used_empty
        end
    end

    -- Soul/black hearts
    if soul_black_halves > 0 and slot <= C.HP_SLOTS then
        local i = 0
        while i < soul_black_halves and slot <= C.HP_SLOTS do
            local has_second = (i + 1) < soul_black_halves
            local bit_i = (black_bits >> i) & 1

            if has_second then
                local bit_j = (black_bits >> (i + 1)) & 1
                if bit_i ~= 0 or bit_j ~= 0 then
                    hp_feat[slot] = C.HP_BLACK_FULL
                else
                    hp_feat[slot] = C.HP_SOUL_FULL
                end
                slot = slot + 1
                i = i + 2
            else
                if bit_i ~= 0 then
                    hp_feat[slot] = C.HP_BLACK_HALF
                else
                    hp_feat[slot] = C.HP_SOUL_HALF
                end
                slot = slot + 1
                i = i + 1
            end
        end
    end

    -- Overflow approximation
    local approx_icons =
        rotten_hearts +
        max_red_containers +
        bone_hearts +
        math.ceil(soul_black_halves / 2)

    hp_feat[13] = (approx_icons > C.HP_SLOTS) and 1.0 or 0.0
    hp_feat[14] = holy_mantle_available and 1.0 or 0.0
    hp_feat[15] = (eternal_hearts > 0) and 1.0 or 0.0
    hp_feat[16] = (golden_hearts > 0) and 1.0 or 0.0

    return hp_feat
end
