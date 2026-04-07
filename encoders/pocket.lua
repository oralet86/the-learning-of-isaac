local P = {}
local game = Game()

-- An offset to Pill IDs to prevent them from overlapping with Card IDs.
local PILL_ID_OFFSET = 100

function P.Encode(player)
    local pocket_vector = { 0.0, 0.0 }
    local item_pool = game:GetItemPool()

    -- Helper function to get the standardized ID of an item in a slot
    local function getSlotContent(slot_index)
        -- 1. Check for Cards/Runes
        local card_id = player:GetCard(slot_index)
        if card_id > 0 then
            return card_id
        end

        -- 2. Check for Pills
        local pill_color = player:GetPill(slot_index)
        if pill_color > 0 then
            -- We need to check if the player knows what this pill does.
            -- Using GetPillEffect retrieves the permanent Effect ID (e.g. Health Up), not the random Color.
            local pill_effect = item_pool:GetPillEffect(pill_color)

            -- Check if the pill is identified in the knowledge base
            local is_identified = item_pool:IsPillIdentified(pill_color)

            -- PhD (Item 75), Virgo (Item 409), and False PhD (Item 717) allow you to identify pills visually
            if not is_identified then
                if player:HasCollectible(CollectibleType.COLLECTIBLE_PHD) or
                    player:HasCollectible(CollectibleType.COLLECTIBLE_VIRGO) or
                    player:HasCollectible(CollectibleType.COLLECTIBLE_FALSE_PHD) then
                    is_identified = true
                end
            end

            if is_identified then
                return pill_effect + PILL_ID_OFFSET
            else
                -- The player sees the pill, but doesn't know the effect.
                return -1.0
            end
        end

        -- Slot is empty
        return 0.0
    end

    -- Slot 0: Primary pocket item (Q key)
    pocket_vector[1] = getSlotContent(0)

    -- Slot 1: Secondary pocket item
    pocket_vector[2] = getSlotContent(1)

    return pocket_vector
end

return P
