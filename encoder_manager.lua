local M = {}

local config = {
    hp           = 1,
    stats        = 1,
    chance       = 1,
    collectibles = 1,
    bugs         = 1,
    pocket       = 1,
    map          = 1,
    items        = 1,
}

local encoders = {
    hp           = include("encoders/hp.lua"),
    stats        = include("encoders/stats.lua"),
    chance       = include("encoders/chance.lua"),
    collectibles = include("encoders/collectibles.lua"),
    bugs         = include("encoders/bugs.lua"),
    pocket       = include("encoders/pocket.lua"),
    map          = include("encoders/map.lua"),
    items        = include("encoders/items.lua"),
}

local encoder_order = {
    "hp",
    "stats",
    "chance",
    "collectibles",
    "bugs",
    "pocket",
    "map",
    "items",
}

local enabled_encoders = {}
local enabled_on_new_room = {}
local enabled_on_damage = {}

for _, name in ipairs(encoder_order) do
    if config[name] == 1 then
        local enc = encoders[name]
        enabled_encoders[#enabled_encoders + 1] = enc

        if enc.OnNewRoom then
            enabled_on_new_room[#enabled_on_new_room + 1] = enc.OnNewRoom
        end

        if enc.OnDamage then
            enabled_on_damage[#enabled_on_damage + 1] = enc.OnDamage
        end
    end
end


function M.OnNewRoom(player)
    for i = 1, #enabled_on_new_room do
        enabled_on_new_room[i](player)
    end
end

function M.OnDamage(player, amount, flags, source, countdown)
    for i = 1, #enabled_on_damage do
        enabled_on_damage[i](player, amount, flags, source, countdown)
    end
end

local flat = {}

function M.EncodeAll(player)
    local n = 0

    for i = 1, #enabled_encoders do
        local vec = enabled_encoders[i].Encode(player)
        for j = 1, #vec do
            n = n + 1
            flat[n] = vec[j]
        end
    end

    for i = n + 1, #flat do
        flat[i] = nil
    end

    return flat
end

return M
