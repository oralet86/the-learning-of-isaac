local mod = RegisterMod("The Learning of Isaac", 1)
local game = Game()

local Encoders = include("encoder_manager.lua")

function mod:OnNewRoom()
    local player = Isaac.GetPlayer(0)
    Encoders.OnNewRoom(player)
end

mod:AddCallback(ModCallbacks.MC_POST_NEW_ROOM, mod.OnNewRoom)

function mod:OnDamage(entity, amount, flags, source, countdown)
    if entity.Type == EntityType.ENTITY_PLAYER then
        local player = entity:ToPlayer()
        Encoders.OnDamage(player, amount, flags, source, countdown)
    end
end

mod:AddCallback(ModCallbacks.MC_ENTITY_TAKE_DMG, mod.OnDamage)

function mod:OnUpdate()
    local player = Isaac.GetPlayer(0)
    local features = Encoders.EncodeAll(player)

    local buf = {}
    for i = 1, #features do
        buf[#buf + 1] = string.format("%.3f", features[i])
    end
    print("[FEATURES]", table.concat(buf, " "))
end

mod:AddCallback(ModCallbacks.MC_POST_UPDATE, mod.OnUpdate)
