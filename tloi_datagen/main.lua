local mod = RegisterMod("TLOI Data Generator", 1)

local Config = include("config")
local Exchange = include("ipc.exchange")
local ScenarioBuilderM1 = include("scenario.build_m1")
local GroundTruthQuery = include("query.ground_truth")
local StageRoute = include("data.stage_route")
local Positioning = include("scenario.positioning")
local Viewports = include("scenario.viewports")
local Pose = include("scenario.pose")
local HeartState = include("spawning.hearts")
local ItemSpawning = include("spawning.items")
local Consumables = include("spawning.consumables")
local Doors = include("spawning.doors")
local Log = include("log")

-- M1 state machine. Only one scenario is in flight at a time. The
-- generation loop starts automatically when a run begins (MC_POST_GAME_STARTED)
-- and keeps posing, settling, and exchanging scenarios with the Python
-- capture daemon for as long as the game is open -- there is no manual
-- start/stop step.
local STATE_IDLE = "IDLE"
local STATE_WAITING_FOR_ROOM = "WAITING_FOR_ROOM"
local STATE_SETTLING = "SETTLING"
local STATE_WAITING_FOR_CAPTURE = "WAITING_FOR_CAPTURE"

-- Hush (stage 9) is a single 2x2 boss room with no normal rooms to visit
-- (https://bindingofisaacrebirth.wiki.gg/wiki/Hush) -- capped to exactly 2
-- captures of that one room rather than a randomized count.
local STAGE_ROOM_COUNT_OVERRIDES = {
    [9] = 2, -- Hush
}

-- Rolled once per stage entry (not once per room visit -- the result is
-- cached on runner.rooms_this_stage_target and stays fixed while
-- runner.rooms_this_stage counts up toward it), from Config's
-- ROOMS_PER_STAGE_MIN/MAX range, unless the stage has a hardcoded override.
---@param stage_command string
---@param rng RNG
---@return integer
local function roll_rooms_for_stage(stage_command, rng)
    local stage_num = tonumber(stage_command:match("^s%.(%d+)%."))
    local override = STAGE_ROOM_COUNT_OVERRIDES[stage_num]
    if override ~= nil then
        return override
    end
    return Config.ROOMS_PER_STAGE_MIN + rng:RandomInt(Config.ROOMS_PER_STAGE_MAX - Config.ROOMS_PER_STAGE_MIN + 1)
end

-- How often (in ticks) to log a heartbeat while WAITING_FOR_CAPTURE, so a
-- stuck daemon/ack is visible without waiting for the full timeout. 60
-- ticks ~= 1s.
local CAPTURE_HEARTBEAT_TICKS = 60

-- PlayerType 0-40 inclusive is the full playable roster including Tainted
-- characters (41 = NUM_PLAYER_TYPES, exclusive).
local NUM_PLAYER_TYPES = 41

---@class RunnerState
---@field state string
---@field session_prefix string Set once per run start, to keep scenario ids unique across runs.
---@field scenario_id string?
---@field seed integer Per-room-visit RNG seed, drawn fresh from master_rng each visit -- never reused across visits or runs.
---@field master_rng RNG? Seeded from Random() once per run (true run start, after the character restart lands); every other RNG used this run is seeded from a value drawn from this one.
---@field awaiting_character_restart boolean True between issuing our own "restart <char>" and the resulting MC_POST_GAME_STARTED firing, so that second firing is recognized as "the real one" instead of triggering another restart.
---@field settle_ticks_remaining integer
---@field capture_ticks_remaining integer
---@field room_wait_ticks_remaining integer
---@field current_capture_mode string "idle" or "animated", rolled once per room visit (see pose_and_settle).
---@field pose string? "animated"-mode pose key ("walk_*"/"shoot_*"), rolled once per room visit alongside current_capture_mode. Purely visual -- the player never moves or fires, and the pose is never logged in ground truth.
---@field pose_frame integer Frame index into the pose's animation cycle, re-rolled per viewport so consecutive captures show different mid-cycle sprites.
---@field current_item_delta {action: string, item_id: integer?}?
---@field current_active_item {held: boolean, item_id: integer?, charge: number?}?
---@field route string[] Built once per run by StageRoute.BuildRandomRoute.
---@field route_index integer Which stage in `route` we're currently on.
---@field rooms_this_stage integer Rooms visited on the current stage so far.
---@field rooms_this_stage_target integer Rooms to visit on the current stage before advancing, rolled once per stage entry (see roll_rooms_for_stage).
---@field baseline_applied boolean Whether the once-per-run baseline passive-item count has been applied yet. Reset at the start of every new run.
---@field viewport_positions {name: string, position: Vector}[] This room visit's representative camera-viewport positions, computed once per room entry by Viewports.GetPositions.
---@field viewport_half_extent Vector? Half-extent of a single 1x1 room's worth of view around any viewport position this room visit, for ground-truth enemy/door frame-filtering.
---@field viewport_index integer 1-based progress through viewport_positions for the current room visit.
local runner = {
    state = STATE_IDLE,
    session_prefix = "",
    scenario_id = nil,
    seed = 0,
    master_rng = nil,
    awaiting_character_restart = false,
    settle_ticks_remaining = 0,
    capture_ticks_remaining = 0,
    room_wait_ticks_remaining = 0,
    current_capture_mode = "idle",
    pose = nil,
    pose_frame = 0,
    current_item_delta = nil,
    current_active_item = nil,
    route = { "s.1.0" },
    route_index = 1,
    rooms_this_stage = 0,
    rooms_this_stage_target = 1,
    baseline_applied = false,
    viewport_positions = {},
    viewport_half_extent = nil,
    viewport_index = 1,
}

---@return string
local function current_stage()
    return runner.route[runner.route_index]
end

---@param new_state string
local function set_state(new_state)
    Log.Info("state: " .. runner.state .. " -> " .. new_state)
    runner.state = new_state
end

-- Enters WAITING_FOR_ROOM with a fresh timeout window -- callers must not
-- do anything else after this that assumes MC_POST_NEW_ROOM will
-- definitely fire (see room-change-ordering comments elsewhere in this
-- file); the timeout is what makes a rejected/ignored room change
-- self-healing instead of hanging forever (this happens for boss rooms:
-- their intro cutscene blocks the *next* room change while it's waiting
-- for a skip input this pipeline never sends).
local function enter_waiting_for_room()
    runner.room_wait_ticks_remaining = Config.ROOM_WAIT_TIMEOUT_TICKS
    set_state(STATE_WAITING_FOR_ROOM)
end

-- Issues a fresh, unpredictable "restart <char>" -- both the very first run
-- of a game session and every subsequent one (once a route is exhausted)
-- go through this, so every run gets a totally random character, not just
-- the first. `Random()` (not os.time(), which only has 1-second
-- resolution) is the game's own source of true runtime entropy.
--
-- Sets state to IDLE and clears scenario_id/route before issuing the
-- restart: if "restart" fires MC_POST_GAME_STARTED synchronously (as
-- Game():ChangeRoom() does for MC_POST_NEW_ROOM), OnGameStarted runs nested
-- inside this call and takes over completely; if it's asynchronous instead,
-- this leaves the runner in a safe idle state for however many frames pass
-- until it fires, rather than something that would crash (e.g.
-- STATE_WAITING_FOR_CAPTURE polling Exchange with a nil scenario_id).
local function start_new_run()
    if runner.scenario_id ~= nil then
        Exchange.Cleanup(runner.scenario_id)
    end
    runner.scenario_id = nil
    set_state(STATE_IDLE)

    local char_rng = RNG()
    char_rng:SetSeed(Random(), 35)
    local character = char_rng:RandomInt(NUM_PLAYER_TYPES)

    runner.awaiting_character_restart = true
    Log.Info("start_new_run: issuing restart " .. character)
    Isaac.ExecuteCommand("restart " .. tostring(character))
end

-- Jacob & Esau and Tainted Forgotten are the only characters with a second
-- EntityPlayer at index 1 (the "sub-player") -- every other character has
-- only Isaac.GetPlayer(0). Whether Isaac.GetPlayer(1) reliably returns nil
-- (rather than erroring) for those other characters is unverified; the
-- pcall wrap makes either outcome safe.
---@return EntityPlayer?
local function get_second_player()
    local ok, p2 = pcall(Isaac.GetPlayer, 1)
    if ok and p2 ~= nil then
        return p2
    end
    return nil
end

-- Poses the current room and moves straight to SETTLING: repositions
-- player 1 (and player 2 for duo characters), applies the once-per-run item
-- baseline, randomizes health/currency/consumables, and rolls this visit's
-- item-delta and active-item state. Orchestrated directly here (rather than
-- via a single bundled scenario-builder call) so each step can be extended
-- independently. Called from OnNewRoom() once a room/stage transition has
-- finished loading, or directly from begin_scenario() on the rare occasion
-- no alternative room was available to teleport to.
--
-- Boss rooms get Room:SetClear(true) immediately, before anything else --
-- the boss-intro cutscene has no auto-timeout and waits indefinitely for a
-- skip input (Escape/Space) that this automated pipeline never sends,
-- silently blocking the *next* scenario's room change (its
-- MC_POST_NEW_ROOM/ack never arrives, the loop just hangs) rather than
-- blocking this room's own capture. There is no API to simulate the skip
-- keypress from pure Lua, so marking the room pre-cleared prevents the
-- encounter/cutscene from engaging at all. This fully suppresses the intro
-- cutscene, but boss rooms still get Config.BOSS_SETTLE_TICKS_BONUS added
-- on top of the normal settle window below -- scripted multi-phase bosses
-- (e.g. Mom's Heart) need extra real time beyond the suppressed cutscene to
-- finish their own internal setup/teardown before it's safe to ChangeRoom
-- away again, and skipping that wait can crash the game.
local function pose_and_settle()
    local room = Game():GetRoom()
    if room:GetType() == RoomType.ROOM_BOSS then
        room:SetClear(true)
        Log.Info("pose_and_settle: boss room, called SetClear(true) to preempt the intro cutscene lock")
    end

    local rng = RNG()
    rng:SetSeed(runner.seed, 0)

    runner.current_capture_mode = rng:RandomFloat() < Config.CAPTURE_MODE_ANIMATED_PCT and "animated" or "idle"
    Log.Info("pose_and_settle: capture_mode=" .. runner.current_capture_mode)

    -- "animated" mode additionally rolls a forced walk/shoot pose: the
    -- player sprite is held mid-animation (never actually moving or firing)
    -- for the whole settle+capture window. Pose is deliberately not part of
    -- ground truth -- it exists only so captured frames show the model the
    -- player's non-idle sprites.
    runner.pose = nil
    runner.pose_frame = 0
    if runner.current_capture_mode == "animated" then
        runner.pose = Pose.RollPose(rng)
        runner.pose_frame = rng:RandomInt(64)
        Log.Info("pose_and_settle: pose=" .. runner.pose .. " frame=" .. runner.pose_frame)
    end

    Doors.RandomizeDoors(room, rng)

    local player = Isaac.GetPlayer(0)
    local player2 = get_second_player()

    runner.viewport_positions, runner.viewport_half_extent = Viewports.GetPositions(room)
    runner.viewport_index = 1
    local pos = runner.viewport_positions[1].position
    player.Position = pos
    Log.Info(
        "pose_and_settle: positioned player at viewport '"
            .. runner.viewport_positions[1].name
            .. "' ("
            .. tostring(pos.X)
            .. ", "
            .. tostring(pos.Y)
            .. "), "
            .. #runner.viewport_positions
            .. " viewport(s) this room visit"
    )

    -- Jacob & Esau have fully independent passive-item and pill/card
    -- inventories; Tainted Forgotten/Tainted Soul share a single item pool.
    -- Rather than branch on which duo character is active, player2 always
    -- gets its own independent baseline/consumable rolls below -- for
    -- Tainted Forgotten this may fold into the same shared state the engine
    -- already maintains, and per this project's "log actual queried state"
    -- principle that's fine either way, since ground truth is always read
    -- back live. player2's item-delta and active item are deliberately not
    -- rolled at all -- passive items and pills are enough diversity without
    -- that extra tracking complexity.
    if not runner.baseline_applied then
        local applied = ItemSpawning.ApplyBaseline(player, rng)
        runner.baseline_applied = true
        Log.Info("pose_and_settle: applied item baseline, " .. #applied .. " items")

        if player2 ~= nil then
            local applied2 = ItemSpawning.ApplyBaseline(player2, rng)
            Log.Info("pose_and_settle: applied player2 item baseline, " .. #applied2 .. " items")
        end
    end

    HeartState.RandomizeHealth(player, rng)
    HeartState.RandomizeCurrency(player, rng)
    Consumables.RollConsumables(player, rng)

    runner.current_item_delta = ItemSpawning.RollDelta(player, rng)
    Log.Info(
        "pose_and_settle: item_delta action="
            .. runner.current_item_delta.action
            .. " item_id="
            .. tostring(runner.current_item_delta.item_id)
    )

    runner.current_active_item = ItemSpawning.RollActiveItem(player, rng)
    Log.Info(
        "pose_and_settle: active_item item_id="
            .. tostring(runner.current_active_item.item_id)
            .. " charge="
            .. tostring(runner.current_active_item.charge)
    )

    -- Duo characters (Jacob & Esau, Tainted Forgotten): reposition (close
    -- to player1, not anywhere valid in the room -- they're a pair, not
    -- independently roaming), re-health, and re-roll consumables for the
    -- second twin too, so it's never left behind in a room that no longer
    -- exists after a teleport. Coins/bombs/keys are a genuinely shared
    -- team-wide pool, so RandomizeCurrency is deliberately not called a
    -- second time.
    if player2 ~= nil then
        local pos2 = Positioning.SampleNearbyPosition(
            room,
            player.Position,
            Config.PLAYER2_NEARBY_MIN_RADIUS,
            Config.PLAYER2_NEARBY_MAX_RADIUS,
            rng
        )
        player2.Position = pos2
        Log.Info("pose_and_settle: positioned player2 at (" .. tostring(pos2.X) .. ", " .. tostring(pos2.Y) .. ")")

        HeartState.RandomizeHealth(player2, rng)
        Consumables.RollConsumables(player2, rng)
    end

    local base_settle_ticks = runner.current_capture_mode == "animated" and Config.SETTLE_TICKS_ANIMATED
        or Config.SETTLE_TICKS
    local boss_bonus = room:GetType() == RoomType.ROOM_BOSS and Config.BOSS_SETTLE_TICKS_BONUS or 0
    runner.settle_ticks_remaining = base_settle_ticks + boss_bonus
    set_state(STATE_SETTLING)
end

-- Moves player 1 to the next camera-viewport position within the same room
-- visit and re-enters SETTLING with a short settle window -- unlike
-- pose_and_settle, doors/items/health are deliberately left untouched here,
-- since only the camera framing changed, not the scene itself. player2
-- (duo characters) is repositioned too, close to player1's new viewport
-- position, same as pose_and_settle's own placement -- otherwise it would
-- stay wherever pose_and_settle originally put it, drifting arbitrarily far
-- from player1's new position/frame as the sweep continues. Health/
-- consumables are deliberately not re-rolled for player2 here either,
-- matching player1's "only position changes" treatment for this function.
local function advance_viewport()
    -- Re-roll the pose's cycle frame per viewport (while the pose itself
    -- stays fixed for the whole room visit), so consecutive captures show
    -- different mid-walk/mid-aim sprites instead of the same frozen frame.
    if runner.pose ~= nil then
        local frame_rng = RNG()
        frame_rng:SetSeed(runner.seed + runner.viewport_index, 0)
        runner.pose_frame = frame_rng:RandomInt(64)
        Log.Info("advance_viewport: pose=" .. runner.pose .. " frame=" .. runner.pose_frame)
    end

    local player = Isaac.GetPlayer(0)
    local target = runner.viewport_positions[runner.viewport_index]
    player.Position = target.position
    Log.Info(
        "advance_viewport: positioned player at viewport '"
            .. target.name
            .. "' ("
            .. tostring(target.position.X)
            .. ", "
            .. tostring(target.position.Y)
            .. ") ["
            .. runner.viewport_index
            .. "/"
            .. #runner.viewport_positions
            .. "]"
    )

    local player2 = get_second_player()
    if player2 ~= nil then
        local rng = RNG()
        rng:SetSeed(runner.seed + runner.viewport_index, 0)
        local pos2 = Positioning.SampleNearbyPosition(
            Game():GetRoom(),
            target.position,
            Config.PLAYER2_NEARBY_MIN_RADIUS,
            Config.PLAYER2_NEARBY_MAX_RADIUS,
            rng
        )
        player2.Position = pos2
        Log.Info("advance_viewport: positioned player2 at (" .. tostring(pos2.X) .. ", " .. tostring(pos2.Y) .. ")")
    end

    runner.settle_ticks_remaining = runner.current_capture_mode == "animated"
            and Config.VIEWPORT_SETTLE_TICKS_ANIMATED
        or Config.VIEWPORT_SETTLE_TICKS
    set_state(STATE_SETTLING)
end

local function begin_scenario()
    runner.seed = runner.master_rng:Next()
    runner.scenario_id = runner.session_prefix .. "_" .. tostring(runner.seed)
    local target_stage = current_stage()
    Log.Info(
        "begin_scenario: "
            .. runner.scenario_id
            .. " (route "
            .. runner.route_index
            .. "/"
            .. #runner.route
            .. " = "
            .. target_stage
            .. ", room "
            .. runner.rooms_this_stage
            .. "/"
            .. runner.rooms_this_stage_target
            .. ")"
    )

    if not ScenarioBuilderM1.IsOnStage(target_stage) then
        -- Either a real stage advance (rooms_this_stage_target reached) or
        -- the rare case of a "continue" that resumed on a different floor:
        -- goto the target stage and wait for MC_POST_NEW_ROOM, same as a
        -- normal stage load.
        enter_waiting_for_room()
        ScenarioBuilderM1.GoToStage(target_stage)
        return
    end

    -- Set state to WAITING_FOR_ROOM *before* calling TeleportToRandomRoom,
    -- not after it returns: Game():ChangeRoom() fires MC_POST_NEW_ROOM
    -- synchronously, inside the call, not on some later frame. Setting
    -- state afterward would mean the real room-load event fires and gets
    -- ignored (state still whatever it was before this scenario began),
    -- then permanently stuck waiting for an event that already happened.
    enter_waiting_for_room()
    local rng = RNG()
    rng:SetSeed(runner.seed, 0)
    if not ScenarioBuilderM1.TeleportToRandomRoom(rng) then
        -- No alternative room found (or the room list couldn't be read) --
        -- pose the current room directly instead of waiting for a room
        -- load that isn't coming. Overrides the WAITING_FOR_ROOM set above.
        Log.Info("begin_scenario: no alternative room found, posing current room directly")
        pose_and_settle()
    end
end

-- Cleans up the exchange files for the current scenario (if any) and
-- immediately poses the next one, so a daemon hiccup (missed ack, timeout)
-- is self-healing rather than a dead end -- there's no toggle key left to
-- kick the loop back into life.
local function abort_scenario(reason)
    Log.Info("ABORT: " .. reason .. " (scenario " .. tostring(runner.scenario_id) .. ") -- retrying")
    Exchange.Cleanup(runner.scenario_id)
    runner.scenario_id = nil
    begin_scenario()
end

-- MC_POST_GAME_STARTED: fires for every new run (new game, continue, or a
-- restart after death) -- including the ones this mod itself triggers via
-- start_new_run(). Every run is forced to go through exactly one
-- "restart <char>" (see start_new_run) before real setup happens here, so
-- the very first firing after a game launch/continue is *not* acted on
-- directly -- it's redirected into start_new_run() too, and only the
-- resulting second firing (awaiting_character_restart already true) does
-- the actual route-building/baseline-item work. This guarantees every run
-- gets a fresh random character, not just ones following an exhausted
-- route.
function mod:OnGameStarted()
    Log.Info("MC_POST_GAME_STARTED fired (awaiting_character_restart=" .. tostring(runner.awaiting_character_restart) .. ")")

    if not runner.awaiting_character_restart then
        start_new_run()
        return
    end
    runner.awaiting_character_restart = false

    -- True run start: a fresh, unpredictable master seed for this run --
    -- everything else this run (route choice, per-room-visit seeds) is
    -- derived from repeated master_rng:Next() calls.
    runner.master_rng = RNG()
    runner.master_rng:SetSeed(Random(), 35)
    runner.session_prefix = tostring(Random())

    runner.baseline_applied = false

    local route_rng = RNG()
    route_rng:SetSeed(runner.master_rng:Next(), 0)
    runner.route = StageRoute.BuildRandomRoute(route_rng)
    runner.route_index = 1
    runner.rooms_this_stage = 0

    local stage_count_rng = RNG()
    stage_count_rng:SetSeed(runner.master_rng:Next(), 0)
    runner.rooms_this_stage_target = roll_rooms_for_stage(current_stage(), stage_count_rng)

    Log.Info("run started (session " .. runner.session_prefix .. "), route = " .. table.concat(runner.route, " -> "))
    begin_scenario()
end

mod:AddCallback(ModCallbacks.MC_POST_GAME_STARTED, mod.OnGameStarted)

-- MC_POST_UPDATE: room-change timeout, settle countdown, and polling for
-- the daemon's capture ack.
function mod:OnUpdate()
    -- Pose forcing must run on every tick through WAITING_FOR_CAPTURE, not
    -- just SETTLING: the engine reverts the player's animation each frame,
    -- and the daemon grabs its frame after settling already ended, so
    -- stopping here would leave the captured frame showing the idle
    -- revert.
    if runner.pose ~= nil and (runner.state == STATE_SETTLING or runner.state == STATE_WAITING_FOR_CAPTURE) then
        Pose.Apply(Isaac.GetPlayer(0), runner.pose, runner.pose_frame)
        local player2 = get_second_player()
        if player2 ~= nil then
            Pose.Apply(player2, runner.pose, runner.pose_frame)
        end
    end

    if runner.state == STATE_WAITING_FOR_ROOM then
        runner.room_wait_ticks_remaining = runner.room_wait_ticks_remaining - 1
        if runner.room_wait_ticks_remaining <= 0 then
            abort_scenario("room change timed out -- MC_POST_NEW_ROOM never fired (rejected/ignored change?)")
        end
        return
    end

    if runner.state == STATE_SETTLING then
        runner.settle_ticks_remaining = runner.settle_ticks_remaining - 1
        if runner.settle_ticks_remaining > 0 then
            return
        end

        Log.Info("settle complete for " .. tostring(runner.scenario_id) .. ", collecting ground truth")
        local player = Isaac.GetPlayer(0)
        local ground_truth = GroundTruthQuery.CollectState(
            player,
            runner.scenario_id,
            runner.seed,
            current_stage(),
            runner.viewport_half_extent,
            runner.current_item_delta,
            get_second_player()
        )
        if not Exchange.WritePending(runner.scenario_id, ground_truth) then
            abort_scenario("failed writing exchange file -- check Config.EXCHANGE_DIR")
            return
        end
        runner.capture_ticks_remaining = Config.CAPTURE_TIMEOUT_TICKS
        set_state(STATE_WAITING_FOR_CAPTURE)
        return
    end

    if runner.state == STATE_WAITING_FOR_CAPTURE then
        if Exchange.PollAck(runner.scenario_id) then
            Exchange.Cleanup(runner.scenario_id)
            runner.pose = nil
            Log.Info("scenario " .. runner.scenario_id .. " captured")

            if runner.viewport_index < #runner.viewport_positions then
                runner.viewport_index = runner.viewport_index + 1
                advance_viewport()
                return
            end

            runner.scenario_id = nil

            runner.rooms_this_stage = runner.rooms_this_stage + 1
            if runner.rooms_this_stage >= runner.rooms_this_stage_target then
                runner.rooms_this_stage = 0
                runner.route_index = runner.route_index + 1
                if runner.route_index > #runner.route then
                    -- Route exhausted (just finished Home): start a whole
                    -- new run with a new random character, rather than
                    -- looping back to stage 1 on the same character/items.
                    -- start_new_run()'s restart *is* the item/hearts wipe
                    -- -- a brand-new player object already has zero items
                    -- and default hearts, no manual clearing needed.
                    Log.Info("route exhausted, starting a new run")
                    start_new_run()
                    return
                end

                local stage_count_rng = RNG()
                stage_count_rng:SetSeed(runner.master_rng:Next(), 0)
                runner.rooms_this_stage_target = roll_rooms_for_stage(current_stage(), stage_count_rng)
            end

            begin_scenario()
            return
        end

        runner.capture_ticks_remaining = runner.capture_ticks_remaining - 1
        if runner.capture_ticks_remaining % CAPTURE_HEARTBEAT_TICKS == 0 then
            Log.Info(
                "still waiting for daemon ack on "
                    .. tostring(runner.scenario_id)
                    .. " ("
                    .. runner.capture_ticks_remaining
                    .. "/"
                    .. Config.CAPTURE_TIMEOUT_TICKS
                    .. " ticks left)"
            )
        end
        if runner.capture_ticks_remaining <= 0 then
            abort_scenario("capture timed out -- is the Python daemon running?")
        end
    end
end

mod:AddCallback(ModCallbacks.MC_POST_UPDATE, mod.OnUpdate)

-- MC_POST_NEW_ROOM: advances WAITING_FOR_ROOM -> SETTLING once the stage
-- change from begin_scenario() has actually loaded a room.
function mod:OnNewRoom()
    if runner.state ~= STATE_WAITING_FOR_ROOM then
        Log.Info("MC_POST_NEW_ROOM fired while in state " .. runner.state .. " -- ignoring")
        return
    end
    -- Log the entered room's descriptor in full -- the game's own "Room
    -- X.Y(Name)" line omits the shape, which is exactly what's needed to
    -- correlate room loads with the teleport log in
    -- scenario/build_m1.lua after a ChangeRoom crash.
    local descriptor = Game():GetLevel():GetCurrentRoomDesc()
    if descriptor ~= nil and descriptor.Data ~= nil then
        Log.Info(
            "MC_POST_NEW_ROOM fired, entered room idx="
                .. tostring(descriptor.GridIndex)
                .. " type="
                .. tostring(descriptor.Data.Type)
                .. " variant="
                .. tostring(descriptor.Data.Variant)
                .. " shape="
                .. tostring(descriptor.Data.Shape)
                .. " name=\""
                .. tostring(descriptor.Data.Name)
                .. "\""
        )
    else
        Log.Info("MC_POST_NEW_ROOM fired, posing new room")
    end
    pose_and_settle()
end

mod:AddCallback(ModCallbacks.MC_POST_NEW_ROOM, mod.OnNewRoom)

-- MC_POST_GAME_END fires when the game-over screen appears (IsGameOver
-- true) or an ending sequence starts (IsGameOver false, e.g. beating Mom or
-- a later boss) -- only the true case is a death. This is a safety net
-- independent of the MC_ENTITY_TAKE_DMG cancellation below: that callback
-- only stops *combat* damage, but a death can also come directly from the
-- health-randomization code in spawning/hearts.lua (e.g. AddHearts/
-- AddMaxHearts briefly hitting 0 on some character-specific edge case),
-- which never goes through MC_ENTITY_TAKE_DMG at all. Reuses
-- start_new_run() (the same restart path used once a route is exhausted)
-- rather than a bespoke recovery path -- a fresh random character/route is
-- exactly what should happen next either way.
--
-- Whether calling Isaac.ExecuteCommand("restart ...") while the actual
-- game-over screen is still showing dismisses it cleanly (rather than
-- queuing behind it or getting ignored) is unverified.
---@param is_game_over boolean
function mod:OnGameEnd(is_game_over)
    if not is_game_over then
        return
    end
    Log.Info("MC_POST_GAME_END fired (game over) -- player died, starting a new run")
    start_new_run()
end

mod:AddCallback(ModCallbacks.MC_POST_GAME_END, mod.OnGameEnd)

-- MC_ENTITY_TAKE_DMG, filtered to EntityType.ENTITY_PLAYER: unconditionally
-- cancels all player damage by returning false ("false" cancels the damage
-- entirely, before it's applied). Needed so an accidental hit during
-- randomization/repositioning never changes the player's actual heart
-- count out from under the ground-truth capture for that room visit.
---@param _entity Entity
---@param _amount number
---@param _flags integer
---@param _source EntityRef
---@param _countdown integer
---@return boolean
function mod:OnPlayerTakeDamage(_entity, _amount, _flags, _source, _countdown)
    return false
end

mod:AddCallback(ModCallbacks.MC_ENTITY_TAKE_DMG, mod.OnPlayerTakeDamage, EntityType.ENTITY_PLAYER)
