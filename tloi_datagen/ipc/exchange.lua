local JSON = include("vendor.json")
local Config = include("config")
local Log = include("log")

-- File-based exchange with the Python capture daemon (datagen/exchange.py),
-- made possible by the game's `--luadebug` launch flag, which unlocks the
-- `io`/`os` standard libraries REPENTOGON mods otherwise lack (see
-- datagen/README.md). Protocol:
--   - `WritePending` writes `<scenario_id>.json` (the ground truth) once a
--     posed scene has settled -- this file's existence is the "ready for
--     capture" signal the daemon polls for.
--   - The daemon claims the file (reads then deletes it), captures the
--     frame, and writes back `<scenario_id>.done` (empty) once saved.
--   - `PollAck` / `Cleanup` on the mod side watch for and remove that file.
---@class Exchange
local Exchange = {}

---@param path string
---@return boolean
local function file_exists(path)
    local f = io.open(path, "r")
    if f == nil then
        return false
    end
    f:close()
    return true
end

---@param scenario_id string
---@return string
local function pending_path(scenario_id)
    return Config.EXCHANGE_DIR .. scenario_id .. ".json"
end

---@param scenario_id string
---@return string
local function ack_path(scenario_id)
    return Config.EXCHANGE_DIR .. scenario_id .. ".done"
end

-- Writes to a ".tmp" file first, then renames into place, so the daemon
-- (which watches for ".json" files) never observes a partially written file.
---@param scenario_id string
---@param ground_truth table
---@return boolean ok
function Exchange.WritePending(scenario_id, ground_truth)
    local tmp_path = pending_path(scenario_id) .. ".tmp"
    local final_path = pending_path(scenario_id)

    local f, open_err = io.open(tmp_path, "w")
    if f == nil then
        Log.Info("Exchange.WritePending: io.open FAILED for " .. tmp_path .. " -- " .. tostring(open_err))
        return false
    end

    f:write(JSON.encode(ground_truth))
    f:close()

    local ok, rename_err = os.rename(tmp_path, final_path)
    if ok ~= true then
        Log.Info("Exchange.WritePending: os.rename FAILED " .. tmp_path .. " -> " .. final_path .. " -- " .. tostring(rename_err))
        return false
    end

    Log.Info("Exchange.WritePending: wrote " .. final_path)
    return true
end

---@param scenario_id string
---@return boolean
function Exchange.PollAck(scenario_id)
    return file_exists(ack_path(scenario_id))
end

-- Removes both exchange files for `scenario_id`. Safe to call even if the
-- daemon already claimed (deleted) the pending file itself.
---@param scenario_id string
function Exchange.Cleanup(scenario_id)
    os.remove(pending_path(scenario_id))
    os.remove(ack_path(scenario_id))
end

return Exchange
