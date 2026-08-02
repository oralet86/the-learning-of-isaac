-- Shared dual-output debug logging: every line goes to both log.txt
-- (Isaac.DebugString -- persistent, but not visible in-game) and the
-- visible in-game debug console overlay (Isaac.ConsoleOutput -- ephemeral,
-- but lets the state machine be watched live without alt-tabbing to a log
-- file).
---@class Log
local Log = {}

---@param msg string
function Log.Info(msg)
    local line = "[tloi_datagen] " .. msg
    Isaac.DebugString(line)
    Isaac.ConsoleOutput(line .. "\n")
end

return Log
