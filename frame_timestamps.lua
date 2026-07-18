obs = obslua

local buf = {}
local file = nil
local recording = false

local SIGNAL_FILE = "/tmp/isaac_autopilot_output"

function get_outdir()
    local f = io.open(SIGNAL_FILE, "r")
    if f then
        local dir = f:read("*l")
        f:close()
        if dir and dir ~= "" then
            return dir
        end
    end
    return nil
end

function frame_callback()
    if not recording or not file then
        return
    end
    local t = obs.obs_get_video_frame_time()
    table.insert(buf, t)
    if #buf >= 500 then
        flush()
    end
end

function flush()
    if not file or #buf == 0 then
        return
    end
    for _, v in ipairs(buf) do
        file:write(v, "\n")
    end
    buf = {}
end

function on_event(event)
    if event == obs.OBS_FRONTEND_EVENT_RECORDING_STARTED then
        local outdir = get_outdir()
        if not outdir then
            return
        end
        recording = true
        local outdir = get_outdir()
        local ts = os.date("%Y%m%d_%H%M%S")
        local path = outdir .. "/frame_times_" .. ts .. ".log"
        file = io.open(path, "w")
        if file then
            print("[frame_timestamps] recording started -> " .. path)
            file:setvbuf("line")
        end
    elseif event == obs.OBS_FRONTEND_EVENT_RECORDING_STOPPED then
        recording = false
        flush()
        if file then
            file:close()
            file = nil
        end
        print("[frame_timestamps] recording stopped")
    end
end

function script_load(settings)
    obs.obs_frontend_add_event_callback(on_event)
    obs.obs_add_main_render_callback(frame_callback)
    print("[frame_timestamps] loaded, waiting for recording start")
end

function script_unload()
    recording = false
    obs.obs_remove_main_render_callback(frame_callback)
    flush()
    if file then
        file:close()
    end
end
