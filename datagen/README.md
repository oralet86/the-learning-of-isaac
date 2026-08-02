# datagen — synthetic training-data generator

Dev-only tool. Drives a REPENTOGON Lua mod (`tloi_datagen/`) to pose real in-engine
scenes and capture labeled screenshots for CV training. Not part of `live/`, which stays
mod-free for the vanilla-compatible RL runtime.

## What it generates

Each run picks a fully random playable character and a random valid stage route from
Basement through Home (including Hush and The Void), then visits 5-10 random rooms per
stage. Every room visit randomizes:

- Player position (and, for Jacob & Esau / Tainted Forgotten, a second player positioned
  close to the first), hearts/coins/bombs/keys, held passive items (a once-per-run
  baseline plus a per-visit add/remove/none delta, logged explicitly as `item_delta`),
  active item + charge, and pocket-slot card/pill.
- Door state per slot (open/closed/locked/barred/broken, or a separate blown-open roll
  for secret/super-secret/ultra-secret room doors, which are otherwise plain wall and
  correctly absent from ground truth until blown).
- Camera viewport: every vanilla room shape is a valid teleport target, and each shape
  gets a representative set of viewport positions (a 3x3 grid for 2x2/L-shapes, 3 points
  along the long axis for long rooms, a few random positions for 1x1), so one room visit
  yields 1-9 captures.
- `capture_mode`: "idle" (settled frame) or "animated" (shorter settle window, so the
  frame lands mid-animation instead).

A handful of characters have real gameplay constraints that would otherwise produce
health/item combinations impossible in real play (e.g. Keeper's coin-heart cap, The
Lost's zero health, Tainted Isaac's 8-item cap) -- these are enforced in
`tloi_datagen/spawning/hearts.lua` and `spawning/items.lua`.

Ground truth is always read back live from the game's own state after posing rather than
assumed from what was requested, so a partial/clamped spawn is reflected accurately.

Not implemented: output directory restructuring by stage/room-type/shape (flat
`images/`/`labels/` is used instead), and the separate paired-delta door dataset
described in the project's original plan doc -- both were explicitly descoped.

## How it works

The game is launched with the `--luadebug` Steam launch option, which unlocks the
`io`/`os` Lua standard libraries REPENTOGON mods otherwise lack. That makes plain
file-based IPC between the mod and Python possible, so the whole loop runs unattended —
no manual console typing, no log-file scraping, and no manual start/stop step:

1. As soon as a run begins (`MC_POST_GAME_STARTED`), the mod (`tloi_datagen/main.lua`)
   starts its auto-generation loop: teleport to a random room on the current stage (or
   `goto`/`stage` there first if not already on it), pose the scene, and settle.
2. Once settled, the mod collects ground truth (`query/ground_truth.lua`) and writes it
   as `<scenario_id>.json` into the shared exchange directory (`ipc/exchange.lua`) — this
   file's existence is the "ready for capture" signal.
3. `datagen/daemon.py`, already running and polling that directory, claims the file
   (reads it, then deletes it — so a slow capture can never double-process it), grabs a
   frame via the existing OBS/ffmpeg pipeline (`live/stream_game.py`), and writes the
   paired image + label under `datagen/output/`. It then writes back
   `<scenario_id>.done`.
4. The mod polls for that `.done` file, cleans up, and immediately poses the next
   scenario (sweeping through the rest of the current room's viewport positions before
   moving to the next room). This repeats for as long as the game is open — a daemon
   hiccup (missed ack, timeout) is logged and retried rather than halting the loop, since
   there's no key to restart it by hand.

See `datagen/exchange.py` and `tloi_datagen/ipc/exchange.lua` for the exact protocol.

## Setup

1. Add `--luadebug` to the game's Steam launch options (Properties → General → Launch
   Options). This is separate from REPENTOGON and from `EnableDebugConsole` — it's what
   grants mods `io`/`os` access. Note it also lifts the sandbox for *any* enabled mod,
   and (like any mod) gates achievements on this save until Mom is defeated with mods
   off.
2. Symlink or copy `tloi_datagen/` into your Isaac mods folder, and enable it plus
   REPENTOGON in-game.
3. The exchange directory is `datagen/runtime/` on the Linux side
   (`datagen/run_config.py`'s `DEFAULT_EXCHANGE_DIR`), and the mod reaches the same
   folder via its Wine `Z:` drive path (`tloi_datagen/config.lua`'s
   `Config.EXCHANGE_DIR`). If you move to a different machine/install, update both
   together.
4. Set up OBS with a scene capturing the game window and the virtual camera running
   (see `live/stream_game.py`'s module docstring for the `v4l2loopback` setup).

## Running

Start the daemon, then start or continue a run — generation begins automatically:

```
uv run python -m datagen.daemon -v
```

It watches `datagen/runtime/` and logs each capture as it happens. Generation keeps
going for as long as the game is open (each new run restarts the scenario counter and
picks up where the loop left off). Stop the daemon itself with Ctrl-C. Output lands in
`datagen/output/images/` and `datagen/output/labels/`.

Use `datagen/viewer.py` (`uv run python -m datagen.viewer`) to page through captured
image/label pairs and sanity-check what actually got written.

## Known limitations

- Whether calling `restart` while the actual game-over screen is showing dismisses it
  cleanly (rather than queuing behind it or getting ignored) is unverified.
- Whether `Isaac.GetPlayer(1)` reliably returns `nil` (rather than erroring) for
  non-duo characters is unverified — the mod wraps the call in `pcall` regardless, so
  either outcome is safe.
- Door state randomization can't get a distinct "locked"/"barred" sprite variant
  (`GridEntityDoor.Variant` is read-only in this REPENTOGON build) — the live door state
  ground truth reports is still correct, this is a cosmetic gap only.
- `RoomShape`s other than the common ones (1x1, long rooms, 2x2, L-shapes) haven't been
  separately confirmed to actually spawn as normal traversable rooms rather than
  internal-only shapes.

## Why file-based IPC

REPENTOGON exposes no render-to-file API, so screenshot capture has no in-engine
equivalent — Python + OBS/ffmpeg is the capture mechanism regardless of how the mod and
daemon talk to each other. The `--luadebug` launch flag unlocks the `io`/`os` Lua
standard libraries REPENTOGON mods otherwise lack, which is what makes plain JSON
file-based IPC between the two processes possible: the mod writes a ground-truth file
once a scene is settled, the daemon claims it, captures, and acks. This lets the mod
drive its own generation loop automatically from the moment a run starts, with no
per-scenario human step and no synthetic-input automation needed — `datagen/daemon.py`
runs as a persistent background watcher rather than a per-scenario CLI invocation.
