# The Learning of Isaac

Teaching an agent to play *The Binding of Isaac: Repentance+* from the screen
alone: it sees pixels and presses keys, exactly like a human player. Nothing is
injected into the game and no mod is required.

The long-term goal is an agent that can complete daily runs on a normal,
unmodified copy of the game.

## What is in here

The repository has two halves that share one action space (13 keyboard keys,
see `KEY_ORDER` in `constants.py`).

**1. Live reinforcement-learning loop** (`live/`, `container/`)

- A Gymnasium environment, `IsaacEnv`, that returns a preprocessed frame as the
  observation and takes a 13-key binary vector as the action, 20 times a second.
- Two interchangeable backends behind it:
  - **Training**: several copies of the game, each in its own container with a
    virtual display and software rendering. A small Python sidecar in each
    container captures the window and injects key presses over TCP.
  - **Inference**: your real game on your desktop, captured through OBS and
    controlled through a virtual keyboard.
- Scripted menu navigation, quick restarts, and automatic pausing whenever the
  trainer stops sending actions.

**2. Inverse-dynamics pipeline** (`inverse_dynamics/`, `scripts/`, `components/`)

- Tools to record your own gameplay with frame-accurate key logs.
- A model that learns "which keys were pressed between these frames", intended
  for labelling unlabelled gameplay video so a policy can be pretrained on it.

### Current status

| Piece | State |
|---|---|
| Containerised game, sidecar, pause/reset handling | Working |
| Vectorised training environment (`live.train`) | Working, driven by a random placeholder agent |
| Inference loop on the real game (`live.play`) | Written, driven by the same placeholder agent |
| Recording and inverse-dynamics training | Working |
| Reward and episode-end detection | Interface only (`live/events.py`); episodes currently end by step limit |
| The actual RL algorithm and policy network | Not implemented yet |

## Requirements

- Linux with an X11 session (inference uses `xdotool` and `/dev/uinput`).
- Python 3.14 and [uv](https://docs.astral.sh/uv/).
- [Podman](https://podman.io/) for training containers.
- `ffmpeg` for video decoding.
- For recording and inference: OBS Studio with obs-websocket enabled, and the
  `v4l2loopback` kernel module for the OBS virtual camera.
- A legally owned copy of *The Binding of Isaac: Rebirth* with the Repentance+
  DLC. The actual game is git-ignored for obvious reasons. If you'd like to run this
  yourself, find yourself a suitable version of Isaac, place it under `container/isaac/`
  and place a %100 completed savefile under `container/save/`.

Each containerised game needs roughly two CPU cores and 0.6 GB of RAM. An
8-core machine runs four comfortably, which is the default.

## Installation

```bash
git clone git@github.com:oralet86/the-learning-of-isaac.git
cd the-learning-of-isaac
uv sync --extra dev
```

Run every command below from the repository root, either inside the
environment (`source .venv/bin/activate`) or prefixed with `uv run`.

## Training mode: games in containers

### 1. Provide the game and a save

```
container/
├── isaac/    # the game folder, containing isaac-ng.exe
└── save/     # exactly one .dat save file
```

- `container/isaac/` must be a copy of the game that starts **without the Steam
  client running**, because the containers have no Steam.
- `container/save/` holds one persistent save file (any file name ending in
  `.dat`). It is installed as save slot 1 before every game launch, so each run
  starts from the same unlocks. A fully unlocked save makes "Random Character"
  draw from the whole roster.

### 2. Build the image

```bash
python -m container.build
```

This builds `localhost/tloi-isaac:dev`. Always use this script instead of
calling `podman build` yourself: it passes paths and the sidecar port from the
Python code into the image, and the build fails without them. Rebuild after
any change under `container/sidecar/` or to `constants.py`.

### 3. Run a rollout

```bash
python -m live.train
```

This starts four containers, navigates each game to the first room of a run,
steps them all at 20 Hz with the placeholder agent, and removes the containers
on exit (including on Ctrl+C or a crash).

| Option | Default | Meaning |
|---|---|---|
| `--num-envs` | `4` | Number of concurrent games |
| `--total-steps` | `1000` | Vector steps to run |
| `--max-episode-steps` | `600` | Truncate an episode after this many steps |
| `--game-width`, `--game-height` | `480`, `270` | Game window size in each container |
| `--base-port` | `5600` | Host port of the first container; the rest follow |
| `--image` | `localhost/tloi-isaac:dev` | Image to run |

The log reports vector steps per second and the worst lateness. Anything other
than about 20 steps/s and 0 ms outside of resets means the machine is
overloaded; lower `--num-envs`.

### Using the environment from your own code

```python
import numpy as np

from live.fleet import ContainerFleet, FleetConfig
from live.vector import make_vector_env, pause_all

with ContainerFleet(FleetConfig(num_envs=4)) as fleet:
    envs = make_vector_env(fleet.endpoints, max_episode_steps=600)
    try:
        observations, _ = envs.reset()  # menu navigation, about 45 s
        for _ in range(1000):
            actions = envs.action_space.sample()
            observations, rewards, terminated, truncated, infos = envs.step(actions)

        pause_all(envs)  # before any long gap, e.g. a policy update
        ...              # the next envs.step() resumes the games
    finally:
        envs.close()
```

- **Observation**: `(128, 224)` float16 grayscale in `[0, 1]`.
- **Action**: `MultiDiscrete([2] * 13)` over `KEY_ORDER`; `1` holds the key
  until the next action.
- **`infos["late_ms"]`**: how far a step missed its 50 ms tick.

### How the games behave

- **Real time**: the game never waits for the trainer. Keys stay held between
  steps.
- **Pausing**: if a sidecar receives nothing for 0.25 s it releases all keys
  and opens the pause menu; the next step closes it. The sidecar checks the
  screen after each press, so the toggle cannot drift out of sync.
- **Resets**: the first reset walks the menus (skip intro, file select, new
  run, "Random Character"). Later resets hold `R` for a quick restart. If a
  step fails, the next reset relaunches the game.

### Tuning a container

Pass these as environment variables to a container (for example
`podman run -e MENU_BOOT_WAIT_S=30 ...`):

| Variable | Default | Meaning |
|---|---|---|
| `GAME_WIDTH`, `GAME_HEIGHT` | `480`, `270` | Game window size |
| `IDLE_PAUSE_S` | `0.25` | Silence before auto-pause; `0` disables it |
| `MENU_BOOT_WAIT_S` | `20` | Wait after the window appears before pressing keys |
| `MENU_RANDOM_LEFT_PRESSES` | `1` | Left presses from the default character to "Random" |
| `MENU_RUN_START_WAIT_S` | `10` | Wait after confirming the character |
| `LOG_LEVEL` | `INFO` | Set to `DEBUG` to log pause-menu checks |

All menu timings are fields of `MenuConfig` in `container/sidecar/menu.py`;
each can be overridden as `MENU_<FIELD_NAME>`.

## Inference mode: the real game

The agent plays the game installed on your machine. Nothing is changed in the
game; frames are read from OBS and keys are sent through a virtual keyboard.

One-time setup:

1. Load the virtual camera module:
   `sudo modprobe v4l2loopback exclusive_caps=0`
2. In OBS, add the game window as a source and enable the WebSocket server
   (default `localhost:4455`, no password).
3. Allow your user to create virtual input devices: add yourself to the
   `input` group (or run as root) so `/dev/uinput` is writable.

Then:

1. Start OBS and the game, and begin the run you want played.
2. Run `python -m live.play`.
3. Press `h` to hand control to the agent, `h` again to take it back, and
   `Ctrl+C` to quit. All keys are released whenever the agent is switched off.

To check the capture on its own:

```bash
python -m live.stream_game --probe   # report the virtual camera's state
python -m live.stream_game --debug   # live preview window with frame stats
```

The agent used by both modes lives in `live/model.py`. Replace `RandomAgent`
with a class that has the same `predict(frame) -> list[int]` method.

## Recording gameplay and training the inverse-dynamics model

### 1. Record

Install `frame_timestamps.lua` in OBS (Tools → Scripts); it logs the exact
time of every recorded frame. Then:

```bash
python -m scripts.autopilot
```

It starts an OBS recording and a key logger together. Press `Ctrl+C` to stop;
the video, key log and frame timestamps are collected into
`data/recorded_gameplay/run_<timestamp>/` and aligned into a per-frame
`keys.json`. Use `--host` / `--password` for a non-default OBS connection and
`--no-sync` to skip the alignment.

Review a recording with `python -m scripts.view_recording`.

### 2. Build the dataset

```bash
python -m inverse_dynamics.chunk_video   # recordings -> raw frame chunks
python -m inverse_dynamics.preprocess    # chunks -> filtered train/val/test shards
```

The first step decodes, downsizes and grayscales each run into
`data/recorded_gameplay_chunks/`. The second drops idle and ambiguous frames,
splits each chunk into train/validation/test, and writes
`data/processed_recorded_gameplay/`. Both accept `--help` for paths, chunk
size, split fractions and worker count.

### 3. Train

```bash
python -m inverse_dynamics.train --model resnet_lite
```

| Option | Meaning |
|---|---|
| `--model {resnet_lite,resnet_vpt}` | Architecture |
| `--epochs` | Number of epochs |
| `--mode {default,profile,ablation}` | Normal run, profiler run, or a run on 10% of the data |
| `--window-size`, `--stride`, `--label-index` | Override the model's frame-window settings |

Runs are written to `data/id_runs/` with checkpoints, curves and metrics.

### Unlabelled video

`python -m scripts.download_videos` downloads the playlists listed in
`playlists.txt` into `data/downloaded_gameplay/`, as raw material for
labelling with a trained inverse-dynamics model.

## Repository layout

```
constants.py           Shared constants: key order, control rate, paths, hyperparameters
container/
  Containerfile        Image: Wine, Xvfb, Mesa software rendering, sidecar
  build.py             Builds the image with values taken from the Python code
  sidecar/             Runs inside the container: capture, key injection,
                       pacing, menu scripts, pausing, TCP server
live/
  env.py               IsaacEnv (Gymnasium)
  backend.py           Interface both backends implement
  remote.py            Training backend: TCP client for a sidecar
  local.py             Inference backend: OBS capture + virtual keyboard
  fleet.py             Starts and removes training containers
  vector.py            One worker process per container
  events.py            Reward and episode-end interface
  train.py, play.py    Entry points for the two modes
inverse_dynamics/      Dataset building and training for the inverse-dynamics model
components/            Model architectures, loss, shard storage, data loading
scripts/               Recording, sync, viewers, video download, resizing
```

## Development

```bash
ruff format . && ruff check .
pyright
pytest components container/tests live/tests inverse_dynamics
```

The tests need neither a container nor the game: the sidecar's logic is
exercised with a fake keyboard, clock and game, and the TCP client is tested
against the real request handler on a loopback socket.

## Notes and limitations

- The sidecar listens without authentication; `live/fleet.py` publishes its
  port on `127.0.0.1` only. Do not expose it on a network.
- Software rendering is the CPU bottleneck. Busy rooms cost more than the
  empty starting room, so leave headroom when choosing `--num-envs`.
- The sidecar pre-accepts the game's first-run beta notice through an option
  whose name contains the game version (`container/sidecar/game.py`). After a
  game update the notice returns and the menu script will stall until that
  name is updated.
- Frames from the containers are rendered in software at a small window size,
  while inference frames come from your real game through OBS. Compare the two
  after preprocessing before trusting a policy trained in one and run in the
  other.
- This project is not affiliated with the game's developers or publisher.
