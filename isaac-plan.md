# The Binding of Isaac: Rebirth — RL Training Plan

## Goal

Train a neural network to play The Binding of Isaac: Rebirth and beat bosses, using reinforcement learning with pure visual observation (no game modding). The trained agent should be capable of completing daily runs automatically.

---

## Hardware

- GPU: NVIDIA RTX 4070 Laptop
- CPU: AMD Ryzen 7840HS (8 cores / 16 threads)
- RAM: 64 GB

---

## 1. Frame Capture

### Method: OBS Virtual Camera (V4L2)

Isaac runs fullscreen. OBS captures the game window and exposes it as a V4L2 device (`/dev/videoN`) via `v4l2loopback`. Python reads frames from the V4L2 device using FFmpeg. This avoids X11/Wayland/compositor issues entirely.

- **Production:** FFmpeg reads from V4L2 device, encodes with AV1 NVENC, writes to mp4
- **RL training loop:** Read raw frames from V4L2 device directly in Python (no encoding)

### Recording Pipeline (`record.py`)

Implemented and working:

1. OBS Virtual Camera → `/dev/video2` at 1920×1080 YUYV
2. FFmpeg captures at 60fps, applies filter: `fps=20,scale=284:160,format=gray`
3. Encodes with `av1_nvenc` (CQ=32, VBR)
4. Simultaneously logs keyboard state via `pynput` at 20fps
5. Output: `output.mp4` + `keys.json` + `meta.json` per session

### RL Frame Preprocessing Pipeline

For the training loop, frames are read live from V4L2 (no encoding) and processed:

1. Capture raw frame at 1920×1080 from `/dev/videoN`
2. Crop to gameplay area (remove HUD borders — hearts, map, etc.)
3. Convert to grayscale
4. Resize to 128×128
5. Stack last 4 frames as input channels (grayscale × 4)

### Note on Isaac Resolution

Isaac renders at the display resolution (1920×1080 fullscreen). The playfield is the inner portion; HUD elements are at fixed positions along the edges. Cropping to the playfield and resizing to 128×128 preserves sufficient detail for bullet hell while minimizing VRAM usage.

---

## 2. No Modding — Pure Visual State Extraction

**No Lua mod is used.** All game state is extracted from visual frames via the neural network itself. This eliminates modding complexity, Isaac API dependencies, and runtime coupling.

### Visual State Sources

| Data | Source | Method |
|---|---|---|
| Player position / movement | Frame pixels | Impala CNN backbone |
| Enemy positions / bullets | Frame pixels | Impala CNN backbone |
| Health (hearts) | HUD crop | Hearts micro-CNN |
| Coins / bombs / keys | HUD crop | Resource micro-CNNs |
| Active item charge | HUD crop | Charge micro-CNN |
| Trinket identity | HUD crop | Trinket micro-CNN |
| Usable item (card/pill/rune) | HUD crop | Pocket item micro-CNN |
| Map layout | HUD crop | Map micro-CNN |

### Reward Signals (Visual Detection)

Without mod events, rewards are detected from visual changes between frames:

| Reward Signal | Detection Method |
|---|---|
| Room clear | Room transition animation (black frame sequence) / door opening detection |
| Boss kill | Boss HP bar disappearance + room clear transition |
| Taking damage | HUD hearts count decrease between frames |
| Death | Game over screen detection |
| Coin/bomb/key pickup | HUD resource count increase between frames |
| Enemy hit | Visual tear-impact flash on enemies (optional; adds noise) |

### Optional: Pause-Screen Item Extraction

As an extension (not required for the core architecture), a small classifier could read the pause-screen item list to identify held items. This would restore item embedding functionality in a no-modding framework. However, item effects are already observable through HUD stat changes and visual consequences, so the LSTM + visual backbone can learn them implicitly.

---

## 3. Neural Network Architecture

### Overview Diagram

```
                       ┌──────────────────┐
                       │  Frame Stack     │
                       │  4 × 128 × 128   │
                       │  (grayscale)     │
                       └────────┬─────────┘
                                │
        ┌───────────────────────┼───────────────────────┐
        │                       │                       │
 ┌──────▼──────┐   ┌────────────▼────────────┐  ┌──────▼──────┐
 │  HUD Micro- │   │     Impala CNN          │  │  Visual     │
 │  CNNs       │   │     (Backbone)          │  │  Feature    │
 │  (per ele-  │   │                         │  │  Vector     │
 │   ment)     │   └────────────┬────────────┘  └──────┬──────┘
 └──────┬──────┘                │                       │
        │                       │                       │
        └───────────────────────┼───────────────────────┘
                                │
                       ┌────────▼─────────┐
                       │  Concatenate +   │
                       │  FC Projection   │
                       └────────┬─────────┘
                                │
                       ┌────────▼─────────┐
                       │  LSTM (256)      │
                       └────────┬─────────┘
                                │
                 ┌──────────────┴──────────────┐
                 │                             │
        ┌────────▼────────┐           ┌───────▼────────┐
        │  Actor Head     │           │  Value Head    │
        │  (multi-        │           │  (scalar)      │
        │   discrete)     │           │                │
        └─────────────────┘           └────────────────┘
```

### 3.1 Visual Backbone: Impala-Style CNN

A custom CNN designed for game screens from scratch (no ImageNet pretraining — pixel art renders transfer learning useless).

**Structure:**
- Input: 4-channel grayscale stack, 128×128
- 3 residual blocks with progressively increasing channels (16 → 32 → 64)
- Large initial kernels (7×7, stride 2)
- Output: 512-dim feature vector

This architecture is battle-tested in DeepMind environments and handles 2D game visuals well with low parameter count (~1-2M).

### 3.2 Item Awareness (Implicit)

**No explicit item ID input.** Without a mod reporting item IDs, the agent learns item effects implicitly:

- **Item effects on stats** are reflected in HUD micro-CNN outputs (health changes, stat changes)
- **Item effects on gameplay** (tear modifiers, familiar behaviors, room effects) are observed in the main frame and learned through the visual backbone + LSTM
- **Item synergies** are discovered through reward signals over time

This is a trade-off: the agent must learn to recognize items through their visual consequences rather than having them explicitly encoded. The 128×128 frame + HUD micro-CNNs + LSTM temporal memory provide enough signal for this.

**Optional extension:** A pause-screen item classifier (small CNN reading the pause menu item list) could restore explicit item embeddings if needed. Run during room transitions where the game is paused anyway.

### 3.3 HUD Micro-CNNs

Separate tiny CNNs for each HUD element. Each runs on a small hardcoded crop of the raw frame. These are NOT fed into the main backbone — they operate independently for maximum accuracy.

| HUD Element | Crop Size (approx) | Output |
|---|---|---|
| Hearts | ~100×30 | Scalar (health fraction) |
| Coins | ~30×15 | Scalar (count normalized) |
| Bombs | ~30×15 | Scalar (count normalized) |
| Keys | ~30×15 | Scalar (count normalized) |
| Active Charge | ~40×15 | Scalar (charge fraction) |
| Trinket | ~30×30 | Small vector (identity embedding) |
| Usable item (card/pill/rune) | ~30×30 | Small vector (identity embedding) |
| Map | ~80×60 | 64-dim feature vector |

Each micro-CNN has 2-3 conv layers and average pools to its output. Combined parameter count is negligible (~5-10K total).

**Hardcoded crop coordinates** are determined once from a screenshot and remain constant — Isaac's HUD layout is static.

### 3.4 Concatenation and Projection

All feature sources are concatenated:
- Visual backbone output (512-dim)
- HUD micro-CNN outputs (hearts, coins, bombs, keys, charge, trinket, usable, map: ~100-dim total)

A single FC layer projects this to 256-dim before the LSTM.

### 3.5 LSTM

- Single layer, 256 hidden units
- Handles memory across time: room transitions (black screen), item effects that manifest over time, tactical context

**Hidden state reset:** LSTM hidden states are zeroed when an episode terminates (floor clear → boss kill → floor transition). Not reset within the same floor.

### 3.6 Action Head: Multi-Discrete

Isaac's keyboard controls are factorized into independent action groups:

| Action Group | Cardinality | Keys |
|---|---|---|
| Movement | 9 (8 dirs + still) | WASD or arrows |
| Shooting | 5 (4 dirs + no fire) | Arrow keys |
| Bomb | 2 (place / no-op) | E |
| Active item | 2 (use / no-op) | Space |
| Pill/Card/Rune | 2 (use / no-op) | Q |

**Why multi-discrete:** Flattening would produce a 9×5×2×2×2 = 360-action space, most of which is unreachable or nonsensical. Multi-discrete outputs 4 independent softmax heads (movement, shooting) and 3 independent Bernoulli heads (bomb, item, card) with a total of ~17 logits. Far more efficient to learn.

Each action is sampled independently. The joint action is the combination.

### 3.7 Value Head

Single scalar output from the LSTM hidden state. Estimates expected return from the current state.

---

## 4. Training

### 4.1 RL Framework

SB3 will be used.

### 4.2 Algorithm

**PPO** (Proximal Policy Optimization) with:
- GAE (Generalized Advantage Estimation), λ = 0.95
### 4.3 Episode Structure

- **One episode = one floor** (Basement I → boss kill → floor transition)
- An Isaac run (5-6 floors) produces 5-6 sequential episodes
- When the boss dies, signal `done=True` → PPO flush → LSTM reset
- The game instance continues naturally to the next floor (items, health, coins persist)
- Floor transition dead time (2-3 sec black screen and stage name): read frames but don't act

### 4.4 Frame Skip & Action Repeat

**Frame skip = 3.** The model outputs an action every 3 frames (effective ~10 actions/sec at 30fps). The action is repeated for those 3 frames. This provides enough temporal resolution for bullet hell dodging while keeping LSTM unroll length manageable.

### 4.5 Parallel Environments

Target: 4-8 parallel Isaac instances. The 7840HS CPU (8 cores) is the likely bottleneck since Isaac is single-threaded per instance.

- Start with 4 instances and measure CPU headroom
- Scale to 6-8 if possible
- Each instance requires its own OBS Virtual Camera device (configure multiple v4l2loopback devices)
- Each instance sends actions via a virtual input device or direct keyboard emulation

### 4.6 Reward Design (Dense Rewards → Visual Detection)

All rewards are detected visually from frame changes. Dense rewards are critical — Isaac has long stretches between room clears with no natural reward signal.

| Event | Reward | Detection Method |
|---|---|---|
| Room clear | +1.0 | Room transition animation / door opening |
| Boss kill | +10.0 | Boss HP bar gone + room clear |
| Taking damage (half heart) | -0.5 | HUD hearts count decrease |
| Death (game over) | -5.0, episode end | Game over screen / death animation |
| Picking up coin | +0.01 | HUD coin count increase |
| Picking up bomb | +0.01 | HUD bomb count increase |
| Picking up key | +0.01 | HUD key count increase |
| Active item used | -0.01 | Charge bar went from full to empty |
| Standing still (no dmg taken) | 0.0 | — |

**Enemy hit detection** is omitted without mod data — too noisy to detect visually (tear impacts may not cause visible flashes). The agent learns to target enemies through room clear and boss kill rewards instead.

### 4.7 Termination & Timeout

- Episode end conditions: boss kill (success) or player death (failure)
- Max step timeout: 10,000 steps per episode (approximately 10 minutes at 3-frame skip). If reached, episode terminates to prevent excessive rollout lengths breaking PPO's batching.

---

## 5. Daily Run Automation

The trained agent should be able to complete Isaac's **Daily Run** — a fixed-seed run available once per day — automatically, from menu navigation to run completion.

### Menu Navigation

A simple state machine navigates Isaac's menus using template matching or a small classifier:

1. **Title screen** → detect "Press Start" → press Enter
2. **File select** → detect save slot → press Enter  
3. **Main menu** → detect "Daily Run" option → navigate and select
4. **Character select** → always choose Isaac (default) → confirm
5. **Run begins** → hand control to the RL agent

Template matching on static menu elements (title logo, menu text, save slot highlight) works reliably since Isaac's UI is pixel-art with fixed layouts.

### Run Lifecycle

1. **Menu entry** → state machine navigates to Daily Run
2. **Active play** → RL agent controls movement and actions
3. **Floor transitions** → agent continues (LSTM state persists across floors within a run)
4. **Run end** (death or completion) → return to menu
5. **Next run** → either quit (daily run already completed) or start a normal run for additional training

### Automation Script

A wrapper script (`auto_run.py`) handles the full cycle:

```
while True:
    navigate_menus()           # get to gameplay
    run_agent()                # RL policy loop until done
    if daily_run_completed():
        wait_until_next_day()  # or start normal runs for training
```

### Scheduling

- **Daily Run resets** at midnight UTC / local time
- Cron job or systemd timer triggers the automation script at the appropriate time
- The agent can also run normal (non-daily) runs continuously for training

---

## 6. Human Demonstrations (Pretraining)

### 6.1 Recording Your Own Play

**What to capture:**
- Screen frames at the acting rate (every 3 frames)
- Keyboard state mapped to the multi-discrete action format

**Tools (implemented and working):**
- Frames: `record.py` — OBS Virtual Camera → FFmpeg (V4L2 + AV1 NVENC)
- Inputs: `pynput` keyboard listener, synchronized at 20fps
- Output: `labeled_runs/run_TIMESTAMP/output.mp4` + `keys.json` + `meta.json`
- Viewer: `view_recording.py` — tkinter GUI for playback, frame stepping, and key state overlay

**YouTube pipeline (implemented):**
- `download_videos.py` — downloads Isaac playlists via yt-dlp
- `compress.py` — batch re-encodes to 284×160 grayscale at 20fps for consistency

Record 30-60 minutes of decent play including both good runs and deaths. Death episodes are important — the value network needs to see terminal states.

### 6.2 Leveraging Web Videos (Inverse Dynamics Model)

Isaac runs are widely available on YouTube/Twitch, but they lack keyboard input data.

**The Inverse Dynamics Approach:**

1. **Record your own (frame, action) dataset** — 30-60 min
2. **Train an inverse dynamics model** on this data: a small classifier that predicts the multi-discrete action given `(s_t, s_{t+1})` — two consecutive frames
   - Input: two 128×128 grayscale frames = 2 channels
   - Architecture: small CNN → 256-dim → 5 parallel heads for each action group
   - This model learns "between these two frames, the player moved right and fired down"
   - Cheap to train — small model, moderate dataset size
3. **Download Isaac runs from the web**, extract frames at the acting rate
4. **Pseudo-label the web videos:** for each adjacent frame pair, run the inverse dynamics model to estimate the human action
5. **Combine datasets:** your labeled data + pseudo-labeled web videos → now you have a large (frame, estimated-action) corpus
6. **Behavior Cloning pre-training:** train the full policy network on this corpus using cross-entropy loss (one per action head, summed)

**Limitation:** Inverse dynamics may be ambiguous during intense combat (multiple actions produce similar frame transitions). Prioritize your own labeled data with higher loss weight, and use web-video pseudo-labels as auxiliary data with lower weight.

### 6.3 BC + PPO Integration

After BC pretraining, switch to PPO:

- **Option A (Recommended): Demo buffer injection** — insert human demo trajectories into PPO's rollout buffer with high advantage estimates (GAE computed with value targets from the episode). The policy is pulled toward human actions during PPO updates without gradient interference.
- **Option B: Joint loss** — add a decaying BC loss term to the PPO objective: `L = L_PPO + λ(t) · L_BC` where λ decays linearly to 0 over training.

### 6.4 Fallback: State-Only Imitation (GAIL)

If inverse dynamics proves too noisy, fall back to state-only imitation:
- Train a discriminator to distinguish expert states (from web videos) from agent states
- Use the discriminator's output as an **imitation reward bonus** during PPO
- The agent is guided toward human-like state distributions without needing to know the exact human actions

---

## 7. Misc. Implementation Notes

### 7.1 LSTM Masking

When an episode terminates (boss kill or death), the LSTM hidden state for that environment must be zeroed before the next rollout step. Track a `done` mask per environment and zero the corresponding hidden states. Failing to do this carries a dead player's memory into a new floor and produces nonsense gradients.

### 7.2 Normalization

- **Observations:** Running mean/std normalization of frame pixels (or feature vectors post-backbone)
- **Rewards:** Scale by running standard deviation (commonly used divisor: `reward_std + 1e-8`)
- **Value targets:** Normalize advantages within each minibatch

### 7.3 Gradient Clipping

- Global gradient norm clip at 0.5 (PPO standard)
- Prevents single-batch explosion from rare Isaac events (e.g., sudden death with many enemies on screen)

### 7.4 Float16 / AMP

Isaac training will likely be model-throughput-limited (not GPU-memory-limited on a 4070). Automatic mixed precision can speed up training ~1.5x with negligible accuracy loss. Enable via PyTorch's `torch.cuda.amp`.

### 7.5 Debugging Tools

- Log episode returns, episode lengths, value loss, policy loss, entropy to TensorBoard or Weights & Biases
- Periodically save rollout videos (actual gameplay) for visual inspection
- Log per-action-head entropy separately to detect mode collapse in specific action groups

### 7.6 Reproducibility

- Seed everything (Python, NumPy, PyTorch, environment)
- However: Isaac's procedural generation makes true reproducibility impossible. Focus on mean/std over multiple runs with different seeds.

---

## 8. Development Phases

### Phase 1: Data Pipeline
- [x] Screen capture via OBS Virtual Camera (V4L2) — `record.py`
- [x] Input recording with `pynput` — `record.py` key logging synchronized at 20fps
- [x] Recording viewer — `view_recording.py` tkinter playback with key overlay
- [x] YouTube video download pipeline — `download_videos.py` + `compress.py`
- [ ] HUD crop coordinate extraction (define bounding boxes for hearts, coins, etc.)
- [ ] Visual reward detection (room clear, boss kill, damage, death from frame changes)
- [ ] Daily run menu navigation state machine
- [ ] Action sending (keyboard emulation to control Isaac)

### Phase 2: Human Recording
- [ ] Record 30-60 min of (frame, action) pairs
- [ ] Download Isaac web videos, extract frames

### Phase 3: Inverse Dynamics Model
- [ ] Train on own recordings (frame_t, frame_t+1 → action)
- [ ] Pseudo-label web videos
- [ ] Combine into full BC dataset

### Phase 4: Network Implementation
- [ ] Impala CNN backbone
- [ ] HUD micro-CNNs (hearts, coins, bombs, keys, charge, trinket, pocket, map)
- [ ] LSTM core
- [ ] Multi-discrete actor head
- [ ] Value head
- [ ] Optional: pause-screen item classifier

### Phase 5: BC Pretraining
- [ ] Train full policy network on combined human dataset
- [ ] Evaluate: does the model produce human-like actions?

### Phase 6: RL Training
- [ ] PPO implementation (adapt CleanRL)
- [ ] Parallel Isaac environments (4-8)
- [ ] Visual reward detection pipeline
- [ ] Train until convergence on basement boss

### Phase 7: Iteration
- [ ] Hyperparameter tuning
- [ ] Demo buffer injection (if BC behaviors are being lost)
- [ ] Scale to more floors if performance is good

---

## 9. Expected Challenges

| Challenge | Mitigation |
|---|---|
| Sparse natural rewards | Visual reward detection (room clear, damage, etc.) |
| Procedural room generation | Diverse training via parallel envs + long training |
| Bullet hell visual complexity | Frame stack gives velocity; LSTM gives dodging patterns |
| Item effect discovery without mod | Visual backbone + LSTM learn effects from gameplay feedback |
| Visual reward detection noise | Smooth over multiple frames; use HUD micro-CNNs for reliable signals (health, resources) |
| Daily run menu navigation | Template matching on static pixel-art UI; robust state machine with timeouts |
| Inverse dynamics noise in combat | Weight own labeled data higher; GAIL fallback |
| CPU bottleneck with parallel envs | Reduce env count; offload preprocessing to GPU |
| Long episodes (10 min) | Floor-based episodes + max step timeout |
| LSTM gradient issues | Done masking + gradient clipping |
