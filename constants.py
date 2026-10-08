from pathlib import Path

SEED = 42

# Target resolution for training data. Video is resized to this during preprocessing.
TARGET_WIDTH = 224
TARGET_HEIGHT = 128
TARGET_SIZE: tuple[int, int] = (TARGET_WIDTH, TARGET_HEIGHT)

# Fraction of width/height kept by the center crop applied to training frames,
# to exclude the non-gameplay HUD border.
FRAME_CROP_FRAC = 1.0

# Duration threshold for AFK detection (idle frames removed).
AFK_DURATION_SECONDS = 3.0

MASTER_FPS = 60

# Sync self-check threshold (ms), reported in drift_report.txt.
SYNC_DROPPED_FRAME_GAP_MS = 25.0

# Directories
DATA_DIR = Path("data")

DOWNLOAD_DIR: Path = DATA_DIR / Path("downloaded_gameplay")
RUNS_DIR: Path = DATA_DIR / Path("recorded_gameplay")
PROCESSED_DATA_DIR: Path = DATA_DIR / Path("processed_recorded_gameplay")
DATA_CHUNKS_DIR: Path = DATA_DIR / Path("recorded_gameplay_chunks")
INVERSE_DYNAMIC_RUNS_DIR: Path = DATA_DIR / Path("id_runs")

KEY_ORDER = [
    "w",
    "a",
    "s",
    "d",
    "up",
    "down",
    "left",
    "right",
    "e",
    "space",
    "q",
    "tab",
    "ctrl",
]

# Model Training

VAL_FRAC = 0.10
TEST_FRAC = 0.10
TRIM_EDGE_FRAC = 0.01
DUPE_EPSILON = 1.0
MAX_SIMUL_KEYS = 5

# --- Chunked video preprocessing ---
RAW_CHUNK_SIZE_GB = 5

# Number of run_* directories processed concurrently by chunk_video.py's
# ThreadPoolExecutor — the real ceiling here is CPU/RAM: each worker holds
# ~RAW_CHUNK_SIZE_GB of grayscale frames and spawns 1 ffmpeg subprocess.
CHUNK_VIDEO_WORKERS = 8

EPOCHS = 100
LR = 2e-3
GT_BATCH_SIZE = 256
PLATEAU_PATIENCE = 4
LR_SCHEDULE_FACTOR = 0.2
PATIENCE = 10
NUM_WORKERS = 4

# Dynamic Weight Averaging (DWA) per-head loss weights, applied once per
# epoch after validation. The EMA smooths per-head val losses (masked heads
# average over few samples and are noisy); the temperature controls how
# aggressively weights rebalance (lower = more); the floor keeps a
# permanently stuck head's weight from collapsing toward zero.
DWA_EMA_ALPHA = 0.85
DWA_TEMPERATURE = 1.0
DWA_FLOOR_FRAC = 0.05

WORST_PRED_BATCH_SIZE = 256
TOP_K_WORST_PREDS = 4
EPSILON = 1e-8
THRESHOLD = 0.5
RUN_TIMESTAMP_FORMAT = "%Y%m%d_%H%M"

# Profiling

PROFILE_WAIT = 10
PROFILE_WARMUP = 10
PROFILE_STEPS = 50

# Run modes

DEFAULT_MODE = "default"
ABLATION_DATA_FRAC = 0.10
