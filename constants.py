from pathlib import Path

SEED = 42

# Target resolution for training data. Video is resized to this during
# preprocessing. 640x360 (360p) preserves the 16:9 aspect ratio.
TARGET_WIDTH = 640
TARGET_HEIGHT = 360
TARGET_SIZE: tuple[int, int] = (TARGET_WIDTH, TARGET_HEIGHT)
FPS = 20

# Subsampling step from master recording fps to training fps.
# Master is recorded at 60 fps; every 3rd frame gives ~20 fps.
SUBSAMPLE_STEP = 3

# Duration threshold for AFK detection (idle frames removed).
AFK_DURATION_SECONDS = 4.0

# Tolerance for detecting pixel-identical consecutive frames within a window.
# Windows where all consecutive frame pairs match below this threshold are
# discarded (they carry no motion signal).
STATIC_WINDOW_TOLERANCE = 0.0

CODEC = "av1_nvenc"
CQ = "32"

# Master recording options (used by record.py). The master is a high-quality
# 720p60 backup from which all training-rate subsets are derived later.
MASTER_WIDTH = 1280
MASTER_HEIGHT = 720
MASTER_FPS = 60
MASTER_CODEC = "av1_nvenc"
MASTER_CQ = "20"
MASTER_FALLBACK_CODEC = "hevc_nvenc"
MASTER_FALLBACK_CQ = "23"
MASTER_PRESET = "p5"

# Sync self-check thresholds (ms). Driving the desync detection logic in
# record.py and reported in drift_report.txt.
SYNC_DELTA_INTERNAL_MAX_MS = 1.0
SYNC_DELTA_PIPELINE_STD_MAX_MS = 5.0
SYNC_DELTA_PIPELINE_SPAN_MAX_MS = 20.0
SYNC_DROPPED_FRAME_GAP_MS = 25.0

# Directories
DATA_DIR = Path("data")

DOWNLOAD_DIR: Path = DATA_DIR / Path("downloaded_gameplay")
COMPRESSED_DIR: Path = DATA_DIR / Path("compressed_downloaded_gameplay")
RUNS_DIR: Path = DATA_DIR / Path("recorded_gameplay")
PROCESSED_DATA_DIR: Path = DATA_DIR / Path("processed_recorded_gameplay")
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

INPUT_CHANNELS = 2
VAL_FRAC = 0.10
TEST_FRAC = 0.10
TRIM_EDGE_FRAC = 0.01
DUPE_EPSILON = 1.0
CHUNK_SIZE = 10000

EPOCHS = 100
LR = 1e-3
MIN_LR = 1e-6
PATIENCE = 5
PLATEAU_EPOCHS = 10
NUM_WORKERS = 0

ADAMW_FUSED = True

COSINE_T_MAX_DIVISOR = 3

THRESHOLD_CANDIDATES = 81
WORST_PRED_BATCH_SIZE = 256
TOP_K = 20
EPSILON = 1e-8
THRESHOLD = 0.5
CHECKPOINT_FREQ = 10
RUN_TIMESTAMP_FORMAT = "%Y%m%d_%H%M"

# Profiling

PROFILE_WAIT = 5
PROFILE_WARMUP = 5
PROFILE_STEPS = 100
