from pathlib import Path

SEED = 42

# Codec options
TARGET_WIDTH = 284
TARGET_HEIGHT = 160
TARGET_SIZE: tuple[int, int] = (TARGET_WIDTH, TARGET_HEIGHT)
FPS = 20
FRAME_SKIP = 3
CODEC = "av1_nvenc"
CQ = "32"

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

VAL_FRAC = 0.10
TEST_FRAC = 0.10
MAX_SAME_FRAMES = 60
TRIM_EDGE_FRAC = 0.01
DUPE_EPSILON = 1.0
CHUNK_SIZE = 10000

EPOCHS = 100
LR = 4e-3
MIN_LR = 5e-6
PATIENCE = 5
PLATEAU_EPOCHS = 10
NUM_WORKERS = 4

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
