from pathlib import Path

DEFAULT_SEED = 42

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
DEFAULT_RUNS_DIR: Path = DATA_DIR / Path("recorded_gameplay")
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

DEFAULT_VAL_FRAC = 0.10
DEFAULT_TEST_FRAC = 0.10
DEFAULT_MAX_SAME_FRAMES = 60
DEFAULT_TRIM_EDGE_FRAC = 0.01
DEFAULT_DUPE_EPSILON = 1.0
DEFAULT_CHUNK_SIZE = 10000

DEFAULT_EPOCHS = 100
DEFAULT_LR = 4e-3
DEFAULT_MIN_LR = 5e-6
DEFAULT_PATIENCE = 5
DEFAULT_PLATEAU_EPOCHS = 10
DEFAULT_NUM_WORKERS = 0

ADAMW_FUSED = True

COSINE_T_MAX_DIVISOR = 3

THRESHOLD_CANDIDATES = 81
WORST_PRED_BATCH_SIZE = 256
DEFAULT_TOP_K = 20
METRICS_EPS = 1e-8
DEFAULT_THRESHOLD = 0.5
CHECKPOINT_FREQ = 10
RUN_TIMESTAMP_FORMAT = "%Y%m%d_%H%M"

# Profiling

PROFILE_WAIT = 5
PROFILE_WARMUP = 5
PROFILE_STEPS = 20
