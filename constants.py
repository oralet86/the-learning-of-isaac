from pathlib import Path

TARGET_WIDTH = 284
TARGET_HEIGHT = 160
TARGET_SIZE = (TARGET_WIDTH, TARGET_HEIGHT)

FPS = 20
FRAME_SKIP = 3

CODEC = "av1_nvenc"
CQ = "32"

DOWNLOAD_DIR = Path("downloads")
COMPRESSED_DIR = Path("compressed")

PROCESSED_DATA_DIR = Path("processed_data")
TRAINING_RUNS_DIR = Path("training_runs")

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

# ---------------------------------------------------------------------------
# Shared
# ---------------------------------------------------------------------------

DEFAULT_SEED = 42

# ---------------------------------------------------------------------------
# Preprocessing
# ---------------------------------------------------------------------------

DEFAULT_RUNS_DIR = Path("labeled_runs")
DEFAULT_VAL_FRAC = 0.10
DEFAULT_TEST_FRAC = 0.10
DEFAULT_MAX_SAME_FRAMES = 60
DEFAULT_TRIM_EDGE_FRAC = 0.01
DEFAULT_DUPE_EPSILON = 1.0
POS_WEIGHT_CLAMP = (0.1, 50.0)

# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

DEFAULT_BATCH_SIZE = 512
DEFAULT_EPOCHS = 100
DEFAULT_LR = 4e-3
DEFAULT_MIN_LR = 5e-6
DEFAULT_DROPOUT = 0.4
DEFAULT_PATIENCE = 5
DEFAULT_PLATEAU_EPOCHS = 10
DEFAULT_NUM_WORKERS = 0
DEFAULT_FOCAL_GAMMA = 2.0

ADAMW_WEIGHT_DECAY = 5e-3
ADAMW_FUSED = True

INPUT_CHANNELS = 2
RESIDUAL_DROPOUT = 0.2
VAL_BATCH_MULTIPLIER = 2

THRESHOLD_CANDIDATES = 81
WORST_PRED_BATCH_SIZE = 256
DEFAULT_TOP_K = 20
METRICS_EPS = 1e-8
DEFAULT_THRESHOLD = 0.5
CHECKPOINT_FREQ = 10
RUN_TIMESTAMP_FORMAT = "%Y%m%d_%H%M"

# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

FIG_DPI = 150
TRAINING_CURVES_FIGSIZE = (14, 10)
PER_KEY_FIGSIZE = (14, 5)
WORST_PRED_N_COLS = 5
WORST_PRED_MAX_ROWS = 4
WORST_PRED_CELL_WIDTH = 4.5
WORST_PRED_CELL_HEIGHT = 2.8
