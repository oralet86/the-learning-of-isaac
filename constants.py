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
