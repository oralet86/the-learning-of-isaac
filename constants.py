from pathlib import Path

TARGET_WIDTH = 284
TARGET_HEIGHT = 160
TARGET_SIZE = (TARGET_WIDTH, TARGET_HEIGHT)

FPS = 20
FRAME_SKIP = 3

CODEC = "av1_nvenc"

DOWNLOAD_DIR = Path("downloads")
COMPRESSED_DIR = Path("compressed")
