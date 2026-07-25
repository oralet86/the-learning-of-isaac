import os
from pathlib import Path
import subprocess
import sys

from tqdm import tqdm

from constants import (
    CODEC,
    COMPRESSED_DIR,
    CQ,
    DOWNLOAD_DIR,
    TARGET_HEIGHT,
    TARGET_WIDTH,
)

VIDEO_EXTENSIONS = {".mp4", ".webm", ".mkv", ".flv", ".avi", ".mov", ".3gp"}

VIDEO_FILTERS = f"scale={TARGET_WIDTH}:{TARGET_HEIGHT},format=gray,mpdecimate=hi=225:lo=15:frac=0.25,setpts=N/FRAME_RATE/TB"


def find_videos(root: Path) -> list[Path]:
    videos = []
    for dirpath, _, filenames in os.walk(root):
        for name in filenames:
            if Path(name).suffix.lower() in VIDEO_EXTENSIONS:
                videos.append(Path(dirpath) / name)
    return videos


def reencode(src: Path, dst: Path, _codec: str) -> bool:
    dst.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        "ffmpeg",
        "-i",
        str(src),
        "-an",
        "-vf",
        VIDEO_FILTERS,
        "-c:v",
        CODEC,
        "-cq",
        CQ,
        "-rc",
        "vbr",
        "-preset",
        "p5",
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        "-y",
        "-loglevel",
        "error",
        str(dst),
    ]

    try:
        subprocess.run(cmd, check=True, capture_output=True, text=True)
        return True
    except subprocess.CalledProcessError as e:
        tqdm.write(f"  ffmpeg: {e.stderr.strip()[:200]}")
        return False


def main() -> None:
    if not DOWNLOAD_DIR.exists():
        sys.exit(f"'{DOWNLOAD_DIR}/' not found. Run videos.py first.")

    print(f"Encoder: {CODEC}")
    videos = find_videos(DOWNLOAD_DIR)

    if not videos:
        sys.exit(f"No videos found in '{DOWNLOAD_DIR}/'.")
    print(f"Found {len(videos)} video(s) to compress.\n")

    for src in tqdm(videos, desc="Compressing", unit="vid", dynamic_ncols=True):
        rel = src.relative_to(DOWNLOAD_DIR)
        dst = COMPRESSED_DIR / rel.with_suffix(".mp4")

        if dst.exists() and dst.stat().st_size > 0:
            continue

        if not reencode(src, dst, CODEC):
            tqdm.write(f"  FAILED: {rel}")

    orig_gb = sum(v.stat().st_size for v in find_videos(DOWNLOAD_DIR)) / 1e9
    comp_gb = sum(v.stat().st_size for v in find_videos(COMPRESSED_DIR)) / 1e9

    if orig_gb > 0 and comp_gb > 0:
        pct = (1 - comp_gb / orig_gb) * 100
        print(f"\n{orig_gb:.1f} GB  ->  {comp_gb:.1f} GB  ({pct:.0f}% smaller)")

    print(f"Done. Compressed files are in '{COMPRESSED_DIR}/'.")


if __name__ == "__main__":
    main()
