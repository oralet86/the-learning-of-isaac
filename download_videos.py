import os
import random
import re
import sys
import time

from tqdm import tqdm
import yt_dlp

from constants import DOWNLOAD_DIR

RATE_LIMIT = int(os.environ.get("RATE_LIMIT", str(4 * 1024 * 1024)))

PLAYLISTS_FILE = "playlists.txt"
DOWNLOAD_DIR_STR = str(DOWNLOAD_DIR)
FAILED_LOG = os.path.join(DOWNLOAD_DIR_STR, "failed.log")
ARCHIVE_FILE = os.path.join(DOWNLOAD_DIR_STR, ".archive.txt")


def ensure_playlists_file() -> list[str]:
    if not os.path.exists(PLAYLISTS_FILE):
        with open(PLAYLISTS_FILE, "w") as f:
            f.write("# Add one playlist URL per line, then run this script again.\n")
        print(f"Created '{PLAYLISTS_FILE}'. Add your playlist URLs and run again.")
        sys.exit(0)

    with open(PLAYLISTS_FILE) as f:
        urls = [line.strip() for line in f if line.strip() and not line.startswith("#")]

    if not urls:
        print(f"'{PLAYLISTS_FILE}' is empty. Add playlist URLs and run again.")
        sys.exit(1)

    return urls


def sanitize_name(name: str, max_len: int = 80) -> str:
    name = re.sub(r'[<>:"/\\|?*]', "_", name)
    name = name.strip(". ")
    if len(name) > max_len:
        name = name[:max_len].rstrip()
    return name or "Unknown"


def log_error(msg: str) -> None:
    os.makedirs(DOWNLOAD_DIR, exist_ok=True)
    with open(FAILED_LOG, "a") as f:
        f.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')}  {msg}\n")


def make_download_opts(folder_slug: str) -> dict:
    return {
        "format": "bestvideo[height<=360][fps<=20]/bestvideo[height<=360]/best[height<=360]",
        "format_sort": ["codec:av1:av01", "codec:vp9.2", "vcodec:vp9", "vcodec:h264"],
        "outtmpl": f"{DOWNLOAD_DIR}/{folder_slug}/%(title).100s.%(ext)s",
        "ignoreerrors": False,
        "quiet": True,
        "no_warnings": True,
        "continuedl": True,
        "retries": 3,
        "fragment_retries": 3,
        "sleep_interval_requests": 2,
        "ratelimit": RATE_LIMIT,
        "download_archive": ARCHIVE_FILE,
        "noplaylist": True,
    }


def extract_playlist(url: str) -> tuple[str, list[dict]]:
    opts = {
        "extract_flat": "in_playlist",
        "quiet": True,
        "ignoreerrors": True,
    }
    with yt_dlp.YoutubeDL(opts) as ydl:
        info = ydl.extract_info(url, download=False)

    if info is None:
        raise RuntimeError(f"Could not extract info from: {url}")

    if "entries" in info:
        title = info.get("title") or info.get("playlist_title") or "Unknown_Playlist"
        entries = [e for e in info["entries"] if e is not None]
    else:
        title = info.get("title") or "Unknown"
        entries = [info]

    return title, entries


def download_one(video_url: str, folder_slug: str) -> bool:
    opts = make_download_opts(folder_slug)
    try:
        with yt_dlp.YoutubeDL(opts) as ydl:
            ydl.download([video_url])
        return True
    except Exception as e:
        log_error(f"FAILED  {video_url}  --  {e}")
        return False


def main() -> None:
    urls = ensure_playlists_file()
    os.makedirs(DOWNLOAD_DIR, exist_ok=True)

    tqdm.write(f"Found {len(urls)} playlist(s).\n")

    all_failed: list[tuple[str, str, str]] = []

    for playlist_url in tqdm(urls, desc="Playlists", unit="pl", dynamic_ncols=True):
        try:
            raw_title, raw_entries = extract_playlist(playlist_url)
        except Exception as e:
            log_error(f"PLAYLIST EXTRACTION FAILED  {playlist_url}  --  {e}")
            tqdm.write(f"  SKIPPED: {playlist_url} ({e})")
            continue

        folder = sanitize_name(raw_title)
        entries = [e for e in raw_entries if e is not None and e.get("id")]

        if not entries:
            tqdm.write(f"  {raw_title}: 0 videos")
            continue

        tqdm.write(
            f"  {raw_title}  ->  {DOWNLOAD_DIR}/{folder}/  ({len(entries)} videos)"
        )

        failures: list[tuple[str, str, str]] = []
        desc = folder[:25].ljust(25)

        for entry in tqdm(
            entries, desc=desc, unit="vid", leave=False, dynamic_ncols=True
        ):
            video_id = entry["id"]
            video_url = (
                entry.get("url") or f"https://www.youtube.com/watch?v={video_id}"
            )
            video_title = entry.get("title", video_id)

            if download_one(video_url, folder):
                pass
            else:
                failures.append((video_url, video_title, folder))

            time.sleep(random.uniform(1.5, 3.5))

        if failures:
            tqdm.write(f"    {len(failures)} failed (will retry)")
            all_failed.extend(failures)

    if all_failed:
        tqdm.write(f"\nRetrying {len(all_failed)} failed video(s)...\n")
        still_failed = 0

        for url, title, folder in tqdm(
            all_failed, desc="Retry  ", unit="vid", dynamic_ncols=True
        ):
            if download_one(url, folder):
                pass
            else:
                still_failed += 1
                log_error(f"STILL FAILED  {title}  --  {url}")
            time.sleep(random.uniform(2, 4))

        if still_failed:
            tqdm.write(f"\n{still_failed} still failed. See {FAILED_LOG}")
        else:
            tqdm.write("\nAll retries succeeded!")

    tqdm.write("\nDone.")


if __name__ == "__main__":
    main()
