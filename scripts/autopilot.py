#!/usr/bin/env python3
"""Single-click OBS recording + key logging with automatic file collection and sync.

Replaces the manual flow of starting OBS, running record.py, stopping both,
and manually moving files.  Connects to OBS via obs-websocket, starts key
logging via pynput, and on Ctrl+C stops both, collects the video + frame
timestamps log into a run directory, and runs post-sync.

Usage:
  python autopilot.py                        # defaults localhost:4455, no password
  python autopilot.py --host 192.168.1.5     # remote OBS
  python autopilot.py --password mypass      # authenticated
  python autopilot.py --no-sync              # skip post_sync step
"""

from __future__ import annotations

import argparse
from contextlib import suppress
from datetime import datetime
import json
from pathlib import Path
import shutil
import sys
import time
from typing import Any

import obsws_python as obs
from obsws_python.error import OBSSDKError
from pynput import keyboard

from constants import MASTER_FPS, RUNS_DIR
from scripts.post_sync import sync_run

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SIGNAL_FILE = Path("/tmp/isaac_autopilot_output")
RECORDING_POLL_INTERVAL_S = 0.25
MAX_POST_STOP_WAIT_S = 10.0


# ---------------------------------------------------------------------------
# Key normalization (mirrors record.py)
# ---------------------------------------------------------------------------


def key_to_name(k: keyboard.Key | keyboard.KeyCode | None) -> str:
    """Normalize a pynput key event to a stable string name."""
    if k is None:
        return "None"
    try:
        return k.char  # type: ignore[union-attr]
    except AttributeError:
        return str(k).replace("Key.", "")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _connect_obs(
    host: str, port: int, password: str, timeout: int | None
) -> obs.ReqClient:
    """Connect to OBS WebSocket and return the request client."""
    return obs.ReqClient(host=host, port=port, password=password, timeout=timeout)


def _check_not_already_recording(cl: obs.ReqClient) -> None:
    """Raise SystemExit if OBS is already recording."""
    status = cl.get_record_status()
    if status.output_active:  # type: ignore[union-attr]
        sys.exit("OBS is already recording. Stop the current recording first.")


def _start_key_listener(
    events: list[tuple[float, str, bool]],
) -> keyboard.Listener:
    """Start pynput keyboard listener, appending press/release events."""

    def on_press(k: keyboard.Key | keyboard.KeyCode | None) -> None:
        events.append((time.monotonic(), key_to_name(k), True))

    def on_release(k: keyboard.Key | keyboard.KeyCode | None) -> None:
        events.append((time.monotonic(), key_to_name(k), False))

    listener = keyboard.Listener(on_press=on_press, on_release=on_release)
    listener.start()
    return listener


def _wait_for_recording_stop(cl: obs.ReqClient, timeout: float | None = None) -> None:
    """Poll OBS until recording output is confirmed inactive."""
    timeout = timeout or MAX_POST_STOP_WAIT_S
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        status = cl.get_record_status()
        if not status.output_active:  # type: ignore[union-attr]
            return
        time.sleep(RECORDING_POLL_INTERVAL_S)
    print("  WARN: recording did not confirm stopped within timeout")


def _find_latest_video(directory: Path) -> Path | None:
    """Return the newest video file in *directory*, or None."""
    video_exts = {".mp4", ".mkv", ".mov", ".avi", ".flv", ".ts"}
    candidates: list[Path] = []
    for ext in video_exts:
        candidates.extend(directory.glob(f"*{ext}"))
    if not candidates:
        return None
    return max(candidates, key=lambda p: p.stat().st_mtime)


def _find_frame_times_log(search_dirs: list[Path]) -> Path | None:
    """Return the newest frame_times*.log across *search_dirs*, or None."""
    candidates: list[Path] = []
    for d in search_dirs:
        candidates.extend(d.glob("frame_times*.log"))
    if not candidates:
        return None
    return max(candidates, key=lambda p: p.stat().st_mtime)


# ---------------------------------------------------------------------------
# Signal file (LUA bridge)
# ---------------------------------------------------------------------------


def _write_signal(run_dir: Path) -> None:
    """Write the target run directory for the LUA frame-timestamps script."""
    SIGNAL_FILE.write_text(str(run_dir))


def _clear_signal() -> None:
    """Remove the signal file so stale state does not affect manual OBS use."""
    with suppress(OSError):
        SIGNAL_FILE.unlink(missing_ok=True)


# ---------------------------------------------------------------------------
# Run directory & meta
# ---------------------------------------------------------------------------


def _create_run_dir() -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = RUNS_DIR / f"run_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def _save_run_meta(
    run_dir: Path,
    key_events: list[tuple[float, str, bool]],
    start_epoch: float,
    start_monotonic: float,
) -> dict[str, Any]:
    """Write key_events.json and meta.json.  Returns meta dict."""
    key_events.sort(key=lambda x: x[0])
    elapsed_s = time.monotonic() - start_monotonic

    with open(run_dir / "key_events.json", "w") as f:
        json.dump(key_events, f)

    meta: dict[str, Any] = {
        "start_epoch": start_epoch,
        "start_monotonic": start_monotonic,
        "start_time": datetime.fromtimestamp(start_epoch).isoformat(),
        "end_time": datetime.now().isoformat(),
        "duration_seconds": round(elapsed_s, 1),
        "fps": MASTER_FPS,
        "key_event_count": len(key_events),
    }
    with open(run_dir / "meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    rate = len(key_events) / elapsed_s if elapsed_s > 0 else 0
    print(f"  Keys: {len(key_events)} events ({rate:.1f}/s) over {elapsed_s:.1f}s")
    return meta


# ---------------------------------------------------------------------------
# File collection
# ---------------------------------------------------------------------------


def _collect_video(cl: obs.ReqClient, run_dir: Path, dir_ok: bool) -> None:
    """Rename OBS video to master.mp4 inside run_dir.

    If *dir_ok* is True the video was written directly into *run_dir*.
    Otherwise it is fetched from OBS's default record directory and moved.
    """
    if dir_ok:
        src_dir = run_dir
    else:
        src_dir = Path(cl.get_record_directory().record_directory)  # type: ignore[union-attr]

    video = _find_latest_video(src_dir)
    if video is None:
        print("  WARN: no video file found")
        return

    target = run_dir / "master.mp4"
    if video.resolve() == target.resolve():
        # Already named master.mp4
        return

    if video.parent == run_dir:
        video.rename(target)
    else:
        shutil.move(str(video), str(target))
    print(f"  Video: {video.name} -> master.mp4")


def _collect_frame_log(run_dir: Path) -> None:
    """Find the frame_times*.log and move it into *run_dir*."""
    search_dirs = [run_dir, Path.cwd()]
    log = _find_frame_times_log(search_dirs)
    if log is None:
        print("  WARN: no frame_times*.log found")
        return
    if log.parent == run_dir:
        print(f"  Frame times: {log.name} (already in run dir)")
        return
    dest = run_dir / log.name
    shutil.move(str(log), str(dest))
    print(f"  Frame times: {log.name} -> run dir")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:

    parser = argparse.ArgumentParser(
        description="Single-click OBS + key recording with automatic sync."
    )
    parser.add_argument(
        "--host", default="localhost", help="OBS WebSocket host (default: localhost)"
    )
    parser.add_argument(
        "--port", type=int, default=4455, help="OBS WebSocket port (default: 4455)"
    )
    parser.add_argument("--password", default="", help="OBS WebSocket password")
    parser.add_argument(
        "--timeout",
        type=int,
        default=5,
        help="OBS WebSocket connect timeout in seconds (default: 5)",
    )
    parser.add_argument(
        "--no-sync", action="store_true", help="Skip automatic post-sync run"
    )
    args = parser.parse_args()

    # ── setup ──────────────────────────────────────────────────────────

    run_dir = _create_run_dir().resolve()
    _write_signal(run_dir)
    cleanup_done = False

    print(f"Run dir: {run_dir}")

    # ── connect OBS ─────────────────────────────────────────────────────

    try:
        cl = _connect_obs(args.host, args.port, args.password, args.timeout)
    except (OBSSDKError, OSError) as e:
        _clear_signal()
        sys.exit(f"Cannot connect to OBS at {args.host}:{args.port} — {e}")

    print(f"  Connected. OBS version: {cl.get_version().obs_version}")  # type: ignore[union-attr]
    _check_not_already_recording(cl)

    # ── set record directory ────────────────────────────────────────────

    dir_ok = True
    try:
        cl.set_record_directory(str(run_dir))
    except Exception:
        dir_ok = False
        print("  WARN: set_record_directory failed, video will land in OBS default dir")

    # ── start OBS recording ─────────────────────────────────────────────

    try:
        cl.start_record()
    except Exception as e:
        _clear_signal()
        cl.disconnect()
        sys.exit(f"Failed to start OBS recording: {e}")
    print("  OBS recording started.")

    # ── start key listener ──────────────────────────────────────────────

    key_events: list[tuple[float, str, bool]] = []
    listener = _start_key_listener(key_events)

    start_epoch = time.time()
    start_monotonic = time.monotonic()

    try:
        # ── main wait ───────────────────────────────────────────────────

        print("Recording keys + video.  Press Ctrl+C to stop.")
        try:
            while True:
                time.sleep(3600)
        except KeyboardInterrupt:
            pass

        # ── stop ────────────────────────────────────────────────────────

        print("\nStopping...")
        listener.stop()

        print("  Stopping OBS recording...")
        cl.stop_record()
        _wait_for_recording_stop(cl)

        # Give OBS a moment to flush the file to disk.
        time.sleep(0.5)

        # ── save & collect ──────────────────────────────────────────────

        _save_run_meta(run_dir, key_events, start_epoch, start_monotonic)
        _collect_video(cl, run_dir, dir_ok)
        _collect_frame_log(run_dir)

        _clear_signal()
        cleanup_done = True

        # ── post-sync ───────────────────────────────────────────────────

        if not args.no_sync:
            master_mp4 = run_dir / "master.mp4"
            if not master_mp4.exists():
                print("  SKIP post_sync: master.mp4 not found")
            elif not list(run_dir.glob("frame_times*.log")):
                print("  SKIP post_sync: frame_times*.log not found")
            else:
                print("\nRunning post_sync...")
                sync_run(run_dir)

        print(f"\nDone.  Output: {run_dir}")

    finally:
        # Ensure we never leave the signal file lying around.
        if not cleanup_done:
            with suppress(Exception):
                _clear_signal()
        with suppress(Exception):
            cl.disconnect()


if __name__ == "__main__":
    main()
