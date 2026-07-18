#!/usr/bin/env python3
"""Post-hoc sync: reconstruct per-frame key state from raw recording data.

Reads the output of record.py + OBS recording + frame_times*.log and produces:
  keys.json         list[list[str]] — one entry per frame of master.mp4
  drift_report.txt  human-readable summary
  meta.json         updated with frame count, sync stats, video resolution

Usage:
  python post_sync.py data/recorded_gameplay/run_<TS>/
"""

import argparse
from dataclasses import dataclass, field
import json
from pathlib import Path
import subprocess
import sys
from typing import Any

from constants import RUNS_DIR, SYNC_DROPPED_FRAME_GAP_MS

RECON_TOLERANCE_MS = 4.0


# ---------------------------------------------------------------------------
# Data containers
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class DriftStats:
    """Inter-frame gap statistics."""

    dt_arrival_mean_ms: float
    dt_arrival_std_ms: float
    dt_arrival_min_ms: float
    dt_arrival_max_ms: float
    dropped_frame_count: int
    frame_count: int


@dataclass(slots=True)
class SyncValidation:
    """Result of the index-count cross-check."""

    keys_count: int
    frame_meta_count: int
    ok: bool
    message: str = field(default_factory=str)


# ---------------------------------------------------------------------------
# Frame times reader
# ---------------------------------------------------------------------------


def read_frame_times(path: Path) -> list[float]:
    """Read OBS frame timestamps from Lua log (ns, CLOCK_MONOTONIC). Returns seconds."""
    times: list[float] = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            ns = int(line)
            times.append(ns / 1e9)
    if not times:
        print(f"  ERROR: empty frame_times log: {path}")
    return times


# ---------------------------------------------------------------------------
# Key reconstruction
# ---------------------------------------------------------------------------


def reconstruct_keys(
    key_events: list[tuple[float, str, bool]],
    frame_capture_times: list[float],
    tolerance_s: float = RECON_TOLERANCE_MS / 1000.0,
) -> list[list[str]]:
    """Reconstruct per-frame key state from event log and frame capture times.

    For each frame i, replays key events up to frame_capture_times[i] +
    tolerance_s (monotonic). The tolerance prevents hard boundary artifacts
    from sub-ms timing jitter between the key listener and OBS frame clock.
    """
    keys_list: list[list[str]] = []
    state: set[str] = set()
    event_idx = 0
    n = len(frame_capture_times)

    for i in range(n):
        deadline = frame_capture_times[i] + tolerance_s
        while event_idx < len(key_events) and key_events[event_idx][0] <= deadline:
            _t, key, is_press = key_events[event_idx]
            if is_press:
                state.add(key)
            else:
                state.discard(key)
            event_idx += 1
        keys_list.append(sorted(state))

    return keys_list


# ---------------------------------------------------------------------------
# Drift computation
# ---------------------------------------------------------------------------


def compute_drift(frame_times: list[float]) -> DriftStats:
    """Compute inter-frame gap statistics from frame timestamps."""
    n = len(frame_times)
    if n < 2:
        return DriftStats(
            dt_arrival_mean_ms=0.0,
            dt_arrival_std_ms=0.0,
            dt_arrival_min_ms=0.0,
            dt_arrival_max_ms=0.0,
            dropped_frame_count=0,
            frame_count=n,
        )

    dt_arrival_ms = [
        (frame_times[i] - frame_times[i - 1]) * 1000.0 for i in range(1, n)
    ]

    mean = sum(dt_arrival_ms) / len(dt_arrival_ms)
    var = sum((v - mean) ** 2 for v in dt_arrival_ms) / len(dt_arrival_ms)
    dropped = sum(1 for v in dt_arrival_ms if v > SYNC_DROPPED_FRAME_GAP_MS)

    return DriftStats(
        dt_arrival_mean_ms=mean,
        dt_arrival_std_ms=var**0.5,
        dt_arrival_min_ms=min(dt_arrival_ms),
        dt_arrival_max_ms=max(dt_arrival_ms),
        dropped_frame_count=dropped,
        frame_count=n,
    )


def get_video_frame_count(path: Path) -> int | None:
    """Return the number of video frames in an mp4 using ffprobe."""
    if not path.exists():
        return None

    for flag, key in [("-count_packets", "nb_read_packets"), ("", "nb_frames")]:
        cmd = [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            f"stream={key}",
            "-of",
            "csv=p=0",
        ]
        if flag:
            cmd.insert(3, flag)
        cmd.append(str(path))

        proc = subprocess.run(
            cmd, capture_output=True, text=True, timeout=10, check=False
        )
        if proc.returncode == 0 and proc.stdout.strip().isdigit():
            return int(proc.stdout.strip())

    return None


def validate_sync(
    keys_count: int,
    frame_count: int,
    video_frame_count: int | None = None,
) -> SyncValidation:
    """Cross-check that key and frame counts match.

    If video_frame_count is provided, it is used as the authoritative
    frame count for the comparison.  A mismatch between frame_count
    (from the log) and video_frame_count is informational only and
    does not affect the ok flag — the adjustment already aligns
    keys_count to video_frame_count.
    """
    ref = video_frame_count if video_frame_count is not None else frame_count
    ok = keys_count == ref
    if video_frame_count is not None and video_frame_count != frame_count:
        msg = f"log frames ({frame_count}) != video frames ({video_frame_count})"
    elif not ok:
        msg = f"keys ({keys_count}) != frames ({ref})"
    else:
        msg = ""
    return SyncValidation(
        keys_count=keys_count,
        frame_meta_count=ref,
        ok=ok,
        message=msg,
    )


def write_drift_report(
    path: str,
    stats: DriftStats,
    sync: SyncValidation,
    *,
    adjusted: int = 0,
    video_frame_count: int | None = None,
) -> None:
    """Write a human-readable drift_report.txt."""
    lines: list[str] = []
    lines.append("=== Sync validation ===")
    lines.append(f"  synced_keys : {sync.keys_count}")
    lines.append(f"  frames      : {sync.frame_meta_count}")
    if video_frame_count is not None:
        lines.append(f"  video frames: {video_frame_count}")
    if adjusted != 0:
        lines.append(f"  adjusted    : {adjusted:+d}")
    lines.append(
        f"  status      : {'OK' if sync.ok else f'MISMATCH -> {sync.message}'}"
    )
    lines.append("")
    lines.append("=== Inter-frame timing (ms) ===")
    lines.append(
        f"  mean / std : {stats.dt_arrival_mean_ms:.3f} / {stats.dt_arrival_std_ms:.3f}"
    )
    lines.append(
        f"  min / max  : {stats.dt_arrival_min_ms:.3f} / {stats.dt_arrival_max_ms:.3f}"
    )
    lines.append(
        f"  dropped (> {SYNC_DROPPED_FRAME_GAP_MS}ms gap): {stats.dropped_frame_count} / {stats.frame_count}"
    )
    lines.append("")
    all_ok = sync.ok and stats.dropped_frame_count == 0
    lines.append(f"=== Overall: {'OK' if all_ok else 'WARNING'} ===")
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def sync_run(run_dir: Path) -> None:
    """Reconstruct per-frame keys, adjusting to match master.mp4 frame count.

    If the video frame count differs from the frame_times log, keys.json is
    truncated or padded so that keys.json[i] always corresponds to
    master.mp4 frame i.
    """
    key_events_path = run_dir / "key_events.json"
    meta_path = run_dir / "meta.json"
    drift_path = run_dir / "drift_report.txt"
    keys_path = run_dir / "keys.json"

    # Find frame_times log (accept frame_times_*.log or just frame_times.log).
    frame_logs = sorted(run_dir.glob("frame_times*.log"))
    if not frame_logs:
        print(f"  ERROR: no frame_times*.log found in {run_dir}")
        print("  Copy the frame_times_<TS>.log from the repo root into this dir.")
        return
    frame_times_path = frame_logs[-1]

    for p, name in [(key_events_path, "key_events.json"), (meta_path, "meta.json")]:
        if not p.exists():
            print(f"  ERROR: {name} not found in {run_dir}")
            return

    with open(key_events_path) as f:
        key_events: list[tuple[float, str, bool]] = json.load(f)
    with open(meta_path) as f:
        meta: dict[str, Any] = json.load(f)

    frame_capture_times = read_frame_times(frame_times_path)
    n_frames = len(frame_capture_times)
    if n_frames == 0:
        return

    # Reconstruct key state at each frame's capture time.
    keys_list = reconstruct_keys(key_events, frame_capture_times)

    # Get actual video frame count from master.mp4.
    master_mp4 = run_dir / "master.mp4"
    video_frame_count = get_video_frame_count(master_mp4)

    # Adjust keys_list to match video frame count exactly.
    adjusted = 0
    if video_frame_count is not None:
        diff = video_frame_count - len(keys_list)
        if diff < 0:
            adjusted = diff
            keys_list = keys_list[:video_frame_count]
        elif diff > 0:
            adjusted = diff
            keys_list.extend([] for _ in range(diff))
        else:
            adjusted = 0

    # Validation.
    sync = validate_sync(
        keys_count=len(keys_list),
        frame_count=n_frames,
        video_frame_count=video_frame_count,
    )

    # Drift.
    drift = compute_drift(frame_capture_times)
    write_drift_report(
        str(drift_path),
        drift,
        sync,
        adjusted=adjusted,
        video_frame_count=video_frame_count,
    )

    # Write keys.json.
    with open(keys_path, "w") as f:
        json.dump(keys_list, f)

    # Probe video for resolution & fps.
    if master_mp4.exists():
        probe = subprocess.run(
            [
                "ffprobe",
                "-v",
                "error",
                "-select_streams",
                "v:0",
                "-show_entries",
                "stream=width,height,r_frame_rate",
                "-of",
                "csv=p=0",
                str(master_mp4),
            ],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
        if probe.returncode == 0 and probe.stdout.strip():
            parts = probe.stdout.strip().split(",")
            meta["master_resolution"] = [int(parts[0]), int(parts[1])]
            num, den = parts[2].split("/")
            meta["master_fps"] = round(int(num) / int(den))

    # Update meta.json.
    meta["frame_count"] = n_frames
    if video_frame_count is not None:
        meta["master_frame_count"] = video_frame_count
    meta["synced_key_count"] = len(keys_list)
    meta["sync_status"] = "OK" if sync.ok else "MISMATCH"
    meta["dt_arrival_mean_ms"] = round(drift.dt_arrival_mean_ms, 3)
    meta["dt_arrival_std_ms"] = round(drift.dt_arrival_std_ms, 3)
    meta["dropped_frame_count"] = drift.dropped_frame_count
    meta["recon_tolerance_ms"] = RECON_TOLERANCE_MS
    if adjusted != 0:
        meta["sync_adjustment"] = adjusted
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)

    print(f"  Source: {frame_times_path.name}")
    print(f"  Frames (log): {n_frames}  Keys synced: {len(keys_list)}")
    if video_frame_count is not None:
        print(f"  Video frames: {video_frame_count}")
    if adjusted != 0:
        print(f"  Adjusted: {adjusted:+d}")
    print(f"  Sync: {'OK' if sync.ok else f'MISMATCH ({sync.message})'}")
    if drift.dropped_frame_count > 0:
        print(f"  WARN: {drift.dropped_frame_count} dropped frame(s) detected")
    print(f"  Drift report: {drift_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Post-hoc sync: reconstruct keys from raw recording data."
    )
    parser.add_argument(
        "run_dir",
        nargs="?",
        help="Path to a run_<TS>/ directory (default: latest in RUNS_DIR)",
    )
    args = parser.parse_args()

    if args.run_dir:
        run_dir = Path(args.run_dir)
        if not run_dir.exists():
            sys.exit(f"Directory not found: {run_dir}")
        print(f"Syncing: {run_dir}")
        sync_run(run_dir)
    else:
        runs = sorted(RUNS_DIR.iterdir())
        if not runs:
            sys.exit("No runs found.")
        latest = runs[-1]
        print(f"Syncing latest run: {latest}")
        sync_run(latest)


if __name__ == "__main__":
    main()
