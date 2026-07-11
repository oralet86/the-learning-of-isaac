#!/usr/bin/env python3
"""EDA for labeled Isaac runs — keys.json + output.mp4."""

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import subprocess

GAMEPLAY_KEYS = {"w", "a", "s", "d", "D", "left", "right", "up", "down", "e", "space", "q", "ctrl", "shift_r", "tab"}
MENU_KEYS = {"esc", "r", "enter"}
ALL_KEYS = GAMEPLAY_KEYS | MENU_KEYS | {"alt"}

TIMELINE_BIN_SECONDS = 1.0
VIDEO_STATS_SUBSAMPLE = 10
ALT_BUFFER_SECONDS = 2.0
SAMPLE_WINDOW = 8


def load_keys(run_dir: Path) -> tuple[list[list[str]], float]:
    path = run_dir / "keys.json"
    with open(path) as f:
        keys = json.load(f)
    with open(run_dir / "meta.json") as f:
        meta = json.load(f)
    return keys, meta["fps"]


def classify_frames(keys: list[list[str]], fps: float) -> np.ndarray:
    """Return string array of per-frame classifications."""
    n = len(keys)
    labels = np.full(n, "gameplay", dtype=object)
    alt_buffer = int(ALT_BUFFER_SECONDS * fps)

    alt_zones = np.zeros(n, dtype=bool)

    for i, k in enumerate(keys):
        ks = set(k)

        if "alt" in ks:
            lo = max(0, i - alt_buffer)
            hi = min(n, i + alt_buffer + 1)
            alt_zones[lo:hi] = True

    for i, k in enumerate(keys):
        ks = set(k)
        if alt_zones[i]:
            labels[i] = "alt_tab"
        elif ks and ks <= MENU_KEYS:
            labels[i] = "menu"
        elif ks == set():
            labels[i] = "idle"
        elif "tab" in ks and (ks - {"tab"}) & GAMEPLAY_KEYS:
            labels[i] = "map"
        else:
            labels[i] = "gameplay"

    return labels


def find_events(keys: list[list[str]], labels: np.ndarray, fps: float) -> dict:
    """Find contiguous blocks of non-gameplay events."""
    n = len(keys)
    events: dict[str, list[dict]] = {"alt": [], "menu": [], "idle_long": [], "map": []}

    i = 0
    while i < n:
        label = labels[i]
        if label not in ("alt_tab", "menu", "map"):
            i += 1
            continue

        start = i
        while i < n and labels[i] == label:
            i += 1
        end = i - 1

        if label == "alt_tab":
            events["alt"].append({"start": start, "end": end, "keys_at_peak": keys[start]})
        elif label == "menu":
            events["menu"].append({"start": start, "end": end, "keys": keys[start:end + 1]})
        elif label == "map":
            events["map"].append({"start": start, "end": end})

    i = 0
    while i < n:
        if labels[i] != "idle":
            i += 1
            continue
        start = i
        while i < n and labels[i] == "idle":
            i += 1
        end = i - 1
        duration = (end - start + 1) / fps
        if duration >= 2.0:
            events["idle_long"].append({"start": start, "end": end, "duration_s": round(duration, 1)})

    return events


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def plot_key_frequencies(keys: list[list[str]], out_dir: Path):
    kc = Counter()
    for k in keys:
        kc.update(k)

    names, counts = zip(*kc.most_common())
    pcts = [c / len(keys) * 100 for c in counts]

    fig, ax = plt.subplots(figsize=(12, 5))
    bars = ax.bar(names, pcts, color="steelblue", edgecolor="white")
    ax.set_ylabel("% of frames")
    ax.set_title("Key Frequency (per-frame hold rate)")
    ax.set_ylim(0, max(pcts) * 1.1)
    for bar, pct in zip(bars, pcts):
        if pct > 1:
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5, f"{pct:.1f}%",
                    ha="center", va="bottom", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_dir / "key_frequencies.png", dpi=150)
    plt.close(fig)


def plot_key_combinations(keys: list[list[str]], out_dir: Path):
    combos = Counter()
    for k in keys:
        combos[tuple(sorted(k)) or ("[none]",)] += 1

    top = combos.most_common(25)
    names = ["+".join(c[0]) if c[0] != ("[none]",) else "(none)" for c in top]
    counts = [c[1] for c in top]
    pcts = [c / len(keys) * 100 for c in counts]

    fig, ax = plt.subplots(figsize=(12, 6))
    y = range(len(names))
    ax.barh(y, pcts, color="steelblue", edgecolor="white")
    ax.set_yticks(y)
    ax.set_yticklabels(names, fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel("% of frames")
    ax.set_title("Top 25 Key Combinations")
    for yi, pct in zip(y, pcts):
        ax.text(pct + 0.1, yi, f"{pct:.1f}%", va="center", fontsize=7)
    fig.tight_layout()
    fig.savefig(out_dir / "key_combinations.png", dpi=150)
    plt.close(fig)


def plot_timeline(keys: list[list[str]], fps: float, labels: np.ndarray, out_dir: Path):
    n = len(keys)
    total_s = n / fps
    bin_frames = int(TIMELINE_BIN_SECONDS * fps)
    n_bins = int(np.ceil(n / bin_frames))

    display_keys = ["w", "a", "s", "d", "D", "left", "right", "up", "down", "e", "space", "q", "tab", "esc", "r", "enter", "alt"]
    key_to_row = {k: i for i, k in enumerate(display_keys)}
    grid = np.zeros((len(display_keys), n_bins), dtype=np.float32)

    for i, k in enumerate(keys):
        bin_idx = i // bin_frames
        if bin_idx >= n_bins:
            break
        for key_name in k:
            row = key_to_row.get(key_name)
            if row is not None:
                grid[row, bin_idx] += 1.0 / bin_frames

    fig, ax = plt.subplots(figsize=(20, len(display_keys) * 0.35 + 1))
    im = ax.imshow(grid, aspect="auto", cmap="YlOrRd", vmin=0, vmax=1, origin="upper",
                   extent=[0, total_s, len(display_keys) - 0.5, -0.5])
    ax.set_yticks(range(len(display_keys)))
    ax.set_yticklabels(display_keys, fontsize=9)
    ax.set_xlabel("Time (seconds)")
    ax.set_title("Keys Over Time (1-second bins, fraction pressed)")
    fig.colorbar(im, ax=ax, label="fraction pressed")

    alt_events = []
    for i, k in enumerate(keys):
        if "alt" in k:
            alt_events.append(i / fps)

    alt_seen = set()
    for t in alt_events:
        t_round = round(t)
        if t_round not in alt_seen:
            ax.axvline(x=t, color="red", alpha=0.3, linewidth=0.5)
            alt_seen.add(t_round)

    fig.tight_layout()
    fig.savefig(out_dir / "timeline.png", dpi=150)
    plt.close(fig)


def plot_segments(labels: np.ndarray, fps: float, out_dir: Path):
    n = len(labels)
    total_s = n / fps

    color_map = {"gameplay": "#2ecc71", "alt_tab": "#e74c3c", "menu": "#f39c12",
                 "idle": "#95a5a6", "map": "#3498db"}
    seq_order = ["gameplay", "idle", "map", "menu", "alt_tab"]

    fig, ax = plt.subplots(figsize=(20, 2.5))

    i = 0
    while i < n:
        lbl = labels[i]
        start = i / fps
        while i < n and labels[i] == lbl:
            i += 1
        end = i / fps
        ax.axvspan(start, end, color=color_map.get(lbl, "#333333"), alpha=0.85)

    from matplotlib.patches import Patch
    legend_patches = [Patch(color=color_map[k], label=k.replace("_", " ")) for k in seq_order]
    ax.legend(handles=legend_patches, loc="upper right", fontsize=8, ncols=5)

    ax.set_xlim(0, total_s)
    ax.set_xlabel("Time (seconds)")
    ax.set_title("Recording Segmentation")
    ax.set_yticks([])

    counts = {k: int(np.sum(labels == k)) for k in seq_order}
    annotation = "  ".join(f"{k.replace('_',' ')}: {v/20/60:.1f}min" for k, v in counts.items())
    ax.text(0.5, -0.35, annotation, transform=ax.transAxes, ha="center", fontsize=8)

    fig.tight_layout()
    fig.savefig(out_dir / "segments.png", dpi=150)
    plt.close(fig)


def plot_hold_durations(keys: list[list[str]], fps: float, out_dir: Path):
    durations: dict[str, list[float]] = defaultdict(list)

    i = 0
    n = len(keys)
    while i < n:
        combo = set(keys[i])
        start = i
        while i < n and set(keys[i]) == combo:
            i += 1
        dur_frames = i - start

        dur_s = dur_frames / fps
        for key_name in combo:
            durations[key_name].append(dur_s)
        if not combo:
            durations["(none)"].append(dur_s)

    main_keys = ["w", "a", "s", "d", "D", "left", "right", "up", "down", "e", "space", "q", "tab", "(none)"]
    fig, axes = plt.subplots(4, 4, figsize=(14, 10))
    axes = axes.flatten()

    for idx, key_name in enumerate(main_keys):
        ax = axes[idx]
        vals = durations.get(key_name, [])
        if vals:
            ax.hist(vals, bins=50, color="steelblue", edgecolor="white", alpha=0.8)
            ax.set_title(f"{key_name}  (n={len(vals)}, mean={np.mean(vals):.2f}s)", fontsize=9)
        else:
            ax.set_title(f"{key_name}  (none)", fontsize=9)
            ax.text(0.5, 0.5, "N/A", ha="center", va="center", transform=ax.transAxes)
        ax.set_xlabel("hold duration (s)", fontsize=7)
        if idx == 0:
            ax.set_ylabel("count", fontsize=7)

    for idx in range(len(main_keys), len(axes)):
        axes[idx].set_visible(False)

    fig.suptitle("Key Hold Duration Distributions", fontsize=13)
    fig.tight_layout()
    fig.savefig(out_dir / "hold_durations.png", dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Video
# ---------------------------------------------------------------------------

def _collect_sample_indices(keys: list[list[str]], labels: np.ndarray, events: dict, fps: float) -> dict:
    """Return dict mapping event type -> list of (label_str, list_of_frame_indices)."""
    n = len(keys)
    samples: dict[str, list[tuple[str, list[int]]]] = defaultdict(list)

    for ev in events["alt"]:
        s, e = ev["start"], ev["end"]
        lo = max(0, s - SAMPLE_WINDOW)
        hi = min(n, e + SAMPLE_WINDOW)
        desc = f"alt @ frame {s}"
        samples["alt"].append((desc, list(range(lo, hi))))

    for ev in events["menu"]:
        s, e = ev["start"], ev["end"]
        lo = max(0, s - SAMPLE_WINDOW)
        hi = min(n, e + SAMPLE_WINDOW)
        ks = set()
        for klist in ev["keys"]:
            ks.update(klist)
        desc = f"{'+'.join(sorted(ks))} @ frame {s}"
        samples["menu"].append((desc, list(range(lo, hi))))

    game_segments = []
    i = 0
    while i < n:
        if labels[i] == "gameplay":
            start = i
            while i < n and labels[i] == "gameplay":
                i += 1
            if i - start >= 30:
                game_segments.append((start, i - 1))
        else:
            i += 1

    rng = np.random.default_rng(42)
    for seg_start, seg_end in game_segments[:5] if len(game_segments) >= 5 else game_segments:
        pts = sorted(rng.choice(range(seg_start, seg_end), size=min(4, seg_end - seg_start), replace=False))
        for pt in pts:
            lo = max(0, pt - SAMPLE_WINDOW)
            hi = min(n, pt + SAMPLE_WINDOW)
            desc = f"gameplay @ frame {pt}"
            samples["gameplay"].append((desc, list(range(lo, hi))))

    return dict(samples)


def _video_resolution(video_path: Path) -> tuple[int, int]:
    result = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0",
         "-show_entries", "stream=width,height", "-of", "csv=p=0",
         str(video_path)],
        capture_output=True, text=True, check=True,
    )
    w, h = result.stdout.strip().split(",")
    return int(w), int(h)


def _video_pass(video_path: Path, keys: list[list[str]], fps: float,
                labels: np.ndarray, sample_spec: dict, out_dir: Path) -> dict:
    """One pass over video using ffmpeg: collect stats & sample frames."""
    width, height = _video_resolution(video_path)
    frame_bytes = width * height * 3

    n = len(keys)

    all_needed = set()
    for groups in sample_spec.values():
        for _, indices in groups:
            all_needed.update(indices)

    sample_buffer: dict[int, np.ndarray] = {}

    means = []
    diffs = []
    prev_gray = None

    cmd = [
        "ffmpeg", "-v", "error",
        "-i", str(video_path),
        "-f", "rawvideo",
        "-pix_fmt", "rgb24",
        "pipe:1",
    ]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)

    print(f"  Scanning video ({width}×{height}, extracting {len(all_needed)} sample frames)...")

    idx = 0
    while True:
        raw = proc.stdout.read(frame_bytes)
        if len(raw) < frame_bytes:
            break

        rgb = np.frombuffer(raw, dtype=np.uint8).reshape(height, width, 3)
        gray = rgb.astype(np.float32).mean(axis=2)

        if idx < n and idx % VIDEO_STATS_SUBSAMPLE == 0:
            means.append(float(gray.mean()))

        if idx in all_needed:
            sample_buffer[idx] = rgb.copy()

        if idx < n and idx % VIDEO_STATS_SUBSAMPLE == 0 and prev_gray is not None:
            diffs.append(float(np.abs(gray - prev_gray).mean()))

        prev_gray = gray
        idx += 1

        if idx % 5000 == 0:
            print(f"    frame {idx}")

    proc.stdout.close()
    proc.wait()

    print(f"  Done. Extracted {len(sample_buffer)} sample frames from {idx} video frames.")

    _write_sample_sheets(sample_spec, sample_buffer, fps, out_dir)

    return {"means": means, "diffs": diffs, "subsample": VIDEO_STATS_SUBSAMPLE, "fps": fps}


def _write_sample_sheets(spec: dict, buffer: dict, fps: float, out_dir: Path):
    for event_type, groups in spec.items():
        if not groups:
            continue

        items = []
        for desc, indices in groups:
            frames = [buffer[i] for i in indices if i in buffer]
            if not frames:
                continue
            items.append((desc, indices, frames))

        if not items:
            continue

        max_cols = SAMPLE_WINDOW * 2
        if event_type in ("alt",):
            n_cols = max_cols
        elif event_type == "menu":
            frame_counts = [len(f) for _, _, f in items]
            n_cols = max(frame_counts) if frame_counts else max_cols
        else:
            n_cols = 10

        n_cols = min(n_cols, 16)

        for page_idx in range(0, len(items), 20):
            page = items[page_idx:page_idx + 20]
            n_rows = len(page)
            fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 0.8, n_rows * 0.85))
            if n_rows == 1:
                axes = axes.reshape(1, -1)
            if n_rows == 0:
                plt.close(fig)
                continue

            desc_offset = []
            for row_idx, (desc, indices, frames) in enumerate(page):
                desc_offset.append(f"f{indices[0]}: {desc}")
                for col in range(n_cols):
                    ax = axes[row_idx, col]
                    if col < len(frames):
                        ax.imshow(frames[col])
                        ax.set_title(f"f{indices[col]}", fontsize=6)
                    ax.set_xticks([])
                    ax.set_yticks([])
                    ax.set_frame_on(False)

            for row_idx, txt in enumerate(desc_offset):
                axes[row_idx, 0].set_ylabel(txt, fontsize=7, rotation=0, ha="right", va="center",
                                            labelpad=20)

            page_tag = f"_{page_idx // 20}" if len(items) > 20 else ""
            fig.suptitle(f"{event_type} frames", fontsize=10)
            fig.tight_layout()
            fig.savefig(out_dir / f"sample_{event_type}{page_tag}.png", dpi=150)
            plt.close(fig)


def plot_video_stats(stats: dict, labels: np.ndarray, fps: float, out_dir: Path):
    means = np.array(stats["means"])
    diffs = np.array(stats["diffs"])
    subsample = stats["subsample"]
    time_means = np.arange(len(means)) * subsample / fps
    time_diffs = np.arange(len(diffs)) * subsample / fps + subsample / fps

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(20, 7), sharex=True)

    ax1.plot(time_means, means, linewidth=0.4, color="#2c3e50")
    ax1.set_ylabel("Mean pixel value")
    ax1.set_title("Frame Mean Pixel Over Time")

    ax2.plot(time_diffs, diffs, linewidth=0.4, color="#e67e22")
    ax2.set_ylabel("Frame-to-frame L1 diff")
    ax2.set_title("Frame Difference Over Time (activity proxy)")
    ax2.set_xlabel("Time (seconds)")

    n = len(labels)
    alt_times = [(i + 0.5) / fps for i in range(n) if labels[i] == "alt_tab"]
    if alt_times:
        from itertools import groupby
        alt_ranges = []
        for k, g in groupby(range(n), key=lambda i: labels[i] == "alt_tab"):
            if k:
                grp = list(g)
                alt_ranges.append((grp[0] / fps, grp[-1] / fps))
        for lo, hi in alt_ranges:
            ax1.axvspan(lo, hi, color="red", alpha=0.08)
            ax2.axvspan(lo, hi, color="red", alpha=0.08)

    fig.tight_layout()
    fig.savefig(out_dir / "video_stats.png", dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

def save_summary(keys: list[list[str]], labels: np.ndarray, events: dict, fps: float, out_dir: Path):
    n = len(keys)
    kc = Counter()
    for k in keys:
        kc.update(k)

    combos = Counter()
    for k in keys:
        combos[tuple(sorted(k)) or ("[none]",)] += 1

    changes = sum(1 for i in range(1, n) if set(keys[i]) != set(keys[i - 1]))

    summary = {
        "total_frames": n,
        "fps": fps,
        "duration_minutes": round(n / fps / 60, 1),
        "unique_keys": sorted(kc.keys()),
        "unique_combinations": len(combos),
        "key_change_count": changes,
        "key_change_rate": round(changes / (n - 1) * 100, 1),
        "key_frequencies_pct": {k: round(v / n * 100, 1) for k, v in kc.most_common()},
        "top_combinations": [{"keys": list(c), "count": cnt, "pct": round(cnt / n * 100, 1)}
                             for c, cnt in combos.most_common(50)],
        "segment_counts": {
            "gameplay": int(np.sum(labels == "gameplay")),
            "alt_tab": int(np.sum(labels == "alt_tab")),
            "menu": int(np.sum(labels == "menu")),
            "idle": int(np.sum(labels == "idle")),
            "map": int(np.sum(labels == "map")),
        },
        "events": {
            "alt_count": len(events["alt"]),
            "menu_count": len(events["menu"]),
            "long_idle_count": len(events["idle_long"]),
            "map_count": len(events["map"]),
            "alt_detail": [{"start_frame": e["start"], "end_frame": e["end"],
                            "duration_s": round((e["end"] - e["start"] + 1) / fps, 1)}
                           for e in events["alt"]],
            "menu_detail": [{"start_frame": e["start"], "end_frame": e["end"],
                             "duration_s": round((e["end"] - e["start"] + 1) / fps, 1)}
                            for e in events["menu"]],
            "long_idle_ranges": [{"start_frame": e["start"], "end_frame": e["end"],
                                  "duration_s": e["duration_s"]}
                                 for e in events["idle_long"]],
        },
    }
    with open(out_dir / "eda_summary.json", "w") as f:
        json.dump(summary, f, indent=2)


def print_console_report(keys: list[list[str]], labels: np.ndarray, events: dict, fps: float):
    n = len(keys)
    kc = Counter()
    for k in keys:
        kc.update(k)

    combos = Counter()
    for k in keys:
        combos[tuple(sorted(k)) or ("[none]",)] += 1

    changes = sum(1 for i in range(1, n) if set(keys[i]) != set(keys[i - 1]))

    print()
    print("=" * 60)
    print("  EDA REPORT")
    print("=" * 60)
    print(f"\nTotal frames:      {n:,}")
    print(f"Duration:          {n / fps / 60:.1f} min")
    print(f"FPS:               {fps}")
    print(f"Unique keys:       {len(kc)}")
    print(f"Unique combos:     {len(combos)}")
    print(f"Key changes:       {changes:,} ({changes / (n - 1) * 100:.1f}% of frames)")
    print()

    print("--- Key Frequencies (% of frames) ---")
    for key_name, count in kc.most_common():
        print(f"  {key_name:10s} {count:>6,}  ({count / n * 100:5.1f}%)")
    print()

    print("--- Top 15 Key Combinations ---")
    for combo, count in combos.most_common(15):
        name = "+".join(combo) if combo != ("[none]",) else "(none)"
        print(f"  {name:24s} {count:>6,}  ({count / n * 100:5.1f}%)")
    print()

    print("--- Segment Breakdown ---")
    for lbl in ["gameplay", "alt_tab", "menu", "idle", "map"]:
        cnt = int(np.sum(labels == lbl))
        pct = cnt / n * 100
        print(f"  {lbl:10s} {cnt:>6,} frames  ({pct:5.1f}%)  = {cnt/fps/60:5.1f} min")
    print()

    print("--- Events ---")
    print(f"  Alt-tab events:       {len(events['alt'])}")
    for e in events["alt"]:
        dur = (e["end"] - e["start"] + 1) / fps
        print(f"    frames {e['start']}-{e['end']} ({dur:.1f}s)")
    print(f"  Menu events:          {len(events['menu'])}")
    for e in events["menu"]:
        dur = (e["end"] - e["start"] + 1) / fps
        ks = set()
        for klist in e["keys"]:
            ks.update(klist)
        print(f"    frames {e['start']}-{e['end']} ({dur:.1f}s)  keys: {sorted(ks)}")
    print(f"  Long idle gaps (>2s): {len(events['idle_long'])}")
    for e in events["idle_long"]:
        print(f"    frames {e['start']}-{e['end']} ({e['duration_s']}s)")
    print(f"  Map overlay events:   {len(events['map'])}")
    print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="EDA for labeled Isaac runs")
    parser.add_argument("run_dir", type=Path, help="Path to run directory (e.g., labeled_runs/run_20260711_141244)")
    parser.add_argument("--no-video", action="store_true", help="Skip video analysis (keys.json only)")
    args = parser.parse_args()

    run_dir = args.run_dir
    if not run_dir.is_dir():
        sys.exit(f"Not a directory: {run_dir}")

    keys_path = run_dir / "keys.json"
    video_path = run_dir / "output.mp4"
    if not keys_path.exists():
        sys.exit(f"keys.json not found in {run_dir}")

    out_dir = run_dir / "eda"
    out_dir.mkdir(exist_ok=True)

    print(f"Loading data from {run_dir} ...")
    keys, fps = load_keys(run_dir)
    n = len(keys)
    print(f"  {n:,} frames at {fps} fps ({n / fps / 60:.1f} min)")

    print("Classifying frames...")
    labels = classify_frames(keys, fps)

    print("Finding events...")
    events = find_events(keys, labels, fps)

    print("Generating plots...")
    plot_key_frequencies(keys, out_dir)
    plot_key_combinations(keys, out_dir)
    plot_timeline(keys, fps, labels, out_dir)
    plot_segments(labels, fps, out_dir)
    plot_hold_durations(keys, fps, out_dir)

    if not args.no_video and video_path.exists():
        print("Collecting sample frame indices...")
        sample_spec = _collect_sample_indices(keys, labels, events, fps)
        total_event_samples = sum(len(groups) for groups in sample_spec.values())
        print(f"  {total_event_samples} sample groups to extract")

        print("Scanning video (stats + sample extraction)...")
        video_stats = _video_pass(video_path, keys, fps, labels, sample_spec, out_dir)

        print("Plotting video stats...")
        plot_video_stats(video_stats, labels, fps, out_dir)
    elif not video_path.exists():
        print("output.mp4 not found — skipping video analysis")
    else:
        print("Skipping video analysis (--no-video)")

    print("Saving summary...")
    save_summary(keys, labels, events, fps, out_dir)

    print_console_report(keys, labels, events, fps)

    print(f"Done. Output in {out_dir}/")
    for f in sorted(out_dir.iterdir()):
        size_kb = f.stat().st_size / 1024
        print(f"  {f.name}  ({size_kb:.0f} KB)")


if __name__ == "__main__":
    main()
