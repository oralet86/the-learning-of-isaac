#!/usr/bin/env python3
"""Interactive tkinter viewer for recorded runs (mp4 + keys.json + meta.json).

Supports both legacy runs (output.mp4, 284x160@20) and new runs (master.mp4,
1280x720@60) by preferring master.mp4 when it exists and reading the frame
size / fps from meta.json.
"""

import json
from pathlib import Path
import subprocess
import sys
import tkinter as tk
from typing import Any

import numpy as np
from PIL import Image, ImageTk

from constants import MASTER_FPS, RUNS_DIR, TARGET_SIZE

SCALE = 4  # scale frames up for visibility (used when src is 284 wide)
SCALE_MASTER = 1  # 720p @ 1x is already display-sized
KEY_COLOR = "#00ff00"
BG_COLOR = "#1a1a2e"


def load_frames(video_path: str, width: int, height: int) -> np.ndarray | None:
    proc = subprocess.run(
        [
            "ffmpeg",
            "-v",
            "error",
            "-i",
            video_path,
            "-f",
            "rawvideo",
            "-pix_fmt",
            "rgb24",
            "-",
        ],
        capture_output=True,
        timeout=30,
        check=False,
    )
    if proc.returncode != 0:
        return None
    frame_bytes = width * height * 3
    total = len(proc.stdout)
    if total == 0 or total % frame_bytes != 0:
        return None
    num_frames = total // frame_bytes
    return np.frombuffer(proc.stdout, dtype=np.uint8).reshape(
        num_frames, height, width, 3
    )


class Viewer:
    def __init__(self, run_dir: str) -> None:
        self.run_dir = Path(run_dir)
        self.keys_path = self.run_dir / "keys.json"
        self.meta_path = self.run_dir / "meta.json"
        self.frame_meta_path = self.run_dir / "frame_meta.json"

        # Prefer master.mp4, fall back to output.mp4 (legacy).
        master_path = self.run_dir / "master.mp4"
        legacy_path = self.run_dir / "output.mp4"
        if master_path.exists():
            self.video_path = master_path
        elif legacy_path.exists():
            self.video_path = legacy_path
        else:
            sys.exit(f"No video found ({master_path} / {legacy_path})")

        if not self.keys_path.exists():
            sys.exit(f"Keys not found: {self.keys_path}")

        self.meta: dict[str, Any] = {}
        if self.meta_path.exists():
            with open(self.meta_path) as f:
                self.meta = json.load(f)

        # Resolution & FPS: probe the actual video; fall back to meta / constants.
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
                str(self.video_path),
            ],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
        if probe.returncode == 0 and probe.stdout.strip():
            parts = probe.stdout.strip().split(",")
            self.src_width = int(parts[0])
            self.src_height = int(parts[1])
            num, den = parts[2].split("/")
            self.fps = round(int(num) / int(den))
        else:
            # Fall back to meta.json or constants.
            master_res = self.meta.get("master_resolution")
            if master_res is not None:
                self.src_width, self.src_height = (
                    int(master_res[0]),
                    int(master_res[1]),
                )
            else:
                target = self.meta.get("target_size", list(TARGET_SIZE))
                self.src_width, self.src_height = int(target[0]), int(target[1])
            self.fps = int(
                self.meta.get("master_fps", self.meta.get("fps", MASTER_FPS))
            )

        print(f"Loading frames from {self.video_path} ...")
        self.frames = load_frames(str(self.video_path), self.src_width, self.src_height)
        if self.frames is None:
            raise SystemExit("Failed to load video frames.")

        with open(self.keys_path) as f:
            self.keys_raw: list[list[str]] = json.load(f)

        self.num_frames = self.frames.shape[0]
        self.initial_scale = SCALE if self.src_width <= 400 else SCALE_MASTER
        self.display_w = self.src_width * self.initial_scale
        self.display_h = self.src_height * self.initial_scale

        self.current_frame = 0
        self.playing = False
        self.direction = 1  # 1=forward, -1=backward
        self.play_timer: str | None = None

        # Optional frame_meta (lazy-loaded for per-frame drift display).
        self._frame_meta: list[dict[str, float]] | None = None

        self.root = tk.Tk()
        self.root.title(f"view_recording - {self.run_dir.name}")
        self.root.configure(bg=BG_COLOR)

        self._build_controls()
        self._build_display()
        self._build_info()
        self._bind_keys()

        self.root.after(10, self._show_frame)

    # ------------------------------------------------------------------
    #  Frame-meta (drift info) lazy loader
    # ------------------------------------------------------------------

    def _ensure_frame_meta(self) -> list[dict[str, float]] | None:
        if self._frame_meta is not None:
            return self._frame_meta
        if not self.frame_meta_path.exists():
            self._frame_meta = []  # not None → don't re-probe
            return None
        try:
            with open(self.frame_meta_path) as f:
                raw = json.load(f)
            self._frame_meta = raw
            return self._frame_meta
        except Exception:
            self._frame_meta = []
            return None

    # ------------------------------------------------------------------
    #  GUI construction
    # ------------------------------------------------------------------

    def _build_controls(self) -> None:
        ctrl = tk.Frame(self.root, bg=BG_COLOR, pady=4)
        ctrl.pack(side=tk.TOP, fill=tk.X)

        btn_opts: dict[str, Any] = {
            "bg": "#2d2d5e",
            "fg": "white",
            "relief": tk.FLAT,
            "activebackground": "#4d4d8e",
            "padx": 8,
            "pady": 2,
        }

        tk.Button(ctrl, text="|<", command=self.goto_first, **btn_opts).pack(
            side=tk.LEFT, padx=2
        )
        tk.Button(ctrl, text="<<", command=self.step_back, **btn_opts).pack(
            side=tk.LEFT, padx=2
        )
        self.back_play_btn = tk.Button(
            ctrl, text="◀", command=self.play_backward, **btn_opts
        )
        self.back_play_btn.pack(side=tk.LEFT, padx=2)
        self.play_btn = tk.Button(ctrl, text="▶", command=self.play_forward, **btn_opts)
        self.play_btn.pack(side=tk.LEFT, padx=2)
        tk.Button(ctrl, text=">>", command=self.step_forward, **btn_opts).pack(
            side=tk.LEFT, padx=2
        )
        tk.Button(ctrl, text=">|", command=self.goto_last, **btn_opts).pack(
            side=tk.LEFT, padx=2
        )

        tk.Label(ctrl, text="  FPS:", fg="white", bg=BG_COLOR).pack(side=tk.LEFT)
        self.fps_var = tk.StringVar(value=str(self.fps))
        self.fps_entry = tk.Entry(
            ctrl,
            textvariable=self.fps_var,
            width=4,
            bg="#2d2d5e",
            fg="white",
            insertbackground="white",
            relief=tk.FLAT,
            justify=tk.CENTER,
        )
        self.fps_entry.pack(side=tk.LEFT, padx=2)
        self.fps_entry.bind("<Return>", self._set_fps)
        self.fps_entry.bind("<FocusOut>", self._set_fps)

        tk.Button(ctrl, text="−", command=self.fps_down, **btn_opts).pack(side=tk.LEFT)
        tk.Button(ctrl, text="+", command=self.fps_up, **btn_opts).pack(side=tk.LEFT)

        self.key_label = tk.Label(
            ctrl,
            text="",
            fg=KEY_COLOR,
            bg=BG_COLOR,
            font=("monospace", 11),
            anchor=tk.E,
        )
        self.key_label.pack(side=tk.RIGHT, padx=8)

    def _build_display(self) -> None:
        self.canvas = tk.Canvas(
            self.root,
            width=self.display_w,
            height=self.display_h,
            bg="black",
            highlightthickness=0,
        )
        self.canvas.pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=8, pady=4)
        self.canvas.bind("<Configure>", self._on_canvas_resize)

    def _build_info(self) -> None:
        info = tk.Frame(self.root, bg=BG_COLOR, pady=2)
        info.pack(side=tk.TOP, fill=tk.X, padx=8)

        self.frame_label = tk.Label(
            info,
            text="",
            fg="white",
            bg=BG_COLOR,
            font=("monospace", 11),
        )
        self.frame_label.pack(side=tk.LEFT)

        self.info_label = tk.Label(
            info,
            text="",
            fg="#8888aa",
            bg=BG_COLOR,
            font=("monospace", 9),
        )
        self.info_label.pack(side=tk.LEFT, padx=16)

    def _bind_keys(self) -> None:
        self.root.bind("<space>", lambda e: self.play_forward())
        self.root.bind("<Left>", lambda e: self.step_back())
        self.root.bind("<Right>", lambda e: self.step_forward())
        self.root.bind("<Up>", lambda e: self.fps_up())
        self.root.bind("<Down>", lambda e: self.fps_down())
        self.root.bind("<Home>", lambda e: self.goto_first())
        self.root.bind("<End>", lambda e: self.goto_last())
        self.root.bind("<Escape>", lambda e: self.root.quit())
        self.root.bind("q", lambda e: self.root.quit())

    def _resize_display(self, cw: int, ch: int) -> None:
        ratio = self.src_width / self.src_height
        if cw / ch > ratio:
            self.display_w = int(ch * ratio)
            self.display_h = ch
        else:
            self.display_w = cw
            self.display_h = int(cw / ratio)

    def _on_canvas_resize(self, _event: tk.Event | None = None) -> None:
        cw = self.canvas.winfo_width()
        ch = self.canvas.winfo_height()
        if cw < 10 or ch < 10:
            return
        self._resize_display(cw, ch)
        self._show_frame()

    # ------------------------------------------------------------------
    #  Frame rendering
    # ------------------------------------------------------------------

    def _show_frame(self) -> None:
        if self.current_frame < 0:
            self.current_frame = 0
        if self.current_frame >= self.num_frames:
            self.current_frame = self.num_frames - 1

        if self.frames is None:
            return
        frame = self.frames[self.current_frame]

        img = Image.fromarray(frame, mode="RGB")
        img = img.resize((self.display_w, self.display_h), Image.Resampling.NEAREST)
        self._photo = ImageTk.PhotoImage(img)

        self.canvas.delete("all")
        self.canvas.create_image(0, 0, anchor=tk.NW, image=self._photo)

        self.frame_label.config(
            text=f"Frame: {self.current_frame + 1} / {self.num_frames}"
        )

        keys = self._current_keys()
        self.key_label.config(text="Keys: " + (", ".join(keys) if keys else "\u2014"))

        # Build info text.
        d = self.meta.get("duration_seconds", 0)
        info_parts: list[str] = []
        if d > 0:
            info_parts.append(f"{d:.0f}s")
        fps_val = self.meta.get("master_fps", self.meta.get("fps", self.fps))
        if self.frames is not None and self.frames.shape[0] > 0 and fps_val > 0:
            info_parts.append(f"{self.frames.shape[0] / fps_val:.0f}s video")

        # Append per-frame drift info if frame_meta is available.
        fm_data = self._ensure_frame_meta()
        if fm_data and self.current_frame < len(fm_data):
            cur = fm_data[self.current_frame]
            t_arrival = cur.get("t_arrival", 0.0)
            if self.current_frame > 0:
                prev = fm_data[self.current_frame - 1]
                dt_frame = (t_arrival - prev.get("t_arrival", t_arrival)) * 1000.0
                info_parts.append(f"\u0394t={dt_frame:.1f}ms")

        self.info_label.config(text="  ".join(info_parts))

    def _current_keys(self) -> list[str]:
        idx = self.current_frame
        if idx < len(self.keys_raw):
            return self.keys_raw[idx]
        return []

    # ------------------------------------------------------------------
    #  Navigation
    # ------------------------------------------------------------------

    def _set_frame(self, n: int) -> None:
        old = self.current_frame
        self.current_frame = max(0, min(n, self.num_frames - 1))
        if self.current_frame != old:
            self._show_frame()

    def step_forward(self) -> None:
        self._set_frame(self.current_frame + 1)

    def step_back(self) -> None:
        self._set_frame(self.current_frame - 1)

    def goto_first(self) -> None:
        self._set_frame(0)

    def goto_last(self) -> None:
        self._set_frame(self.num_frames - 1)

    # ------------------------------------------------------------------
    #  Playback
    # ------------------------------------------------------------------

    def _stop_playing(self) -> None:
        self.playing = False
        if self.play_timer:
            self.root.after_cancel(self.play_timer)
        self.play_timer = None

    def play_forward(self) -> None:
        if self.playing and self.direction == 1:
            self._stop_playing()
            return
        self._stop_playing()
        self.direction = 1
        self.playing = True
        self._tick()

    def play_backward(self) -> None:
        if self.playing and self.direction == -1:
            self._stop_playing()
            return
        self._stop_playing()
        self.direction = -1
        self.playing = True
        self._tick()

    def _tick(self) -> None:
        if not self.playing:
            return
        new_frame = self.current_frame + self.direction
        if new_frame < 0:
            self.current_frame = self.num_frames - 1
        elif new_frame >= self.num_frames:
            self.current_frame = 0
        else:
            self.current_frame = new_frame
        self._show_frame()

        interval = int(1000 / max(1, self.fps))
        self.play_timer = self.root.after(interval, self._tick)

    def _set_fps(self, _event: tk.Event | None = None) -> None:
        try:
            val = int(self.fps_var.get())
            if 1 <= val <= 120:
                self.fps = val
                if self.playing:
                    self._stop_playing()
                    self._tick()
            else:
                self.fps_var.set(str(self.fps))
        except ValueError:
            self.fps_var.set(str(self.fps))

    def fps_up(self) -> None:
        self.fps = min(120, self.fps + 5)
        self.fps_var.set(str(self.fps))
        if self.playing:
            self._stop_playing()
            self._tick()

    def fps_down(self) -> None:
        self.fps = max(1, self.fps - 5)
        self.fps_var.set(str(self.fps))
        if self.playing:
            self._stop_playing()
            self._tick()

    def run(self) -> None:
        self.root.mainloop()


def main() -> None:
    if len(sys.argv) < 2:
        sys.exit(f"Usage: {sys.argv[0]} {RUNS_DIR}/run_timestamp/")

    run_dir = sys.argv[1]
    viewer = Viewer(run_dir)
    viewer.run()


if __name__ == "__main__":
    main()
