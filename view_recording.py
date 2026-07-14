#!/usr/bin/env python3
"""Interactive tkinter viewer for recorded runs (mp4 + keys.json + meta.json)."""

import json
from pathlib import Path
import subprocess
import sys
import tkinter as tk
from typing import Any

import numpy as np
from PIL import Image, ImageTk

from constants import FPS, TARGET_SIZE

SCALE = 4  # scale frames up for visibility
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
            "gray",
            "-",
        ],
        capture_output=True,
        timeout=30,
        check=False,
    )
    if proc.returncode != 0:
        return None
    frame_bytes = width * height
    total = len(proc.stdout)
    if total == 0 or total % frame_bytes != 0:
        return None
    num_frames = total // frame_bytes
    return np.frombuffer(proc.stdout, dtype=np.uint8).reshape(num_frames, height, width)


class Viewer:
    def __init__(self, run_dir: str):
        self.run_dir = Path(run_dir)
        self.video_path = self.run_dir / "output.mp4"
        self.keys_path = self.run_dir / "keys.json"
        self.meta_path = self.run_dir / "meta.json"

        if not self.video_path.exists():
            sys.exit(f"Video not found: {self.video_path}")
        if not self.keys_path.exists():
            sys.exit(f"Keys not found: {self.keys_path}")

        self.meta = {}
        if self.meta_path.exists():
            with open(self.meta_path) as f:
                self.meta = json.load(f)

        self.target_size = self.meta.get("target_size", list(TARGET_SIZE))
        self.src_width, self.src_height = self.target_size[0], self.target_size[1]

        print(f"Loading frames from {self.video_path} ...")
        self.frames = load_frames(str(self.video_path), self.src_width, self.src_height)
        if self.frames is None:
            raise SystemExit("Failed to load video frames.")

        with open(self.keys_path) as f:
            self.keys_raw: list[list[str]] = json.load(f)

        self.num_frames = self.frames.shape[0]
        self.display_w = self.src_width * SCALE
        self.display_h = self.src_height * SCALE

        self.current_frame = 0
        self.playing = False
        self.direction = 1  # 1=forward, -1=backward
        self.fps = FPS
        self.play_timer: str | None = None

        self.root = tk.Tk()
        self.root.title(f"view_recording - {self.run_dir.name}")
        self.root.configure(bg=BG_COLOR)
        self.root.resizable(False, False)

        self._build_controls()
        self._build_display()
        self._build_info()
        self._bind_keys()

        self._show_frame()

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
        self.play_btn = tk.Button(ctrl, text="▶", command=self.toggle_play, **btn_opts)
        self.play_btn.pack(side=tk.LEFT, padx=2)
        tk.Button(ctrl, text=">>", command=self.step_forward, **btn_opts).pack(
            side=tk.LEFT, padx=2
        )
        tk.Button(ctrl, text=">|", command=self.goto_last, **btn_opts).pack(
            side=tk.LEFT, padx=2
        )

        tk.Button(ctrl, text="⟲", command=self.reverse, **btn_opts).pack(
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

    def _build_display(self) -> None:
        self.canvas = tk.Canvas(
            self.root,
            width=self.display_w,
            height=self.display_h,
            bg="black",
            highlightthickness=0,
        )
        self.canvas.pack(side=tk.TOP, padx=8, pady=4)

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

        self.key_label = tk.Label(
            info,
            text="",
            fg=KEY_COLOR,
            bg=BG_COLOR,
            font=("monospace", 11),
            anchor=tk.E,
        )
        self.key_label.pack(side=tk.RIGHT)

        self.info_label = tk.Label(
            info,
            text="",
            fg="#8888aa",
            bg=BG_COLOR,
            font=("monospace", 9),
        )
        self.info_label.pack(side=tk.LEFT, padx=16)

    def _bind_keys(self) -> None:
        self.root.bind("<space>", lambda e: self.toggle_play())
        self.root.bind("<Left>", lambda e: self.step_back())
        self.root.bind("<Right>", lambda e: self.step_forward())
        self.root.bind("<Up>", lambda e: self.fps_up())
        self.root.bind("<Down>", lambda e: self.fps_down())
        self.root.bind("<Home>", lambda e: self.goto_first())
        self.root.bind("<End>", lambda e: self.goto_last())
        self.root.bind("<Escape>", lambda e: self.root.quit())
        self.root.bind("q", lambda e: self.root.quit())

    def _show_frame(self) -> None:
        if self.current_frame < 0:
            self.current_frame = 0
        if self.current_frame >= self.num_frames:
            self.current_frame = self.num_frames - 1

        frame = self.frames[self.current_frame]  # type: ignore[index]

        img = Image.fromarray(frame, mode="L")
        img = img.resize((self.display_w, self.display_h), Image.NEAREST)  # type: ignore[attr-defined]
        self._photo = ImageTk.PhotoImage(img)

        self.canvas.delete("all")
        self.canvas.create_image(0, 0, anchor=tk.NW, image=self._photo)

        self.frame_label.config(
            text=f"Frame: {self.current_frame + 1} / {self.num_frames}"
        )

        keys = self._current_keys()
        self.key_label.config(text="Keys: " + (", ".join(keys) if keys else "—"))

        duration = self.meta.get("duration_seconds", 0)
        fps = self.meta.get("fps", self.fps)
        info_parts = []
        if duration > 0:
            info_parts.append(f"{duration:.0f}s")
        if self.frames.shape[0] > 0 and fps > 0:  # type: ignore[union-attr]
            info_parts.append(f"{self.frames.shape[0] / fps:.0f}s video")  # type: ignore[union-attr]
        self.info_label.config(text="  ".join(info_parts))

    def _current_keys(self) -> list[str]:
        idx = self.current_frame
        if idx < len(self.keys_raw):
            return self.keys_raw[idx]
        return []

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

    def reverse(self) -> None:
        self.direction = -self.direction
        self._update_button_text()

    def _update_button_text(self) -> None:
        if self.playing:
            if self.direction == 1:
                label = "▶⏸"
            else:
                label = "◀⏸"
        else:
            if self.direction == 1:
                label = "▶"
            else:
                label = "◀"
        self.play_btn.config(text=label)

    def toggle_play(self) -> None:
        if self.playing:
            self._stop_playing()
        else:
            self._start_playing()

    def _start_playing(self) -> None:
        self.playing = True
        self._update_button_text()
        self._tick()

    def _stop_playing(self) -> None:
        self.playing = False
        if self.play_timer:
            self.root.after_cancel(self.play_timer)
        self.play_timer = None
        self._update_button_text()

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
                    self._start_playing()
            else:
                self.fps_var.set(str(self.fps))
        except ValueError:
            self.fps_var.set(str(self.fps))

    def fps_up(self) -> None:
        self.fps = min(120, self.fps + 5)
        self.fps_var.set(str(self.fps))
        if self.playing:
            self._stop_playing()
            self._start_playing()

    def fps_down(self) -> None:
        self.fps = max(1, self.fps - 5)
        self.fps_var.set(str(self.fps))
        if self.playing:
            self._stop_playing()
            self._start_playing()

    def run(self) -> None:
        self.root.mainloop()


def main() -> None:
    if len(sys.argv) < 2:
        sys.exit(f"Usage: {sys.argv[0]} labeled_runs/run_timestamp/")

    run_dir = sys.argv[1]
    viewer = Viewer(run_dir)
    viewer.run()


if __name__ == "__main__":
    main()
