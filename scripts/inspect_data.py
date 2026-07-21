#!/usr/bin/env python3
"""Qt-based inspector for processed recorded gameplay frame windows.

Displays strided frame windows (window_size=4, stride=2) from the
processed dataset alongside their signed-difference representation.
"""

from __future__ import annotations

from pathlib import Path
import sys
from typing import cast

import numpy as np
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QImage, QKeyEvent, QPixmap
from PySide6.QtWidgets import (
    QApplication,
    QComboBox,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QScrollArea,
    QSlider,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)
import torch

# -- Project root so we can import from inverse_dynamics -----------------
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

from components.signed_diff import SignedDiff  # noqa: E402

# -- Constants -----------------------------------------------------------
DATA_DIR = _PROJECT_ROOT / "data" / "processed_recorded_gameplay"
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
WINDOW_SIZE = 4
STRIDE = 2
ZOOM = 2  # integer scale factor per frame
FRAME_W = 320
FRAME_H = 180


def _numpy_to_qimage(arr: np.ndarray, zoom: int) -> QImage:
    """Convert a (H, W) uint8 grayscale array to a zoomed QImage."""
    img = QImage(
        arr.tobytes(),
        arr.shape[1],
        arr.shape[0],
        arr.shape[1],
        QImage.Format.Format_Grayscale8,
    )
    if zoom != 1:
        img = img.scaled(
            arr.shape[1] * zoom,
            arr.shape[0] * zoom,
            Qt.AspectRatioMode.IgnoreAspectRatio,
            Qt.TransformationMode.FastTransformation,
        )
    return img


def _make_row_image(frames: np.ndarray, zoom: int) -> QPixmap:
    """Concatenate 4 (H, W) frames horizontally and return a QPixmap."""
    row = np.hstack([frames[i] for i in range(frames.shape[0])])
    qimg = _numpy_to_qimage(row, zoom=1)
    if zoom != 1:
        qimg = qimg.scaled(
            row.shape[1] * zoom,
            row.shape[0] * zoom,
            Qt.AspectRatioMode.IgnoreAspectRatio,
            Qt.TransformationMode.FastTransformation,
        )
    return QPixmap.fromImage(qimg)


_signed_diff = SignedDiff()


def _compute_signed_diff(frames: np.ndarray) -> np.ndarray:
    """Apply signed-diff transform for visualisation.

    Returns a (window_size, H, W) uint8 array where frame 0 is raw and
    frames 1..T-1 are ``clamp(128 + frame[t] - frame[t-1], 0, 255)``.
    """
    t = torch.from_numpy(frames.copy()).unsqueeze(0)  # (1, T, H, W) uint8
    t = t.float().div_(255)  # normalise -> [0, 1]
    t = _signed_diff(t)
    t = t.squeeze(0).mul_(255).clamp_(0, 255).byte()  # back to uint8
    return t.numpy()


def _format_keys(label: np.ndarray) -> str:
    """Format a label vector as a compact string of active key names."""
    active = [k for k, v in zip(KEY_ORDER, label, strict=False) if v]
    return ", ".join(active) if active else "\u2014"  # em dash


class _Run:
    """Lightweight handle to one run's memory-mapped data."""

    def __init__(self, name: str, frames_path: Path, labels_path: Path) -> None:
        self.name = name
        self._frames_path = frames_path
        self._labels_path = labels_path
        self._frames: np.ndarray | None = None
        self._labels: np.ndarray | None = None
        self._n_frames: int = 0

    def open(self) -> None:
        if self._frames is not None:
            return
        frames = np.load(self._frames_path, mmap_mode="r")
        labels = np.load(self._labels_path, mmap_mode="r")
        self._frames = frames
        self._labels = labels
        self._n_frames = int(frames.shape[0])

    @property
    def n_frames(self) -> int:
        return self._n_frames

    @property
    def max_offset(self) -> int:
        """Maximum valid frame_start for the current window/stride config."""
        return max(0, self._n_frames - (WINDOW_SIZE - 1) * STRIDE - 1)

    def get_window(self, offset: int) -> tuple[np.ndarray, np.ndarray]:
        """Return (window_size, H, W) frames and (window_size, K) labels."""
        indices = np.arange(
            offset,
            offset + WINDOW_SIZE * STRIDE,
            STRIDE,
            dtype=np.int32,
        )
        frames = cast("np.ndarray", self._frames)
        labels = cast("np.ndarray", self._labels)
        return np.asarray(frames[indices]), np.asarray(labels[indices])


def _discover_runs() -> list[_Run]:
    runs: list[_Run] = []
    for fp in sorted(DATA_DIR.glob("run_*_frames.npy")):
        stem = fp.name.replace("_frames.npy", "")
        lp = DATA_DIR / f"{stem}_labels.npy"
        if lp.exists():
            runs.append(_Run(stem, fp, lp))
    return runs


class InspectorWindow(QMainWindow):
    """Main window for browsing frame windows and their signed-diff views."""

    _offset_changed = Signal(int)

    def __init__(self, runs: list[_Run]) -> None:
        super().__init__()
        self._runs = runs
        self._active_run: _Run | None = None
        self._active_offset = 0

        self.setWindowTitle("Data Inspector — window_size=4, stride=2")
        self.setMinimumWidth(FRAME_W * WINDOW_SIZE * 2 // 2)

        central = QWidget()
        self.setCentralWidget(central)
        root = QVBoxLayout(central)

        # --- top bar -------------------------------------------------------
        top = QHBoxLayout()

        self._run_combo = QComboBox()
        for r in runs:
            self._run_combo.addItem(r.name)
        self._run_combo.currentIndexChanged.connect(self._on_run_changed)
        top.addWidget(QLabel("Run:"))
        top.addWidget(self._run_combo)

        top.addWidget(QLabel("  Offset:"))

        self._offset_spin = QSpinBox()
        self._offset_spin.setMinimumWidth(100)
        self._offset_spin.setMaximum(999999)
        self._offset_spin.valueChanged.connect(self._on_offset_spin)
        top.addWidget(self._offset_spin)

        self._max_label = QLabel()
        top.addWidget(self._max_label)

        top.addStretch()

        self._sample_info = QLabel()
        top.addWidget(self._sample_info)

        root.addLayout(top)

        # --- key display ---------------------------------------------------
        self._keys_label = QLabel()
        self._keys_label.setStyleSheet("font-family: monospace; font-size: 12px;")
        root.addWidget(self._keys_label)

        # --- raw frames row ------------------------------------------------
        raw_title = QLabel("Raw Frames")
        raw_title.setStyleSheet("font-weight: bold; font-size: 13px;")
        root.addWidget(raw_title)

        self._raw_scroll = QScrollArea()
        self._raw_scroll.setWidgetResizable(False)
        self._raw_label = QLabel()
        self._raw_label.setAlignment(
            Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop
        )
        self._raw_scroll.setWidget(self._raw_label)
        root.addWidget(self._raw_scroll, stretch=1)

        # --- signed-diff row -----------------------------------------------
        diff_title = QLabel("Signed Diff  (128 + frame[t] \u2212 frame[t\u22121])")
        diff_title.setStyleSheet("font-weight: bold; font-size: 13px;")
        root.addWidget(diff_title)

        self._diff_scroll = QScrollArea()
        self._diff_scroll.setWidgetResizable(False)
        self._diff_label = QLabel()
        self._diff_label.setAlignment(
            Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop
        )
        self._diff_scroll.setWidget(self._diff_label)
        root.addWidget(self._diff_scroll, stretch=1)

        # --- slider --------------------------------------------------------
        self._slider = QSlider(Qt.Orientation.Horizontal)
        self._slider.valueChanged.connect(self._on_slider_changed)
        root.addWidget(self._slider)

        # --- finalise ------------------------------------------------------
        if runs:
            self._switch_run(0)
            self._set_offset(0)

        self._offset_changed.connect(self._render)

    # -- navigation --------------------------------------------------------

    def _on_run_changed(self, idx: int) -> None:
        self._switch_run(idx)

    def _switch_run(self, idx: int) -> None:
        run = self._runs[idx]
        run.open()
        self._active_run = run

        mx = run.max_offset
        self._offset_spin.setMaximum(mx)
        self._slider.setMaximum(mx)
        self._max_label.setText(f"/ {mx}")

        self._set_offset(0)

    def _on_offset_spin(self, val: int) -> None:
        self._slider.blockSignals(True)
        self._slider.setValue(val)
        self._slider.blockSignals(False)
        self._active_offset = val
        self._offset_changed.emit(val)

    def _on_slider_changed(self, val: int) -> None:
        self._offset_spin.blockSignals(True)
        self._offset_spin.setValue(val)
        self._offset_spin.blockSignals(False)
        self._active_offset = val
        self._offset_changed.emit(val)

    def _set_offset(self, val: int) -> None:
        self._offset_spin.setValue(val)
        self._slider.setValue(val)
        self._active_offset = val
        self._offset_changed.emit(val)

    # -- rendering ---------------------------------------------------------

    def _render(self, offset: int) -> None:
        run = self._active_run
        if run is None:
            return

        frames, labels = run.get_window(offset)

        # --- keys text ----------------------------------------------------
        lines = []
        for i in range(WINDOW_SIZE):
            frame_id = offset + i * STRIDE
            lines.append(f"Frame {i} (pos {frame_id:>6d}):  {_format_keys(labels[i])}")
        self._keys_label.setText("\n".join(lines))

        self._sample_info.setText(f"pos {offset} / {run.max_offset}")

        # --- raw row ------------------------------------------------------
        raw_pix = _make_row_image(frames, ZOOM)
        self._raw_label.setPixmap(raw_pix)
        self._raw_label.resize(raw_pix.size())

        # --- signed-diff row ----------------------------------------------
        diff_frames = _compute_signed_diff(frames)
        diff_pix = _make_row_image(diff_frames, ZOOM)
        self._diff_label.setPixmap(diff_pix)
        self._diff_label.resize(diff_pix.size())

    # -- keyboard ----------------------------------------------------------

    def keyPressEvent(self, event: QKeyEvent) -> None:  # noqa: N802
        delta = 0
        if event.key() == Qt.Key.Key_Left:
            delta = -1
        elif event.key() == Qt.Key.Key_Right:
            delta = 1
        elif event.key() == Qt.Key.Key_PageUp:
            delta = -50
        elif event.key() == Qt.Key.Key_PageDown:
            delta = 50
        elif event.key() == Qt.Key.Key_Home:
            self._set_offset(0)
            return
        elif event.key() == Qt.Key.Key_End:
            if self._active_run is not None:
                self._set_offset(self._active_run.max_offset)
            return
        elif event.key() == Qt.Key.Key_W and (
            event.modifiers() & Qt.KeyboardModifier.ControlModifier
        ):
            self.close()
            return
        else:
            super().keyPressEvent(event)
            return

        if self._active_run is not None:
            new = max(0, min(self._active_run.max_offset, self._active_offset + delta))
            self._set_offset(new)


def main() -> None:
    runs = _discover_runs()
    if not runs:
        print(f"No runs found in {DATA_DIR}")
        sys.exit(1)

    app = QApplication(sys.argv)
    win = InspectorWindow(runs)
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
