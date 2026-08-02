"""Qt viewer for browsing captured image/label pairs in ``datagen/output/``.

Dev-only inspection tool: lets you page through ``images/<stem>.png`` /
``labels/<stem>.json`` pairs to sanity-check what the capture daemon
actually wrote. Written defensively -- label JSON files may be missing
entirely, fail to parse, or lack expected fields, and images may be
missing, corrupt, or zero-byte. None of that should crash the viewer; it
should just say so on screen.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import logging
from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtGui import QKeySequence, QPixmap, QResizeEvent, QShortcut
from PySide6.QtWidgets import (
    QApplication,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QPlainTextEdit,
    QSizePolicy,
    QSplitter,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
    QWidget,
)

from datagen.run_config import DEFAULT_OUTPUT_DIR

logger = logging.getLogger(__name__)

IMAGES_SUBDIR = "images"
LABELS_SUBDIR = "labels"

JSONValue = dict[str, "JSONValue"] | list["JSONValue"] | str | int | float | bool | None


@dataclass
class Pair:
    """One image/label pairing discovered under an output directory.

    Attributes:
        stem: Shared filename stem (the scenario's ``image_id``).
        image_path: Path to the ``.png``, or ``None`` if no image file
            with this stem was found.
        label_path: Path to the ``.json``, or ``None`` if no label file
            with this stem was found.
    """

    stem: str
    image_path: Path | None
    label_path: Path | None


def discover_pairs(output_dir: Path) -> list[Pair]:
    """Find every image/label pairing under `output_dir`, by filename stem.

    Deliberately union-based rather than intersection-based: a stem present
    on only one side (an image with no label, or vice versa) is still
    reported, with the missing side left ``None``, so those gaps are visible
    in the UI instead of being silently dropped.

    Args:
        output_dir: Root directory expected to contain ``images/`` and
            ``labels/`` subdirectories.

    Returns:
        Pairs sorted by stem.
    """
    images_dir = output_dir / IMAGES_SUBDIR
    labels_dir = output_dir / LABELS_SUBDIR

    images_by_stem = (
        {p.stem: p for p in images_dir.glob("*.png")} if images_dir.is_dir() else {}
    )
    labels_by_stem = (
        {p.stem: p for p in labels_dir.glob("*.json")} if labels_dir.is_dir() else {}
    )

    if not images_dir.is_dir():
        logger.warning("Images directory not found: %s", images_dir)
    if not labels_dir.is_dir():
        logger.warning("Labels directory not found: %s", labels_dir)

    stems = sorted(set(images_by_stem) | set(labels_by_stem))
    return [
        Pair(
            stem=stem,
            image_path=images_by_stem.get(stem),
            label_path=labels_by_stem.get(stem),
        )
        for stem in stems
    ]


def load_label(label_path: Path) -> tuple[dict[str, JSONValue] | None, str | None]:
    """Load and parse a label JSON file, never raising.

    Args:
        label_path: Path to the label file.

    Returns:
        A ``(data, error)`` tuple: on success, ``(parsed_dict, None)``; on
        any failure, ``(None, human_readable_error)``. A label that parses
        but isn't a JSON object (e.g. a bare list) is also treated as an
        error, since the top-level shape is always a record.
    """
    try:
        raw = label_path.read_text(encoding="utf-8")
    except OSError as exc:
        return None, f"Could not read label file: {exc}"

    try:
        data = json.loads(raw)
    except json.JSONDecodeError as exc:
        return None, f"Malformed JSON: {exc}"

    if not isinstance(data, dict):
        return None, f"Expected a JSON object at top level, got {type(data).__name__}"

    return data, None


def populate_tree(parent: QTreeWidgetItem | QTreeWidget, value: JSONValue) -> None:
    """Recursively render an arbitrary JSON value into a QTreeWidget.

    No assumption is made about which fields exist or what shape they take
    -- this walks whatever structure is actually present, so partially
    missing or malformed records still render as best-effort trees instead
    of raising.

    Args:
        parent: The tree or tree item to attach rendered children to.
        value: The JSON-decoded value (dict, list, or scalar) to render.
    """
    if isinstance(value, dict):
        for key, sub_value in value.items():
            item = QTreeWidgetItem([str(key), ""])
            _attach(parent, item)
            populate_tree(item, sub_value)
    elif isinstance(value, list):
        if not value:
            item = QTreeWidgetItem(["(empty)", ""])
            _attach(parent, item)
            return
        for index, sub_value in enumerate(value):
            item = QTreeWidgetItem([f"[{index}]", ""])
            _attach(parent, item)
            populate_tree(item, sub_value)
    else:
        display = "null" if value is None else str(value)
        leaf = parent
        if isinstance(leaf, QTreeWidgetItem):
            leaf.setText(1, display)
        else:
            # Top-level scalar (shouldn't happen for well-formed records,
            # but render it rather than dropping it silently).
            _attach(parent, QTreeWidgetItem(["(value)", display]))


def _attach(parent: QTreeWidgetItem | QTreeWidget, item: QTreeWidgetItem) -> None:
    """Add `item` as a child of `parent`, whether it's a tree or a tree item."""
    if isinstance(parent, QTreeWidget):
        parent.addTopLevelItem(item)
    else:
        parent.addChild(item)


class PairViewer(QMainWindow):
    """Main window: pair list, field tree, and raw JSON on the left; image on the right."""

    def __init__(self, output_dir: Path) -> None:
        super().__init__()
        self.output_dir = output_dir
        self.pairs: list[Pair] = []

        self.setWindowTitle(f"datagen pair viewer — {output_dir}")
        self.resize(1400, 900)

        self.pair_list = QListWidget()
        self.pair_list.currentRowChanged.connect(self._on_row_changed)

        self.status_label = QLabel("")
        self.status_label.setStyleSheet("color: #b00; font-weight: bold;")
        self.status_label.setWordWrap(True)

        self.field_tree = QTreeWidget()
        self.field_tree.setHeaderLabels(["Field", "Value"])
        self.field_tree.setColumnWidth(0, 220)

        self.raw_text = QPlainTextEdit()
        self.raw_text.setReadOnly(True)
        self.raw_text.setMaximumHeight(160)
        self.raw_text.setPlaceholderText("Raw label JSON appears here.")

        left_layout = QVBoxLayout()
        left_layout.addWidget(QLabel("Pairs (by scenario/image id):"))
        left_layout.addWidget(self.pair_list, stretch=1)
        left_layout.addWidget(self.status_label)
        left_layout.addWidget(QLabel("Fields:"))
        left_layout.addWidget(self.field_tree, stretch=2)
        left_layout.addWidget(QLabel("Raw JSON:"))
        left_layout.addWidget(self.raw_text)
        left_widget = QWidget()
        left_widget.setLayout(left_layout)

        self.image_label = QLabel("No image")
        self.image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.image_label.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding
        )
        self.image_label.setMinimumSize(200, 200)
        self.image_label.setStyleSheet("background-color: #222; color: #aaa;")

        self.image_meta_label = QLabel("")
        self.image_meta_label.setAlignment(Qt.AlignmentFlag.AlignCenter)

        right_layout = QVBoxLayout()
        right_layout.addWidget(self.image_label, stretch=1)
        right_layout.addWidget(self.image_meta_label)
        right_widget = QWidget()
        right_widget.setLayout(right_layout)

        splitter = QSplitter()
        splitter.addWidget(left_widget)
        splitter.addWidget(right_widget)
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        splitter.setSizes([520, 880])

        container = QWidget()
        outer_layout = QHBoxLayout()
        outer_layout.addWidget(splitter)
        container.setLayout(outer_layout)
        self.setCentralWidget(container)

        next_shortcut = QShortcut(QKeySequence(Qt.Key.Key_Down), self)
        next_shortcut.activated.connect(self._select_next)
        previous_shortcut = QShortcut(QKeySequence(Qt.Key.Key_Up), self)
        previous_shortcut.activated.connect(self._select_previous)

        self.reload()

    def reload(self) -> None:
        """Re-scan the output directory and repopulate the pair list."""
        self.pairs = discover_pairs(self.output_dir)
        self.pair_list.clear()

        if not self.pairs:
            self.setWindowTitle(
                f"datagen pair viewer — {self.output_dir} (no pairs found)"
            )
            return

        for pair in self.pairs:
            label = pair.stem
            if pair.image_path is None:
                label += "  [no image]"
            if pair.label_path is None:
                label += "  [no label]"
            item = QListWidgetItem(label)
            self.pair_list.addItem(item)

        self.pair_list.setCurrentRow(0)

    def _select_next(self) -> None:
        row = self.pair_list.currentRow()
        if row < self.pair_list.count() - 1:
            self.pair_list.setCurrentRow(row + 1)

    def _select_previous(self) -> None:
        row = self.pair_list.currentRow()
        if row > 0:
            self.pair_list.setCurrentRow(row - 1)

    def _on_row_changed(self, row: int) -> None:
        if row < 0 or row >= len(self.pairs):
            self._show_empty()
            return
        self._show_pair(self.pairs[row])

    def _show_empty(self) -> None:
        self.status_label.setText("")
        self.field_tree.clear()
        self.raw_text.clear()
        self.image_label.setText("No pair selected")
        self.image_label.setPixmap(QPixmap())
        self.image_meta_label.setText("")

    def _show_pair(self, pair: Pair) -> None:
        self.field_tree.clear()
        self.raw_text.clear()
        problems: list[str] = []

        if pair.label_path is None:
            problems.append("No label file found for this image.")
        else:
            data, error = load_label(pair.label_path)
            if error is not None:
                problems.append(f"{pair.label_path.name}: {error}")
            else:
                assert data is not None
                populate_tree(self.field_tree, data)
                self.field_tree.expandToDepth(0)
                self.raw_text.setPlainText(json.dumps(data, indent=2, sort_keys=True))

        if pair.image_path is None:
            problems.append("No image file found for this label.")
            self.image_label.setPixmap(QPixmap())
            self.image_label.setText("(missing image)")
            self.image_meta_label.setText("")
        else:
            self._load_image(pair.image_path, problems)

        self.status_label.setText("\n".join(problems))

    def _load_image(self, image_path: Path, problems: list[str]) -> None:
        pixmap = QPixmap(str(image_path))
        if pixmap.isNull():
            problems.append(f"{image_path.name}: could not be decoded as an image.")
            self.image_label.setPixmap(QPixmap())
            self.image_label.setText("(unreadable image)")
            self.image_meta_label.setText(image_path.name)
            return

        self.image_label.setText("")
        self._current_pixmap = pixmap
        self._rescale_image()
        self.image_meta_label.setText(
            f"{image_path.name}  ({pixmap.width()}x{pixmap.height()})"
        )

    def _rescale_image(self) -> None:
        pixmap = getattr(self, "_current_pixmap", None)
        if pixmap is None or pixmap.isNull():
            return
        scaled = pixmap.scaled(
            self.image_label.size(),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        self.image_label.setPixmap(scaled)

    def resizeEvent(self, event: QResizeEvent) -> None:  # noqa: N802 (Qt override)
        super().resizeEvent(event)
        self._rescale_image()


def main() -> None:
    """CLI entry point: launch the viewer over an output directory."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        default=None,
        help=f"Output directory to browse (default: {DEFAULT_OUTPUT_DIR})",
    )
    parser.add_argument(
        "-v", "--verbose", action="store_true", help="Enable debug logging"
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    output_dir = (
        Path(args.output_dir) if args.output_dir is not None else DEFAULT_OUTPUT_DIR
    )
    if not output_dir.is_dir():
        logger.warning("Output directory does not exist yet: %s", output_dir)

    app = QApplication([])
    window = PairViewer(output_dir)
    window.show()
    app.exec()


if __name__ == "__main__":
    main()
