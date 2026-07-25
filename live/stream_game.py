#!/usr/bin/env python3
"""Real-time frame stream from OBS virtual camera via v4l2 + ffmpeg.

Connects to OBS via obs-websocket, starts the virtual camera, and pipes
decoded rgb24 frames from ffmpeg into Python as fp16 numpy arrays.

Requires the ``v4l2loopback`` kernel module loaded with ``exclusive_caps=0``::

    sudo modprobe v4l2loopback exclusive_caps=0

Usage:
  # Probe device state (diagnostics)
  python stream_game.py --probe

  # Debug mode — Qt window showing live frames + stats
  python stream_game.py --debug

  # Library mode — import into training loop
  from stream_game import StreamGame
  with StreamGame() as stream:
      for frame in stream:          # fp16 (H, W, 3), raw uint8 values
          ...
"""

from __future__ import annotations

import argparse
from contextlib import suppress
import os
import queue
import select
import subprocess
import sys
import threading
import time
from typing import Self

import numpy as np
import obsws_python as obs
from obsws_python.error import OBSSDKError
from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QCloseEvent, QFont, QImage, QPixmap
from PySide6.QtWidgets import (
    QApplication,
    QLabel,
    QMainWindow,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEFAULT_VIDEO_DEVICE = "/dev/video2"
FRAME_READ_CHUNK = 65536
FRAME_QUEUE_SIZE = 4
DEVICE_SETTLE_S = 0.3
DEBUG_REFRESH_MS = 16
INFO_FONT_SIZE = 11
VERBOSE_LOG_INTERVAL_S = 5.0
READER_POLL_TIMEOUT_S = 0.5


# ---------------------------------------------------------------------------
# v4l2 probing
# ---------------------------------------------------------------------------


def _probe_device(device: str) -> tuple[int, int, int] | None:
    """Run ffprobe on the v4l2 device and return (width, height, fps).

    Returns ``None`` if the device is inactive or ffprobe fails.
    """
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-f",
        "v4l2",
        "-show_entries",
        "stream=width,height,avg_frame_rate",
        "-of",
        "csv=p=0",
        device,
    ]
    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=10, check=False
        )
    except OSError, subprocess.TimeoutExpired:
        return None

    if result.returncode != 0 or not result.stdout.strip():
        return None

    try:
        w_str, h_str, rate_str = result.stdout.strip().split(",")
        w = int(w_str)
        h = int(h_str)
        num, den = rate_str.split("/")
        fps = float(num) / float(den)
        return w, h, round(fps)
    except ValueError, ZeroDivisionError:
        return None


def _probe_and_print(device: str) -> None:
    """Probe the v4l2 device and print its state to stdout."""
    import shutil

    print(f"Probe: {device}\n")

    # v4l2-ctl dump
    v4l2_path = shutil.which("v4l2-ctl")
    if v4l2_path:
        try:
            result = subprocess.run(
                [v4l2_path, "-d", device, "--all"],
                capture_output=True,
                text=True,
                timeout=10,
                check=False,
            )
            if result.returncode == 0:
                print(result.stdout)
            else:
                print(result.stderr)
        except OSError as e:
            print(f"  v4l2-ctl: {e}")
    else:
        print("  v4l2-ctl not found in PATH")

    print("---")

    # ffprobe
    probed = _probe_device(device)
    if probed is not None:
        w, h, fps = probed
        print(f"ffprobe: {w}x{h} @ {fps}fps")
    else:
        print("ffprobe: no active signal on device")


# ---------------------------------------------------------------------------
# ffmpeg helpers
# ---------------------------------------------------------------------------


def _launch_ffmpeg(device: str, fps: int) -> subprocess.Popen[bytes]:
    """Launch ffmpeg reading from a v4l2 device, piping raw rgb24 to stdout.

    Resolution is auto-detected from the v4l2 device (OBS must already be
    pushing frames).  Low-latency flags minimise pipeline buffering.
    """
    cmd = [
        "ffmpeg",
        "-nostdin",
        "-loglevel",
        "error",
        "-fflags",
        "nobuffer",
        "-flags",
        "low_delay",
        "-avioflags",
        "direct",
        "-thread_queue_size",
        "4",
        "-f",
        "v4l2",
        "-framerate",
        str(fps),
        "-i",
        device,
        "-f",
        "rawvideo",
        "-pix_fmt",
        "rgb24",
        "-vsync",
        "0",
        "pipe:1",
    ]
    return subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)


def _kill_process(proc: subprocess.Popen[bytes] | None) -> None:
    """Kill a subprocess and wait for it to exit gracefully."""
    if proc is None:
        return
    with suppress(Exception):
        proc.kill()
    with suppress(Exception):
        proc.wait(timeout=2)


def _read_stderr(proc: subprocess.Popen[bytes], sink: list[str]) -> None:
    """Daemon thread target: read stderr lines from *proc* into *sink*."""
    assert proc.stderr is not None
    for line in proc.stderr:
        sink.append(line.decode(errors="replace").rstrip())


# ---------------------------------------------------------------------------
# StreamGame
# ---------------------------------------------------------------------------


class StreamGame:
    """Real-time frame stream from OBS virtual camera.

    Connects to OBS, starts the virtual camera, probes the v4l2 device
    for the actual resolution, then launches ffmpeg to decode raw rgb24
    frames into fp16 numpy arrays.

    Typical usage::

        with StreamGame() as stream:
            for frame in stream:
                # frame: fp16 (H, W, 3), raw uint8 values
                ...

    or imperative::

        stream = StreamGame()
        stream.start()
        while True:
            frame = stream.get_frame()
            if frame is None:
                break
        stream.stop()
    """

    def __init__(
        self,
        host: str = "localhost",
        port: int = 4455,
        password: str = "",
        video_device: str = DEFAULT_VIDEO_DEVICE,
        normalize: bool = False,
        verbose: bool = False,
    ) -> None:
        self._host = host
        self._port = port
        self._password = password
        self._video_device = video_device
        self._normalize = normalize
        self._verbose = verbose

        self._width = 0
        self._height = 0
        self._fps = 0
        self._frame_size = 0

        self._cl: obs.ReqClient | None = None
        self._ffmpeg: subprocess.Popen[bytes] | None = None
        self._ffmpeg_stderr: list[str] = []
        self._stderr_thread: threading.Thread | None = None
        self._frame_size = self._width * self._height * 3
        self._frame_queue: queue.Queue[np.ndarray | None] = queue.Queue(
            maxsize=FRAME_QUEUE_SIZE
        )
        self._reader_thread: threading.Thread | None = None
        self._reader_running = threading.Event()
        self._started = False
        self._frame_count = 0
        self._start_monotonic = 0.0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def start(self) -> None:
        """Connect OBS, start virtual camera, probe device, launch ffmpeg.

        Raises:
            FileNotFoundError: if the v4l2 device does not exist.
            ConnectionError: if the OBS WebSocket is unreachable.
            RuntimeError: if the virtual camera fails to start.
        """
        if self._started:
            return

        if not os.path.exists(self._video_device):
            raise FileNotFoundError(
                f"{self._video_device} not found. "
                f"Run: sudo modprobe v4l2loopback exclusive_caps=0"
            )

        # Step 1 — connect OBS.
        try:
            self._cl = obs.ReqClient(
                host=self._host,
                port=self._port,
                password=self._password,
                timeout=5,
            )
        except (OBSSDKError, OSError) as e:
            raise ConnectionError(
                f"Cannot connect to OBS at {self._host}:{self._port} — {e}"
            ) from e

        # Step 2 — start virtual camera (so v4l2 device has a signal).
        self._start_virtual_cam()

        # Step 3 — probe actual resolution (mandatory — no fallback).
        time.sleep(DEVICE_SETTLE_S)
        probed = _probe_device(self._video_device)
        if probed is None:
            raise RuntimeError(
                f"Could not probe {self._video_device}. "
                f"Is the OBS virtual camera running and outputting frames?"
            )
        self._width, self._height, self._fps = probed
        self._frame_size = self._width * self._height * 3

        # Step 4 — launch ffmpeg (now that the device format is stable).
        self._ffmpeg = _launch_ffmpeg(self._video_device, self._fps)

        # Step 5 — start background reader thread.
        self._reader_running.set()
        self._reader_thread = threading.Thread(target=self._reader_loop, daemon=True)
        self._reader_thread.start()

        self._started = True
        self._start_monotonic = time.monotonic()
        print(
            f"  Streaming {self._width}x{self._height} @ {self._fps}fps "
            f"from {self._video_device}"
        )

    def get_frame(self, timeout: float | None = None) -> np.ndarray | None:
        """Block until the next frame is ready.

        Args:
            timeout: Seconds to wait before returning ``None``.
                     ``None`` blocks indefinitely.

        Returns:
            fp16 array ``(H, W, 3)``, or ``None`` on timeout or
            end-of-stream.  Values are raw uint8 unless ``normalize=True``
            was passed to the constructor, in which case they are in
            ``[0, 1]``.
        """
        if not self._started:
            return None

        try:
            frame = self._frame_queue.get(timeout=timeout)
        except queue.Empty:
            return None

        if frame is None:
            return None

        return frame

    def stop(self) -> None:
        """Stop streaming, kill ffmpeg, disconnect OBS."""
        self._started = False

        # Signal reader thread to stop.
        self._reader_running.clear()

        # Unblock queue consumer by pushing a sentinel.
        with suppress(queue.Full):
            self._frame_queue.put(None, timeout=0.5)

        if self._reader_thread is not None:
            self._reader_thread.join(timeout=3.0)
            self._reader_thread = None

        # Drain any remaining frames.
        while True:
            try:
                self._frame_queue.get_nowait()
            except queue.Empty:
                break

        had_error = self._frame_count == 0 and bool(self._ffmpeg_stderr)

        if self._cl is not None:
            with suppress(Exception):
                self._cl.stop_virtual_cam()
            with suppress(Exception):
                self._cl.disconnect()
            self._cl = None

        _kill_process(self._ffmpeg)
        self._ffmpeg = None

        if had_error:
            print("  ffmpeg stderr (last 20 lines):")
            for line in self._ffmpeg_stderr[-20:]:
                print(f"    {line}")

    # ------------------------------------------------------------------
    # Iterator / context manager
    # ------------------------------------------------------------------

    def __iter__(self) -> Self:
        return self

    def __next__(self) -> np.ndarray:
        frame = self.get_frame()
        if frame is None:
            raise StopIteration
        return frame

    def __enter__(self) -> Self:
        self.start()
        return self

    def __exit__(self, *args: object) -> None:
        self.stop()

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def frame_count(self) -> int:
        """Total frames received since ``start()``."""
        return self._frame_count

    @property
    def elapsed(self) -> float:
        """Seconds elapsed since ``start()``."""
        if self._start_monotonic == 0.0:
            return 0.0
        return time.monotonic() - self._start_monotonic

    @property
    def effective_fps(self) -> float:
        """Rolling average fps over the entire stream."""
        elapsed = self.elapsed
        if elapsed <= 0 or self._frame_count == 0:
            return 0.0
        return self._frame_count / elapsed

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _start_virtual_cam(self) -> None:
        """Start OBS virtual camera if not already running."""
        assert self._cl is not None
        try:
            status = self._cl.get_virtual_cam_status()
            if getattr(status, "output_active", False):
                print("  Note: virtual camera was already active")
                return
        except Exception:
            pass

        try:
            self._cl.start_virtual_cam()
        except Exception as e:
            self._cl.disconnect()
            raise RuntimeError(f"Failed to start OBS virtual camera: {e}") from e

    def _reader_loop(self) -> None:
        """Background thread: drain ffmpeg stdout, push frames into the queue."""
        assert self._ffmpeg is not None and self._ffmpeg.stdout is not None

        # Spin up the stderr reader inside this thread so it shares
        # the process lifetime.
        stderr_thread = threading.Thread(
            target=_read_stderr,
            args=(self._ffmpeg, self._ffmpeg_stderr),
            daemon=True,
        )
        stderr_thread.start()

        buf = bytearray()
        count = 0
        last_log = time.monotonic()

        try:
            while self._reader_running.is_set():
                try:
                    ready, _, _ = select.select(
                        [self._ffmpeg.stdout], [], [], READER_POLL_TIMEOUT_S
                    )
                except ValueError, OSError:
                    break

                if not ready:
                    if (
                        self._verbose
                        and time.monotonic() - last_log > VERBOSE_LOG_INTERVAL_S
                    ):
                        last_log = time.monotonic()
                        elapsed = time.monotonic() - self._start_monotonic
                        print(
                            f"  [stream] idle ({elapsed:.0f}s elapsed, {count} frames)"
                        )
                    continue

                try:
                    chunk = os.read(self._ffmpeg.stdout.fileno(), FRAME_READ_CHUNK)
                except OSError:
                    break

                if not chunk:
                    if self._verbose:
                        print(f"  [stream] ffmpeg stdout closed ({count} frames)")
                    break

                buf.extend(chunk)

                while len(buf) >= self._frame_size:
                    frame_bytes = buf[: self._frame_size]
                    buf = buf[self._frame_size :]
                    count += 1
                    self._frame_count = count

                    if self._verbose and count == 1:
                        print(
                            f"  [stream] first frame: "
                            f"({self._height}, {self._width}, 3)"
                        )

                    arr = np.frombuffer(frame_bytes, dtype=np.uint8).reshape(
                        self._height, self._width, 3
                    )
                    result = arr.astype(np.float16)
                    if self._normalize:
                        result /= 255.0

                    self._frame_queue.put(result, timeout=1.0)

        finally:
            with suppress(queue.Full):
                self._frame_queue.put(None, timeout=1.0)


# ---------------------------------------------------------------------------
# Qt debug viewer
# ---------------------------------------------------------------------------


class StreamViewer(QMainWindow):
    """Live view of StreamGame frames with overlaid stats."""

    def __init__(self, stream: StreamGame) -> None:
        super().__init__()
        self._stream = stream
        self._stream.start()

        self._current_array: np.ndarray | None = None

        self.setWindowTitle("stream_game — live")
        self.setStyleSheet("background-color: #1a1a2e;")

        central = QWidget()
        self.setCentralWidget(central)
        layout = QVBoxLayout(central)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        # Info bar
        self._info_label = QLabel()
        self._info_label.setFont(QFont("monospace", INFO_FONT_SIZE))
        self._info_label.setStyleSheet(
            "color: #00ff88; background-color: #0d0d1a; padding: 4px 8px;"
        )
        self._info_label.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed
        )
        layout.addWidget(self._info_label)

        # Frame display
        self._frame_label = QLabel()
        self._frame_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._frame_label.setMinimumSize(100, 100)
        self._frame_label.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding
        )
        self._frame_label.setStyleSheet("background-color: #000000;")
        layout.addWidget(self._frame_label)

        # Timer
        self._timer = QTimer()
        self._timer.timeout.connect(self._tick)
        self._timer.start(DEBUG_REFRESH_MS)

        # Initial window size
        self.resize(self._stream._width, self._stream._height + 30)

    def _tick(self) -> None:
        """Poll for a new frame and refresh the display."""
        frame = self._stream.get_frame(timeout=0.0)
        if frame is None:
            return

        self._current_array = frame
        display = (
            (frame * 255).astype(np.uint8)
            if self._stream._normalize
            else frame.astype(np.uint8)
        )
        h, w, _ = display.shape

        qimg = QImage(display.tobytes(), w, h, w * 3, QImage.Format.Format_RGB888)
        pixmap = QPixmap.fromImage(qimg)

        label_size = self._frame_label.size()
        scaled = pixmap.scaled(
            label_size,
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        self._frame_label.setPixmap(scaled)

        self._update_info()

    def _update_info(self) -> None:
        """Refresh the info bar."""
        fps = self._stream.effective_fps
        elapsed = self._stream.elapsed
        shape_str = f"({self._stream._height}, {self._stream._width}, 3)"
        norm = "norm" if self._stream._normalize else "raw"

        self._info_label.setText(
            f"Frame: {self._stream.frame_count:>6}   "
            f"FPS: {fps:5.1f}   "
            f"Shape: {shape_str}   "
            f"float16 ({norm})   "
            f"Elapsed: {elapsed:.0f}s"
        )

    def closeEvent(self, event: QCloseEvent) -> None:  # noqa: N802
        """Stop the stream and timer on window close."""
        self._timer.stop()
        self._stream.stop()
        super().closeEvent(event)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Stream OBS virtual camera frames to Python via ffmpeg + v4l2."
    )
    parser.add_argument(
        "--host", default="localhost", help="OBS WebSocket host (default: localhost)"
    )
    parser.add_argument(
        "--port", type=int, default=4455, help="OBS WebSocket port (default: 4455)"
    )
    parser.add_argument("--password", default="", help="OBS WebSocket password")
    parser.add_argument(
        "--device",
        default=DEFAULT_VIDEO_DEVICE,
        help=f"v4l2 device path (default: {DEFAULT_VIDEO_DEVICE})",
    )
    parser.add_argument(
        "--normalize",
        action="store_true",
        help="Divide frame values by 255 (default: raw uint8 in fp16)",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Log data-flow events (stalls, first frame, EOF)",
    )
    parser.add_argument(
        "--probe",
        action="store_true",
        help="Probe the v4l2 device and exit (diagnostic)",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Open a Qt window showing live frames with stats",
    )
    args = parser.parse_args()

    if args.probe:
        _probe_and_print(args.device)
        return

    stream = StreamGame(
        host=args.host,
        port=args.port,
        password=args.password,
        video_device=args.device,
        normalize=args.normalize,
        verbose=args.verbose,
    )

    if args.debug:
        app = QApplication(sys.argv)
        viewer = StreamViewer(stream)
        viewer.show()
        sys.exit(app.exec())
    else:
        print("Starting stream (library mode).  Press Ctrl+C to stop.")
        stream.start()
        try:
            while True:
                frame = stream.get_frame(timeout=1.0)
                if frame is None:
                    time.sleep(0.01)
                    continue
                if stream.frame_count % 60 == 0:
                    print(
                        f"  frame {stream.frame_count:>6}"
                        f"  fps {stream.effective_fps:5.1f}"
                        f"  shape {frame.shape}"
                    )
        except KeyboardInterrupt:
            print("\nStopping...")
        finally:
            stream.stop()


if __name__ == "__main__":
    main()
