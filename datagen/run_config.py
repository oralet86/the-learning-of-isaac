"""Run-level configuration for the synthetic data generator."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

DEFAULT_OUTPUT_DIR = Path(__file__).parent / "output"

# Shared file-exchange directory with the mod. The mod writes ground-truth
# files here via its Windows-style Z:-drive path
# (tloi_datagen/config.lua's Config.EXCHANGE_DIR) -- confirmed on this
# machine's default Wine drive mapping (host / -> Z:\). If you move to a
# different machine/install, update both together.
DEFAULT_EXCHANGE_DIR = Path(__file__).parent / "runtime"


@dataclass
class RunConfig:
    """Settings shared by a single capture-daemon run.

    Attributes:
        exchange_dir: Shared file-exchange directory (see
            :mod:`datagen.exchange`). The mod writes ``<scenario_id>.json``
            ground-truth files here once a scene is settled; the daemon
            writes back ``<scenario_id>.done`` once it has captured and
            saved the paired frame.
        output_dir: Root directory for captured images and ground-truth
            labels.
        poll_interval_s: Sleep between exchange-directory polls.
        video_device: v4l2 device path passed through to
            ``live.stream_game.StreamGame``.
    """

    exchange_dir: Path = field(default_factory=lambda: DEFAULT_EXCHANGE_DIR)
    output_dir: Path = field(default_factory=lambda: DEFAULT_OUTPUT_DIR)
    poll_interval_s: float = 0.05
    video_device: str = "/dev/video2"
