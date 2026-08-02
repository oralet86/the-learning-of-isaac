"""File-based exchange protocol with the ``tloi_datagen`` REPENTOGON mod.

Protocol (must stay in sync with ``tloi_datagen/ipc/exchange.lua``):

- The mod writes ``<scenario_id>.json`` (ground truth) into the shared
  exchange directory once a posed scene has settled -- this file's
  existence is the "ready for capture" signal. It writes to a ``.tmp`` path
  first and renames into place, so a ``.json`` file is always complete by
  the time it appears under that name.
- The daemon (:mod:`datagen.daemon`) claims a pending file by reading then
  immediately deleting it, so a slow capture step can never cause the same
  scenario to be processed twice. Once the paired image + label are saved,
  it writes back an empty ``<scenario_id>.done`` file, which the mod polls
  for as its capture-acknowledged signal before cleaning up and (if still
  running) posing the next scenario.

Both sides see the same directory through different path forms: the mod
runs as a Windows process under Proton/Wine and writes via its Z:-drive
path (``tloi_datagen/config.lua``'s ``Config.EXCHANGE_DIR``); this
module reads the same folder via its native Linux path
(``RunConfig.exchange_dir``). This is possible at all because the game is
launched with ``--luadebug``, which unlocks the ``io``/``os`` Lua standard
libraries REPENTOGON mods otherwise lack.
"""

from __future__ import annotations

import json
import logging
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from pathlib import Path

logger = logging.getLogger(__name__)


def find_pending(exchange_dir: Path) -> list[Path]:
    """Return ground-truth files currently waiting for capture, oldest first.

    Args:
        exchange_dir: The shared exchange directory.
    """
    if not exchange_dir.exists():
        return []
    return sorted(exchange_dir.glob("*.json"), key=lambda p: p.stat().st_mtime)


def read_ground_truth(path: Path) -> dict[str, Any]:
    """Parse a ground-truth file written by the mod.

    Raises:
        json.JSONDecodeError: If the file is not valid JSON.
    """
    return json.loads(path.read_text(encoding="utf-8"))


def write_ack(exchange_dir: Path, scenario_id: str) -> None:
    """Signal the mod that `scenario_id` has been captured and may be cleaned up."""
    (exchange_dir / f"{scenario_id}.done").touch()
