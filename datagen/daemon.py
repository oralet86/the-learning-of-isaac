"""Capture daemon: watches the exchange directory and captures frames as the mod poses scenes.

The mod starts posing scenarios on its own as soon as a run begins (see
``tloi_datagen/main.lua``'s ``MC_POST_GAME_STARTED`` handler) and keeps
going for as long as the game is open -- this process only needs to
already be running and watching the exchange directory before that
happens. There is no per-scenario confirmation step; it runs until
interrupted (Ctrl-C).
"""

from __future__ import annotations

import argparse
import dataclasses
from datetime import UTC, datetime
import json
import logging
from pathlib import Path
import time
from typing import TYPE_CHECKING

from datagen.capture import capture_frame
from datagen.exchange import find_pending, read_ground_truth, write_ack
from datagen.schema import (
    ActiveItem,
    Consumable,
    CurrencyState,
    Door,
    EnemyState,
    GroundTruthRecordM1,
    ItemDelta,
    PlayerHealth,
    PlayerStateM1,
    PlayerStats,
)

if TYPE_CHECKING:
    from datagen.run_config import RunConfig
    from live.stream_game import StreamGame

logger = logging.getLogger(__name__)


def _drain_stream(stream: StreamGame) -> None:
    """Discard all currently buffered frames.

    ``StreamGame``'s background reader thread pushes frames into a small
    bounded queue at the capture frame rate regardless of whether anything
    is consuming them; while the daemon is idle between scenarios, nothing
    else calls ``get_frame``, so the queue fills and the reader thread
    crashes on the next full push. Called once per poll loop iteration to
    keep it drained.
    """
    while stream.get_frame(timeout=0) is not None:
        pass


def _process_one(
    config: RunConfig, stream: StreamGame, pending_path: Path
) -> GroundTruthRecordM1:
    """Capture and save one scenario, then ack it back to the mod.

    Claims `pending_path` by deleting it before the (slower) capture step,
    so a mod poll that fires while capture is still in flight can never see
    the same file twice.

    Args:
        config: Run configuration (exchange dir, output dir).
        stream: An already-started ``StreamGame`` instance.
        pending_path: The ground-truth file to process.

    Returns:
        The ground-truth record that was written to disk.
    """
    raw_ground_truth = read_ground_truth(pending_path)
    scenario_id = raw_ground_truth["scenario_id"]
    pending_path.unlink(missing_ok=True)

    viewport = raw_ground_truth["camera_viewport_position"]
    image_id = f"{scenario_id}_{viewport}"
    image_path = config.output_dir / "images" / f"{image_id}.png"
    capture_frame(stream, image_path)

    player_raw = raw_ground_truth["player"]
    player2_raw = raw_ground_truth.get("player2")
    currency_raw = raw_ground_truth["currency"]
    consumable_raw = raw_ground_truth["consumable"]
    stats_raw = raw_ground_truth["stats"]
    health_raw = raw_ground_truth["health"]
    item_delta_raw = raw_ground_truth["item_delta"]
    active_item_raw = raw_ground_truth["active_item"]
    record = GroundTruthRecordM1(
        image_id=image_id,
        scenario_id=scenario_id,
        seed=raw_ground_truth["seed"],
        stage=raw_ground_truth["stage"],
        room_shape=raw_ground_truth["room_shape"],
        room_type=raw_ground_truth["room_type"],
        camera_viewport_position=raw_ground_truth["camera_viewport_position"],
        player=PlayerStateM1(
            character=player_raw["character"],
            position=(player_raw["position"][0], player_raw["position"][1]),
        ),
        player2=PlayerStateM1(
            character=player2_raw["character"],
            position=(player2_raw["position"][0], player2_raw["position"][1]),
        )
        if player2_raw is not None
        else None,
        items_held=raw_ground_truth["items_held"],
        item_delta=ItemDelta(
            action=item_delta_raw["action"],
            item_id=item_delta_raw.get("item_id"),
        ),
        active_item=ActiveItem(
            held=active_item_raw["held"],
            item_id=active_item_raw.get("item_id"),
            charge=active_item_raw.get("charge"),
        ),
        currency=CurrencyState(
            coins=currency_raw["coins"],
            bombs=currency_raw["bombs"],
            keys=currency_raw["keys"],
        ),
        consumable=Consumable(
            kind=consumable_raw["kind"],
            card_id=consumable_raw.get("card_id"),
            pill_color=consumable_raw.get("pill_color"),
        ),
        stats=PlayerStats(
            damage=stats_raw["damage"],
            fire_rate=stats_raw["fire_rate"],
            speed=stats_raw["speed"],
            range=stats_raw["range"],
            shot_speed=stats_raw["shot_speed"],
            luck=stats_raw["luck"],
        ),
        health=PlayerHealth(
            unknown=health_raw["unknown"],
            hearts=health_raw.get("hearts"),
            max_hearts=health_raw.get("max_hearts"),
            soul_hearts=health_raw.get("soul_hearts"),
            black_hearts=health_raw.get("black_hearts"),
            bone_hearts=health_raw.get("bone_hearts"),
            rotten_hearts=health_raw.get("rotten_hearts"),
            eternal_hearts=health_raw.get("eternal_hearts"),
            golden_hearts=health_raw.get("golden_hearts"),
            broken_hearts=health_raw.get("broken_hearts"),
        ),
        enemies=[
            EnemyState(
                type=enemy_raw["type"],
                variant=enemy_raw["variant"],
                subtype=enemy_raw["subtype"],
                position=(enemy_raw["position"][0], enemy_raw["position"][1]),
            )
            for enemy_raw in raw_ground_truth["enemies"]
        ],
        doors=[
            Door(
                slot=door_raw["slot"],
                is_open=door_raw["is_open"],
                is_locked=door_raw["is_locked"],
                is_busted=door_raw["is_busted"],
            )
            for door_raw in raw_ground_truth["doors"]
        ],
        capture_mode=raw_ground_truth["capture_mode"],
        timestamp=datetime.now(UTC).isoformat(),
    )

    label_path = config.output_dir / "labels" / f"{image_id}.json"
    label_path.parent.mkdir(parents=True, exist_ok=True)
    label_path.write_text(json.dumps(dataclasses.asdict(record), indent=2))

    write_ack(config.exchange_dir, scenario_id)

    return record


def run_daemon(config: RunConfig, stream: StreamGame) -> None:
    """Poll the exchange directory forever, capturing each scenario as the mod poses it.

    Args:
        config: Run configuration (exchange dir, output dir, poll interval).
        stream: An already-started ``StreamGame`` instance.
    """
    config.exchange_dir.mkdir(parents=True, exist_ok=True)
    logger.info(
        "watching %s -- generation starts automatically when a run begins",
        config.exchange_dir,
    )
    captured_count = 0
    while True:
        _drain_stream(stream)
        for pending_path in find_pending(config.exchange_dir):
            try:
                record = _process_one(config, stream, pending_path)
            except (KeyError, TypeError, ValueError, json.JSONDecodeError):
                logger.exception(
                    "malformed ground-truth file %s, skipping", pending_path
                )
                pending_path.unlink(missing_ok=True)
            else:
                captured_count += 1
                logger.info(
                    "[%d] captured %s (seed=%d, stage=%s, room_type=%d, items=%d, "
                    "delta=%s, active=%s, consumable=%s, enemies=%d, doors=%d)",
                    captured_count,
                    record.image_id,
                    record.seed,
                    record.stage,
                    record.room_type,
                    record.items_held,
                    record.item_delta.action,
                    record.active_item.item_id,
                    record.consumable.kind,
                    len(record.enemies),
                    len(record.doors),
                )
            _drain_stream(stream)
        time.sleep(config.poll_interval_s)


def main() -> None:
    """Entry point: run the capture daemon until interrupted.

    Requires the game running with ``--luadebug`` set, the ``tloi_datagen``
    mod enabled, plus OBS with the virtual camera pointed at the game
    window (see ``live/stream_game.py``'s module docstring for setup).
    """
    from datagen.run_config import RunConfig
    from live.stream_game import StreamGame

    parser = argparse.ArgumentParser(description="Run the capture daemon.")
    parser.add_argument(
        "--video-device", default="/dev/video2", help="v4l2 device path"
    )
    parser.add_argument("--output-dir", default=None, help="Output directory override")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO)

    config = RunConfig(video_device=args.video_device)
    if args.output_dir is not None:
        config.output_dir = Path(args.output_dir)

    with StreamGame(video_device=args.video_device) as stream:
        try:
            run_daemon(config, stream)
        except KeyboardInterrupt:
            logger.info("stopped")


if __name__ == "__main__":
    main()
