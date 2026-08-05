#!/usr/bin/env python3
"""Train `GroundTruthNet`: screenshot -> `labels.LabelBundle`.

Requires a completed `datagen` capture run (`datagen/output/{images,labels}`)
-- see `tloi_datagen/MOD_CHANGES_NEEDED.md` for the ground-truth fields this
still depends on that the mod doesn't produce yet. Not invoked automatically
by anything else in this repo; run by hand once real data exists.
"""

from __future__ import annotations

import argparse
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime
import json
import logging
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from components.ground_truth_losses import MultiHeadLoss
from components.ground_truth_net import USES_CHANNELS_LAST, GroundTruthNet
from visual_extraction.dataset import (
    IMAGE_HEIGHT,
    IMAGE_WIDTH,
    GroundTruthDataset,
    collate_label_bundles,
)
from visual_extraction.labels import LabelBundle, encode_record

logger = logging.getLogger(__name__)

_DEFAULT_EPOCHS = 30
_DEFAULT_BATCH_SIZE = 64
_DEFAULT_LR = 1e-3
_DEFAULT_SEED = 0
_RUNS_DIR = Path(__file__).parent / "runs"

StepFn = Callable[
    [torch.Tensor, LabelBundle], tuple[torch.Tensor, dict[str, torch.Tensor]]
]


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)


def get_device() -> torch.device:
    """Picks the best available device, falling back gracefully -- never assumes a GPU."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


@dataclass
class EpochResult:
    """Mean total loss and mean per-head losses for one pass over a loader."""

    total_loss: float
    per_head: dict[str, float]


def _move_bundle(bundle: LabelBundle, device: torch.device) -> LabelBundle:
    fields = {name: tensor.to(device) for name, tensor in vars(bundle).items()}
    return LabelBundle(**fields)


def to_model_memory_format(images: torch.Tensor) -> torch.Tensor:
    """Converts a batch to `channels_last` if `GroundTruthNet` uses it, eagerly.

    Called by every caller of `step_fn` (`run_epoch`, `_dummy_batch`)
    *before* `step_fn` runs, rather than branching on `USES_CHANNELS_LAST`
    inside `step_fn` itself -- `step_fn` is the function `torch.compile`
    actually wraps, so keeping it branch-free means there's no Python-level
    conditional anywhere inside the traced/compiled region, full stop,
    rather than relying on `USES_CHANNELS_LAST` being constant-folded away.
    """
    if USES_CHANNELS_LAST:
        return images.to(memory_format=torch.channels_last)
    return images


def make_step_fn(model: GroundTruthNet, criterion: MultiHeadLoss) -> StepFn:
    """Bundles model forward + loss forward into one function, `torch.compile`'d as a unit.

    Keeping this as a single function (rather than compiling the model and
    criterion separately) lets Dynamo trace across both calls as one graph
    instead of two. Callers must pass `images` already in the model's
    memory format (see `to_model_memory_format`) -- this function itself
    does no conversion, so it has zero conditionals of any kind.
    """

    def step_fn(
        images: torch.Tensor, bundle: LabelBundle
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        predictions = model(images)
        return criterion(predictions, bundle)

    return step_fn


def _dummy_record() -> dict[str, object]:
    """Minimal ground-truth dict matching `labels.encode_record`'s target contract.

    Used only to build a correctly-shaped one-sample batch to probe
    `torch.compile` with at startup -- not real data.
    """
    return {
        "stage_bucket": 0,
        "player": {"frame_position": [0.5, 0.5]},
        "player2": None,
        "item_delta": {"action": 0},
        "active_item": {"held": False, "charge": None},
        "currency": {"coins": 0, "bombs": 0, "keys": 0},
        "consumable": {
            "kind": 0,
            "card_id": None,
            "pill_color": None,
            "pill_known": None,
        },
        "stats": {
            "damage": 0.0,
            "fire_rate": 0.0,
            "speed": 0.0,
            "range": 0.0,
            "shot_speed": 0.0,
            "luck": 0.0,
        },
        "health": {
            "unknown": False,
            "hearts": 0,
            "max_hearts": 0,
            "soul_hearts": 0,
            "black_hearts": 0,
            "bone_hearts": 0,
            "rotten_hearts": 0,
            "eternal_hearts": 0,
            "golden_hearts": 0,
            "broken_hearts": 0,
        },
        "doors": [],
        "enemies": [],
    }


def _dummy_batch(device: torch.device) -> tuple[torch.Tensor, LabelBundle]:
    image = torch.zeros(3, IMAGE_HEIGHT, IMAGE_WIDTH)
    images, bundle = collate_label_bundles([(image, encode_record(_dummy_record()))])
    images = to_model_memory_format(images.to(device))
    return images, _move_bundle(bundle, device)


def compile_step_fn(
    model: GroundTruthNet, step_fn: StepFn, device: torch.device
) -> StepFn:
    """Wraps `step_fn` in `torch.compile`, probed once with a dummy batch.

    `torch.compile` only actually compiles on first call, so wrapping
    alone can't catch a broken backend -- this runs one real (dummy,
    grad-free) forward through it immediately so a compile failure surfs
    here, at startup, instead of mid-epoch. Falls back to the uncompiled
    `step_fn` on any failure, since `torch.compile` support is
    environment-dependent (backend/GPU/OS-specific) and this must not be
    a hard requirement to train at all.
    """
    was_training = model.training
    model.eval()
    try:
        compiled = torch.compile(step_fn, mode="max-autotune")
        with torch.no_grad():
            compiled(*_dummy_batch(device))
        logger.info("torch.compile: enabled")
        return compiled
    except Exception:
        logger.exception("torch.compile failed, running uncompiled")
        return step_fn
    finally:
        model.train(mode=was_training)


def run_epoch(
    model: GroundTruthNet,
    step_fn: StepFn,
    loader: DataLoader[tuple[torch.Tensor, LabelBundle]],
    device: torch.device,
    optimizer: torch.optim.Optimizer | None,
    epoch: int,
    desc: str,
) -> EpochResult:
    """Runs one epoch. Trains if `optimizer` is given, otherwise evaluates under `no_grad`."""
    model.train(mode=optimizer is not None)
    total_losses: list[float] = []
    per_head_totals: dict[str, list[float]] = {}

    pbar = tqdm(loader, desc=f"epoch {epoch:3d} {desc}", unit="batch", leave=False)
    context = torch.enable_grad() if optimizer is not None else torch.no_grad()
    with context:
        for images, bundle in pbar:
            images = to_model_memory_format(images.to(device, non_blocking=True))
            bundle = _move_bundle(bundle, device)

            total_loss, per_head = step_fn(images, bundle)

            if optimizer is not None:
                optimizer.zero_grad(set_to_none=True)
                total_loss.backward()
                optimizer.step()

            total_losses.append(total_loss.item())
            for name, value in per_head.items():
                per_head_totals.setdefault(name, []).append(value.item())
            pbar.set_postfix(loss=f"{total_loss.item():.4f}")

    return EpochResult(
        total_loss=float(np.mean(total_losses)) if total_losses else float("nan"),
        per_head={
            name: float(np.mean(values)) for name, values in per_head_totals.items()
        },
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train the ground-truth perception model."
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path(__file__).parent.parent / "datagen" / "output",
        help="datagen output dir containing images/ and labels/",
    )
    parser.add_argument("--epochs", type=int, default=_DEFAULT_EPOCHS)
    parser.add_argument("--batch-size", type=int, default=_DEFAULT_BATCH_SIZE)
    parser.add_argument("--lr", type=float, default=_DEFAULT_LR)
    parser.add_argument("--seed", type=int, default=_DEFAULT_SEED)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO)
    set_seed(args.seed)

    device = get_device()
    logger.info("device: %s", device)

    train_ds = GroundTruthDataset(args.data_dir, split="train")
    val_ds = GroundTruthDataset(args.data_dir, split="val")
    logger.info("train=%d val=%d", len(train_ds), len(val_ds))
    if len(train_ds) == 0:
        msg = (
            f"No training data found under {args.data_dir}. Run a datagen capture "
            "session first (see tloi_datagen/MOD_CHANGES_NEEDED.md for still-open "
            "ground-truth fields this model depends on)."
        )
        raise SystemExit(msg)

    train_loader: DataLoader[tuple[torch.Tensor, LabelBundle]] = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collate_label_bundles,
        drop_last=True,
    )
    val_loader: DataLoader[tuple[torch.Tensor, LabelBundle]] = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_label_bundles,
    )

    model: GroundTruthNet = (
        GroundTruthNet().to(device, memory_format=torch.channels_last)  # type: ignore
        if USES_CHANNELS_LAST
        else GroundTruthNet().to(device)
    )
    criterion = MultiHeadLoss().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info("model params: %d", n_params)

    step_fn = compile_step_fn(model, make_step_fn(model, criterion), device)

    run_dir = _RUNS_DIR / f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    run_dir.mkdir(parents=True, exist_ok=True)

    best_val_loss = float("inf")
    history: list[dict[str, object]] = []

    for epoch in range(1, args.epochs + 1):
        train_result = run_epoch(
            model, step_fn, train_loader, device, optimizer, epoch, "train"
        )
        val_result = run_epoch(model, step_fn, val_loader, device, None, epoch, "val")

        logger.info(
            "epoch %3d train_loss=%.4f val_loss=%.4f",
            epoch,
            train_result.total_loss,
            val_result.total_loss,
        )
        history.append(
            {
                "epoch": epoch,
                "train_loss": train_result.total_loss,
                "val_loss": val_result.total_loss,
                "val_per_head": val_result.per_head,
            }
        )

        if val_result.total_loss < best_val_loss:
            best_val_loss = val_result.total_loss
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "val_loss": best_val_loss,
                },
                run_dir / "best_model.pt",
            )

    (run_dir / "history.json").write_text(json.dumps(history, indent=2))
    logger.info("best val loss: %.4f -- run dir: %s", best_val_loss, run_dir)


if __name__ == "__main__":
    main()
