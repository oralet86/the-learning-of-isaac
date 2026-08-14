#!/usr/bin/env python3
"""Train `GroundTruthNet`: screenshot -> `labels.LabelBundle`.

Loop modeled on `inverse_dynamics/train.py`: mixed-precision (fp16
autocast + GradScaler), `torch.compile`d train/eval steps with graceful
fallback, ReduceLROnPlateau, early stopping, per-epoch console table,
training-curve plots, per-head torchmetrics metrics (macro F1 for
classification/binary heads, R2 for regression heads; one subplot per
head), and a JSON history/metrics dump. Data is split train/val/test
80/10/10 by `scenario_id` (`visual_extraction.dataset`); the held-out
test set is evaluated once with the best checkpoint after training.

Requires a completed `datagen.augment` run -- `visual_extraction.dataset`
reads the 480x270 augmented set under
`constants.DATAGEN_AUGMENTED_OUTPUT_DIR` (see
`tloi_datagen/MOD_CHANGES_NEEDED.md` for the ground-truth fields this still
depends on that the mod doesn't produce yet). Not invoked automatically by
anything else in this repo; run by hand once real data exists.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from collections.abc import Callable  # noqa: TC003
from contextlib import nullcontext
from datetime import datetime
import json
import os
from pathlib import Path
import sys
from typing import TYPE_CHECKING, Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.amp import GradScaler, autocast
import torch.nn as nn
from torch.profiler import ProfilerActivity
from torchmetrics.classification import (
    BinaryF1Score,
    MulticlassF1Score,
    MultilabelF1Score,
)
from torchmetrics.regression import R2Score
from tqdm import tqdm

from components.data_loading import build_loader
from components.ground_truth_losses import MultiHeadLoss
from components.ground_truth_net import USES_CHANNELS_LAST, GroundTruthNet
from components.helpers import broadcast_mask
from constants import (
    DATAGEN_AUGMENTED_OUTPUT_DIR,
    DEFAULT_MODE,
    EPOCHS,
    GT_BATCH_SIZE,
    LR,
    LR_SCHEDULE_FACTOR,
    NUM_WORKERS,
    PATIENCE,
    PLATEAU_PATIENCE,
    PROFILE_STEPS,
    PROFILE_WAIT,
    PROFILE_WARMUP,
    RUN_TIMESTAMP_FORMAT,
    SEED,
)
from visual_extraction.dataset import (
    GroundTruthDataset,
    collate_label_bundles,
    partition_label_paths,
)
from visual_extraction.labels import HEAD_SPECS, LabelBundle

torch.set_float32_matmul_precision(precision="high")
torch.backends.cudnn.benchmark = True

if TYPE_CHECKING:
    from torch import Tensor
    from torch.utils.data import DataLoader
    from torchmetrics import Metric

_RUNS_DIR = Path(__file__).parent / "runs"

# Heads whose targets are class labels (`classification`) or binary
# (`binary`): their masked accuracy is tracked per epoch. Regression heads
# have no meaningful accuracy. Keyed from `HEAD_SPECS` so net/loss/metrics
# can never disagree about which heads exist.
_HEAD_KINDS = {spec.name: spec.kind for spec in HEAD_SPECS}
_ACCURACY_HEADS = tuple(
    name for name, kind in _HEAD_KINDS.items() if kind in ("classification", "binary")
)
# Regression heads get R2 ("goodness of fit") instead of F1, with the
# same per-head display/subplot treatment.
_REGRESSION_HEADS = tuple(
    name
    for name, kind in _HEAD_KINDS.items()
    if kind in ("regression", "bounded_regression")
)

# Short display names for the console table and plot legends/titles --
# the full head names (e.g. `consumable_pill_color`) don't fit on one
# line alongside 7 F1 and 5 R2 columns.
_DISPLAY_NAMES: dict[str, str] = {
    "stage": "Stage",  # stage bucket (floor grouping)
    "active_item_held": "Held",  # active-item slot showing an item
    "active_item_charge": "Charge",  # active item charge %
    "currency": "Currency",  # coins/bombs/keys counts
    "consumable_kind": "Kind",  # pocket slot: none/card/pill
    "consumable_card_id": "CardId",  # which of the 98 cards
    "consumable_pill_color": "PillColor",  # pill color
    "stats": "Stats",  # 6 combat/movement stat values
    "health_unknown": "HealthUnk",  # Curse of the Unknown hiding hearts
    "health_counts": "HealthCounts",  # heart-type counts
    "detection_class": "Detection",  # player/enemy/none per grid cell
    "detection_offset": "Offset",  # sub-cell entity position
}


def _display_name(head: str) -> str:
    """Short label for `head`, falling back to the full name if unmapped."""
    return _DISPLAY_NAMES.get(head, head)


def _build_f1_metrics(device: torch.device) -> dict[str, Metric]:
    """One torchmetrics F1 per classification/binary head, all on `device`.

    `classification` heads get a `MulticlassF1Score` over their class
    count (the spatial detection head feeds per-cell classes via
    multidim inputs); single-output `binary` heads get a `BinaryF1Score`;
    multi-output `binary` heads (`doors_presence` 8 slots, `doors_state`
    24 (slot, state) pairs) get a `MultilabelF1Score`. Regression heads
    have no F1. All use macro averaging, like the ID loop's per-key
    metrics.
    """
    metrics: dict[str, Metric] = {}
    for spec in HEAD_SPECS:
        name = spec.name
        if spec.kind == "classification":
            num_classes = spec.pred_shape[0] if spec.spatial else spec.out_dim
            metric = MulticlassF1Score(num_classes=num_classes, average="macro")
        elif spec.kind == "binary":
            if spec.out_dim == 1:
                metric = BinaryF1Score()
            else:
                metric = MultilabelF1Score(num_labels=spec.out_dim, average="macro")
        else:
            continue
        metrics[name] = metric.to(device)
    return metrics


def _update_f1_metric(
    name: str,
    metric: Metric,
    pred: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
) -> bool:
    """Feeds one batch to head `name`'s F1 metric, applying its mask.

    Masked-out elements are filtered *before* the metric update
    (torchmetrics has no native masking): per-sample masks drop whole
    samples (`card_id`/`pill_color` only where that consumable kind is
    shown). Returns whether anything was fed -- heads whose mask is
    always empty stay NaN.
    """
    if name == "detection_class":
        # pred is already channel-second (N, 3, ROWS, COLS) -- the
        # multidim multiclass form torchmetrics expects.
        metric.update(pred.float(), target)
        return True
    if _HEAD_KINDS[name] == "classification":
        keep = mask.bool()
        if not keep.any():
            return False
        metric.update(pred.float()[keep], target[keep])
        return True
    pred = torch.sigmoid(pred.float())
    if pred.ndim == 2 and pred.shape[1] == 1:
        pred = pred[:, 0]
    keep = mask.bool() if mask.ndim == 1 else mask.bool().all(dim=1)
    if not keep.any():
        return False
    metric.update(pred[keep], target[keep].int())
    return True


def _build_regression_metrics(device: torch.device) -> dict[str, Metric]:
    """One `R2Score` per regression head, all on `device`.

    R2 (coefficient of determination) is the regression analog of F1:
    `1.0` = perfect fit, `0` = predicting the mean, negative = worse
    than the mean. `multioutput="raw_values"` keeps per-output scores so
    the caller can average only the well-defined ones (an output with
    zero target variance has an undefined R2 and would otherwise poison
    the average).
    """
    metrics: dict[str, Metric] = {}
    for spec in HEAD_SPECS:
        if spec.kind in ("regression", "bounded_regression"):
            metrics[spec.name] = R2Score(multioutput="raw_values").to(device)
    return metrics


def _update_regression_metric(
    name: str,
    metric: Metric,
    pred: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
) -> bool:
    """Feeds one batch to head `name`'s R2 metric, applying its mask.

    Masked-out elements are filtered before the update, like the F1
    metrics: per-sample masks drop whole samples (`active_item_charge`
    unless held, `health_counts` under Curse of the Unknown), and
    `detection_offset` keeps only occupied grid cells. Bounded-regression
    heads (`charge`, `offset`) predict logits, so they're sigmoid'd into
    `[0, 1]` before comparing against their targets.
    """
    if name == "detection_offset":
        pred = torch.sigmoid(pred.float()).reshape(-1, 2)
        target = target.reshape(-1, 2)
        keep = mask.reshape(-1).bool()
        if not keep.any():
            return False
        metric.update(pred[keep], target[keep])
        return True
    pred = pred.float()
    if _HEAD_KINDS[name] == "bounded_regression":
        pred = torch.sigmoid(pred)
    if pred.ndim == 2 and pred.shape[1] == 1:
        pred = pred[:, 0]
    keep = mask.bool()
    if not keep.any():
        return False
    metric.update(pred[keep], target[keep])
    return True


def set_seed(seed: int = SEED) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device() -> torch.device:
    """Returns the CUDA device, or fails loudly when no GPU is available."""
    if torch.cuda.is_available():
        return torch.device(device="cuda")
    raise RuntimeError("CUDA not available!!")


# ---------------------------------------------------------------------------
# Training / validation loops
# ---------------------------------------------------------------------------


def _move_bundle(bundle: LabelBundle, device: torch.device) -> LabelBundle:
    """Moves every target/mask tensor in a `LabelBundle` to `device`."""
    fields = {
        name: tensor.to(device, non_blocking=True)
        for name, tensor in vars(bundle).items()
    }
    return LabelBundle(**fields)


def train_epoch(
    model: nn.Module,
    loader: DataLoader[tuple[torch.Tensor, LabelBundle]],
    train_step: Callable[[torch.Tensor, LabelBundle], torch.Tensor],
    optimizer: torch.optim.Optimizer,
    scaler: GradScaler,
    device: torch.device,
    epoch: int,
    profiler: torch.profiler.profile | None = None,
    max_batches: int | None = None,
) -> dict[str, Any]:
    """Runs one training epoch over `loader`, returning mean batch loss."""
    model.train()
    running_loss: Tensor = torch.zeros(size=(), device=device)
    n_batches = 0
    pbar = tqdm(loader, desc=f"Epoch {epoch:3d} train", unit="batch", leave=False)
    for images, bundle in pbar:
        images = images.to(device, non_blocking=True)
        bundle = _move_bundle(bundle, device)
        optimizer.zero_grad(set_to_none=True)

        with autocast(device_type="cuda"):
            loss = train_step(images, bundle)

        scaler.scale(outputs=loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()

        running_loss += loss.detach()
        n_batches += 1
        if profiler is not None:
            profiler.step()
        if (max_batches is not None) and max_batches == n_batches:
            break

    if n_batches == 0:
        return {"loss": torch.tensor(float("nan"))}
    return {"loss": running_loss / n_batches}


def make_train_step(
    model: nn.Module,
    criterion: nn.Module,
    uses_channels_last: bool,
) -> Callable[[torch.Tensor, LabelBundle], torch.Tensor]:
    """Bundles model forward + loss into one `torch.compile`-able step.

    Dataset images are already float in `[0, 1]` (torchvision
    `to_tensor`), so only the fp16 cast happens here -- no `div(255)`.
    """

    def train_step(images: torch.Tensor, bundle: LabelBundle) -> torch.Tensor:
        x = images.to(dtype=torch.float16)
        if uses_channels_last:
            x = x.to(memory_format=torch.channels_last)
        predictions = model(x)
        total_loss, _per_head = criterion(predictions, bundle)
        return total_loss

    return train_step


def make_eval_step(
    model: nn.Module,
    criterion: nn.Module,
    uses_channels_last: bool,
) -> Callable[
    [torch.Tensor, LabelBundle],
    tuple[dict[str, torch.Tensor], dict[str, torch.Tensor], torch.Tensor],
]:
    """As `make_train_step`, but also returns predictions + per-head losses for metrics."""

    def eval_step_fn(
        images: torch.Tensor, bundle: LabelBundle
    ) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor], torch.Tensor]:
        x = images.to(dtype=torch.float16)
        if uses_channels_last:
            x = x.to(memory_format=torch.channels_last)
        predictions = model(x)
        total_loss, per_head = criterion(predictions, bundle)
        return predictions, per_head, total_loss

    return eval_step_fn


def _masked_correct(
    pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor, kind: str
) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-element correctness and mask, broadcast like `_masked_mean` in the losses module.

    `classification` heads compare `argmax` against the class target
    (channel dim second, so the spatial detection head works unchanged);
    `binary` heads threshold logits at 0. The mask is broadcast to the
    correctness tensor's shape via the shared `broadcast_mask` helper --
    the same masking semantics the loss applies.
    """
    if kind == "classification":
        correct = (pred.argmax(dim=1) == target).float()
    else:
        if pred.ndim == 2 and pred.shape[1] == 1:
            pred = pred[:, 0]
        correct = ((pred > 0).long() == target.long()).float()
    mask = broadcast_mask(mask, correct.ndim).expand_as(correct)
    return correct * mask, mask


@torch.no_grad()
def eval_epoch(
    eval_step: Callable[
        [torch.Tensor, LabelBundle],
        tuple[dict[str, torch.Tensor], dict[str, torch.Tensor], torch.Tensor],
    ],
    loader: DataLoader[tuple[Tensor, LabelBundle]],
    device: torch.device,
    f1_metrics: dict[str, Metric],
    r2_metrics: dict[str, Metric],
    epoch: int,
) -> dict[str, Any]:
    """Runs one validation epoch: mean loss, per-head loss, masked per-head accuracy, F1, and R2.

    `f1_metrics`/`r2_metrics` are built once in `main()` and reset here,
    so the same metric objects accumulate state for one epoch at a time
    and their device buffers are never re-allocated per epoch.
    """
    n_batches = len(loader)
    if n_batches == 0:
        return {
            "loss": float("nan"),
            "acc": float("nan"),
            "per_head_loss": {name: float("nan") for name in _HEAD_KINDS},
            "per_head_acc": {name: float("nan") for name in _ACCURACY_HEADS},
            "per_head_f1": {name: float("nan") for name in _ACCURACY_HEADS},
            "per_head_r2": {name: float("nan") for name in _REGRESSION_HEADS},
        }

    for metric in f1_metrics.values():
        metric.reset()
    for metric in r2_metrics.values():
        metric.reset()
    f1_seen = dict.fromkeys(f1_metrics, False)
    r2_seen = dict.fromkeys(r2_metrics, False)

    running_loss = 0.0
    per_head_loss_totals: dict[str, float] = defaultdict(float)
    per_head_correct: dict[str, float] = defaultdict(float)
    per_head_counts: dict[str, float] = defaultdict(float)

    pbar = tqdm(loader, desc=f"Epoch {epoch:3d} val  ", unit="batch", leave=False)
    for images, bundle in pbar:
        images = images.to(device, non_blocking=True)
        bundle = _move_bundle(bundle, device)
        with autocast(device_type="cuda"):
            predictions, per_head, loss = eval_step(images, bundle)

        running_loss += loss.item()
        for name, value in per_head.items():
            per_head_loss_totals[name] += value.item()
        for name in _ACCURACY_HEADS:
            correct, mask = _masked_correct(
                predictions[name],
                getattr(bundle, name),
                getattr(bundle, f"{name}_mask"),
                _HEAD_KINDS[name],
            )
            per_head_correct[name] += correct.sum().item()
            per_head_counts[name] += mask.sum().item()
        for name, metric in f1_metrics.items():
            if _update_f1_metric(
                name,
                metric,
                predictions[name],
                getattr(bundle, name),
                getattr(bundle, f"{name}_mask"),
            ):
                f1_seen[name] = True
        for name, metric in r2_metrics.items():
            if _update_regression_metric(
                name,
                metric,
                predictions[name],
                getattr(bundle, name),
                getattr(bundle, f"{name}_mask"),
            ):
                r2_seen[name] = True
        pbar.set_postfix(loss=f"{loss.item():.4f}")

    per_head_acc = {
        name: (
            per_head_correct[name] / per_head_counts[name]
            if per_head_counts[name] > 0
            else float("nan")
        )
        for name in _ACCURACY_HEADS
    }
    valid_accs = [acc for acc in per_head_acc.values() if acc == acc]
    overall_acc = sum(valid_accs) / len(valid_accs) if valid_accs else float("nan")

    per_head_f1 = {
        name: (float(metric.compute()) if f1_seen[name] else float("nan"))
        for name, metric in f1_metrics.items()
    }

    per_head_r2: dict[str, float] = {}
    for name, metric in r2_metrics.items():
        if not r2_seen[name]:
            per_head_r2[name] = float("nan")
            continue
        # raw_values -> one R2 per output; average only well-defined
        # (finite) outputs, since zero target variance makes R2 undefined.
        outputs = metric.compute()
        outputs = outputs[torch.isfinite(outputs)]
        per_head_r2[name] = (
            float(outputs.mean()) if outputs.numel() > 0 else float("nan")
        )

    return {
        "loss": running_loss / n_batches,
        "acc": overall_acc,
        "per_head_loss": {
            name: total / n_batches for name, total in per_head_loss_totals.items()
        },
        "per_head_acc": per_head_acc,
        "per_head_f1": per_head_f1,
        "per_head_r2": per_head_r2,
    }


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------


def plot_training_curves(history: dict[str, list[float]], out_path: Path) -> None:
    """Plots train/val loss, overall val accuracy, and per-head val accuracies/F1."""
    epochs = range(1, len(history["train_loss"]) + 1)
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    ax = axes[0, 0]
    ax.plot(epochs, history["train_loss"], label="train", linewidth=1.5)
    ax.plot(epochs, history["val_loss"], label="val", linewidth=1.5)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("Loss")
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax = axes[0, 1]
    ax.plot(epochs, history["val_acc"], label="val", linewidth=1.5)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Accuracy")
    ax.set_title("Overall val accuracy")
    ax.grid(True, alpha=0.3)

    ax = axes[1, 0]
    for key in sorted(history):
        if key.startswith("val_acc_"):
            ax.plot(
                epochs,
                history[key],
                label=_display_name(key.removeprefix("val_acc_")),
                linewidth=1,
            )
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Accuracy")
    ax.set_title("Per-head val accuracy")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    ax = axes[1, 1]
    for key in sorted(history):
        if key.startswith("val_f1_"):
            ax.plot(
                epochs,
                history[key],
                label=_display_name(key.removeprefix("val_f1_")),
                linewidth=1,
            )
    ax.set_xlabel("Epoch")
    ax.set_ylabel("F1")
    ax.set_title("Per-head val F1")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def plot_per_head_f1(history: dict[str, list[float]], out_path: Path) -> Path | None:
    """Plots one subplot per head's val F1 curve into a single figure at `out_path`.

    Heads whose F1 is always NaN (masked-out every epoch, e.g. today's
    `consumable_pill_known`) get no subplot. Returns `out_path`, or `None`
    if no head has any F1 history.
    """
    keys = [
        key
        for key in sorted(history)
        if key.startswith("val_f1_") and any(v == v for v in history[key])
    ]
    if not keys:
        return None

    n_heads = len(keys)
    n_cols = 4
    n_rows = (n_heads + n_cols - 1) // n_cols
    epochs = range(1, len(history[keys[0]]) + 1)
    fig, axes = plt.subplots(
        n_rows, n_cols, figsize=(4.5 * n_cols, 3.5 * n_rows), squeeze=False
    )
    for ax, key in zip(axes.flat[:n_heads], keys, strict=True):
        ax.plot(epochs, history[key], linewidth=1.5)
        ax.set_title(_display_name(key.removeprefix("val_f1_")))
        ax.set_xlabel("Epoch")
        ax.set_ylabel("F1")
        ax.grid(True, alpha=0.3)
    for ax in axes.flat[n_heads:]:
        ax.set_visible(False)

    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    return out_path


def plot_per_head_r2(history: dict[str, list[float]], out_path: Path) -> Path | None:
    """Plots one subplot per regression head's val R2 curve into `out_path`.

    Heads whose R2 is always NaN (never any valid, finite output) get no
    subplot. Returns `out_path`, or `None` if no head has any R2 history.
    """
    keys = [
        key
        for key in sorted(history)
        if key.startswith("val_r2_") and any(v == v for v in history[key])
    ]
    if not keys:
        return None

    n_heads = len(keys)
    n_cols = 4
    n_rows = (n_heads + n_cols - 1) // n_cols
    epochs = range(1, len(history[keys[0]]) + 1)
    fig, axes = plt.subplots(
        n_rows, n_cols, figsize=(4.5 * n_cols, 3.5 * n_rows), squeeze=False
    )
    for ax, key in zip(axes.flat[:n_heads], keys, strict=True):
        ax.plot(epochs, history[key], linewidth=1.5)
        ax.set_title(_display_name(key.removeprefix("val_r2_")))
        ax.set_xlabel("Epoch")
        ax.set_ylabel("R2")
        ax.grid(True, alpha=0.3)
    for ax in axes.flat[n_heads:]:
        ax.set_visible(False)

    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    return out_path


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description="Train ground-truth perception model")
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument(
        "--mode",
        type=str,
        default=DEFAULT_MODE,
        choices=["default", "profile"],
        help=f"Run mode: default or profile (default: {DEFAULT_MODE})",
    )
    args = parser.parse_args()

    out_dir = _RUNS_DIR / f"run_{datetime.now().strftime(RUN_TIMESTAMP_FORMAT)}"

    set_seed(args.seed)

    device = get_device()
    print(f"Device: {device}")

    partitioned = partition_label_paths(DATAGEN_AUGMENTED_OUTPUT_DIR / "labels")
    train_ds = GroundTruthDataset(
        DATAGEN_AUGMENTED_OUTPUT_DIR / "images", partitioned["train"]
    )
    val_ds = GroundTruthDataset(
        DATAGEN_AUGMENTED_OUTPUT_DIR / "images", partitioned["val"]
    )
    test_ds = GroundTruthDataset(
        DATAGEN_AUGMENTED_OUTPUT_DIR / "images", partitioned["test"]
    )
    print(f"Split: train={len(train_ds):,}  val={len(val_ds):,}  test={len(test_ds):,}")
    if len(train_ds) == 0:
        sys.exit(
            f"No training data found under {DATAGEN_AUGMENTED_OUTPUT_DIR}. "
            "Run `datagen.augment` first (see tloi_datagen/MOD_CHANGES_NEEDED.md "
            "for still-open ground-truth fields this model depends on)."
        )

    model: GroundTruthNet = (
        GroundTruthNet().to(device, memory_format=torch.channels_last)  # type: ignore
        if USES_CHANNELS_LAST
        else GroundTruthNet().to(device)
    )
    criterion = MultiHeadLoss().to(device)
    print(f"Model: GroundTruthNet (heads: {', '.join(_HEAD_KINDS)})")

    # Built once and reset per epoch inside eval_epoch -- rebuilding them
    # (plus a .to(device) transfer per metric) every epoch is pure waste.
    f1_metrics = _build_f1_metrics(device)
    r2_metrics = _build_regression_metrics(device)

    train_loader: DataLoader[tuple[Tensor, LabelBundle]] = build_loader(
        train_ds,
        batch_size=GT_BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        drop_last=True,
        collate_fn=collate_label_bundles,
    )
    val_loader: DataLoader[tuple[Tensor, LabelBundle]] = build_loader(
        val_ds,
        batch_size=GT_BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        drop_last=False,
        collate_fn=collate_label_bundles,
    )
    test_loader: DataLoader[tuple[Tensor, LabelBundle]] = build_loader(
        test_ds,
        batch_size=GT_BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        drop_last=False,
        collate_fn=collate_label_bundles,
    )

    train_step = make_train_step(model, criterion, USES_CHANNELS_LAST)
    try:
        train_step = torch.compile(train_step, mode="max-autotune")
        print("  torch.compile: enabled")
    except Exception as e:
        print(f"  torch.compile: failed ({e}), running uncompiled")
    eval_step = make_eval_step(model, criterion, USES_CHANNELS_LAST)
    try:
        eval_step = torch.compile(eval_step, dynamic=False, mode="max-autotune")
        print("  eval torch.compile: enabled")
    except Exception as e:
        print(f"  eval torch.compile: failed ({e}), running uncompiled")
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model params: {n_params:,}")

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LR,
        fused=True,
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=PLATEAU_PATIENCE, factor=LR_SCHEDULE_FACTOR
    )
    scaler = GradScaler(device="cuda")

    out_dir.mkdir(parents=True, exist_ok=True)

    if args.mode == "profile":
        os.environ["TORCH_LOGS_OUT"] = str(out_dir / "compile_log.txt")
        torch._logging.set_logs(cudagraphs=True, recompiles=True)
        prof = torch.profiler.profile(
            activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
            schedule=torch.profiler.schedule(
                wait=PROFILE_WAIT,
                warmup=PROFILE_WARMUP,
                active=PROFILE_STEPS,
                repeat=1,
            ),
            record_shapes=True,
            profile_memory=True,
        )
        prof_ctx: Any = prof
        print(
            f"Profiling enabled: wait={PROFILE_WAIT} warmup={PROFILE_WARMUP} "
            f"active={PROFILE_STEPS}"
        )
    else:
        prof = None
        prof_ctx = nullcontext()

    history: dict[str, list[float]] = defaultdict(list)
    best_val_loss = float("inf")
    best_epoch = 0
    patience_counter = 0

    print(f"\nTraining for up to {args.epochs} epochs (patience={PATIENCE})...")

    f1_headers = [_display_name(name) for name in _ACCURACY_HEADS]
    r2_headers = [f"r2/{_display_name(name)}" for name in _REGRESSION_HEADS]
    n_f1 = len(f1_headers)
    widths = [
        5,
        10,
        10,
        8,
        *[max(8, len(header)) for header in f1_headers],
        *[max(8, len(header)) for header in r2_headers],
        10,
    ]
    columns = [
        "Epoch",
        "TrainLoss",
        "ValLoss",
        "ValAcc",
        *f1_headers,
        *r2_headers,
        "LR",
    ]
    f1_widths = widths[4 : 4 + n_f1]
    r2_widths = widths[4 + n_f1 : -1]
    header_line = " ".join(
        f"{col:>{w}}" for col, w in zip(columns, widths, strict=False)
    )
    print(f"\n{header_line}")
    print("-" * len(header_line))

    for epoch in range(1, args.epochs + 1):
        with prof_ctx:
            train_metrics: dict[str, Any] = train_epoch(
                model,
                train_loader,
                train_step,
                optimizer,
                scaler,
                device,
                epoch,
                profiler=prof,
                max_batches=None
                if (prof is None)
                else (PROFILE_WAIT + PROFILE_WARMUP + PROFILE_STEPS),
            )
        if prof is not None:
            prof.stop()
            trace_path: Path = out_dir / "train_trace.json"
            table_path: Path = out_dir / "averages_table.txt"
            prof.export_chrome_trace(path=str(trace_path))
            with open(file=table_path, mode="w+") as f:
                f.write(
                    prof.key_averages().table(
                        sort_by="cuda_time_total",
                        max_name_column_width=200,
                        max_shapes_column_width=200,
                    )
                )

            print(f"Profiler trace saved to {trace_path}")
            prof_ctx = nullcontext()
            prof = None
            return
        model.eval()
        val_metrics = eval_epoch(
            eval_step, val_loader, device, f1_metrics, r2_metrics, epoch
        )
        scheduler.step(val_metrics["loss"])

        history["train_loss"].append(train_metrics["loss"].item())
        history["val_loss"].append(val_metrics["loss"])
        history["val_acc"].append(val_metrics["acc"])
        for name, acc in val_metrics["per_head_acc"].items():
            history[f"val_acc_{name}"].append(acc)
        for name, f1 in val_metrics["per_head_f1"].items():
            history[f"val_f1_{name}"].append(f1)
        for name, r2 in val_metrics["per_head_r2"].items():
            history[f"val_r2_{name}"].append(r2)
        for name, loss in val_metrics["per_head_loss"].items():
            history[f"val_loss_{name}"].append(loss)

        cells = [
            f"{epoch:>5d}",
            f"{train_metrics['loss']:>10.4f}",
            f"{val_metrics['loss']:>10.4f}",
            f"{val_metrics['acc']:>8.4f}",
            *[
                f"{val_metrics['per_head_f1'][name]:>{w}.4f}"
                for name, w in zip(_ACCURACY_HEADS, f1_widths, strict=True)
            ],
            *[
                f"{val_metrics['per_head_r2'][name]:>{w}.4f}"
                for name, w in zip(_REGRESSION_HEADS, r2_widths, strict=True)
            ],
            f"{scheduler.get_last_lr()[0]:>10.2e}",
        ]
        print(" ".join(cells))

        if val_metrics["loss"] < best_val_loss:
            best_val_loss = val_metrics["loss"]
            best_epoch = epoch
            patience_counter = 0
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_metrics": val_metrics,
                    "head_names": [spec.name for spec in HEAD_SPECS],
                },
                out_dir / "best_model.pt",
            )
        else:
            patience_counter += 1

        if patience_counter >= PATIENCE:
            print(
                f"\nEarly stopping at epoch {epoch} (no improvement for {PATIENCE} epochs)"
            )
            break

    print(f"\nBest validation loss: {best_val_loss:.4f} at epoch {best_epoch}")

    # Test evaluation
    print("\nLoading best model for test evaluation...")
    ckpt = torch.load(
        out_dir / "best_model.pt", map_location=device, weights_only=False
    )
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    test_metrics = eval_epoch(eval_step, test_loader, device, f1_metrics, r2_metrics, 0)
    print(f"Test loss: {test_metrics['loss']:.4f}")
    print(f"Test acc: {test_metrics['acc']:.4f}")
    f1_strs = [
        f"{_display_name(name)}={test_metrics['per_head_f1'][name]:.3f}"
        for name in _ACCURACY_HEADS
    ]
    print(f"Per-head test F1: {', '.join(f1_strs)}")
    r2_strs = [
        f"{_display_name(name)}={test_metrics['per_head_r2'][name]:.3f}"
        for name in _REGRESSION_HEADS
    ]
    print(f"Per-head test R2: {', '.join(r2_strs)}")

    plot_training_curves(history, out_dir / "training_curves.png")
    print(f"Training curves saved to {out_dir / 'training_curves.png'}")

    f1_plot_path = plot_per_head_f1(history, out_dir / "per_head_f1.png")
    if f1_plot_path is not None:
        print(f"Per-head F1 saved to {f1_plot_path}")

    r2_plot_path = plot_per_head_r2(history, out_dir / "per_head_r2.png")
    if r2_plot_path is not None:
        print(f"Per-head R2 saved to {r2_plot_path}")

    metrics_json = {
        "best_epoch": best_epoch,
        "best_val_loss": best_val_loss,
        "test_metrics": {
            "loss": test_metrics["loss"],
            "acc": test_metrics["acc"],
            "per_head_loss": test_metrics["per_head_loss"],
            "per_head_acc": test_metrics["per_head_acc"],
            "per_head_f1": test_metrics["per_head_f1"],
            "per_head_r2": test_metrics["per_head_r2"],
        },
        "config": {
            "model": "GroundTruthNet",
            "batch_size": GT_BATCH_SIZE,
            "lr": LR,
            "epochs": len(history["train_loss"]),
            "patience": PATIENCE,
            "plateau_patience": PLATEAU_PATIENCE,
            "model_params": n_params,
            "train_frames": len(train_ds),
            "val_frames": len(val_ds),
            "test_frames": len(test_ds),
        },
        "history": dict(history),
    }
    with open(out_dir / "metrics.json", "w") as f:
        json.dump(metrics_json, f, indent=2)
    print(f"Metrics saved to {out_dir / 'metrics.json'}")


if __name__ == "__main__":
    main()
