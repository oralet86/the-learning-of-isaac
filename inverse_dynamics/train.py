#!/usr/bin/env python3
"""Train inverse dynamics model: (frame_t, frame_{t+1}, frame_{t+2}, frame_{t+3}) -> keys_{t+1}."""

import argparse
from collections import defaultdict
from contextlib import nullcontext
from datetime import datetime
from functools import partial
import json
import math
from pathlib import Path  # noqa: TC003
import sys
from typing import TYPE_CHECKING, Any

import matplotlib
import numpy as np
import torch
from torch.amp import GradScaler, autocast
import torch.nn as nn
from torch.profiler import ProfilerActivity
from tqdm import tqdm

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

from components import InverseDynamicsResNet, focal_loss_with_logits
from components.inverse_dynamics_resnet import (
    ADAMW_WEIGHT_DECAY,
    BATCH_SIZE,
    DROPOUT,
    VAL_BATCH_MULTIPLIER,
)
from constants import (
    ADAMW_FUSED,
    CHECKPOINT_FREQ,
    COSINE_T_MAX_DIVISOR,
    EPOCHS,
    EPSILON,
    INVERSE_DYNAMIC_RUNS_DIR,
    LR,
    MIN_LR,
    NUM_WORKERS,
    PATIENCE,
    PLATEAU_EPOCHS,
    PROCESSED_DATA_DIR,
    PROFILE_STEPS,
    PROFILE_WAIT,
    PROFILE_WARMUP,
    RUN_TIMESTAMP_FORMAT,
    SEED,
    THRESHOLD,
    THRESHOLD_CANDIDATES,
    TOP_K,
    WORST_PRED_BATCH_SIZE,
)
from inverse_dynamics.dataloader import ChunkedDataset, build_loader

torch.set_float32_matmul_precision(precision="high")
torch.backends.cudnn.benchmark = True

if TYPE_CHECKING:
    from torch import Tensor
    from torch.utils.data import Dataset


def set_seed(seed: int = SEED) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device(device="cuda")
    raise BaseException("CUDA not available!!")


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


@torch.no_grad()
def compute_metrics(
    logits: torch.Tensor,
    targets: torch.Tensor,
    key_names: list[str],
    thresholds: dict[str, float] | None = None,
) -> dict[str, Any]:
    probs = torch.sigmoid(logits)
    if thresholds is None:
        preds = (probs > THRESHOLD).float()
    else:
        threshold_tensor = torch.tensor(
            [thresholds[k] for k in key_names], device=probs.device, dtype=probs.dtype
        )
        preds = (probs > threshold_tensor.unsqueeze(0)).float()

    correct = (preds == targets).float()
    per_key_acc = correct.mean(dim=0).cpu().numpy()

    tp = (preds * targets).sum(dim=0).float()
    fp = (preds * (1 - targets)).sum(dim=0).float()
    fn = ((1 - preds) * targets).sum(dim=0).float()

    eps = EPSILON
    per_key_precision = tp / (tp + fp + eps)
    per_key_recall = tp / (tp + fn + eps)
    per_key_f1 = (
        2
        * per_key_precision
        * per_key_recall
        / (per_key_precision + per_key_recall + eps)
    )

    macro_f1 = per_key_f1.mean().item()
    micro_tp = tp.sum()
    micro_fp = fp.sum()
    micro_fn = fn.sum()
    micro_precision = micro_tp / (micro_tp + micro_fp + eps)
    micro_recall = micro_tp / (micro_tp + micro_fn + eps)
    micro_f1 = (
        2 * micro_precision * micro_recall / (micro_precision + micro_recall + eps)
    )

    overall_acc = correct.mean().item()

    all_zeros_mask = targets.sum(dim=1) == 0
    n_zeros = all_zeros_mask.sum().item()
    no_action_acc = correct[all_zeros_mask].mean().item() if n_zeros > 0 else 0.0
    n_no_action = n_zeros

    all_ones_pred = preds.sum(dim=1) == targets.shape[1]
    n_all_active_pred = all_ones_pred.sum().item()

    return {
        "loss": 0.0,
        "overall_acc": overall_acc,
        "macro_f1": macro_f1,
        "micro_f1": micro_f1.item(),
        "per_key_acc": {
            k: float(v) for k, v in zip(key_names, per_key_acc, strict=True)
        },
        "per_key_f1": {
            k: float(v)
            for k, v in zip(key_names, per_key_f1.cpu().numpy(), strict=True)
        },
        "no_action_acc": no_action_acc,
        "no_action_count": int(n_no_action),
        "all_active_pred_count": int(n_all_active_pred),
    }


def aggregate_metrics(metrics_list: list[dict[str, Any]]) -> dict[str, Any]:
    result = {}
    if metrics_list and "per_key_acc" in metrics_list[0]:
        keys = list(metrics_list[0]["per_key_acc"].keys())
        for k in keys:
            result[f"acc_{k}"] = float(
                np.mean([m["per_key_acc"][k] for m in metrics_list])
            )
            result[f"f1_{k}"] = float(
                np.mean([m["per_key_f1"][k] for m in metrics_list])
            )
    result["overall_acc"] = float(np.mean([m["overall_acc"] for m in metrics_list]))
    result["macro_f1"] = float(np.mean([m.get("macro_f1", 0.0) for m in metrics_list]))
    result["micro_f1"] = float(np.mean([m.get("micro_f1", 0.0) for m in metrics_list]))
    na = [m["no_action_acc"] for m in metrics_list if m.get("no_action_count", 0) > 0]
    result["no_action_acc"] = float(np.mean(na)) if na else 0.0
    result["loss"] = float(np.mean([m["loss"] for m in metrics_list]))
    return result


# ---------------------------------------------------------------------------
# Training / validation loops
# ---------------------------------------------------------------------------


@torch.no_grad()
def find_optimal_thresholds(
    model: nn.Module,
    val_loader: DataLoader[tuple[torch.Tensor, torch.Tensor]],
    device: torch.device,
    key_names: list[str],
) -> tuple[dict[str, float], dict[str, float]]:
    model.eval()
    all_logits: list[torch.Tensor] = []
    all_targets: list[torch.Tensor] = []
    for x, y in val_loader:
        x: Tensor = (
            x.to(device, non_blocking=True)
            .to(dtype=torch.float16)
            .div_(255.0)
            .to(memory_format=torch.channels_last)
        )
        with autocast(device_type="cuda"):
            logits: Tensor = model(x)

        all_logits.append(logits.cpu())
        all_targets.append(y)
    logits_cat: Tensor = torch.cat(all_logits)
    targets_cat: Tensor = torch.cat(all_targets)
    probs: Tensor = torch.sigmoid(logits_cat)

    best_thresholds: dict[str, float] = {}
    best_f1s: dict[str, float] = {}

    thresholds_candidates = np.linspace(0.0, 1.0, THRESHOLD_CANDIDATES)[1:-1]

    for k, key_name in enumerate(iterable=key_names):
        p = probs[:, k].numpy()
        t = targets_cat[:, k].numpy()

        best_t = THRESHOLD
        best_f1 = 0.0
        for threshold in thresholds_candidates:
            preds = (p >= threshold).astype(np.float32)
            tp = float((preds * t).sum())
            fp = float((preds * (1.0 - t)).sum())
            fn = float(((1.0 - preds) * t).sum())
            precision = tp / (tp + fp + 1e-8)
            recall = tp / (tp + fn + 1e-8)
            f1 = 2.0 * precision * recall / (precision + recall + 1e-8)
            if f1 > best_f1:
                best_f1 = f1
                best_t = threshold

        best_thresholds[key_name] = float(best_t)
        best_f1s[key_name] = float(best_f1)

    return best_thresholds, best_f1s


def train_epoch(
    model: nn.Module,
    loader: DataLoader[tuple[torch.Tensor, torch.Tensor]],
    train_step: Any,
    optimizer: torch.optim.Optimizer,
    scaler: GradScaler,
    device: torch.device,
    epoch: int,
    profiler: Any = None,
) -> dict[str, Any]:
    model.train()
    running_loss: Tensor = torch.zeros(size=(), device=device)
    n_batches = 0
    pbar = tqdm(loader, desc=f"Epoch {epoch:3d} train", unit="batch", leave=False)
    for x, y in pbar:
        x: Tensor = x.to(device, non_blocking=True)
        y: Tensor = y.to(device, non_blocking=True, dtype=torch.float32)
        optimizer.zero_grad(set_to_none=True)

        with autocast(device_type="cuda"):
            loss: Tensor = train_step(x, y)

        scaler.scale(outputs=loss).backward()
        scaler.step(optimizer)
        scaler.update()

        running_loss += loss.detach()
        n_batches += 1
        if n_batches % 200 == 0:
            pbar.set_postfix(loss=f"{loss.detach():.4f}")
        if profiler is not None:
            profiler.step()

    return {"loss": running_loss / n_batches}


def make_train_step(model: nn.Module, criterion):
    def train_step(x_raw, y):
        x: Tensor = (
            x_raw.to(dtype=torch.float16)
            .div(255.0)
            .to(memory_format=torch.channels_last)
        )
        logits: Tensor = model(x)
        loss: Tensor = criterion(logits, y)
        return loss

    return train_step


@torch.no_grad()
def eval_epoch(
    model: nn.Module,
    loader: DataLoader[tuple[torch.Tensor, torch.Tensor]],
    criterion: Any,
    device: torch.device,
    key_names: list[str],
    epoch: int,
    thresholds: dict[str, float] | None = None,
) -> dict[str, Any]:
    model.eval()
    metrics_list = []
    running_loss = 0.0
    pbar = tqdm(loader, desc=f"Epoch {epoch:3d} val  ", unit="batch", leave=False)
    for x, y in pbar:
        x: Tensor = (
            x.to(device, non_blocking=True)
            .to(dtype=torch.float16)
            .div_(255.0)
            .to(memory_format=torch.channels_last)
        )
        y: Tensor = y.to(device, non_blocking=True, dtype=torch.float32)
        with autocast(device_type="cuda"):
            logits: Tensor = model(x)
            loss: Tensor = criterion(logits, y)

        running_loss += loss.item()
        m = compute_metrics(logits, y, key_names, thresholds=thresholds)
        m["loss"] = loss.item()
        metrics_list.append(m)
        pbar.set_postfix(loss=f"{loss.item():.4f}")
    agg = aggregate_metrics(metrics_list)
    agg["loss"] = running_loss / len(loader)
    return agg


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------


def plot_training_curves(
    history: dict[str, list[float]], out_path: Path, _key_names: list[str]
) -> None:
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
    ax.plot(epochs, history["train_macro_f1"], label="train", linewidth=1.5)
    ax.plot(epochs, history["val_macro_f1"], label="val", linewidth=1.5)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Macro F1")
    ax.set_title("Macro F1")
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax = axes[1, 0]
    ax.plot(epochs, history["val_overall_acc"], label="val", linewidth=1.5)
    ax.plot(
        epochs,
        history["train_no_action_acc"],
        label="train (no-action)",
        linewidth=1,
        linestyle="--",
    )
    ax.plot(
        epochs,
        history["val_no_action_acc"],
        label="val (no-action)",
        linewidth=1,
        linestyle="--",
    )
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Accuracy")
    ax.set_title("Accuracy")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    ax = axes[1, 1]
    ax.plot(epochs, history["train_micro_f1"], label="train", linewidth=1.5)
    ax.plot(epochs, history["val_micro_f1"], label="val", linewidth=1.5)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Micro F1")
    ax.set_title("Micro F1")
    ax.legend()
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_per_key_metrics(
    best_metrics: dict[str, float], out_path: Path, key_names: list[str]
) -> None:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    keys = key_names
    accs = [best_metrics.get(f"acc_{k}", 0) for k in keys]
    f1s = [best_metrics.get(f"f1_{k}", 0) for k in keys]

    y_pos = range(len(keys))
    ax1.barh(y_pos, accs, color="steelblue", edgecolor="white")
    ax1.set_yticks(y_pos)
    ax1.set_yticklabels(keys)
    ax1.invert_yaxis()
    ax1.set_xlabel("Accuracy")
    ax1.set_title("Per-Key Accuracy")
    for i, v in enumerate(accs):
        ax1.text(v + 0.01, i, f"{v:.3f}", va="center", fontsize=8)

    ax2.barh(y_pos, f1s, color="darkorange", edgecolor="white")
    ax2.set_yticks(y_pos)
    ax2.set_yticklabels(keys)
    ax2.invert_yaxis()
    ax2.set_xlabel("F1 Score")
    ax2.set_title("Per-Key F1 Score")
    for i, v in enumerate(f1s):
        ax2.text(v + 0.01, i, f"{v:.3f}", va="center", fontsize=8)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


@torch.no_grad()
def find_worst_predictions_per_key(
    test_dataset: Dataset[tuple[torch.Tensor, torch.Tensor]],
    model: nn.Module,
    device: torch.device,
    key_names: list[str],
    top_k: int = 20,
) -> dict[str, list[dict[str, object]]]:
    model.eval()
    num_keys = len(key_names)

    per_key_items: list[list[tuple[float, np.ndarray, float, float]]] = [
        [] for _ in range(num_keys)
    ]

    loader: DataLoader[tuple[Tensor, Tensor]] = DataLoader(
        test_dataset,
        batch_size=WORST_PRED_BATCH_SIZE,
        shuffle=False,
        num_workers=0,
    )

    pbar = tqdm(loader, desc="Worst preds", unit="batch", leave=False)
    for x_batch, y_batch in pbar:
        x_batch_cpu = x_batch.clone()
        x_batch = (
            x_batch.to(device, non_blocking=True)
            .to(dtype=torch.float16)
            .div_(255.0)
            .to(memory_format=torch.channels_last)
        )
        with autocast("cuda"):
            logits = model(x_batch)
            probs = torch.sigmoid(logits)

        eps = EPSILON
        y_np = y_batch.numpy()
        p_np = np.clip(probs.cpu().numpy(), eps, 1.0 - eps)
        bce = -(y_np * np.log(p_np + eps) + (1 - y_np) * np.log(1 - p_np + eps))

        batch_size = y_np.shape[0]
        for i in range(batch_size):
            frames_i = x_batch_cpu[i].numpy()
            for k in range(num_keys):
                per_key_items[k].append(
                    (
                        float(bce[i, k]),
                        frames_i,
                        float(y_np[i, k]),
                        float(p_np[i, k]),
                    )
                )

    result: dict[str, list[dict[str, object]]] = {}
    for k, key_name in enumerate(key_names):
        items = per_key_items[k]
        items.sort(key=lambda t: t[0], reverse=True)
        top = items[:top_k]
        result[key_name] = [
            {
                "loss": float(loss),
                "frames": frames.tolist(),
                "true": int(true_label),
                "prob": float(prob),
            }
            for loss, frames, true_label, prob in top
        ]

    return result


def plot_worst_predictions_for_key(
    items: list[dict[str, Any]],
    key_name: str,
    out_path: Path,
) -> None:
    n_items = len(items)
    if n_items == 0:
        return

    n_cols = 5
    n_rows = min(4, int(np.ceil(n_items / n_cols)))

    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(n_cols * 4.5, n_rows * 2.8),
    )
    if n_rows == 1:
        axes = axes.reshape(1, -1)

    for idx, item in enumerate(items):
        row, col = divmod(idx, n_cols)
        if row >= n_rows:
            break
        ax = axes[row, col]

        frames = item["frames"]
        true_label = item["true"]
        prob = item["prob"]
        loss = item["loss"]

        top = np.concatenate([frames[0], frames[1]], axis=1)
        bottom = np.concatenate([frames[2], frames[3]], axis=1)
        pair_img = np.concatenate([top, bottom], axis=0)

        ax.imshow(pair_img, cmap="gray", aspect="auto")
        ax.axis("off")

        predicted = int(prob > 0.5)
        wrong = predicted != true_label
        title_color = "red" if wrong else "black"

        ax.set_title(
            f"#{idx + 1}\ntrue={true_label}  pred={prob:.3f}  loss={loss:.4f}",
            fontsize=7,
            color=title_color,
        )

    for idx in range(n_items, n_rows * n_cols):
        row, col = divmod(idx, n_cols)
        axes[row, col].axis("off")

    fig.suptitle(
        f"Worst predictions for key: {key_name}", fontsize=12, fontweight="bold"
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description="Train inverse dynamics model")
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument(
        "--profile",
        action="store_true",
        help="Profile the first epoch of training and dump a Chrome trace.",
    )
    parser.add_argument(
        "--profile-detailed",
        action="store_true",
        help="Like --profile, but also record memory and input shapes.",
    )
    args = parser.parse_args()

    cosine_t_max = max(args.epochs // COSINE_T_MAX_DIVISOR, 1)
    min_lr_ratio = MIN_LR / LR

    out_dir = (
        INVERSE_DYNAMIC_RUNS_DIR
        / f"run_{datetime.now().strftime(RUN_TIMESTAMP_FORMAT)}"
    )

    set_seed(args.seed)

    info_path = PROCESSED_DATA_DIR / "dataset_info.json"
    if not info_path.exists():
        sys.exit(
            f"{info_path} not found. Run preprocess.py first to prepare the dataset."
        )
    with open(info_path) as f:
        ds_info = json.load(f)
    key_names: list[str] = ds_info["key_names"]
    num_keys = len(key_names)
    pos_weight = torch.as_tensor(ds_info["pos_weight"], dtype=torch.float32)

    train_ds = ChunkedDataset(chunk_dir=PROCESSED_DATA_DIR / "train")
    val_ds = ChunkedDataset(chunk_dir=PROCESSED_DATA_DIR / "val")
    test_ds = ChunkedDataset(chunk_dir=PROCESSED_DATA_DIR / "test")
    print(f"Split: train={len(train_ds):,}  val={len(val_ds):,}  test={len(test_ds):,}")

    device = get_device()
    print(f"Device: {device}")

    train_loader: DataLoader[tuple[Tensor, Tensor]] = build_loader(
        train_ds, BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS
    )
    val_loader: DataLoader[tuple[Tensor, Tensor]] = build_loader(
        val_ds,
        BATCH_SIZE * VAL_BATCH_MULTIPLIER,
        shuffle=False,
        num_workers=NUM_WORKERS,
    )
    test_loader: DataLoader[tuple[Tensor, Tensor]] = build_loader(
        test_ds,
        BATCH_SIZE * VAL_BATCH_MULTIPLIER,
        shuffle=False,
        num_workers=NUM_WORKERS,
    )

    criterion: partial[Tensor] = partial(
        focal_loss_with_logits, pos_weight=pos_weight.to(device)
    )
    model = InverseDynamicsResNet(num_keys=num_keys, dropout=DROPOUT)
    model = model.to(device, memory_format=torch.channels_last)  # type: ignore
    train_step = make_train_step(model, criterion)
    try:
        train_step = torch.compile(train_step, mode="reduce-overhead")
        print("  torch.compile: enabled")
    except Exception as e:
        print(f"  torch.compile: failed ({e}), running uncompiled")
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model params: {n_params:,}")

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LR,
        weight_decay=ADAMW_WEIGHT_DECAY,
        fused=ADAMW_FUSED,
    )

    def lr_lambda(epoch: int) -> float:
        if epoch < PLATEAU_EPOCHS:
            return 1.0
        if epoch < PLATEAU_EPOCHS + cosine_t_max:
            progress: int | float = (epoch - PLATEAU_EPOCHS) / max(cosine_t_max, 1)
            return min_lr_ratio + (1.0 - min_lr_ratio) * (
                0.5 * (1.0 + math.cos(math.pi * progress))
            )
        return min_lr_ratio

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    scaler = GradScaler(device="cuda")

    out_dir.mkdir(parents=True, exist_ok=True)

    if args.profile or args.profile_detailed:
        detailed = args.profile_detailed
        prof = torch.profiler.profile(
            activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
            schedule=torch.profiler.schedule(
                wait=PROFILE_WAIT,
                warmup=PROFILE_WARMUP,
                active=PROFILE_STEPS,
                repeat=1,
            ),
            record_shapes=detailed,
            profile_memory=detailed,
        )
        prof_ctx: Any = prof
        print(
            f"Profiling enabled: wait={PROFILE_WAIT} warmup={PROFILE_WARMUP} "
            f"active={PROFILE_STEPS} detailed={detailed}"
        )
    else:
        prof = None
        prof_ctx = nullcontext()

    history: dict[str, list[float]] = defaultdict(list)
    best_val_f1 = -1.0
    best_epoch = 0
    patience_counter = 0

    print(f"\nTraining for up to {args.epochs} epochs (patience={PATIENCE})...")

    print(
        f"\n{'Epoch':>5s} {'TrainLoss':>10s} {'ValLoss':>10s} {'ValF1':>8s} {'ValAcc':>8s} {'NoActAcc':>8s} {'LR':>10s}"
    )
    print("-" * 65)

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
            )
        if prof is not None:
            prof.stop()
            trace_path: Path = out_dir / "train_trace.json"
            table_path: Path = out_dir / "averages_table.txt"
            prof.export_chrome_trace(path=str(object=trace_path))
            with open(file=table_path, mode="w+") as f:
                f.write(prof.key_averages().table(sort_by="cuda_time_total"))

            print(f"Profiler trace saved to {trace_path}")
            prof_ctx = nullcontext()
            prof = None
            return
        val_metrics = eval_epoch(model, val_loader, criterion, device, key_names, epoch)
        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]

        history["train_loss"].append(train_metrics["loss"])
        history["train_macro_f1"].append(val_metrics["macro_f1"])  # val as proxy
        history["train_micro_f1"].append(val_metrics["micro_f1"])
        history["train_no_action_acc"].append(val_metrics["no_action_acc"])
        history["val_loss"].append(val_metrics["loss"])
        history["val_macro_f1"].append(val_metrics["macro_f1"])
        history["val_micro_f1"].append(val_metrics["micro_f1"])
        history["val_overall_acc"].append(val_metrics["overall_acc"])
        history["val_no_action_acc"].append(val_metrics["no_action_acc"])

        print(
            f"{epoch:5d} {train_metrics['loss']:10.4f} {val_metrics['loss']:10.4f} "
            f"{val_metrics['macro_f1']:8.4f} {val_metrics['overall_acc']:8.4f} "
            f"{val_metrics['no_action_acc']:8.4f} {current_lr:10.2e}"
        )

        if val_metrics["macro_f1"] > best_val_f1:
            best_val_f1 = val_metrics["macro_f1"]
            best_epoch = epoch
            patience_counter = 0
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_metrics": val_metrics,
                    "key_names": key_names,
                },
                out_dir / "best_model.pt",
            )
        else:
            patience_counter += 1

        if epoch % CHECKPOINT_FREQ == 0:
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "key_names": key_names,
                },
                out_dir / f"checkpoint_epoch{epoch:03d}.pt",
            )

        if patience_counter >= PATIENCE:
            print(
                f"\nEarly stopping at epoch {epoch} (no improvement for {PATIENCE} epochs)"
            )
            break

    print(f"\nBest validation macro F1: {best_val_f1:.4f} at epoch {best_epoch}")

    # Test evaluation
    print("\nLoading best model for test evaluation...")
    ckpt = torch.load(
        out_dir / "best_model.pt", map_location=device, weights_only=False
    )
    model.load_state_dict(ckpt["model_state_dict"])
    print("\nTuning per-key thresholds on validation set...")
    optimal_thresholds, threshold_f1s = find_optimal_thresholds(
        model, val_loader, device, key_names
    )
    for k in key_names:
        print(f"  {k}: thr={optimal_thresholds[k]:.2f}  val_f1={threshold_f1s[k]:.4f}")
    test_metrics = eval_epoch(
        model,
        test_loader,
        criterion,
        device,
        key_names,
        0,
        thresholds=optimal_thresholds,
    )
    print(f"Test macro F1: {test_metrics['macro_f1']:.4f}")
    print(f"Test micro F1: {test_metrics['micro_f1']:.4f}")
    print(f"Test overall acc: {test_metrics['overall_acc']:.4f}")
    print(f"Test no-action acc: {test_metrics['no_action_acc']:.4f}")
    f1_strs = [f"{k}={test_metrics['f1_' + k]:.3f}" for k in key_names]
    print(f"Per-key test F1: {', '.join(f1_strs)}")

    plot_training_curves(history, out_dir / "training_curves.png", key_names)
    print(f"Training curves saved to {out_dir / 'training_curves.png'}")

    plot_per_key_metrics(test_metrics, out_dir / "per_key_metrics.png", key_names)
    print(f"Per-key metrics saved to {out_dir / 'per_key_metrics.png'}")

    print("\nGenerating worst-prediction visualizations per key...")
    worst_preds_per_key = find_worst_predictions_per_key(
        test_ds, model, device, key_names, top_k=TOP_K
    )
    for key_name in key_names:
        items = worst_preds_per_key.get(key_name, [])
        if not items:
            print(f"  {key_name}: no test samples, skipping")
            continue
        out_path = out_dir / f"worst_preds_{key_name}.png"
        plot_worst_predictions_for_key(items, key_name, out_path)
        print(f"  {key_name}: saved {len(items)} worst predictions to {out_path.name}")

    with open(out_dir / "worst_predictions.json", "w") as f:
        json.dump(worst_preds_per_key, f, indent=2)
    print(f"Worst predictions JSON saved to {out_dir / 'worst_predictions.json'}")

    metrics_json = {
        "best_epoch": best_epoch,
        "best_val_macro_f1": best_val_f1,
        "optimal_thresholds": optimal_thresholds,
        "test_metrics": {
            "macro_f1": test_metrics["macro_f1"],
            "micro_f1": test_metrics["micro_f1"],
            "overall_acc": test_metrics["overall_acc"],
            "no_action_acc": test_metrics["no_action_acc"],
            "per_key_acc": {k: test_metrics[f"acc_{k}"] for k in key_names},
            "per_key_f1": {k: test_metrics[f"f1_{k}"] for k in key_names},
        },
        "history": {k: v[-1] for k, v in history.items() if k.startswith("val_")},
        "config": {
            "loss": "focal",
            "batch_size": BATCH_SIZE,
            "epochs": epoch,
            "lr": LR,
            "min_lr": MIN_LR,
            "dropout": DROPOUT,
            "patience": PATIENCE,
            "plateau_epochs": PLATEAU_EPOCHS,
            "cosine_t_max": cosine_t_max,
            "model_params": n_params,
            "train_windows": len(train_ds),
            "val_pairs": len(val_ds),
            "test_pairs": len(test_ds),
        },
    }
    with open(out_dir / "metrics.json", "w") as f:
        json.dump(metrics_json, f, indent=2)
    print(f"Metrics saved to {out_dir / 'metrics.json'}")


if __name__ == "__main__":
    main()
