#!/usr/bin/env python3
"""Train inverse dynamics model: (frame_t, frame_{t+1}) -> keys_t."""

import argparse
from collections import defaultdict
from collections.abc import Callable  # noqa: TC003
from contextlib import nullcontext
from datetime import datetime
import json
from pathlib import Path  # noqa: TC003
import sys
from typing import TYPE_CHECKING, Any

import matplotlib
import numpy as np
import torch
from torch.amp import GradScaler, autocast
import torch.nn as nn
from torch.profiler import ProfilerActivity
from torchmetrics import MetricCollection
from torchmetrics.classification import MultilabelAccuracy, MultilabelF1Score
from tqdm import tqdm

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

from components import (
    InverseDynamicsModel,
)
from constants import (
    EPOCHS,
    EPSILON,
    INVERSE_DYNAMIC_RUNS_DIR,
    LR,
    NUM_WORKERS,
    PATIENCE,
    PLATEAU_PATIENCE,
    PROCESSED_DATA_DIR,
    PROFILE_STEPS,
    PROFILE_WAIT,
    PROFILE_WARMUP,
    RUN_TIMESTAMP_FORMAT,
    SEED,
    THRESHOLD,
    TOP_K_WORST_PREDS,
    WORST_PRED_BATCH_SIZE,
)
from inverse_dynamics.dataloader import (
    WindowedDataset,
    build_loader,
    signed_diff_collate,
)

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
# Training / validation loops
# ---------------------------------------------------------------------------


def train_epoch(
    model: nn.Module,
    loader: DataLoader[tuple[torch.Tensor, torch.Tensor]],
    train_step: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    optimizer: torch.optim.Optimizer,
    scaler: GradScaler,
    device: torch.device,
    epoch: int,
    profiler: torch.profiler.profile | None = None,
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
        if profiler is not None:
            profiler.step()

    if n_batches == 0:
        return {"loss": torch.tensor(float("nan"))}
    return {"loss": running_loss / n_batches}


def make_train_step(
    model: nn.Module,
    criterion: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    uses_channels_last: bool,
) -> Callable[[torch.Tensor, torch.Tensor], torch.Tensor]:
    def train_step(x_raw: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        x: Tensor = x_raw.to(dtype=torch.float16).div(255.0)
        if uses_channels_last:
            x = x.to(memory_format=torch.channels_last)
        logits: Tensor = model(x)
        loss: Tensor = criterion(logits, y)
        return loss

    return train_step


def make_eval_step(
    model: nn.Module,
    criterion: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    uses_channels_last: bool,
) -> Callable[[torch.Tensor, torch.Tensor], tuple[torch.Tensor, torch.Tensor]]:
    def eval_step_fn(
        x_raw: torch.Tensor, y: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        x: Tensor = x_raw.to(dtype=torch.float16).div(255.0)
        if uses_channels_last:
            x = x.to(memory_format=torch.channels_last)
        logits: Tensor = model(x)
        loss: Tensor = criterion(logits, y)
        return logits, loss

    return eval_step_fn


@torch.no_grad()
def eval_epoch(
    eval_step: Callable[
        [torch.Tensor, torch.Tensor], tuple[torch.Tensor, torch.Tensor]
    ],
    loader: DataLoader[tuple[torch.Tensor, torch.Tensor]],
    device: torch.device,
    key_names: list[str],
    metric_collection: MetricCollection,
    epoch: int,
) -> dict[str, Any]:
    if len(loader) == 0:
        return {
            "loss": float("nan"),
            "macro_f1": float("nan"),
            "micro_f1": float("nan"),
            "overall_acc": float("nan"),
            "no_action_acc": float("nan"),
            "no_action_count": 0,
            "all_active_pred_count": 0,
            **{f"acc_{k}": float("nan") for k in key_names},
            **{f"f1_{k}": float("nan") for k in key_names},
        }

    metric_collection.reset()
    running_loss = 0.0
    no_action_correct = 0.0
    no_action_total = 0
    all_active_preds = 0

    pbar = tqdm(loader, desc=f"Epoch {epoch:3d} val  ", unit="batch", leave=False)
    for x, y in pbar:
        x: Tensor = x.to(device, non_blocking=True)
        y: Tensor = y.to(device, non_blocking=True, dtype=torch.float32)
        with autocast(device_type="cuda"):
            logits, loss = eval_step(x, y)

        running_loss += loss.item()
        probs = torch.sigmoid(logits.detach())
        metric_collection.update(probs.float(), y.int())

        no_action_mask = y.sum(dim=1) == 0
        no_action_count_batch = no_action_mask.sum()
        if no_action_count_batch > 0:
            no_action_total += int(no_action_count_batch.item()) * y.shape[1]
            no_action_correct += int((probs[no_action_mask] <= THRESHOLD).sum().item())

        all_active_preds += int((probs > THRESHOLD).all(dim=1).sum().item())
        pbar.set_postfix(loss=f"{loss.item():.4f}")

    result = metric_collection.compute()

    return {
        "loss": running_loss / len(loader),
        "macro_f1": float(result["macro_f1"]),
        "micro_f1": float(result["micro_f1"]),
        "overall_acc": float(result["overall_acc"]),
        "no_action_acc": float(no_action_correct / no_action_total)
        if no_action_total > 0
        else 0.0,
        "no_action_count": no_action_total,
        "all_active_pred_count": all_active_preds,
        **{
            f"acc_{k}": float(v)
            for k, v in zip(key_names, result["per_key_acc"], strict=True)
        },
        **{
            f"f1_{k}": float(v)
            for k, v in zip(key_names, result["per_key_f1"], strict=True)
        },
    }


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
    uses_channels_last: bool,
    top_k: int = TOP_K_WORST_PREDS,
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
            x_batch.to(device, non_blocking=True).to(dtype=torch.float16).div_(255.0)
        )
        if uses_channels_last:
            x_batch = x_batch.to(memory_format=torch.channels_last)
        with autocast("cuda"):
            logits = model(x_batch)
            probs = torch.sigmoid(logits)

        y_np = y_batch.numpy()
        p_np = np.clip(probs.cpu().numpy(), EPSILON, 1.0 - EPSILON)
        bce = -(y_np * np.log(p_np + EPSILON) + (1 - y_np) * np.log(1 - p_np + EPSILON))

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

        n_ch = len(frames)
        pair_img = np.concatenate([frames[i] for i in range(n_ch)], axis=1)

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
    parser.add_argument(
        "--model",
        type=str,
        default="resnet",
        choices=InverseDynamicsModel.list_models(),
        help=f"Model architecture (default: resnet, available: {InverseDynamicsModel.list_models()})",
    )
    parser.add_argument(
        "--window-size",
        type=int,
        default=None,
        help="Override model window_size",
    )
    parser.add_argument(
        "--label-index",
        type=int,
        default=None,
        help="Override model label_index",
    )
    parser.add_argument(
        "--stride",
        type=int,
        default=None,
        help="Override model stride",
    )
    parser.add_argument(
        "--signed-diff",
        action="store_true",
        help="Use signed-diff collate function (replaces frame_t with frame_t - frame_{t-1})",
    )
    args = parser.parse_args()

    model_cls = InverseDynamicsModel.get(args.model)
    model_config = model_cls.config()

    window_size = (
        args.window_size if args.window_size is not None else model_config.window_size
    )
    label_index_default = (window_size - 1) // 2
    label_index = (
        args.label_index if args.label_index is not None else label_index_default
    )
    stride = args.stride if args.stride is not None else model_config.stride

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

    train_ds = WindowedDataset(
        processed_dir=PROCESSED_DATA_DIR,
        split="train",
        window_size=window_size,
        label_index=label_index,
        stride=stride,
    )
    val_ds = WindowedDataset(
        processed_dir=PROCESSED_DATA_DIR,
        split="val",
        window_size=window_size,
        label_index=label_index,
        stride=stride,
    )
    test_ds = WindowedDataset(
        processed_dir=PROCESSED_DATA_DIR,
        split="test",
        window_size=window_size,
        label_index=label_index,
        stride=stride,
    )
    print(f"Split: train={len(train_ds):,}  val={len(val_ds):,}  test={len(test_ds):,}")

    device = get_device()
    print(f"Device: {device}")

    model_cls_any: Any = model_cls
    model = model_cls_any(window_size=window_size, num_keys=num_keys, dropout=model_config.dropout)  # fmt: skip
    print(f"Model: {model_cls.__name__}")
    if model_config.uses_channels_last:
        model = model.to(device, memory_format=torch.channels_last)  # type: ignore
    else:
        model = model.to(device)  # type: ignore

    collate_fn = signed_diff_collate if args.signed_diff else None

    train_loader: DataLoader[tuple[Tensor, Tensor]] = build_loader(
        train_ds,
        model_config.batch_size,
        shuffle=True,
        num_workers=NUM_WORKERS,
        collate_fn=collate_fn,
    )
    val_loader: DataLoader[tuple[Tensor, Tensor]] = build_loader(
        val_ds,
        model_config.batch_size * model_config.val_batch_multiplier,
        shuffle=False,
        num_workers=NUM_WORKERS,
        drop_last=False,
        collate_fn=collate_fn,
    )
    test_loader: DataLoader[tuple[Tensor, Tensor]] = build_loader(
        test_ds,
        model_config.batch_size * model_config.val_batch_multiplier,
        shuffle=False,
        num_workers=NUM_WORKERS,
        drop_last=False,
        collate_fn=collate_fn,
    )

    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight.to(device))
    train_step = make_train_step(model, criterion, model_config.uses_channels_last)
    try:
        train_step = torch.compile(train_step)
        print("  torch.compile: enabled")
    except Exception as e:
        print(f"  torch.compile: failed ({e}), running uncompiled")
    eval_step = make_eval_step(model, criterion, model_config.uses_channels_last)
    try:
        eval_step = torch.compile(eval_step, dynamic=False)
        print("  eval torch.compile: enabled")
    except Exception as e:
        print(f"  eval torch.compile: failed ({e}), running uncompiled")
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model params: {n_params:,}")

    metric_collection = MetricCollection(
        {
            "macro_f1": MultilabelF1Score(num_labels=num_keys, average="macro"),
            "micro_f1": MultilabelF1Score(num_labels=num_keys, average="micro"),
            "overall_acc": MultilabelAccuracy(num_labels=num_keys, average="micro"),
            "per_key_f1": MultilabelF1Score(num_labels=num_keys, average="none"),
            "per_key_acc": MultilabelAccuracy(num_labels=num_keys, average="none"),
        }
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LR,
        weight_decay=model_config.weight_decay,
        fused=True,
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=PLATEAU_PATIENCE
    )
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
        model.eval()
        val_metrics = eval_epoch(
            eval_step,
            val_loader,
            device,
            key_names,
            metric_collection,
            epoch,
        )
        scheduler.step(val_metrics["loss"])

        history["train_loss"].append(train_metrics["loss"].item())
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
            f"{val_metrics['no_action_acc']:8.4f} {scheduler.get_last_lr()[0]:10.2e}"
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
    model.eval()
    test_metrics = eval_epoch(
        eval_step,
        test_loader,
        device,
        key_names,
        metric_collection,
        0,
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
        test_ds,
        model,
        device,
        key_names,
        model_config.uses_channels_last,
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
        "test_metrics": {
            "macro_f1": test_metrics["macro_f1"],
            "micro_f1": test_metrics["micro_f1"],
            "overall_acc": test_metrics["overall_acc"],
            "no_action_acc": test_metrics["no_action_acc"],
            "per_key_acc": {k: test_metrics[f"acc_{k}"] for k in key_names},
            "per_key_f1": {k: test_metrics[f"f1_{k}"] for k in key_names},
        },
        "config": {
            "loss": "bce",
            "model": args.model,
            "batch_size": model_config.batch_size,
            "epochs": len(history["train_loss"]),
            "lr": LR,
            "dropout": model_config.dropout,
            "weight_decay": model_config.weight_decay,
            "patience": PATIENCE,
            "plateau_patience": PLATEAU_PATIENCE,
            "model_params": n_params,
            "train_windows": len(train_ds),
            "val_windows": len(val_ds),
            "test_windows": len(test_ds),
            "window_size": window_size,
            "label_index": label_index,
            "stride": stride,
            "uses_channels_last": model_config.uses_channels_last,
        },
        "history": dict(history),
    }
    with open(out_dir / "metrics.json", "w") as f:
        json.dump(metrics_json, f, indent=2)
    print(f"Metrics saved to {out_dir / 'metrics.json'}")


if __name__ == "__main__":
    main()
