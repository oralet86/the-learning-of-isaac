#!/usr/bin/env python3
"""Train inverse dynamics model: (frame_t, frame_{t+1}) -> keys_t."""

import argparse
from collections import defaultdict
from collections.abc import Callable  # noqa: TC003
from contextlib import nullcontext
from datetime import datetime
import json
import os
from pathlib import Path  # noqa: TC003
import sys
from typing import TYPE_CHECKING, Any

import numpy as np
import torch
from torch.amp import GradScaler, autocast
import torch.nn as nn
from torch.profiler import ProfilerActivity
from torchmetrics import MetricCollection
from torchmetrics.classification import MultilabelAccuracy, MultilabelF1Score
from tqdm import tqdm

from components import TrainingConfig
from components.inverse_dynamics_model import InverseDynamicsModel
from constants import (
    ABLATION_DATA_FRAC,
    DEFAULT_MODE,
    EPOCHS,
    INVERSE_DYNAMIC_RUNS_DIR,
    LR,
    LR_SCHEDULE_FACTOR,
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
)
from inverse_dynamics.dataloader import (
    WindowedDataset,
    build_loader,
)
from inverse_dynamics.plotting import (
    find_worst_predictions_per_key,
    plot_per_key_metrics,
    plot_training_curves,
    plot_worst_predictions_for_key,
)

torch.set_float32_matmul_precision(precision="high")
torch.backends.cudnn.benchmark = True

if TYPE_CHECKING:
    from torch import Tensor
    from torch.utils.data import DataLoader


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


def train_epoch(
    model: nn.Module,
    loader: DataLoader[tuple[torch.Tensor, torch.Tensor]],
    train_step: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    optimizer: torch.optim.Optimizer,
    scaler: GradScaler,
    device: torch.device,
    epoch: int,
    profiler: torch.profiler.profile | None = None,
    max_batches: int | None = None,
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
) -> Callable[[torch.Tensor, torch.Tensor], torch.Tensor]:
    def train_step(x_raw: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        x: Tensor = x_raw.to(dtype=torch.float16).div(255.0)
        if uses_channels_last:
            x = x.to(memory_format=torch.channels_last)
        logits: tuple[Tensor, Tensor] = model(x)
        loss: Tensor = criterion(logits, y)
        return loss

    return train_step


def make_eval_step(
    model: nn.Module,
    criterion: nn.Module,
    uses_channels_last: bool,
) -> Callable[[torch.Tensor, torch.Tensor], tuple[torch.Tensor, torch.Tensor]]:
    to_probs = getattr(
        criterion, "to_binary_probs", lambda t: torch.sigmoid(t.detach())
    )

    def eval_step_fn(
        x_raw: torch.Tensor, y: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        x: Tensor = x_raw.to(dtype=torch.float16).div(255.0)
        if uses_channels_last:
            x = x.to(memory_format=torch.channels_last)
        logits: tuple[Tensor, Tensor] = model(x)
        loss: Tensor = criterion(logits, y)
        probs: Tensor = to_probs(logits)
        return probs, loss

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
            probs, loss = eval_step(x, y)

        running_loss += loss.item()
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
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description="Train inverse dynamics model")
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument(
        "--mode",
        type=str,
        default=DEFAULT_MODE,
        choices=["default", "profile", "ablation"],
        help=f"Run mode: default, profile or ablation (default: {DEFAULT_MODE})",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="resnet_lite",
        choices=InverseDynamicsModel.list_models(),
        help=f"Model architecture (default: resnet_lite, available: {InverseDynamicsModel.list_models()})",
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
    args = parser.parse_args()

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
    pos_weight_full = torch.as_tensor(ds_info["pos_weight"], dtype=torch.float32)

    data_frac = ABLATION_DATA_FRAC if args.mode == "ablation" else 1.0

    device = get_device()
    print(f"Device: {device}")

    cfg = TrainingConfig.build(
        model_name=args.model,
        window_size_override=args.window_size,
        label_index_override=args.label_index,
        stride_override=args.stride,
        key_names=key_names,
        pos_weight=pos_weight_full,
        device=device,
    )
    criterion: nn.Module = cfg.criterion

    train_ds = WindowedDataset(
        processed_dir=PROCESSED_DATA_DIR,
        split="train",
        window_size=cfg.window_size,
        label_index=cfg.label_index,
        stride=cfg.stride,
        data_frac=data_frac,
    )
    val_ds = WindowedDataset(
        processed_dir=PROCESSED_DATA_DIR,
        split="val",
        window_size=cfg.window_size,
        label_index=cfg.label_index,
        stride=cfg.stride,
    )
    test_ds = WindowedDataset(
        processed_dir=PROCESSED_DATA_DIR,
        split="test",
        window_size=cfg.window_size,
        label_index=cfg.label_index,
        stride=cfg.stride,
    )
    print(f"Split: train={len(train_ds):,}  val={len(val_ds):,}  test={len(test_ds):,}")

    model_cls_any: Any = cfg.model_cls
    model = model_cls_any(window_size=cfg.window_size, dropout=cfg.dropout)
    print(f"Model: {cfg.model_name}")
    if cfg.uses_channels_last:
        model = model.to(device, memory_format=torch.channels_last)  # type: ignore
    else:
        model = model.to(device)  # type: ignore

    train_loader: DataLoader[tuple[Tensor, Tensor]] = build_loader(
        train_ds,
        cfg.batch_size,
        shuffle=True,
        num_workers=NUM_WORKERS,
        drop_last=True,
    )
    val_loader: DataLoader[tuple[Tensor, Tensor]] = build_loader(
        val_ds,
        cfg.batch_size * cfg.val_batch_multiplier,
        shuffle=False,
        num_workers=NUM_WORKERS,
        drop_last=False,
    )
    test_loader: DataLoader[tuple[Tensor, Tensor]] = build_loader(
        test_ds,
        cfg.batch_size * cfg.val_batch_multiplier,
        shuffle=False,
        num_workers=NUM_WORKERS,
        drop_last=False,
    )

    train_step = make_train_step(model, criterion, cfg.uses_channels_last)
    try:
        train_step = torch.compile(train_step, mode="max-autotune")
        print("  torch.compile: enabled")
    except Exception as e:
        print(f"  torch.compile: failed ({e}), running uncompiled")
    eval_step = make_eval_step(model, criterion, cfg.uses_channels_last)
    try:
        eval_step = torch.compile(eval_step, dynamic=False, mode="max-autotune")
        print("  eval torch.compile: enabled")
    except Exception as e:
        print(f"  eval torch.compile: failed ({e}), running uncompiled")
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model params: {n_params:,}")

    metric_collection = MetricCollection(
        {
            "macro_f1": MultilabelF1Score(num_labels=cfg.num_keys, average="macro"),
            "micro_f1": MultilabelF1Score(num_labels=cfg.num_keys, average="micro"),
            "overall_acc": MultilabelAccuracy(num_labels=cfg.num_keys, average="micro"),
            "per_key_f1": MultilabelF1Score(num_labels=cfg.num_keys, average="none"),
            "per_key_acc": MultilabelAccuracy(num_labels=cfg.num_keys, average="none"),
        }
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LR,
        weight_decay=cfg.weight_decay,
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

    print(f"\nBest validation loss: {best_val_loss:.4f} at epoch {best_epoch}")

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

    plot_training_curves(history, out_dir / "training_curves.png")
    print(f"Training curves saved to {out_dir / 'training_curves.png'}")

    plot_per_key_metrics(test_metrics, out_dir / "per_key_metrics.png", key_names)
    print(f"Per-key metrics saved to {out_dir / 'per_key_metrics.png'}")

    print("\nGenerating worst-prediction visualizations per key...")
    worst_preds_per_key = find_worst_predictions_per_key(
        test_ds,
        model,
        device,
        key_names,
        cfg.uses_channels_last,
        to_probs_fn=criterion.to_binary_probs,
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
        "best_val_loss": best_val_loss,
        "test_metrics": {
            "macro_f1": test_metrics["macro_f1"],
            "micro_f1": test_metrics["micro_f1"],
            "overall_acc": test_metrics["overall_acc"],
            "no_action_acc": test_metrics["no_action_acc"],
            "per_key_acc": {k: test_metrics[f"acc_{k}"] for k in key_names},
            "per_key_f1": {k: test_metrics[f"f1_{k}"] for k in key_names},
        },
        "config": cfg.to_dict(
            mode=args.mode,
            epochs=len(history["train_loss"]),
            lr=LR,
            patience=PATIENCE,
            plateau_patience=PLATEAU_PATIENCE,
            model_params=n_params,
            train_windows=len(train_ds),
            val_windows=len(val_ds),
            test_windows=len(test_ds),
        ),
        "history": dict(history),
    }
    with open(out_dir / "metrics.json", "w") as f:
        json.dump(metrics_json, f, indent=2)
    print(f"Metrics saved to {out_dir / 'metrics.json'}")


if __name__ == "__main__":
    main()
