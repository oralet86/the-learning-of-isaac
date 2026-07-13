#!/usr/bin/env python3
"""Train inverse dynamics model: (frame_t, frame_{t+1}) -> keys_t."""

import argparse
from collections import defaultdict
import json
import math
from pathlib import Path
import sys

import matplotlib
import numpy as np
import torch
from torch.amp import GradScaler, autocast
import torch.nn as nn

import torch.utils.data
from tqdm import tqdm

matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from constants import PROCESSED_DATA_DIR, TRAINING_RUNS_DIR


def set_seed(seed: int = 42):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------


class ResidualBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False,
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )
        else:
            self.downsample = nn.Identity()

        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout2d(0.2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = self.downsample(x)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = self.dropout(out)
        out += identity
        return self.relu(out)


class InverseDynamicsResNet(nn.Module):
    def __init__(self, in_channels: int = 2, num_keys: int = 13, dropout: float = 0.3):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, 16, 7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.maxpool = nn.MaxPool2d(3, stride=2, padding=1)

        self.layer1_block0 = ResidualBlock(16, 32, stride=1)
        self.layer1_block1 = ResidualBlock(32, 32, stride=1)
        self.layer2_block0 = ResidualBlock(32, 96, stride=2)
        self.layer2_block1 = ResidualBlock(96, 96, stride=1)
        self.layer3_block0 = ResidualBlock(96, 200, stride=2)
        self.layer3_block1 = ResidualBlock(200, 200, stride=1)
        self.layer4_block0 = ResidualBlock(200, 320, stride=2)
        self.layer4_block1 = ResidualBlock(320, 320, stride=1)

        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(320, num_keys)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.maxpool(x)
        x = self.layer1_block0(x)
        x = self.layer1_block1(x)
        x = self.layer2_block0(x)
        x = self.layer2_block1(x)
        x = self.layer3_block0(x)
        x = self.layer3_block1(x)
        x = self.layer4_block0(x)
        x = self.layer4_block1(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        return self.fc(x)


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------


class PreprocessedDataset(torch.utils.data.Dataset):
    def __init__(self, x_path: str | Path, y_path: str | Path):
        self.X = np.load(x_path)
        self.y = np.load(y_path)

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        x = torch.from_numpy(self.X[idx])
        y = torch.from_numpy(self.y[idx])
        return x, y


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


@torch.no_grad()
def compute_metrics(
    logits: torch.Tensor,
    targets: torch.Tensor,
    key_names: list[str],
    thresholds: dict[str, float] | None = None,
) -> dict:
    probs = torch.sigmoid(logits)
    if thresholds is None:
        preds = (probs > 0.5).float()
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

    eps = 1e-8
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


def aggregate_metrics(metrics_list: list[dict]) -> dict:
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


def compute_light_metrics(logits: torch.Tensor, targets: torch.Tensor) -> dict:
    probs = torch.sigmoid(logits)
    preds = (probs > 0.5).float()
    correct = (preds == targets).float()
    overall_acc = correct.mean().item()
    no_action_mask = targets.sum(dim=1) == 0
    n_zeros = no_action_mask.sum().item()
    no_action_acc = correct[no_action_mask].mean().item() if n_zeros > 0 else 0.0
    return {
        "overall_acc": overall_acc,
        "no_action_acc": no_action_acc,
        "no_action_count": int(n_zeros),
    }


# ---------------------------------------------------------------------------
# Training / validation loops
# ---------------------------------------------------------------------------


@torch.no_grad()
def find_optimal_thresholds(
    model: nn.Module,
    val_loader: torch.utils.data.DataLoader,
    device: torch.device,
    key_names: list[str],
) -> tuple[dict[str, float], dict[str, float]]:
    model.eval()
    all_logits: list[torch.Tensor] = []
    all_targets: list[torch.Tensor] = []
    for x, y in val_loader:
        x = (
            x.to(device, non_blocking=True)
            .to(dtype=torch.float16)
            .div_(255.0)
            .to(memory_format=torch.channels_last)
        )
        with autocast("cuda"):
            logits = model(x)

        all_logits.append(logits.cpu())
        all_targets.append(y)
    logits_cat = torch.cat(all_logits)
    targets_cat = torch.cat(all_targets)
    probs = torch.sigmoid(logits_cat)

    best_thresholds: dict[str, float] = {}
    best_f1s: dict[str, float] = {}

    thresholds_candidates = np.linspace(0.0, 1.0, 81)[1:-1]

    for k, key_name in enumerate(key_names):
        p = probs[:, k].numpy()
        t = targets_cat[:, k].numpy()

        best_t = 0.5
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
    model, loader, criterion, optimizer, scaler, device, _key_names, epoch: int
) -> dict:
    model.train()
    running_loss = 0.0
    running_acc = 0.0
    n_batches = 0
    pbar = tqdm(loader, desc=f"Epoch {epoch:3d} train", unit="batch", leave=False)
    for x, y in pbar:
        x = (
            x.to(device, non_blocking=True)
            .to(dtype=torch.float16)
            .div_(255.0)
            .to(memory_format=torch.channels_last)
        )
        y = y.to(device, non_blocking=True, dtype=torch.float32)
        optimizer.zero_grad(set_to_none=True)

        with autocast("cuda"):
            logits = model(x)
            loss = criterion(logits, y)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        running_loss += loss.item()
        running_acc += compute_light_metrics(logits.detach(), y)["overall_acc"]
        n_batches += 1
        pbar.set_postfix(loss=f"{loss.item():.4f}")

    return {"loss": running_loss / n_batches, "overall_acc": running_acc / n_batches}


@torch.no_grad()
def eval_epoch(
    model,
    loader,
    criterion,
    device,
    key_names,
    epoch: int,
    thresholds: dict[str, float] | None = None,
) -> dict:
    model.eval()
    metrics_list = []
    running_loss = 0.0
    pbar = tqdm(loader, desc=f"Epoch {epoch:3d} val  ", unit="batch", leave=False)
    for x, y in pbar:
        x = (
            x.to(device, non_blocking=True)
            .to(dtype=torch.float16)
            .div_(255.0)
            .to(memory_format=torch.channels_last)
        )
        y = y.to(device, non_blocking=True, dtype=torch.float32)
        with autocast("cuda"):
            logits = model(x)
            loss = criterion(logits, y)

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


def plot_training_curves(history: dict, out_path: Path, _key_names: list[str]):
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
    ax.plot(epochs, history["train_overall_acc"], label="train", linewidth=1.5)
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


def plot_per_key_metrics(best_metrics: dict, out_path: Path, key_names: list[str]):
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
    test_dataset: torch.utils.data.Dataset,
    model: nn.Module,
    device: torch.device,
    key_names: list[str],
    top_k: int = 20,
) -> dict[str, list[dict]]:
    model.eval()
    num_keys = len(key_names)

    per_key_items: list[list[tuple]] = [[] for _ in range(num_keys)]

    loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=256,
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

        eps = 1e-8
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

    result: dict[str, list[dict]] = {}
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
    items: list[dict],
    key_name: str,
    out_path: Path,
):
    n_items = len(items)
    if n_items == 0:
        return

    n_cols = 5
    n_rows = min(4, int(np.ceil(n_items / n_cols)))

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 4.5, n_rows * 2.8))
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

        pair_img = np.concatenate([frames[0], frames[1]], axis=1)

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


def build_loader(
    dataset: torch.utils.data.Dataset,
    batch_size: int,
    shuffle: bool,
    num_workers: int = 8,
    pin_memory: bool = True,
) -> torch.utils.data.DataLoader:
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
        persistent_workers=num_workers > 0,
        prefetch_factor=(2 if num_workers > 0 else None),
    )


def main():
    parser = argparse.ArgumentParser(description="Train inverse dynamics model")
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=3e-3)
    parser.add_argument("--min-lr", type=float, default=1e-6)
    parser.add_argument("--dropout", type=float, default=0.4)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--warmup-epochs", type=int, default=5)
    parser.add_argument("--cosine-t-max", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=0)
    args = parser.parse_args()

    from datetime import datetime

    out_dir = TRAINING_RUNS_DIR / f"run_{datetime.now().strftime('%Y%m%d_%H%M')}"

    set_seed(args.seed)
    torch.set_float32_matmul_precision("high")

    dataset_dir = PROCESSED_DATA_DIR / "dataset"
    info_path = dataset_dir / "dataset_info.json"
    if not info_path.exists():
        sys.exit(
            f"{info_path} not found. Run preprocess.py first to prepare the dataset."
        )
    with open(info_path) as f:
        ds_info = json.load(f)
    key_names: list[str] = ds_info["key_names"]
    num_keys = len(key_names)
    print(f"Key order ({num_keys} keys): {key_names}")

    pos_weight = torch.as_tensor(ds_info["pos_weight"], dtype=torch.float32)
    print(
        f"Pos weights: {dict(zip(key_names, [f'{w:.1f}' for w in pos_weight.tolist()], strict=True))}"
    )

    train_ds = PreprocessedDataset(
        dataset_dir / "train_X.npy", dataset_dir / "train_y.npy"
    )
    val_ds = PreprocessedDataset(dataset_dir / "val_X.npy", dataset_dir / "val_y.npy")
    test_ds = PreprocessedDataset(
        dataset_dir / "test_X.npy", dataset_dir / "test_y.npy"
    )
    print(f"Split: train={len(train_ds):,}  val={len(val_ds):,}  test={len(test_ds):,}")

    device = get_device()
    print(f"Device: {device}")

    train_loader = build_loader(
        train_ds, args.batch_size, shuffle=True, num_workers=args.num_workers
    )
    val_loader = build_loader(
        val_ds, args.batch_size * 2, shuffle=False, num_workers=args.num_workers
    )
    test_loader = build_loader(
        test_ds, args.batch_size * 2, shuffle=False, num_workers=args.num_workers
    )

    model = InverseDynamicsResNet(
        in_channels=2, num_keys=num_keys, dropout=args.dropout
    )
    model = model.to(device, memory_format=torch.channels_last)
    try:
        model = torch.compile(model)
        print("  torch.compile: enabled")
    except Exception as e:
        print(f"  torch.compile: failed ({e}), running uncompiled")
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model params: {n_params:,}")

    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight.to(device))
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=5e-3, fused=True
    )
    cosine_t_max = (
        args.cosine_t_max if args.cosine_t_max is not None else max(args.epochs // 3, 1)
    )
    warmup_epochs = args.warmup_epochs
    start_lr = args.min_lr / 10.0

    def lr_lambda(epoch: int) -> float:
        if epoch < warmup_epochs:
            return start_lr / args.lr + (1.0 - start_lr / args.lr) * (
                epoch / max(warmup_epochs, 1)
            )
        elif epoch < warmup_epochs + cosine_t_max:
            progress = (epoch - warmup_epochs) / max(cosine_t_max, 1)
            return args.min_lr / args.lr + (1.0 - args.min_lr / args.lr) * (
                0.5 * (1.0 + math.cos(math.pi * progress))
            )
        else:
            return args.min_lr / args.lr

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    scaler = GradScaler("cuda")

    out_dir.mkdir(parents=True, exist_ok=True)

    history: dict[str, list[float]] = defaultdict(list)
    best_val_f1 = -1.0
    best_epoch = 0
    patience_counter = 0

    print(f"\nTraining for up to {args.epochs} epochs (patience={args.patience})...")

    model.train()

    print(
        f"\n{'Epoch':>5s} {'TrainLoss':>10s} {'ValLoss':>10s} {'ValF1':>8s} {'ValAcc':>8s} {'NoActAcc':>8s} {'LR':>10s}"
    )
    print("-" * 65)

    for epoch in range(1, args.epochs + 1):
        train_metrics = train_epoch(
            model, train_loader, criterion, optimizer, scaler, device, key_names, epoch
        )
        val_metrics = eval_epoch(model, val_loader, criterion, device, key_names, epoch)
        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]

        history["train_loss"].append(train_metrics["loss"])
        history["train_overall_acc"].append(train_metrics["overall_acc"])
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

        if epoch % 10 == 0:
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "key_names": key_names,
                },
                out_dir / f"checkpoint_epoch{epoch:03d}.pt",
            )

        if patience_counter >= args.patience:
            print(
                f"\nEarly stopping at epoch {epoch} (no improvement for {args.patience} epochs)"
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
        test_ds, model, device, key_names, top_k=20
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
            "batch_size": args.batch_size,
            "epochs": epoch,
            "lr": args.lr,
            "min_lr": args.min_lr,
            "dropout": args.dropout,
            "patience": args.patience,
            "warmup_epochs": warmup_epochs,
            "cosine_t_max": cosine_t_max,
            "model_params": n_params,
            "train_pairs": len(train_ds),
            "val_pairs": len(val_ds),
            "test_pairs": len(test_ds),
        },
    }
    with open(out_dir / "metrics.json", "w") as f:
        json.dump(metrics_json, f, indent=2)
    print(f"Metrics saved to {out_dir / 'metrics.json'}")


if __name__ == "__main__":
    main()
