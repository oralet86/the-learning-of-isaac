"""Python script for matplotlib plot functions."""

from pathlib import Path  # noqa: TC003
from typing import TYPE_CHECKING, Any

import matplotlib
import numpy as np
import torch
from torch.amp import autocast
import torch.nn as nn
from tqdm import tqdm

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

from constants import (
    EPSILON,
    TOP_K_WORST_PREDS,
    WORST_PRED_BATCH_SIZE,
)

torch.set_float32_matmul_precision(precision="high")
torch.backends.cudnn.benchmark = True

if TYPE_CHECKING:
    from torch import Tensor
    from torch.utils.data import Dataset


def plot_training_curves(history: dict[str, list[float]], out_path: Path) -> None:
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
