"""Training configuration, decouples model and loss selection from train.py."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from components.inverse_dynamics_model import InverseDynamicsModel, ModelConfig
from components.loss_factory import LossCriterion, make_criterion

if TYPE_CHECKING:
    import torch


@dataclass(frozen=True)
class TrainingConfig:
    """Fully resolved training configuration.

    Built once from CLI args + dataset metadata, then used everywhere
    in the training loop, logging, and plotting.  train.py never accesses
    ``args.model`` or ``args.loss`` directly after construction.
    """

    model_name: str
    loss_name: str
    model_cls: type[InverseDynamicsModel] = field(repr=False)
    model_config: ModelConfig
    criterion: LossCriterion = field(repr=False)
    num_output_logits: int

    # ---- dataset-derived --------------------------------------------------
    key_names: list[str]
    num_keys: int

    # ---- overridable / optional -------------------------------------------
    window_size: int
    label_index: int
    stride: int

    @property
    def batch_size(self) -> int:
        return self.model_config.batch_size

    @property
    def val_batch_multiplier(self) -> int:
        return self.model_config.val_batch_multiplier

    @property
    def uses_channels_last(self) -> bool:
        return self.model_config.uses_channels_last

    @property
    def dropout(self) -> float:
        return self.model_config.dropout

    @property
    def weight_decay(self) -> float:
        return self.model_config.weight_decay

    # ---- factories --------------------------------------------------------

    @classmethod
    def build(
        cls,
        model_name: str,
        loss_name: str,
        window_size_override: int | None,
        label_index_override: int | None,
        stride_override: int | None,
        key_names: list[str],
        pos_weight: torch.Tensor,
        device: torch.device,
    ) -> TrainingConfig:
        """Construct from raw arguments."""
        model_cls = InverseDynamicsModel.get(model_name)
        model_config = model_cls.config()

        window_size = (
            window_size_override
            if window_size_override is not None
            else model_config.window_size
        )
        label_index = (
            label_index_override
            if label_index_override is not None
            else model_config.label_index
        )
        stride = stride_override if stride_override is not None else model_config.stride

        criterion = make_criterion(loss_name, pos_weight, device)
        num_output_logits = criterion.num_output_logits

        return cls(
            model_name=model_name,
            loss_name=loss_name,
            model_cls=model_cls,
            model_config=model_config,
            criterion=criterion,
            num_output_logits=num_output_logits,
            key_names=key_names,
            num_keys=len(key_names),
            window_size=window_size,
            label_index=label_index,
            stride=stride,
        )

    # ---- serialisation ----------------------------------------------------

    def to_dict(self, **extra: object) -> dict[str, object]:
        """Return a JSON-safe dictionary for metrics.json."""
        return {
            "model": self.model_name,
            "loss": self.loss_name,
            "batch_size": self.batch_size,
            "dropout": self.dropout,
            "weight_decay": self.weight_decay,
            "window_size": self.window_size,
            "label_index": self.label_index,
            "stride": self.stride,
            "uses_channels_last": self.uses_channels_last,
            "num_output_logits": self.num_output_logits,
            "num_keys": self.num_keys,
            **extra,
        }
