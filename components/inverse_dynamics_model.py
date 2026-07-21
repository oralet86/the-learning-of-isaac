"""Abstract base class and registry for inverse-dynamics models.

Every model subclass registers itself by name via ``__init_subclass__`` so
that ``train.py`` can look up any model with a simple string flag, without
importing per-model constants directly.
"""

from __future__ import annotations

from abc import ABCMeta, abstractmethod
from dataclasses import dataclass
from typing import ClassVar


@dataclass(frozen=True)
class ModelConfig:
    """Hyperparameters owned by the model, consumed by the training loop."""

    batch_size: int
    val_batch_multiplier: int
    dropout: float
    weight_decay: float
    window_size: int
    label_index: int
    stride: int
    uses_channels_last: bool = False


class _RegistryABCMeta(ABCMeta):
    """Metaclass that auto-registers subclasses under a ``model_name`` kwarg."""

    def __new__(
        mcs,
        name: str,
        bases: tuple[type, ...],
        namespace: dict[str, object],
        model_name: str | None = None,
        **kwargs: object,
    ) -> type:
        cls = super().__new__(mcs, name, bases, namespace, **kwargs)
        if model_name is not None:
            InverseDynamicsModel._registry[model_name] = cls  # type: ignore[attr-defined]
        return cls


class InverseDynamicsModel(metaclass=_RegistryABCMeta):
    """Abstract base for an inverse-dynamics key-prediction model.

    Subclass with ``model_name=`` to register::

        class MyModel(InverseDynamicsModel, model_name="my_model"): ...
    """

    _registry: ClassVar[dict[str, type[InverseDynamicsModel]]] = {}

    @classmethod
    @abstractmethod
    def config(cls) -> ModelConfig:
        """Return the model's default hyperparameter configuration."""

    @classmethod
    def get(cls, name: str) -> type[InverseDynamicsModel]:
        """Look up a registered model class by name."""
        if name not in cls._registry:
            raise KeyError(f"Unknown model {name!r}. Available: {cls.list_models()}")
        return cls._registry[name]

    @classmethod
    def list_models(cls) -> list[str]:
        """Return the names of all registered model classes."""
        return sorted(cls._registry)
