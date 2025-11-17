from typing import Any

from omegaconf import DictConfig
from timm.layers.mlp import Mlp
import torch

from coach_pl.configuration import configurable

from .build import CLASSIFIER_REGISTRY


@CLASSIFIER_REGISTRY.register()
class MLPClassifier(Mlp):

    @configurable
    def __init__(
        self,
        latent_dim: int,
        hidden_features: int,
        num_classes: int,
    ) -> None:
        super().__init__(
            in_features=latent_dim,
            hidden_features=hidden_features,
            out_features=num_classes,
        )
        self.num_classes = num_classes

    @classmethod
    def from_config(cls, cfg: DictConfig) -> dict[str, Any]:
        return {
            "latent_dim": cfg.MODEL.LATENT_DIM,
            "hidden_features": cfg.MODEL.CLASSIFIER.HIDDEN_DIM,
            "num_classes": cfg.MODEL.CLASSIFIER.NUM_CLASSES,
        }

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        logits = super().forward(x)
        return logits
