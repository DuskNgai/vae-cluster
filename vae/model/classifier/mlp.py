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
        in_features: int,
        hidden_features: int,
        out_features: int,
    ) -> None:
        super().__init__(
            in_features=in_features,
            hidden_features=hidden_features,
            out_features=out_features,
        )

    @classmethod
    def from_config(cls, cfg: DictConfig) -> dict[str, Any]:
        return {
            "in_features": cfg.MODEL.LATENT_DIM,
            "hidden_features": cfg.MODEL.CLASSIFIER.HIDDEN_DIM,
            "out_features": cfg.MODEL.CLASSIFIER.NUM_CLASSES,
        }

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        logits = super().forward(x)
        return logits
