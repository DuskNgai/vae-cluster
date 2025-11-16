from typing import Any, Literal

from omegaconf import DictConfig
import torch

from coach_pl.configuration import configurable

from ..layer.table import LatentTable
from .build import ENCODER_REGISTRY


@ENCODER_REGISTRY.register()
class MeanLogVarTableEncoder(LatentTable):

    @configurable
    def __init__(self, num_latents: int, latent_dim: int, weight_init: Literal["zero"] = "zero") -> None:
        super().__init__(
            num_latents=num_latents,
            latent_dim=latent_dim * 2,
            weight_init=weight_init,
        )

    @classmethod
    def from_config(cls, cfg: DictConfig) -> dict[str, Any]:
        return {
            "img_size": cfg.MODEL.IMG_SIZE,
            "latent_dim": cfg.MODEL.LATENT_DIM,
            "weight_init": cfg.MODEL.WEIGHT_INIT,
        }

    def forward(self, x: torch.Tensor, index: torch.LongTensor) -> tuple[torch.Tensor, torch.Tensor]:
        mean_logvar = super().forward(x, index)
        mean, logvar = mean_logvar.chunk(2, dim=-1)
        return mean, logvar
