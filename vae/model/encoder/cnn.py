from typing import Any

from omegaconf import DictConfig
import torch
import torch.nn as nn

from coach_pl.configuration import configurable

from .build import ENCODER_REGISTRY


@ENCODER_REGISTRY.register()
class CNNEncoder(nn.Module):

    @configurable
    def __init__(self, img_size: int, num_filters: int, latent_dim: int) -> None:
        super().__init__()

        self.net = nn.Sequential(
            nn.Conv2d(1, num_filters, 3, 2, 1),                                # [B, C, 14, 14]
            nn.BatchNorm2d(num_filters),
            nn.ReLU(inplace=True),
            nn.Conv2d(num_filters, num_filters * 2, 3, 1, 1),                  # [B, C * 2, 14, 14]
            nn.BatchNorm2d(num_filters * 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(num_filters * 2, num_filters * 4, 3, 2, 1),              # [B, C * 4, 7, 7]
            nn.BatchNorm2d(num_filters * 4),
            nn.ReLU(inplace=True),
            nn.Conv2d(num_filters * 4, num_filters * 8, 3, 1, 1),              # [B, C * 8, 7, 7]
            nn.BatchNorm2d(num_filters * 8),
            nn.ReLU(inplace=True),
            nn.Flatten(),
            nn.Linear(num_filters * 8 * (img_size // 4) ** 2, latent_dim * 2),
        )

    @classmethod
    def from_config(cls, cfg: DictConfig) -> dict[str, Any]:
        return {
            "img_size": cfg.MODEL.IMG_SIZE,
            "num_filters": cfg.MODEL.NUM_FILTERS,
            "latent_dim": cfg.MODEL.LATENT_DIM,
        }

    def forward(self, x: torch.Tensor, index: torch.LongTensor) -> tuple[torch.Tensor, torch.Tensor]:
        mean_logvar = self.net(x)
        mean, logvar = mean_logvar.chunk(2, dim=-1)
        return mean, logvar
