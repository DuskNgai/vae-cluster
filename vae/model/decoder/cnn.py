from typing import Any

from omegaconf import DictConfig
import torch
import torch.nn as nn

from coach_pl.configuration import configurable

from .build import DECODER_REGISTRY


@DECODER_REGISTRY.register()
class CNNDecoder(nn.Module):

    @configurable
    def __init__(self, img_size: int, num_filters: int, latent_dim: int) -> None:
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(latent_dim, num_filters * 8 * (img_size // 4) ** 2),                   # [B, C * 8, 7, 7]
            nn.Unflatten(-1, (num_filters * 8, img_size // 4, img_size // 4)),
            nn.ConvTranspose2d(num_filters * 8, num_filters * 4, 3, 1, 1),                   # [B, C * 4, 7, 7]
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.ConvTranspose2d(num_filters * 4, num_filters * 2, 3, 2, 1, output_padding=1), # [B, C * 2, 14, 14]
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.ConvTranspose2d(num_filters * 2, num_filters, 3, 1, 1),                       # [B, C, 14, 14]
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.ConvTranspose2d(num_filters, 1, 3, 2, 1, output_padding=1),                   # [B, 1, 28, 28]
            nn.Sigmoid(),
        )

    @classmethod
    def from_config(cls, cfg: DictConfig) -> dict[str, Any]:
        return {
            "img_size": cfg.MODEL.IMG_SIZE,
            "num_filters": cfg.MODEL.NUM_FILTERS,
            "latent_dim": cfg.MODEL.LATENT_DIM,
        }

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
