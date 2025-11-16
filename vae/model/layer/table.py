from typing import Any, Literal

from omegaconf import DictConfig
import torch
import torch.nn as nn


class LatentTable(nn.Module):

    def __init__(self, num_latents: int, latent_dim: int, weight_init: Literal["zero"] = "zero") -> None:
        super().__init__()

        if weight_init == "zero":
            latents = torch.zeros(num_latents, latent_dim)
        else:
            raise NotImplementedError(f"Unknown weight_init: {weight_init}.")

        self.latents = nn.Parameter(latents)

    @classmethod
    def from_config(cls, cfg: DictConfig) -> dict[str, Any]:
        return {
            "num_latents": cfg.MODEL.NUM_LATENTS,
            "latent_dim": cfg.MODEL.LATENT_DIM,
            "weight_init": cfg.MODEL.WEIGHT_INIT,
        }

    def forward(self, x: torch.Tensor, index: torch.LongTensor) -> torch.Tensor:
        return self.latents[index]
