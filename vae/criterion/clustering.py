from typing import Any

from omegaconf import DictConfig
import torch
import torch.nn as nn

from coach_pl.configuration import configurable
from coach_pl.criterion import CRITERION_REGISTRY


@CRITERION_REGISTRY.register()
class ClusteringCriterion(nn.Module):
    """
    Criterion for EDM Diffusion model.
    """

    @configurable
    def __init__(self, weight: float) -> None:
        super().__init__()

        self.loss_fn = nn.MSELoss(reduction="none")
        self.weight = weight

    @classmethod
    def from_config(cls, cfg: DictConfig) -> dict[str, Any]:
        return {
            "weight": cfg.CRITERION.WEIGHT,
        }

    def forward(self, result: dict[str, Any], target: torch.Tensor) -> dict[str, torch.Tensor]:

        reconstruction_loss = self.loss_fn(result["reconstruction"], target).flatten(1).sum(dim=-1).mean(0) # [B]

        posterior_logvar = result["posterior_logvar"].unsqueeze(-2) # [B, 1, D]
        prior_mean = result["prior_mean"].unsqueeze(0)              # [1, K, D]
        prior_logvar = result["prior_logvar"].unsqueeze(0)          # [1, K, D]
        latent = result["latent"].unsqueeze(-2)                     # [B, 1, D]
        logit = result["logit"]                                     # [B, K]
        prob = logit.softmax(dim=-1)                                # [B, K]

        score = ((latent - prior_mean) / torch.exp(0.5 * prior_logvar)).square()                                  # [B, K, D]
        continuous_kl = 0.5 * torch.einsum("bk, bkd -> b", prob, prior_logvar - posterior_logvar + score).mean(0) # [B]
        discrete_kl = (prob * logit.log_softmax(dim=-1)).sum(dim=-1).mean(0)                                      # [B]

        return {
            "loss": reconstruction_loss + self.weight * (continuous_kl + discrete_kl),
            "reconstruction_loss": reconstruction_loss,
            "continuous_kl": continuous_kl,
            "discrete_kl": discrete_kl,
        }
