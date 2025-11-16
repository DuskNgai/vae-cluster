from typing import Any

from omegaconf import DictConfig
import torch
import torch.nn as nn

from coach_pl.configuration import configurable
from coach_pl.model import MODEL_REGISTRY

from ..classifier import build_classifier
from ..decoder import build_decoder
from ..encoder import build_encoder
from ..generator import build_generator


@MODEL_REGISTRY.register()
class SelfSupervisedClusteringVAE(nn.Module):

    @configurable
    def __init__(
        self,
        encoder: nn.Module,
        classifier: nn.Module,
        generator: nn.Module,
        decoder: nn.Module,
    ) -> None:
        super().__init__()

        self.encoder = encoder
        self.classifier = classifier
        self.generator = generator
        self.decoder = decoder

    @classmethod
    def from_config(cls, cfg: DictConfig) -> dict[str, Any]:
        return {
            "encoder": build_encoder(cfg),
            "classifier": build_classifier(cfg),
            "generator": build_generator(cfg),
            "decoder": build_decoder(cfg),
        }

    def forward(self, x: torch.Tensor, index: torch.LongTensor) -> dict[str, torch.Tensor]:
        posterior_mean, posterior_logvar = self.encoder(x, index)
        latent = self.reparameterize(posterior_mean, posterior_logvar)
        logit = self.classifier(latent)
        prediction = logit.argmax(dim=1)
        prior_mean, prior_logvar = self.generator()
        reconstruction = self.decoder(latent)

        return {
            "posterior_mean": posterior_mean,
            "posterior_logvar": posterior_logvar,
            "latent": latent,
            "logit": logit,
            "prediction": prediction,
            "prior_mean": prior_mean,
            "prior_logvar": prior_logvar,
            "reconstruction": reconstruction,
        }

    def reparameterize(self, mean: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mean + eps * std
