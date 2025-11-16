from fvcore.common.registry import Registry
from omegaconf import DictConfig
import torch

__all__ = [
    "ENCODER_REGISTRY",
    "build_encoder",
]

ENCODER_REGISTRY = Registry("ENCODER")
ENCODER_REGISTRY.__doc__ = "Registry for the encoder."


def build_encoder(cfg: DictConfig) -> torch.nn.Module:
    """
    Build the encoder defined by `cfg.MODEL.ENCODER.NAME`.
    """
    encoder_name = getattr(cfg, "NAME", cfg.MODEL.ENCODER.NAME)
    try:
        encoder = ENCODER_REGISTRY.get(encoder_name)(cfg)
    except KeyError as e:
        raise KeyError(ENCODER_REGISTRY) from e

    return encoder
