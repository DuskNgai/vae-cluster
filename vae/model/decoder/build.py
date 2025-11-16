from fvcore.common.registry import Registry
from omegaconf import DictConfig
import torch

__all__ = [
    "DECODER_REGISTRY",
    "build_decoder",
]

DECODER_REGISTRY = Registry("DECODER")
DECODER_REGISTRY.__doc__ = "Registry for the decoder."


def build_decoder(cfg: DictConfig) -> torch.nn.Module:
    """
    Build the decoder defined by `cfg.MODEL.DECODER.NAME`.
    """
    decoder_name = getattr(cfg, "NAME", cfg.MODEL.DECODER.NAME)
    try:
        decoder = DECODER_REGISTRY.get(decoder_name)(cfg)
    except KeyError as e:
        raise KeyError(DECODER_REGISTRY) from e

    return decoder
