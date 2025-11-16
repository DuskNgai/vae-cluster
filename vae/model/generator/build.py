from fvcore.common.registry import Registry
from omegaconf import DictConfig
import torch

__all__ = [
    "GENERATOR_REGISTRY",
    "build_generator",
]

GENERATOR_REGISTRY = Registry("GENERATOR")
GENERATOR_REGISTRY.__doc__ = "Registry for the generator."


def build_generator(cfg: DictConfig) -> torch.nn.Module:
    """
    Build the generator defined by `cfg.MODEL.GENERATOR.NAME`.
    """
    generator_name = getattr(cfg, "NAME", cfg.MODEL.GENERATOR.NAME)
    try:
        generator = GENERATOR_REGISTRY.get(generator_name)(cfg)
    except KeyError as e:
        raise KeyError(GENERATOR_REGISTRY) from e

    return generator
