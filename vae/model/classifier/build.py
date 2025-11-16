from fvcore.common.registry import Registry
from omegaconf import DictConfig
import torch

__all__ = [
    "CLASSIFIER_REGISTRY",
    "build_classifier",
]

CLASSIFIER_REGISTRY = Registry("CLASSIFIER")
CLASSIFIER_REGISTRY.__doc__ = "Registry for the classifier."


def build_classifier(cfg: DictConfig) -> torch.nn.Module:
    """
    Build the classifier defined by `cfg.MODEL.CLASSIFIER.NAME`.
    """
    classifier_name = getattr(cfg, "NAME", cfg.MODEL.CLASSIFIER.NAME)
    try:
        classifier = CLASSIFIER_REGISTRY.get(classifier_name)(cfg)
    except KeyError as e:
        raise KeyError(CLASSIFIER_REGISTRY) from e

    return classifier
