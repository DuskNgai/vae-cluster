from pathlib import Path
from typing import Any

from omegaconf import DictConfig
from pytorch_lightning.trainer.states import RunningStage
from torch.utils.data import Dataset
from torchvision.datasets import MNIST
from torchvision.transforms import Compose, v2

from coach_pl.configuration import configurable
from coach_pl.datamodule import DATASET_REGISTRY


@DATASET_REGISTRY.register()
class MNISTDataset(Dataset):
    """
    A wrapper around torchvision.datasets.MNIST to provide a consistent interface.
    """

    @configurable
    def __init__(
        self,
        root: str,
        transform: v2.Compose | Compose | None = None,
    ) -> None:
        super().__init__()
        self.train_dataset = MNIST(root=root, train=True, transform=transform, download=True)
        self.test_dataset = MNIST(root=root, train=False, transform=transform, download=True)

    @classmethod
    def from_config(cls, cfg: DictConfig, stage: RunningStage) -> dict[str, Any]:
        transform = v2.ToTensor()
        return {
            "root": Path(cfg.DATAMODULE.DATASET.ROOT),
            "transform": transform,
        }

    def __len__(self) -> int:
        return len(self.train_dataset) + len(self.test_dataset)

    def __getitem__(self, index: int) -> tuple[Any, int]:
        if index < len(self.train_dataset):
            return *self.train_dataset[index], index
        else:
            test_index = index - len(self.train_dataset)
            return *self.test_dataset[test_index], index

    @property
    def collate_fn(self) -> None:
        return None
