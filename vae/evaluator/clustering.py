from collections import defaultdict
import io
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from omegaconf import DictConfig
from PIL import Image
from scipy.optimize import linear_sum_assignment
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix
import torch
import umap

from coach_pl.configuration import configurable
from coach_pl.evaluator import EVALUATOR_REGISTRY


def plt_to_image(fig: plt.Figure) -> torch.Tensor:
    buf = io.BytesIO()
    fig.savefig(buf, format="png")
    plt.close(fig)
    buf.seek(0)
    image = Image.open(buf).convert("RGB")
    buf.close()
    image = np.array(image).transpose(2, 0, 1) # [C, H, W]
    return image[None]                         # [1, C, H, W]


@EVALUATOR_REGISTRY.register()
class ClusteringEvaluator:

    @configurable
    def __init__(self) -> None:
        self.result = defaultdict(list)

        self.pca = PCA(n_components=2)
        self.umap = umap.UMAP(n_components=2, random_state=0)

    @classmethod
    def from_config(cls, cfg: DictConfig) -> dict[str, Any]:
        return {}

    def add(self, result: dict[str, Any], label: torch.Tensor) -> None:
        self.result["label"].append(label.cpu().numpy())
        self.result["prediction"].append(result["prediction"].cpu().numpy())
        self.result["latent"].append(result["latent"].cpu().numpy())

    def __call__(self) -> None:
        labels = np.concatenate(self.result["label"])
        predictions = np.concatenate(self.result["prediction"])
        latents = np.concatenate(self.result["latent"])

        # Find best permutation & accuracy
        cm = confusion_matrix(labels, predictions)
        row_ind, col_ind = linear_sum_assignment(cm, maximize=True)
        accuracy = cm[row_ind, col_ind].sum() / cm.sum()
        mapping = {
            original: new
            for original, new in zip(col_ind, row_ind)
        }
        mapped_predictions = np.array([mapping[p] for p in predictions])

        self.result = defaultdict(list)

        return {
            "metric/accuracy": accuracy,
            "pca_image": self.pca_plot(labels, latents, mapped_predictions),
            "umap_image": self.umap_plot(labels, latents, mapped_predictions),
        }

    def pca_plot(self, labels: np.ndarray, latents: np.ndarray, mapped_predictions: np.ndarray) -> torch.Tensor:
        # Latent visualization with PCA
        latents_2d = self.pca.fit_transform(latents)

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 6), tight_layout=True)
        cmap = plt.get_cmap("tab10", 10)

        # Color by label
        scatter1 = ax1.scatter(latents_2d[:, 0], latents_2d[:, 1], c=labels, cmap=cmap, alpha=0.7)
        ax1.set_title("Latent Space Visualization (Colored by True Labels)")
        ax1.set_xlabel("PC 1")
        ax1.set_ylabel("PC 2")
        ax1.grid(True, linestyle="--", alpha=0.6)
        legend1 = ax1.legend(*scatter1.legend_elements(), title="Classes")
        ax1.add_artist(legend1)

        # Color by prediction
        scatter2 = ax2.scatter(latents_2d[:, 0], latents_2d[:, 1], c=mapped_predictions, cmap=cmap, alpha=0.7)
        ax2.set_title("Latent Space Visualization (Colored by Matched Predictions)")
        ax2.set_xlabel("PC 1")
        ax2.set_ylabel("PC 2")
        ax2.grid(True, linestyle="--", alpha=0.6)
        legend2 = ax2.legend(*scatter2.legend_elements(), title="Clusters")
        ax2.add_artist(legend2)

        image = plt_to_image(fig)
        return image

    def umap_plot(self, labels: np.ndarray, latents: np.ndarray, mapped_predictions: np.ndarray) -> torch.Tensor:
        # Latent visualization with UMAP
        latents_2d = self.umap.fit_transform(latents)

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 6), tight_layout=True)
        cmap = plt.get_cmap("tab10", 10)

        # Color by label
        scatter1 = ax1.scatter(latents_2d[:, 0], latents_2d[:, 1], c=labels, cmap=cmap, alpha=0.7)
        ax1.set_title("Latent Space Visualization (Colored by True Labels)")
        ax1.set_xlabel("UMAP 1")
        ax1.set_ylabel("UMAP 2")
        ax1.grid(True, linestyle="--", alpha=0.6)
        legend1 = ax1.legend(*scatter1.legend_elements(), title="Classes")
        ax1.add_artist(legend1)

        # Color by prediction
        scatter2 = ax2.scatter(latents_2d[:, 0], latents_2d[:, 1], c=mapped_predictions, cmap=cmap, alpha=0.7)
        ax2.set_title("Latent Space Visualization (Colored by Matched Predictions)")
        ax2.set_xlabel("UMAP 1")
        ax2.set_ylabel("UMAP 2")
        ax2.grid(True, linestyle="--", alpha=0.6)
        legend2 = ax2.legend(*scatter2.legend_elements(), title="Clusters")
        ax2.add_artist(legend2)

        image = plt_to_image(fig)
        return image
