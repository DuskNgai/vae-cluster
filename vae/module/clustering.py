from pathlib import Path
from typing import Any

import numpy as np
from omegaconf import DictConfig
from PIL import Image
from pytorch_lightning import LightningModule
from pytorch_lightning.utilities.types import LRSchedulerTypeUnion
from timm.optim import create_optimizer_v2
from timm.scheduler import create_scheduler_v2
import torch

from coach_pl.configuration import configurable
from coach_pl.criterion import build_criterion
from coach_pl.evaluator import build_evaluator
from coach_pl.model import build_model
from coach_pl.module import MODULE_REGISTRY
from coach_pl.utils.logging import setup_logger

logger = setup_logger(__name__, rank_zero_only=True)


@MODULE_REGISTRY.register()
class SelfSupervisedClusteringModule(LightningModule):
    """
    Training module for diffusion model.
    """

    @configurable
    def __init__(
        self,
        model: torch.nn.Module,
        criterion: torch.nn.Module,
        evaluator: torch.nn.Module,
        cfg: DictConfig,
    ) -> None:
        super().__init__()

        self.save_hyperparameters(cfg)

        self.model = torch.compile(model) if cfg.MODULE.COMPILE else model
        self.criterion = torch.compile(criterion) if cfg.MODULE.COMPILE else criterion
        self.evaluator = evaluator

        logger.info(self.model)

        self.batch_size = cfg.DATAMODULE.DATALOADER.TRAIN.BATCH_SIZE

        # Optimizer configuration
        self.base_lr = cfg.MODULE.OPTIMIZER.BASE_LR
        self.optimizer_name = cfg.MODULE.OPTIMIZER.NAME
        self.optimizer_params = {
            k.lower(): v
            for k, v in cfg.MODULE.OPTIMIZER.PARAMS.items()
        }

        # Scheduler configuration
        self.step_on_epochs = cfg.MODULE.SCHEDULER.STEP_ON_EPOCHS
        self.scheduler_name = cfg.MODULE.SCHEDULER.NAME
        self.scheduler_params = {
            k.lower(): v
            for k, v in cfg.MODULE.SCHEDULER.PARAMS.items()
        }
        self.scheduler_params["num_epochs"] = cfg.TRAINER.MAX_EPOCHS

    @classmethod
    def from_config(cls, cfg: DictConfig) -> dict[str, Any]:
        return {
            "model": build_model(cfg),
            "criterion": build_criterion(cfg),
            "evaluator": build_evaluator(cfg),
            "cfg": cfg,
        }

    def configure_optimizers(self) -> Any:
        total_batch_size = self.batch_size * self.trainer.accumulate_grad_batches * self.trainer.world_size
        logger.info(
            f"Total training batch size ({total_batch_size}) = single batch size ({self.batch_size}) * accumulate ({self.trainer.accumulate_grad_batches}) * world size ({self.trainer.world_size})"
        )

        base = 125
        lr = self.base_lr * total_batch_size / base
        logger.info(f"Learning rate ({lr:0.6g}) = base_lr ({self.base_lr:0.6g}) * total_batch_size ({total_batch_size}) / base")

        hyperparameters = {
            "lr": lr,
        }

        optimizer = create_optimizer_v2(
            model_or_params=self.model,
            opt=self.optimizer_name,
            lr=hyperparameters["lr"],
            **self.optimizer_params,
        )

        scheduler, _ = create_scheduler_v2(
            optimizer=optimizer,
            sched=self.scheduler_name,
            step_on_epochs=self.step_on_epochs,
            updates_per_epoch=int(len(self.trainer.datamodule.train_dataloader()) / (self.trainer.accumulate_grad_batches * self.trainer.world_size)),
            **self.scheduler_params,
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch" if self.step_on_epochs else "step",
                "frequency": 1,
            }
        }

    def lr_scheduler_step(self, scheduler: LRSchedulerTypeUnion, metric: Any | None) -> None:
        if self.step_on_epochs:
            scheduler.step(self.current_epoch, metric)
        else:
            scheduler.step_update(self.global_step, metric)

    def forward(self, image: torch.Tensor, label: torch.LongTensor, index: torch.LongTensor) -> dict[str, Any]:
        result = self.model(image, index)
        loss = self.criterion(result, image)
        return result, loss

    def training_step(self, batch: Any, batch_idx: int) -> torch.Tensor:
        image, label, index = batch
        result, loss = self.forward(image, label, index)

        self.log_dict({
            "train/" + k: v
            for k, v in loss.items()
        }, prog_bar=True, sync_dist=False, rank_zero_only=True)
        return loss

    def validation_step(self, batch: Any, batch_idx: int) -> None:
        image, label, index = batch
        result, loss = self.forward(image, label, index)

        self.evaluator.add(result, label)

        self.log_dict({
            "val/" + k: v
            for k, v in loss.items()
        }, sync_dist=False, rank_zero_only=True)

    def on_validation_epoch_end(self):
        metric = self.evaluator()

        self.log_dict({
            "val/" + k: v
            for k, v in metric.items()
            if "image" not in k
        }, sync_dist=False, rank_zero_only=True)

        self.logger.experiment.add_images("val/pca_space", metric["pca_image"], self.current_epoch)
        self.logger.experiment.add_images("val/umap_space", metric["umap_image"], self.current_epoch)

    def test_step(self, batch: Any, batch_idx: int) -> None:
        image, label, index = batch
        result, loss = self.forward(image, label, index)

        self.evaluator.add(result, label)

        self.log_dict({
            "test/" + k: v
            for k, v in loss.items()
        }, sync_dist=False, rank_zero_only=True)

    def on_test_epoch_end(self):
        metric = self.evaluator()

        self.log_dict({
            "test/" + k: v
            for k, v in metric.items()
            if "image" not in k
        }, sync_dist=False, rank_zero_only=True)

        output_dir = Path(self.trainer.default_root_dir) / "results"
        output_dir.mkdir(parents=True, exist_ok=True)

        Image.fromarray(metric["pca_image"][0].transpose(1, 2, 0)).save(output_dir / "pca.png")
        Image.fromarray(metric["umap_image"][0].transpose(1, 2, 0)).save(output_dir / "umap.png")

        for category in range(10):
            sample_path = output_dir / f"category_{category}.png"
            sample_image = self.random_sample(category)
            Image.fromarray(sample_image).save(sample_path)

    def random_sample(self, category: int) -> np.ndarray:
        prior_mean, prior_logvar = self.model.generator()
        D = prior_mean.shape[1]

        z = prior_mean[category] + torch.exp(0.5 * prior_logvar[category]) * torch.randn(64, D).to(prior_mean.device)
        x = self.model.decoder(z)

        x = x.cpu().numpy().reshape(8, 8, 28, 28).transpose(0, 2, 1, 3).reshape(8 * 28, 8 * 28) # [H, W]
        x = x * 255.0
        x = x.astype(np.uint8)

        return x

    def on_save_checkpoint(self, checkpoint: dict[str, Any]) -> None:
        model = self.model
        if hasattr(model, "_orig_mod"):
            model = model._orig_mod
        checkpoint["model"] = model.state_dict()
