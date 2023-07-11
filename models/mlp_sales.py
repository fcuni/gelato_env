from pathlib import Path
from typing import List, Optional, Literal, Any
import os

import numpy as np
import torch
import torch.nn as nn

import pytorch_lightning as pl
from torch.optim.lr_scheduler import ReduceLROnPlateau

from utils.config import NetConfig

Activations = Literal[tuple(Literal[act] for act in nn.modules.activation.__all__)]


class SalesMLPBlock(nn.Module):

    def __init__(self,
                 input_dim: int,
                 hidden_dims: List[int],
                 activation: Optional[Activations] = "ReLU"
                 ):
        super().__init__()
        self._model = self._build_model(input_dim, hidden_dims, activation)

    @staticmethod
    def _build_model(input_dim, hidden_dims, activation):
        layers = []
        in_dim = input_dim
        for dim in hidden_dims:
            layers += [nn.LayerNorm(in_dim)]
            layers += [nn.Linear(in_features=in_dim, out_features=dim)]
            in_dim = dim
            if activation is not None:
                layers += [getattr(nn, activation)()]

        layers += [nn.Linear(in_features=in_dim, out_features=1)]

        return nn.Sequential(*layers)

    @staticmethod
    def compute_loss(preds: torch.Tensor, labels: torch.Tensor):
        return {"sales": nn.MSELoss()(preds, labels)}

    def forward(self, input: torch.Tensor):
        return self._model(input)

    def load(self, path: Path):
        self.load_state_dict(torch.load(path / "mlp_sales_model.pt"))

    def save(self, path: Path):
        os.makedirs(path, exist_ok=True)
        torch.save(self.state_dict(), path / "mlp_sales_model.pt")


class MLPLogSalesModel(pl.LightningModule):

    def __init__(self, input_dim: int, name: str, config: NetConfig):
        super().__init__()
        self._config = config
        self._name = name
        self._optim = config.optim
        self._model = SalesMLPBlock(input_dim, config.embedding_dims, config.activation)

        self.save_hyperparameters()

    @property
    def name(self):
        return self._name

    @property
    def config(self):
        return self._config

    def load(self):
        path = self.config.path_to_model / self.name
        self._model.load(path)

    def save(self):
        path = self.config.path_to_model / self.name
        self._model.save(path)

    def get_sales(self, input: torch.Tensor,):
        return np.exp(self._model(input).detach().cpu()) - 1

    def forward(self, input: torch.Tensor):
        return self._model(input)

    def configure_optimizers(self) -> Any:
        optimiser = self._optim(self.parameters(), lr=self._config.lr)

        if self._config.lr_scheduler is not None:
            lr_scheduler_fn = self._config.lr_scheduler
            config = self._config.scheduler_config if self._config.scheduler_config is not None else {}
            lr_scheduler = lr_scheduler_fn(optimiser, **config)

            if isinstance(lr_scheduler, ReduceLROnPlateau):
                return {"optimizer": optimiser,
                        "lr_scheduler": lr_scheduler,
                        "monitor": "mean_loss_train"}
            else:
                return [optimiser], [lr_scheduler]
        return [optimiser]

    def training_step(self, batch, batch_idx,
                      step_mode: str = "train"
                      ):
        x = batch[self._config.input_key]
        preds = self(x)
        step_output = preds

        if step_mode != "test":
            labels = batch[self._config.target]
            losses_dict = self._model.compute_loss(preds=preds, labels=labels)
            losses = []
            for name, loss in losses_dict.items():
                losses += [loss]
                self.log(f"{name}_{step_mode}", loss, on_epoch=True, prog_bar=True)

            step_output = torch.mean(torch.stack(losses), dim=0)
            self.log(f"mean_loss_{step_mode}", step_output, on_epoch=True, prog_bar=True)

        return step_output

    def validation_step(self, batch, batch_idx):
        return self.training_step(batch, batch_idx, step_mode="val")

    def test_step(self, batch, batch_idx):
        return self.training_step(batch, batch_idx, step_mode="test")
