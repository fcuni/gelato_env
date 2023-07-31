import abc
from pathlib import Path
from typing import Optional, Sequence, Dict, Any, Union

import numpy as np
import pandas as pd
import torch
from torch import nn

from utils.enums import Flavour
from utils.types import Activations


class BaseSalesModel(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, name: Optional[str] = None,
                 info: Optional[Dict[str, Any]] = None, **kwargs):
        super().__init__()
        self._input_dim = input_dim
        self._output_dim = output_dim
        self._name = name if name is not None else self.__class__.__name__
        self._repr = f"{self.__class__.__name__}(input_dim={input_dim}, output_dim={output_dim})"
        if info is None:
            info = {}
        self._info = info

    def __repr__(self) -> str:
        return self._repr

    @property
    def name(self) -> str:
        return self._name

    @property
    def info(self) -> Dict[str, Any]:
        return self._info

    def save(self, path: Union[Path, str]):
        if isinstance(path, str):
            path = Path(path)
        if path is None:
            path = Path.cwd() / "trained_models"
        # Create directory if it does not exist
        if not path.exists():
            path.mkdir(parents=True)

        torch.save(self.state_dict(), path / f"{self.name}.pt")

    def load(self, path: Union[Path, str]):
        if isinstance(path, str):
            path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"File {path} does not exist.")
        self.load_state_dict(torch.load(path))
        self.eval()

    @abc.abstractmethod
    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    @abc.abstractmethod
    def get_sales(self, inputs: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


class MLPLogBaseSalesModel(BaseSalesModel):
    def __init__(self, input_dim, output_dim, hidden_layers: Sequence[int] = (64, 128, 64),
                 activation: Optional[Activations] = "ReLU", noise_std: float = 2.0,
                 info: Optional[Dict[str, Any]] = None):
        super().__init__(input_dim=input_dim, output_dim=output_dim, name="MLPLogBaseSalesModel", info=info)
        self._layers = []
        in_dim = input_dim
        for layer_dim in hidden_layers:
            self._layers += [nn.LayerNorm(in_dim)]
            self._layers += [nn.Linear(in_features=in_dim, out_features=layer_dim)]
            in_dim = layer_dim
            if activation is not None:
                self._layers += [getattr(nn, activation)()]
        self._layers += [nn.Linear(in_features=in_dim, out_features=output_dim)]
        self._model = nn.Sequential(*self._layers)
        self._noises = torch.distributions.Normal(loc=0, scale=noise_std)
        self._hidden_layers = hidden_layers
        self._repr = f"{self.__class__.__name__}(input_dim={self._input_dim}, output_dim={self._output_dim}, " \
                     f"hidden_layers={hidden_layers}, activation={getattr(nn, activation)().__class__.__name__}, " \
                     f"noise_std={noise_std})"

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        inputs = torch.atleast_2d(inputs)
        preds = self._model(inputs)
        noises = self._noises.sample(preds.shape).to(preds.device)
        return torch.clip(preds.exp() + noises, min=1e-15).log()

    def get_sales(self, inputs: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            sales = self.forward(inputs).exp()
        return sales


