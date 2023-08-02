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
    def __init__(self, input_dim: Optional[int] = None, output_dim: Optional[int] = None,
                 load_path: Optional[Union[Path, str]] = None, name: Optional[str] = None,
                 info: Optional[Dict[str, Any]] = None, **kwargs):
        super().__init__()

        # Load model if load_path is provided
        if input_dim is None or output_dim is None:
            assert load_path is not None, \
                "Either input_dim and output_dim must be provided or load_path must be provided."
            self.load(load_path)

        # Initialise new model if input_dim and output_dim are provided
        else:
            self._input_dim = input_dim
            self._output_dim = output_dim
            self._name = name if name is not None else self.__class__.__name__
            if info is None:
                info = {}
            self._info = info

        # Set representation of the model
        self._repr = f"{self.name}(input_dim={input_dim}, output_dim={output_dim}, info={info})"

        # Set default file name (used for saving and loading the model if a dir path is provided)
        self._default_file_name = f"{self.name.replace('[', '_').replace(']', '').replace(' ', '_')}.pt"

    def __repr__(self) -> str:
        return self._repr

    @property
    def name(self) -> str:
        return self._name

    @property
    def info(self) -> Dict[str, Any]:
        return self._info

    def save(self, path: Union[Path, str], additional_params: Optional[Dict[str, Any]] = None):
        if isinstance(path, str):
            path = Path(path)
        if path is None:
            dir_name = Path.cwd() / "trained_models"
            file_name = self._default_file_name
        else:
            dir_name = Path(path).parent
            file_name = Path(path).name
        # Create directory if it does not exist
        if not dir_name.exists():
            dir_name.mkdir(parents=True)

        torch.save({
            "model_state_dict": self.state_dict(),
            "info": self._info,
            "input_dim": self._input_dim,
            "output_dim": self._output_dim,
            "name": self.name,
            "additional_info": additional_params
        }, dir_name / file_name)

    def load(self, path: Union[Path, str]):

        # Convert path to Path object if it is a string
        if isinstance(path, str):
            path = Path(path)

        # If path is a directory, append the default file name
        if path.is_dir():
            path = path / self._default_file_name

        # Check if file exists
        if not path.exists():
            raise FileNotFoundError(f"File {path} does not exist.")

        checkpoint = torch.load(path)
        self._input_dim = checkpoint["input_dim"]
        self._output_dim = checkpoint["output_dim"]
        self._name = checkpoint["name"]
        self._info = checkpoint["info"]
        if checkpoint['additional_info'] is not None:
            for param_key, param_value in checkpoint['additional_info'].items():
                setattr(self, param_key, param_value)
        self._model_state_dict_temp = checkpoint["model_state_dict"]

    def _load_model_state_dict(self):
        assert self._model_state_dict_temp is not None, "this should be called after load()"
        self.load_state_dict(self._model_state_dict_temp)
        self.eval()


    @abc.abstractmethod
    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    @abc.abstractmethod
    def get_sales(self, inputs: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


class MLPLogBaseSalesModel(BaseSalesModel):
    def __init__(self, input_dim: Optional[int] = None, output_dim: Optional[int] = None, hidden_layers: Sequence[int] = (256, 512, 1024, 512, 256),  # (64, 256, 64),
                 activation: Optional[Activations] = "ReLU", noise_std: float = 2.0,
                 load_path: Optional[Union[Path, str]] = None,
                 info: Optional[Dict[str, Any]] = None, additional_name: Optional[str] = None):
        additional_name = "" if additional_name is None else f"[{additional_name}]"

        self._hidden_layers: Optional[Sequence[int]] = None
        self._activation: Optional[Activations] = None
        self._noise_std: Optional[float] = None
        self._dynamic_std: Optional[float] = None

        super().__init__(input_dim=input_dim, output_dim=output_dim, load_path=load_path,
                         name=f"MLPLogBaseSalesModel{additional_name}",
                         info=info)
        if self._hidden_layers is None:
            self._hidden_layers = hidden_layers

        if self._activation is None:
            self._activation = activation

        if self._noise_std is None:
            self._noise_std = noise_std

        self._layers = []
        in_dim = self._input_dim
        for layer_dim in self._hidden_layers:
            self._layers += [nn.LayerNorm(in_dim)]
            self._layers += [nn.Linear(in_features=in_dim, out_features=layer_dim)]
            in_dim = layer_dim
            if self._activation is not None:
                self._layers += [getattr(nn, self._activation)()]
        self._layers += [nn.Linear(in_features=in_dim, out_features=self._output_dim)]
        self._model = nn.Sequential(*self._layers)
        self._noises = torch.distributions.Normal(loc=0, scale=self._noise_std)

        self._repr = f"{self.name}(input_dim={self._input_dim}, output_dim={self._output_dim}, " \
                     f"hidden_layers={hidden_layers}, activation={getattr(nn, activation)().__class__.__name__}, " \
                     f"noise_std={noise_std})"

        if self._info is not None and "sales_normalising_factor" in self._info:
            self._sales_normalising_factor = self._info["sales_normalising_factor"]
        else:
            self._sales_normalising_factor = 1.0

        if self._dynamic_std is None:
            self._dynamic_std = noise_std
            self._step_count = 0

        if load_path is not None:
            self._load_model_state_dict()

    def save(self, path: Union[Path, str]):
        additional_params = {
            "_sales_normalising_factor": self._sales_normalising_factor,
            "_hidden_layers": self._hidden_layers,
            "._activation": self._activation,
            "_noise_std": self._noise_std,
            "_dynamic_std": self._dynamic_std,
            "_step_count": self._step_count
        }
        super().save(path, additional_params=additional_params)

    @property
    def dynamic_std(self):
        return self._dynamic_std

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        inputs = torch.atleast_2d(inputs)
        preds = self._model(inputs)

        if self.training:
            noises_dist = torch.distributions.Normal(loc=0, scale=self._dynamic_std)
            noises = noises_dist.sample(preds.shape).to(preds.device)

            self._step_count += 1
            if self._step_count > 10:
                self._dynamic_std = 5 / np.sqrt(self._step_count)  # np.exp(-(self._step_count+2))
            return torch.clip(preds.exp() + noises / self._sales_normalising_factor, min=1e-15).log()
        else:
            # noises = self._noises.sample(preds.shape).to(preds.device)
            # return torch.clip(preds.exp() + noises / self._sales_normalising_factor, min=1e-15).log()
            return preds

    def get_sales(self, inputs: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            sales = (self.forward(inputs).exp() - 1) * self._sales_normalising_factor
        return sales


class CombinedMLPLogBaseSalesModelForEachFlavour(BaseSalesModel):
    def __init__(self, input_dim, output_dim, hidden_layers: Sequence[int] = (64, 256, 64),
                 activation: Optional[Activations] = "ReLU", noise_std: float = 2.0,
                 info: Optional[Dict[str, Any]] = None):

        super().__init__(input_dim=input_dim, output_dim=output_dim, name=f"CombinedMLPLogBaseSalesModelForEachFlavour",
                         info=info)

    @property
    def dynamic_std(self):
        return self._dynamic_std

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        inputs = torch.atleast_2d(inputs)
        preds = self._model(inputs)

        if self.training:
            noises_dist = torch.distributions.Normal(loc=0, scale=self._dynamic_std)
            noises = noises_dist.sample(preds.shape).to(preds.device)

            self._step_count += 1
            if self._step_count > 10:
                self._dynamic_std = 5 / np.sqrt(self._step_count)  # np.exp(-(self._step_count+2))
            return torch.clip(preds.exp() + noises / self._sales_normalising_factor, min=1e-15).log()
        else:
            # noises = self._noises.sample(preds.shape).to(preds.device)
            # return torch.clip(preds.exp() + noises / self._sales_normalising_factor, min=1e-15).log()
            return preds

    def get_sales(self, inputs: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            sales = (self.forward(inputs).exp() - 1) * self._sales_normalising_factor
        return sales
