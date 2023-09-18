import abc
from pathlib import Path
from typing import Optional, Sequence, Dict, Any, Union, Callable

import numpy as np
import torch
from torch import nn

from utils.enums import Flavour
from utils.types import Activations


class BaseSalesModel(nn.Module):
    def __init__(self, input_dim: Optional[int] = None, output_dim: Optional[int] = None,
                 load_from: Optional[Union[Path, str]] = None, name: Optional[str] = None,
                 info: Optional[Dict[str, Any]] = None, **kwargs):
        super().__init__()

        # Load model if load_from is provided
        if load_from is not None:
            self.load(load_from)

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

    @property
    def input_dim(self) -> int:
        return self._input_dim

    @property
    def output_dim(self) -> int:
        return self._output_dim

    # def save(self, path: Union[Path, str], additional_params: Optional[Dict[str, Any]] = None):
    def save(self, path: Optional[Union[Path, str]] = None, save_type: str = "file",
             additional_params: Optional[Dict[str, Any]] = None) -> Union[Path, Dict[str, Any]]:

        model_info = {
            "model_state_dict": self.state_dict(),
            "info": self._info,
            "input_dim": self._input_dim,
            "output_dim": self._output_dim,
            "name": self.name,
            "additional_params": additional_params
        }

        if save_type == "dict":
            return model_info

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

        torch.save(model_info, dir_name / file_name)

        return dir_name / file_name

    def load(self, load_from: Union[Path, str, dict]):

        if isinstance(load_from, dict):
            checkpoint = load_from
        else:
            # Convert path to Path object if it is a string
            if isinstance(load_from, str):
                load_from = Path(load_from)

            # If path is a directory, append the default file name
            if load_from.is_dir():
                load_from = load_from / self._default_file_name

            # Check if file exists
            if not load_from.exists():
                raise FileNotFoundError(f"File {load_from} does not exist.")

            checkpoint = torch.load(load_from)

        self._input_dim = checkpoint["input_dim"]
        self._output_dim = checkpoint["output_dim"]
        self._name = checkpoint["name"]
        self._info = checkpoint["info"]
        if checkpoint['additional_params'] is not None:
            for param_key, param_value in checkpoint['additional_params'].items():
                setattr(self, param_key, param_value)
        self._model_state_dict_temp = checkpoint["model_state_dict"]

    def _load_model_state_dict(self):
        assert self._model_state_dict_temp is not None, "this should be called after load()"
        self.load_state_dict(self._model_state_dict_temp)
        self.eval()


    @abc.abstractmethod
    def forward(self, inputs: torch.Tensor, input_transform_fn: Optional[Callable] = None) -> torch.Tensor:
        raise NotImplementedError

    @abc.abstractmethod
    def get_sales(self, inputs: torch.Tensor, input_transform_fn: Optional[Callable] = None) -> torch.Tensor:
        raise NotImplementedError


class MLPLogBaseSalesModel(BaseSalesModel):
    def __init__(self, input_dim: Optional[int] = None, output_dim: Optional[int] = None, hidden_layers: Sequence[int] = (256, 512, 1024, 512, 256),  # (64, 256, 64),
                 activation: Optional[Activations] = "ReLU", noise_std: float = 2.0,
                 load_from: Optional[Union[Path, str]] = None,
                 info: Optional[Dict[str, Any]] = None, additional_name: Optional[str] = None):
        additional_name = "" if additional_name is None else f"[{additional_name}]"

        self._hidden_layers: Optional[Sequence[int]] = None
        self._activation: Optional[Activations] = None
        self._noise_std: Optional[float] = None
        self._dynamic_std: Optional[float] = None

        super().__init__(input_dim=input_dim, output_dim=output_dim, load_from=load_from,
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
                     f"hidden_layers={self._hidden_layers}, activation={getattr(nn, activation)().__class__.__name__}, " \
                     f"noise_std={noise_std})"

        if self._info is not None and "sales_normalising_factor" in self._info:
            self._sales_normalising_factor = self._info["sales_normalising_factor"]
        else:
            self._sales_normalising_factor = 1.0

        if self._dynamic_std is None:
            self._dynamic_std = noise_std
            self._step_count = 0

        if load_from is not None:
            self._load_model_state_dict()

    def save(self, path: Optional[Union[Path, str]] = None, save_type: str = "file",
             additional_params: Optional[Dict[str, Any]] = None) -> Union[Path, Dict[str, Any]]:
        additional_params = {
            "_sales_normalising_factor": self._sales_normalising_factor,
            "_hidden_layers": self._hidden_layers,
            "._activation": self._activation,
            "_noise_std": self._noise_std,
            "_dynamic_std": self._dynamic_std,
            "_step_count": self._step_count
        }
        saved = super().save(path, save_type=save_type, additional_params=additional_params)
        return saved

    @property
    def dynamic_std(self):
        return self._dynamic_std

    def forward(self, inputs: torch.Tensor, input_transform_fn: Optional[Callable] = None) -> torch.Tensor:
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
            if input_transform_fn is not None:
                inputs = input_transform_fn(inputs, info=self._info)
            # noises = self._noises.sample(preds.shape).to(preds.device)
            # return torch.clip(preds.exp() + noises / self._sales_normalising_factor, min=1e-15).log()
            return preds

    def get_sales(self, inputs: torch.Tensor, input_transform_fn: Optional[Callable] = None) -> torch.Tensor:
        with torch.no_grad():
            if input_transform_fn is not None:
                inputs = input_transform_fn(inputs, info=self._info)
            sales = (self.forward(inputs).exp() - 1) * self._sales_normalising_factor
        return sales


class CombinedMLPLogBaseSalesModel(BaseSalesModel):
    def __init__(self, load_from_dir: Optional[Union[Path, str]] = None,
                 load_from_file: Optional[Union[Path, str]] = None, info: Optional[Dict[str, Any]] = None,
                 model_type: Optional[str] = None,
                 device: Optional[torch.device] = None):

        assert (load_from_dir is not None) != (load_from_file is not None), \
            "Either load_from_dir or load_from_file must be provided (not both)."

        super().__init__(name=f"CombinedMLPLogBaseSalesModel", info=info, device=device)

        # Initialise attributes
        self._models: Optional[Dict[str, BaseSalesModel]] = None

        # Load models of each flavour from directory
        if load_from_dir is not None:
            if model_type is None:
                self._flavour_model_type = "MLPLogBaseSalesModel"
            else:
                self._flavour_model_type = model_type
            if isinstance(load_from_dir, str):
                load_from_dir = Path(load_from_dir)
            self._models = {}
            models_input_dim, models_output_dim = [], []
            for flavour in Flavour.get_all_flavours():
                # File path of the model is defined by the model type and flavour name
                # (with spaces replaced by underscores, e.g. MLPLogBaseSalesModel_<flavour_name>.pt)
                file_path = load_from_dir / f"{self._flavour_model_type}_{flavour.replace(' ', '_')}.pt"
                # Load model using defined model type (model class name must be defined in this file)
                self._models[flavour] = globals()[self._flavour_model_type](load_from=file_path)
                models_input_dim.append(self._models[flavour].input_dim)
                models_output_dim.append(self._models[flavour].output_dim)
                if device is not None:
                    self._models[flavour].to(self._device)
            assert len(set(models_input_dim)) == 1, "All models must have the same input dimension"
            assert len(set(models_output_dim)) == 1, "All models must have the same output dimension"
            self._input_dim = models_input_dim[0]
            self._output_dim = models_output_dim[0]
            self._repr = f"{self.name}(input_dim={self._input_dim}, output_dim={self._output_dim}, " \
                         f"num_models={len(self._models)})"
        else:
            self.load(load_from_file)

        self._info = {**self._info, "model_info": self.model_info()}

    def save(self, path: Optional[Union[Path, str]] = None, save_type: str = "file",
             additional_params: Optional[Dict[str, Any]] = None) -> Union[Path, Dict[str, Any]]:
        # Construct a dictionary of models
        model_dict = {}
        for flavour, model in self._models.items():
            model_dict[flavour] = {
                "model_type": model.__class__.name,
                "model_data": model.save(save_type="dict")
            }
        # Save the model
        save_data = {
            "model_dict": model_dict,
            "info": self._info,
            "name": self.name,
            "input_dim": self._input_dim,
            "output_dim": self._output_dim,
        }

        # == Return the save data as a dictionary ==
        if save_type == "dict":
            return save_data

        # == Save the model as file ==
        if isinstance(path, str):
            path = Path(path)
        if path is None:
            dir_name = Path("experiment_data/trained_models/base_sales_models")
            file_name = self._default_file_name
        else:
            dir_name = Path(path).parent
            file_name = Path(path).name
        # Create directory if it does not exist
        if not dir_name.exists():
            dir_name.mkdir(parents=True)
        # Save the model
        torch.save(save_data, dir_name / file_name)
        # Return the path to the saved file
        return dir_name / file_name

    def load(self):
        raise NotImplementedError

    def _initiate_model(self):
        pass

    @property
    def model_type(self):
        return self._flavour_model_type

    def model_info(self, flavour: Optional[str] = None) -> Union[Dict[str, Any], Dict[str, Dict[str, Any]]]:
        if flavour is None:
            return {flav: self._models[flav].info for flav in Flavour.get_all_flavours()}
        else:
            return self._models[flavour].info

    def forward(self, inputs: torch.Tensor, input_transform_fn: Optional[Callable] = None) -> torch.Tensor:
        inputs = torch.atleast_2d(inputs)
        flavour_one_hot_encoding_size = len(Flavour.get_all_flavours())
        flavours = Flavour.get_flavour_from_one_hot_encoding(inputs[:, -flavour_one_hot_encoding_size:].cpu().numpy())
        inputs_without_encoding = inputs[:, :-flavour_one_hot_encoding_size]
        assert len(flavours) == inputs_without_encoding.shape[0], \
            "Number of flavours must be equal to the number of rows in `inputs`."

        # This model is not trained standalone, so it is always in eval mode
        with torch.no_grad():
            preds = []
            for flavour, input_row in zip(flavours, inputs_without_encoding):
                if input_transform_fn is not None:
                    input_row = input_transform_fn(input_row, info=self.model_info(flavour.value))
                # Get the prediction from the corresponding model
                pred = self._models[flavour.value](input_row).detach().cpu()
                preds.append(pred)
            preds = torch.vstack(preds)
        return preds

    def get_sales(self, inputs: torch.Tensor, input_transform_fn: Optional[Callable] = None) -> torch.Tensor:
        inputs = torch.atleast_2d(inputs)
        flavour_one_hot_encoding_size = len(Flavour.get_all_flavours())
        flavours = Flavour.get_flavour_from_one_hot_encoding(inputs[:, -flavour_one_hot_encoding_size:].cpu().numpy())

        assert len(flavours) == inputs.shape[0], "Number of flavours must be equal to the number of rows in `inputs`."

        # This model is not trained standalone, so it is always in eval mode
        with torch.no_grad():
            sales = []
            for flavour, input_row in zip(flavours, inputs):
                if input_transform_fn is not None:
                    input_row = input_transform_fn(input_row, info=self.model_info(flavour.value))
                inputs_without_encoding = input_row[:, :-flavour_one_hot_encoding_size]
                # Get the prediction from the corresponding model
                single_sales = self._models[flavour.value].get_sales(inputs_without_encoding).detach().cpu()
                sales.append(single_sales)
            sales = torch.vstack(sales)
        return sales
