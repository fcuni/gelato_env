from pathlib import Path
from typing import List, Optional, Literal, Any
import os

import numpy as np
import torch
import torch.nn as nn

from torch.optim.lr_scheduler import ReduceLROnPlateau

from utils.config import NetConfig

def get_simple_prediction(inventory:int, markdown:float):
    return min(inventory,round(2+markdown * (inventory-2)))



class SimpleLinearSalesModel(nn.Module):

    def __init__(self, input_dim: int, name: str, config: NetConfig):
        super().__init__()
        self._config = config
        self._name = name
        self._optim = config.optim
        self._model = self.get_simple_prediction

    def get_simple_prediction(self, inventory:torch.Tensor, markdown:torch.Tensor):
        
        return torch.min(inventory,torch.round(2+markdown * (inventory-2)))
    
    @property
    def name(self):
        return self._name

    @property
    def config(self):
        return self._config

    def load(self):
        # path = self.config.path_to_model / self.name
        # self._model.load(path)
        pass

    def save(self):
        # path = self.config.path_to_model / self.name
        # self._model.save(path)
        pass

    def get_sales(self, input: torch.Tensor,):
        return self.forward(input)

    def forward(self, input: torch.Tensor):
        inventory = input[:,1]
        markdown = input[:,3]
        return self._model(inventory, markdown)


