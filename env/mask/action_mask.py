from abc import abstractmethod
from typing import Union
import torch
from env.gelateria import GelateriaState


class ActionMask:

    def __init__(self, name: str):
        self._name = name

    @property
    def name(self):
        return self._name

    @abstractmethod
    def __call__(self, state: Union[GelateriaState, torch.Tensor]):
        raise NotImplementedError