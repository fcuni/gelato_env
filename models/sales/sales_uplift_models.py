from typing import Union, List, Sequence, Optional
import abc

import numpy as np
import torch
from torch.distributions.gamma import Gamma


class SalesUpliftModel:
    """Base class for sales uplift model"""

    def __init__(self, name: Optional[str] = None, *args, **kwargs):
        self._name = self.__class__.__name__ if name is None else name
        self._repr = f"{self._name}()"

    @property
    def name(self):
        """Name of the model"""
        return self._name

    def __repr__(self):
        """Representation of the model"""
        return self._repr

    @abc.abstractmethod
    def __call__(self, markdown: float):
        raise NotImplementedError

    @abc.abstractmethod
    def get_uplift(self, markdown: float) -> float:
        raise NotImplementedError


class GammaSalesUpliftModel(SalesUpliftModel):
    def __init__(self, rate: float = 1.2):
        """
        Args:
            rate: rate parameter of the gamma distribution
        """
        super().__init__()
        self._rate = rate
        self._repr = f"{self.name}(rate={rate})"

    def __call__(self, markdown: Union[float, Sequence[float]]) -> List[float]:
        """
        Return uplift values given markdown values
        Args:
            markdown: markdown value (between 0 and 1)
        Returns:
            List of uplift values
        """
        if isinstance(markdown, float):
            return self.get_uplift(markdown)
        elif isinstance(markdown, torch.Tensor):
            return [self.get_uplift(m.item()) for m in markdown]
        else:
            return [self.get_uplift(m) for m in markdown]

    def get_uplift(self, markdown: float) -> float:
        if round(markdown, 2) == 0.0:
            return 1.0
        elif round(markdown, 2) == 1.0:
            return np.inf
        return Gamma(concentration=-np.log(1-markdown).item(), rate=self._rate).sample().item()+1
