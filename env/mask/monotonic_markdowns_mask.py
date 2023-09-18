from typing import Union, Optional

import numpy as np
import torch

from env.gelateria import GelateriaState
from env.mask.action_mask import ActionMask


class MonotonicMarkdownsMask(ActionMask):

    def __init__(self):
        super().__init__(name="MonotonicMarkdownMask")

    def __call__(self, state: GelateriaState):
        mask = np.zeros((len(state.products), 101))
        for idx, markdown in enumerate(state.current_markdowns.values()):
            mask[idx, :int(markdown * 100)] = -np.inf
        return mask.squeeze()


class MonotonicMarkdownsBooleanMask(ActionMask):
    # True: keep, False: mask
    def __init__(self):
        super().__init__(name="MonotonicMarkdownsBooleanMask")

    def __call__(self, state: Union[GelateriaState, torch.Tensor], current_dates: Optional[np.ndarray] = None,
                 output_dtype: Optional[type] = None) -> np.ndarray:

        if output_dtype is None:
            output_dtype = bool

        if isinstance(state, GelateriaState):
            mask = np.ones((len(state.products), 101))
            for idx, markdown in enumerate(state.current_markdowns.values()):
                mask[idx, :int(round(markdown * 100))] = 0
        else:
            mask = np.ones((state.shape[0], 101))
            for idx, markdown in enumerate(state[:, 0]):
                markdown = np.clip(markdown, a_min=0.0, a_max=1.0)
                mask[idx, :int(round((markdown * 100).item()))] = 0

        return np.atleast_2d(mask.squeeze().astype(output_dtype))
