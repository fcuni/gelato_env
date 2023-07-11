from typing import Union
import numpy as np
import torch

from env.gelateria import GelateriaState
from env.mask.action_mask import ActionMask


class IdentityMask(ActionMask):

    def __init__(self):
        super().__init__(name="IdentityMask")

    def __call__(self, state: GelateriaState):
        return np.zeros((len(state.products), 101))


class MonotonicMarkdownsMask(ActionMask):

    def __init__(self):
        super().__init__(name="MonotonicMarkdownMask")

    def __call__(self, state: GelateriaState):
        mask = np.zeros((len(state.products), 101))
        for idx, markdown in enumerate(state.current_markdowns.values()):
            mask[idx, :int(markdown * 100)] = -np.inf
        return mask.squeeze()


class BooleanMonotonicMarkdownsMask(ActionMask):
    # True: keep, False: mask
    def __init__(self):
        super().__init__(name="BooleanMonotonicMarkdownMask")

    def __call__(self, state: Union[GelateriaState, torch.Tensor])->np.ndarray:
        if isinstance(state, GelateriaState):
            mask = np.ones((len(state.products), 101))
            for idx, markdown in enumerate(state.current_markdowns.values()):
                mask[idx, :int(markdown * 100)] = 0
        else:
            mask = np.ones((state.shape[0], 101))
            for idx, markdown in enumerate(state[:, 3]):
                mask[idx, :int(markdown * 100)] = 0
       
        return mask.squeeze().astype(bool)