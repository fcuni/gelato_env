from typing import Union, Optional
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

    def __call__(self, state: Union[GelateriaState, torch.Tensor], output_dtype: Optional[type] = None) -> np.ndarray:

        if output_dtype is None:
            output_dtype = bool

        if isinstance(state, GelateriaState):
            mask = np.ones((len(state.products), 101))
            for idx, markdown in enumerate(state.current_markdowns.values()):
                mask[idx, :int(round(markdown * 100))] = 0
        else:
            mask = np.ones((state.shape[0], 101))
            for idx, markdown in enumerate(state[:, 0]):
                mask[idx, :int(round((markdown * 100).item()))] = 0
       
        return mask.squeeze().astype(output_dtype)


class NoRestrictionBooleanMask(ActionMask):
    # True: keep, False: mask
    def __init__(self):
        super().__init__(name="NoRestrictionBooleanMask")

    def __call__(self, state: Union[GelateriaState, torch.Tensor], output_dtype: Optional[type] = None) -> np.ndarray:

        if output_dtype is None:
            output_dtype = bool

        if isinstance(state, GelateriaState):
            mask = np.ones((len(state.products), 101))
        else:
            mask = np.ones((state.shape[0], 101))

        return mask.squeeze().astype(output_dtype)


class OnlyCurrentActionBooleanMask(ActionMask):

    def __init__(self):
        super().__init__(name="OnlyCurrentActionBooleanMask")

    def __call__(self, state: Union[GelateriaState, torch.Tensor], output_dtype: Optional[type] = None) -> np.ndarray:
        """
        Args:
            state: GelateriaState or torch.Tensor
            output_as_int: if True, return the mask as an int array. Otherwise, return a boolean array.
        """

        if output_dtype is None:
            output_dtype = bool

        if isinstance(state, GelateriaState):
            mask = np.zeros((len(state.products), 101))
            for idx, markdown in enumerate(state.current_markdowns.values()):
                mask[idx, int(round(markdown * 100))] = 1
        else:
            mask = np.zeros((state.shape[0], 101))
            for idx, markdown in enumerate(state[:, 0]):
                mask[idx, int(round((markdown * 100).item()))] = 1

        return mask.squeeze().astype(output_dtype)

