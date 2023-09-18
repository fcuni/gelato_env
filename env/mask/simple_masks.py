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


class IdentityBooleanMask(ActionMask):
    # True: keep, False: mask
    def __init__(self):
        super().__init__(name="IdentityBooleanMask")

    def __call__(self, state: Union[GelateriaState, torch.Tensor], current_dates: Optional[np.ndarray] = None,
                 output_dtype: Optional[type] = None) -> np.ndarray:

        if output_dtype is None:
            output_dtype = bool

        if isinstance(state, GelateriaState):
            mask = np.ones((len(state.products), 101))
        else:
            mask = np.ones((state.shape[0], 101))

        return mask.squeeze().astype(output_dtype)
