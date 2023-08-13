from typing import Union, Optional

import numpy as np
import torch

from env.gelateria import GelateriaState
from env.mask.action_mask import ActionMask


class CurrentActionOnlyBooleanMask(ActionMask):

    def __init__(self):
        super().__init__(name="CurrentActionOnlyBooleanMask")

    def __call__(self, state: Union[GelateriaState, torch.Tensor], current_dates: Optional[np.ndarray] = None,
                 output_dtype: Optional[type] = None) -> np.ndarray:
        """
        Args:
            state: GelateriaState or torch.Tensor
            current_dates: np.ndarray of datetime objects (not used for this mask, default None)
            output_dtype: type of the output array (default None: bool)
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
                markdown = torch.clip(markdown, min=0.0, max=1.0)
                mask[idx, int(round((markdown * 100).item()))] = 1

        return mask.squeeze().astype(output_dtype)
