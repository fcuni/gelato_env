from datetime import datetime
from typing import Tuple, Union

import torch
import numpy as np

from env.gelateria import GelateriaState
from env.mask.action_mask import ActionMask


class PhasedMarkdown(ActionMask):
    def __init__(
            self,
            markdown_range: Tuple[float, float]
    ):
        super().__init__(name=f"PhasedMarkdown({markdown_range[0]}, {markdown_range[1]})")
        self.markdown_range = markdown_range

    def __call__(self, state: Union[GelateriaState, torch.Tensor]) -> np.ndarray:
        """
        Simple logic that masks all reductions below the current markdown and outside the
        markdown range.

        Args:
            state (`GelateriaState`/`torch.Tensor`): Current state of the environment.

        Returns:
            Mask with the logic applied.
        """

        lower_bound, upper_bound = self.markdown_range

        if isinstance(state, GelateriaState):
            mask = np.ones((len(state.products), 101))
            for idx, markdown in enumerate(state.current_markdowns.values()):
                lower_bound_per_product = round(max(lower_bound, markdown))
                # block the markdowns that are smaller than the current markdown and lower_bound
                mask[idx, :int(round(lower_bound_per_product * 100))] = 0
                # block the markdowns that are greater than the upper_bound
                mask[idx, int(round(upper_bound * 100)):] = 0
                # ensure the current markdown is not blocked
                mask[idx, int(round(markdown * 100))] = 1
        else:
            mask = np.ones((state.shape[0], 101))
            for idx, markdown in enumerate(state[:, 3]):
                lower_bound_per_product = round(max(lower_bound, markdown))
                # block the markdowns that are smaller than the current markdown and lower_bound
                mask[idx, :int(round((lower_bound_per_product * 100).item()))] = 0
                # block the markdowns that are greater than the upper_bound
                mask[idx, int(round((upper_bound * 100).item())):] = 0
                # ensure the current markdown is not blocked
                mask[idx, int(round((markdown * 100).item()))] = 1

        return mask.squeeze().astype(bool)
