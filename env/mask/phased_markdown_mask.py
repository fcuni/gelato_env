from datetime import datetime
from typing import Tuple, Union, Optional

import pandas as pd
import torch
import numpy as np

from env.gelateria import GelateriaState
from env.mask.action_mask import ActionMask


class PhasedMarkdownMask(ActionMask):
    def __init__(self, markdown_schedule: pd.DataFrame):
        markdown_schedule['start_date'] = pd.to_datetime(markdown_schedule['start_date'])
        markdown_schedule['end_date'] = pd.to_datetime(markdown_schedule['end_date'])
        super().__init__(name=f"PhasedMarkdown(num_phases={len(markdown_schedule)}, "
                              f"from={markdown_schedule['start_date'].min().strftime('%Y-%m-%d')}, "
                              f"to={markdown_schedule['end_date'].max().strftime('%Y-%m-%d')})")
        self.markdown_schedule = markdown_schedule

    def __call__(self, state: Union[GelateriaState, torch.Tensor]) -> np.ndarray:
        """
        Simple logic that masks all reductions below the current markdown and outside the
        markdown range.

        Args:
            state (`GelateriaState`/`torch.Tensor`): Current state of the environment.

        Returns:
            Mask with the logic applied.
        """

        if isinstance(state, torch.Tensor):
            raise NotImplementedError("PhasedMarkdown mask is not implemented for torch.Tensor")

        current_date = state.current_date.date()
        markdown_restriction = self.markdown_schedule[(self.markdown_schedule['start_date'].dt.date <= current_date) &
                                                      (self.markdown_schedule['end_date'].dt.date >= current_date)]

        if len(markdown_restriction) == 0:
            raise ValueError(f"No markdown restriction found for date {current_date}")

        lower_bound = markdown_restriction['lowest_markdown'].iloc[0]
        upper_bound = markdown_restriction['highest_markdown'].iloc[0]

        mask = np.ones((len(state.products), 101))
        for idx, markdown in enumerate(state.current_markdowns.values()):
            lower_bound_per_product = round(max(lower_bound, markdown), 2)
            # block the markdowns that are smaller than the current markdown and lower_bound
            mask[idx, :int(round(lower_bound_per_product * 100))] = 0
            # block the markdowns that are greater than the upper_bound
            mask[idx, int(round(upper_bound * 100))+1:] = 0
            # ensure the current markdown is not blocked
            if int(round(markdown * 100)) >= int(round(upper_bound * 100)):
                mask[idx, int(round(markdown * 100))] = 1
        # else: # ABANDONED LOGIC for state in torch.Tensor format
        #     mask = np.ones((state.shape[0], 101))
        #     for idx, markdown in enumerate(state[:, 3]):
        #         lower_bound_per_product = round(max(lower_bound, markdown))
        #         # block the markdowns that are smaller than the current markdown and lower_bound
        #         mask[idx, :int(round((lower_bound_per_product * 100).item()))] = 0
        #         # block the markdowns that are greater than the upper_bound
        #         mask[idx, int(round((upper_bound * 100).item())):] = 0
        #         # ensure the current markdown is not blocked
        #         mask[idx, int(round((markdown * 100).item()))] = 1

        return np.atleast_2d(mask.squeeze().astype(bool))
