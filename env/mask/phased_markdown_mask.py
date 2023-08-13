from typing import Tuple, Union, Optional
from datetime import datetime
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

    def __call__(self, state: Union[GelateriaState, torch.Tensor], current_dates: Optional[np.ndarray] = None,
                 output_dtype: Optional[type] = None) -> np.ndarray:
        """
        Simple logic that masks all reductions below the current markdown and outside the
        markdown range.

        Args:
            state (`GelateriaState`/`torch.Tensor`): Current state of the environment.
            current_dates (`numpy.ndarray`): Current dates corresponding to states. Only needed when state is a tensor.
            output_dtype (`type`): Output data type of the mask.

        Returns:
            Mask with the logic applied.
        """

        if output_dtype is None:
            output_dtype = bool

        if isinstance(state, GelateriaState):
            current_date = state.current_date.date()
            current_markdowns = state.current_markdowns.values()
            mask = np.ones((len(state.products), 101))
            markdown_restriction = self.markdown_schedule[
                (self.markdown_schedule['start_date'].dt.date <= current_date) &
                (self.markdown_schedule['end_date'].dt.date >= current_date)]

            if len(markdown_restriction) == 0:
                raise ValueError(f"No markdown restriction found for date {current_date}")

            lower_bounds = [markdown_restriction['lowest_markdown'].iloc[0]] * state.n_products
            upper_bounds = [markdown_restriction['highest_markdown'].iloc[0]] * state.n_products

        else:
            assert current_dates is not None, "current_date must be provided when state is a tensor"
            current_date_list = [date.date() for date in current_dates]
            lower_bounds, upper_bounds = [], []
            for date in current_date_list:
                markdown_restriction = self.markdown_schedule[
                    (self.markdown_schedule['start_date'].dt.date <= date) &
                    (self.markdown_schedule['end_date'].dt.date >= date)]

                if len(markdown_restriction) == 0:
                    raise ValueError(f"No markdown restriction found for date {date}")

                lower_bound = markdown_restriction['lowest_markdown'].iloc[0]
                upper_bound = markdown_restriction['highest_markdown'].iloc[0]

                lower_bounds.append(lower_bound)
                upper_bounds.append(upper_bound)

            if isinstance(state, torch.Tensor):
                current_markdowns = torch.clip(state[:, 0], min=0.0, max=1.0).detach().cpu().numpy()
            else:
                current_markdowns = np.clip(state[:, 0], a_min=0.0, a_max=1.0)
            mask = np.ones((state.shape[0], 101))

        for idx, markdown in enumerate(current_markdowns):
            lower_bound_per_product = round(max(lower_bounds[idx], markdown), 2)
            # block the markdowns that are smaller than the current markdown and lower_bound
            mask[idx, :int(round(lower_bound_per_product * 100))] = 0
            # block the markdowns that are greater than the upper_bound
            mask[idx, int(round(upper_bounds[idx] * 100)) + 1:] = 0
            # ensure the current markdown is not blocked
            if int(round(markdown * 100)) >= int(round(upper_bounds[idx] * 100)):
                mask[idx, int(round(markdown * 100))] = 1

        return np.atleast_2d(mask.squeeze().astype(output_dtype))
