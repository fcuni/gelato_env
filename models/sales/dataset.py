from typing import Optional, Dict, Any, List

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from utils.enums import Flavour


class BaseSalesDataset(Dataset):
    """Dataset for base sales model."""

    def __init__(self, data: pd.DataFrame, target_name: str = "sales", info: Optional[Dict[str, Any]] = None):
        """
        Args:
            data (pd.DataFrame): DataFrame containing the data
            target_name (str): Name of the target column
        """
        self._target_name = target_name
        self._n_flavours = len(Flavour.get_all_flavours())
        self._flavour_encoding = Flavour.get_flavour_encoding()
        self._features_columns = None
        self._info = info
        self._features, self._labels = self._prep_data(data)

    def _prep_data(self, data: pd.DataFrame):
        data, info = df_transform_fn(data, inplace=False)
        if self._info is None:
            self._info = info
        df_without_target = data.drop(columns=[self._target_name])
        self._features_columns = df_without_target.columns.to_numpy().tolist()
        features = torch.from_numpy(df_without_target.to_numpy())
        log_labels = torch.log(1 + torch.from_numpy(data[[self._target_name]].to_numpy()))
        return features.float(), log_labels.float()

    @property
    def columns(self) -> List[str]:
        return self._features_columns

    @property
    def info(self):
        return self._info

    def __len__(self):
        return len(self._features)

    def __getitem__(self, idx):
        return self._features[idx], self._labels[idx]


def df_transform_fn(df: pd.DataFrame, info: Optional[Dict[str, Any]] = None, inplace: bool = False):

    if not inplace:
        df = df.copy()

    # Convert 'calendar_date' to datetime type
    df['calendar_date'] = pd.to_datetime(df['calendar_date'])

    df['year'] = df['calendar_date'].dt.year
    df['month'] = df['calendar_date'].dt.month
    df['day'] = df['calendar_date'].dt.day
    df['day_of_week'] = df['calendar_date'].dt.dayofweek

    # Drop the original 'calendar_date' column
    df.drop(columns=['calendar_date'], inplace=True)

    # Encode date components as cyclic representation
    # df['year_sin'] = np.sin(2 * np.pi * df['year'] / df['year'].max())
    # df['year_cos'] = np.cos(2 * np.pi * df['year'] / df['year'].max())

    if info is not None:
        df['year_since_start'] = df['year'] - info['min_year'] + 1
        df['month_sin'] = np.sin(2 * np.pi * df['month'] / info['max_month'])
        df['month_cos'] = np.cos(2 * np.pi * df['month'] / info['max_month'])
        df['day_sin'] = np.sin(2 * np.pi * df['day'] / info['max_day'])
        df['day_cos'] = np.cos(2 * np.pi * df['day'] / info['max_day'])

    else:
        year_min = df['year'].min()
        month_max = 12
        day_max = 31
        df['year_since_start'] = df['year'] - year_min + 1
        df['month_sin'] = np.sin(2 * np.pi * df['month'] / month_max)
        df['month_cos'] = np.cos(2 * np.pi * df['month'] / month_max)
        df['day_sin'] = np.sin(2 * np.pi * df['day'] / day_max)
        df['day_cos'] = np.cos(2 * np.pi * df['day'] / day_max)

        info = {
            "min_year": year_min,
            "max_month": 12,
            "max_day": 31
        }

    # Drop the original date component columns
    df.drop(columns=['year', 'month', 'day', 'day_of_week'], inplace=True)

    # get one-hot flavour encoding
    df["flavour"] = df['flavour'].map(lambda x: Flavour.get_flavour_encoding()[x])
    df = pd.get_dummies(df, columns=["flavour"], dtype=int)

    return df, info


# def df_transform_fn(df: pd.DataFrame, info: Optional[Dict[str, Any]] = None, inplace: bool = False):
#
#     if not inplace:
#         df = df.copy()
#
#     # Convert 'calendar_date' to datetime type
#     df['calendar_date'] = pd.to_datetime(df['calendar_date'])
#     df['day_number_of_year'] = df['calendar_date'].dt.day_of_year
#
#     # Drop the original 'calendar_date' column
#     df.drop(columns=['calendar_date'], inplace=True)
#
#     if info is not None:
#         df['day_number_of_year_sin'] = np.sin(2 * np.pi * df['day_number_of_year'] / info['max_day_number_of_year'])
#         df['day_number_of_year_cos'] = np.cos(2 * np.pi * df['day_number_of_year'] / info['max_day_number_of_year'])
#
#     else:
#         max_day_number_of_year = 365
#         df['day_number_of_year_sin'] = np.sin(2 * np.pi * df['day_number_of_year'] / max_day_number_of_year)
#         df['day_number_of_year_cos'] = np.cos(2 * np.pi * df['day_number_of_year'] / max_day_number_of_year)
#
#         info = {
#             "max_day_number_of_year": max_day_number_of_year
#         }
#
#     # Drop the original date component columns
#     df.drop(columns=['day_number_of_year'], inplace=True)
#
#     # get one-hot flavour encoding
#     df["flavour"] = df['flavour'].map(lambda x: Flavour.get_flavour_encoding()[x])
#     df = pd.get_dummies(df, columns=["flavour"], dtype=int)
#
#     return df, info