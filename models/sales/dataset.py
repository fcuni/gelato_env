from typing import Optional, Dict, Any, List, Callable

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from utils.enums import Flavour


class BaseSalesDataset(Dataset):
    """Dataset for base sales model."""

    def __init__(self, data: pd.DataFrame, target_name: str = "sales", df_transform_fn: Optional[Callable] = None,
                 info: Optional[Dict[str, Any]] = None):
        """
        Args:
            data (pd.DataFrame): DataFrame containing the data
            target_name (str): Name of the target column
            df_transform_fn (Callable): Function to transform the DataFrame
            info (Dict[str, Any]): Additional information for transforming the dataset
        """
        self._target_name = target_name
        self._n_flavours = len(Flavour.get_all_flavours())
        self._flavour_encoding = Flavour.get_flavour_encoding()

        # Assign df_transform_fn
        if df_transform_fn is not None:
            # Use the specified transform function if provided
            self._df_transform_fn = df_transform_fn
        elif info is not None and "df_transform_fn" in info and info["df_transform_fn"] is not None:
            # Use the transform function from the info if provided
            self._df_transform_fn = info["df_transform_fn"]
        else:
            # Use the default transform function if not provided
            self._df_transform_fn = transform_df_with_day_of_year

        self._features_columns = None
        self._info = info
        self._features, self._labels = self._prep_data(data)

    def _prep_data(self, data: pd.DataFrame):
        data, info = self._df_transform_fn(data, inplace=False, info=self._info)
        if self._info is None:
            self._info = info
        else:
            for key, value in info.items():
                if key not in self._info:
                    self._info[key] = value
        self._info["df_transform_fn"] = self._df_transform_fn
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


# def transform_df_with_sin_cos_month_day(df: pd.DataFrame, info: Optional[Dict[str, Any]] = None, inplace: bool = False):
#     if not inplace:
#         df = df.copy()
#
#     # Calculate the available stock before sales
#     # df['stock_before'] = df['stock'] + df['sales']
#     # df.drop(columns=['stock'], inplace=True)
#
#     # Convert 'calendar_date' to datetime type
#     df['calendar_date'] = pd.to_datetime(df['calendar_date'])
#
#     df['year'] = df['calendar_date'].dt.year
#     df['month'] = df['calendar_date'].dt.month
#     df['day'] = df['calendar_date'].dt.day
#     df['day_of_week'] = df['calendar_date'].dt.dayofweek
#
#     # Drop the original 'calendar_date' column
#     df.drop(columns=['calendar_date'], inplace=True)
#
#     # Encode date components as cyclic representation
#     # df['year_sin'] = np.sin(2 * np.pi * df['year'] / df['year'].max())
#     # df['year_cos'] = np.cos(2 * np.pi * df['year'] / df['year'].max())
#
#     if info is not None:
#         # df['year_since_start'] = df['year'] - info['min_year'] + 1
#         df['month_sin'] = np.sin(2 * np.pi * df['month'] / info['max_month'])
#         df['month_cos'] = np.cos(2 * np.pi * df['month'] / info['max_month'])
#         df['day_sin'] = np.sin(2 * np.pi * df['day'] / info['max_day'])
#         df['day_cos'] = np.cos(2 * np.pi * df['day'] / info['max_day'])
#
#     else:
#         # year_min = df['year'].min()
#         month_max = 12
#         day_max = 31
#         # df['year_since_start'] = df['year'] - year_min + 1
#         df['month_sin'] = np.sin(2 * np.pi * df['month'] / month_max)
#         df['month_cos'] = np.cos(2 * np.pi * df['month'] / month_max)
#         df['day_sin'] = np.sin(2 * np.pi * df['day'] / day_max)
#         df['day_cos'] = np.cos(2 * np.pi * df['day'] / day_max)
#
#         info = {
#             # "min_year": year_min,
#             "max_month": 12,
#             "max_day": 31
#         }
#
#     # Drop the original date component columns
#     df.drop(columns=['year', 'month', 'day', 'day_of_week'], inplace=True)
#
#     # get one-hot flavour encoding
#     df["flavour"] = df['flavour'].map(lambda x: Flavour.get_flavour_encoding()[x])
#     df = pd.get_dummies(df, columns=["flavour"], dtype=int)
#
#     return df, info


#
def transform_df_with_day_of_year(df: pd.DataFrame, info: Optional[Dict[str, Any]] = None, inplace: bool = False):
    if not inplace:
        df = df.copy()

    # sort dataset by calendar_date (ascending)
    df.sort_values(by='calendar_date', inplace=True)

    df['last_sales'] = df.groupby("flavour")["sales"].shift(1, fill_value=0)
    df['last_stock'] = df.groupby("flavour")["stock"].shift(1, fill_value=0)

    df['replenish_stock'] = (df['sales'] - df['last_stock']).clip(lower=0)
    df['available_stock'] = df['last_stock'] + df['replenish_stock']
    df.drop(columns=['last_stock', 'replenish_stock', 'stock'], inplace=True)

    # normalise stocks
    if info is not None and "stock_normalising_factor" in info:
        df['available_stock'] /= info['stock_normalising_factor']
    else:
        max_stock = df['available_stock'].max()
        df['available_stock'] /= max_stock
        info = {"stock_normalising_factor": max_stock, **info} if info is not None else {
            "stock_normalising_factor": max_stock}

    # normalise sales
    if info is not None and "sales_normalising_factor" in info:
        df['sales'] = df['sales'] / info['sales_normalising_factor']
        df['last_sales'] = np.log(df['last_sales'] / info['sales_normalising_factor']+1)

    else:
        max_sales = df['sales'].max()
        df['sales'] = df['sales'] / max_sales
        df['last_sales'] = np.log(df['last_sales'] / max_sales + 1)
        info = {"sales_normalising_factor": max_sales, **info} if info is not None else {
            "sales_normalising_factor": max_sales}


    # Convert 'calendar_date' to datetime type
    df['calendar_date'] = pd.to_datetime(df['calendar_date'])
    df['day_number_of_year'] = df['calendar_date'].dt.day_of_year - 1

    # Drop the original 'calendar_date' column
    df.drop(columns=['calendar_date'], inplace=True)

    if info is not None and "max_day_number_of_year" in info:
        df['day_number_of_year_sin'] = np.sin(2 * np.pi * df['day_number_of_year'] / info['max_day_number_of_year'])
        df['day_number_of_year_cos'] = np.cos(2 * np.pi * df['day_number_of_year'] / info['max_day_number_of_year'])

    else:
        max_day_number_of_year = 365
        df['day_number_of_year_sin'] = np.sin(2 * np.pi * df['day_number_of_year'] / max_day_number_of_year)
        df['day_number_of_year_cos'] = np.cos(2 * np.pi * df['day_number_of_year'] / max_day_number_of_year)

        info = {"max_day_number_of_year": max_day_number_of_year, **info} if info is not None else {
            "max_day_number_of_year": max_day_number_of_year}

    # Drop the original date component columns
    df.drop(columns=['day_number_of_year'], inplace=True)

    # get one-hot flavour encoding (if there is only one flavour, then we don't need to do this)
    if len(df["flavour"].unique().tolist()) > 1:
        for flavour in Flavour.get_all_flavours():
            df[f"flavour_{flavour}"] = df['flavour'].map(lambda x: 1 if x == flavour else 0)
    df.drop(columns=["flavour"], inplace=True)

    return df, info


def transform_gym_inputs(public_obs: torch.Tensor, info: Dict[str, Any]) -> torch.Tensor:
    """
    Transform the public observation from the Gelato-Env gym environment to the inputs for the sales model.

    Args:
        public_obs (torch.Tensor): The public observation from the Gelato-Env gym environment.
            The shape is (n_products, 5 + n_flavours). The columns are:
            - 0: current markdown (between 0 and 1)
            - 1: day_number_of_year (between 0 and 365)
            - 2: stock (integer between 0 and max_stock)
            - 3: normalised full price (between 0 and 1)
            - 4: last sales (integer between 0 and max_stock)
            - 5 to 5 + n_flavours: one-hot encoding of the flavour
        info (Dict[str, Any]): Additional information for transforming the dataset. It should contain the keys:
            `sales_normalising_factor` and `stock_normalising_factor`.

    Returns:
        torch.Tensor: The transformed inputs for the sales model.
    """

    # make sure public_obs is at least 2d
    public_obs = torch.atleast_2d(public_obs)

    # get number of flavours
    n_flavours = len(Flavour.get_all_flavours())

    # select the column of markdown
    markdowns = public_obs[:, 0].reshape(-1, 1)
    # select the column of full_price_masked
    full_price_masked = public_obs[:, 3].reshape(-1, 1)
    # calculate log_last_sales: log(normalised_last_sales + 1)
    log_last_sales = ((public_obs[:, 4] / info['sales_normalising_factor'] + 1).log()).reshape(-1, 1)
    # calculate log_avail_stock: log(normalised_stock + 1)
    log_avail_stock = ((public_obs[:, 2] / info['stock_normalising_factor'] + 1).log()).reshape(-1, 1)
    # calculate day_number_of_year_sin and day_number_of_year_cos
    day_number_of_year_sin = (torch.sin(2 * torch.pi * public_obs[:, 1] / info["max_day_number_of_year"])).reshape(-1, 1)
    day_number_of_year_cos = (torch.cos(2 * torch.pi * public_obs[:, 1] / info["max_day_number_of_year"])).reshape(-1, 1)
    # get flavour one-hot encoding
    if n_flavours > 1:
        flavour_one_hot = public_obs[:, -n_flavours:]
    else:
        flavour_one_hot = public_obs[:, -n_flavours:].reshape(-1, 1)

    transformed_inputs = torch.hstack([

        full_price_masked,
        log_last_sales,
        log_avail_stock,
        day_number_of_year_sin,
        day_number_of_year_cos,
        flavour_one_hot
    ]).float()

    return transformed_inputs



