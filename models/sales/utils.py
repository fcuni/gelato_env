from typing import Optional, Tuple

import numpy as np
import pandas as pd
import torch
import plotly.express as px
from plotly.subplots import make_subplots
from plotly.graph_objs import Figure
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

from models.sales.dataset import BaseSalesDataset
from models.sales.base_sales_models import BaseSalesModel


def split_train_and_test_df_by_flavour(df: pd.DataFrame, test_size: float = 0.25, seed: Optional[int] = None) -> \
        Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split the DataFrame into training and test DataFrames at the flavor level.
    Args:
        df (pd.DataFrame): DataFrame containing the data
        test_size (float): Fraction of the data to be used for testing
        seed (int): Random seed for reproducibility
    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]: Training and test DataFrames
    """

    # Identify unique flavors
    unique_flavors = df['flavour'].unique()

    # Create dictionaries to store training and test DataFrames for each flavor
    train_dfs = {}
    test_dfs = {}

    # Perform random split at the flavor level
    for flavor in unique_flavors:
        flavor_df = df[df['flavour'] == flavor]
        train_df, test_df = train_test_split(flavor_df, test_size=test_size, random_state=seed)
        train_dfs[flavor] = train_df
        test_dfs[flavor] = test_df

    # Combine the DataFrames for all flavors into final training and test DataFrames
    train_df_final = pd.concat(train_dfs.values())
    test_df_final = pd.concat(test_dfs.values())

    return train_df_final, test_df_final


def generate_eval_plot(model: BaseSalesModel, df: pd.DataFrame, batch_size: int = 128,
                       device: Optional[torch.device] = None) -> Figure:
    """
    Generate plotly figure for sales vs. predicted sales.
    Args:

        model (BaseSalesModel): Trained model
        df (pd.DataFrame): DataFrame containing the data
        batch_size (int): Batch size for evaluation
        device (torch.device): Device to use for evaluation
    Returns:
        Figure: Plotly figure
    """

    # Create DataLoader for evaluation

    eval_dataset = BaseSalesDataset(df, target_name="sales", info=model.info)
    eval_dataloader = DataLoader(eval_dataset, batch_size=batch_size, shuffle=False)

    model.eval()
    preds = []
    for inputs, _ in eval_dataloader:
        with torch.no_grad():
            if device is not None:
                inputs = inputs.to(device)
            preds.append((np.atleast_1d(model.get_sales(inputs).detach().cpu().numpy().squeeze())))
    df["pred"] = np.concatenate(preds)

    # Create two subplots using make_subplots
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True)

    # df["log_sales"] = np.log(df["sales"])

    sales_fig = px.scatter(df, x='calendar_date', y='sales', color="flavour")
    pred_fig = px.scatter(df, x='calendar_date', y='pred', color="flavour")
    for trace in sales_fig["data"]:
        fig.add_trace(trace, row=1, col=1)

    for trace in pred_fig["data"]:
        fig.add_trace(trace, row=2, col=1)

    # Update layout for both subplots
    fig.update_layout(height=600, title_text='Sales vs. Predicted Sales')
    fig.update_yaxes(title_text='Sales', row=1, col=1)
    fig.update_yaxes(title_text='Predicted Sales', row=2, col=1)
    fig.update_xaxes(title_text='Calendar Date', row=2, col=1)
    fig.update_traces(showlegend=False, row=2, col=1)

    # Show the combined subplots with a common legend
    # fig.show()
    return fig
