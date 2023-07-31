from typing import Optional, Dict, Any

import pandas as pd

from torch.utils.data import DataLoader
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import wandb

from models.sales.dataset import BaseSalesDataset
from models.sales.utils import generate_eval_plot, split_train_and_test_df_by_flavour

from models.sales.base_sales_models import MLPLogBaseSalesModel


# Training loop
def train(model, train_set: BaseSalesDataset, valid_set: BaseSalesDataset, eval_df: Optional[pd.DataFrame] = None,
          num_epochs=100, batch_size: int = 128, learning_rate=0.001, logging_info: Optional[Dict[str, Any]] = None,
          device: Optional[torch.device] = None):

    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if logging_info is None:
        logging_info = {}

    model.to(device)

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(valid_set, batch_size=batch_size, shuffle=True)

    # Define the loss function and optimizer
    loss_function = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    wandb.init(project="base_sales_model", entity="timc", config={
        "learning_rate": learning_rate,
        "architecture": model.__repr__(),
        "optimizer": optimizer.__class__.__name__,
        "epochs": num_epochs,
        "batch_size": batch_size,
        "device": device.type,
        "data_columns": train_set.columns,
        **logging_info
    })

    for epoch in tqdm(range(num_epochs), desc="Epoch", leave=True):
        model.train()  # Set the model to training mode
        running_loss = 0.0

        for inputs, targets in train_loader:
            # for inputs, targets in tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs}', leave=False):
            inputs, targets = inputs.to(device), targets.to(device)

            optimizer.zero_grad()

            # Forward pass
            outputs = model(inputs)
            loss = loss_function(outputs, targets)

            # Backward pass and optimization
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        train_loss = running_loss / len(train_loader)

        # Validation
        model.eval()  # Set the model to evaluation mode
        valid_loss = 0.0
        all_predictions = []
        all_targets = []

        with torch.no_grad():
            for inputs, targets in valid_loader:
                inputs, targets = inputs.to(device), targets.to(device)

                outputs = model(inputs)
                loss = loss_function(outputs, targets)

                valid_loss += loss.item()

                all_predictions.append(outputs.cpu())
                all_targets.append(targets.cpu())

        valid_loss /= len(valid_loader)
        all_predictions = torch.cat(all_predictions, dim=0)
        all_targets = torch.cat(all_targets, dim=0)

        # Calculate metrics
        mae = mean_absolute_error(all_targets, all_predictions)
        rmse = mean_squared_error(all_targets, all_predictions, squared=False)
        r2 = r2_score(all_targets, all_predictions)

        # Logging metrics
        wandb.log({
            "mean_loss_train": train_loss,
            "mean_loss_valid": valid_loss,
            "mae": mae,
            "rmse": rmse,
            "r2": r2,
        })
        print(f'Epoch [{epoch + 1}/{num_epochs}]'
              f' - Train Loss: {train_loss:.4f}, Valid Loss: {valid_loss:.4f}'
              f' - MAE: {mae:.4f}, RMSE: {rmse:.4f}, R2 Score: {r2:.4f}')

    # Evaluation
    if eval_df is not None:
        wandb.log({"sales_vs_predictions": generate_eval_plot(model, eval_df, batch_size, device=device)})
    wandb.finish()


if __name__ == "__main__":

    # Load the dataset
    df = pd.read_csv("masked_dataset.csv")

    # We only use zero markdown
    base_sales_df = df[df["markdown"] == 0.0].copy()
    base_sales_df.drop(columns=["markdown"], inplace=True)

    # Create Dataset objects for training and test datasets
    base_sales_train_df, base_sales_valid_df = split_train_and_test_df_by_flavour(base_sales_df, test_size=0.25)
    train_dataset = BaseSalesDataset(base_sales_train_df, "sales")
    valid_dataset = BaseSalesDataset(base_sales_valid_df, "sales")
    log_info = {"valid_split": 0.25}

    # Create Dataset object for evaluation dataset
    eval_df = base_sales_df.copy()

    # Define the model
    base_sales_model = MLPLogBaseSalesModel(input_dim=len(train_dataset.columns), output_dim=1, info=train_dataset.info)

    # Train the model
    train(base_sales_model, train_dataset, valid_dataset, eval_df=eval_df, num_epochs=2000, batch_size=128,
          learning_rate=0.001, logging_info=log_info)

    # Save the model
    base_sales_model.save("base_sales_model_using_doy_sin_cs.pt")
