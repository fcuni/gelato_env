import pandas as pd
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from torch.utils.data import DataLoader

from models.sales.dataset import BaseSalesDataset
from models.sales.utils import split_train_and_test_df_by_flavour
from utils.config import ExperimentConfig, SupervisedExperimentConfig
from models.mlp_sales import MLPLogSalesModel


def get_model(input_dim, config):
    return MLPLogSalesModel(input_dim=input_dim, name="mlp_sales", config=config)


def get_experiment_config():
    config = SupervisedExperimentConfig()
    config.lightning_config.callbacks = [ModelCheckpoint(monitor="mean_loss_train", mode="min", save_top_k=5)]
    config.lightning_config.max_epochs = 1000
    return config


class BaseSalesModelExperiment:

    def __init__(self):
        self._config: ExperimentConfig = get_experiment_config()

    def _get_dataloaders(self, batch_size: int = 128, target_column: str = "sales"):
        df = pd.read_csv("masked_dataset.csv")
        # We only use zero markdown
        base_sales_df = df[df["markdown"] == 0.0].copy()
        base_sales_df.drop(columns=["markdown"], inplace=True)
        base_sales_train_df, base_sales_test_df = split_train_and_test_df_by_flavour(base_sales_df)
        train_dataset = BaseSalesDataset(base_sales_train_df, target_column)
        test_dataset = BaseSalesDataset(base_sales_test_df, target_column)
        train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)
        return train_dataloader, test_dataloader

    def build_model(self):
        dummy_dataloader, _ = self._get_dataset_generator().get_train_val_dataloaders()
        input_dim = next(iter(dummy_dataloader))["public_obs"].shape[-1]
        return get_model(input_dim=input_dim, config=self._config.net_config)

    def run(self):
        train_dataloader, val_dataloader = self._get_dataloaders(config["batch_size"])
        model = self.build_model()
        trainer = pl.Trainer(**self._config.lightning_config.to_dict())
        trainer.fit(model, train_dataloader, val_dataloader)

        if self._config.net_config.path_to_model is not None:
            model.save()


if __name__ == "__main__":
    SupervisedExperiment().run()
