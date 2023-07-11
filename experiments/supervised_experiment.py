import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint

import os
print(os.getcwd())

import sys
sys.path.append(os.getcwd())

from utils.config import ExperimentConfig
from data_generators.data_generators import DataGenerator
from models.mlp_sales import MLPLogSalesModel


def get_model(input_dim, config):
    return MLPLogSalesModel(input_dim=input_dim, name="mlp_sales", config=config)


def get_experiment_config():
    config = ExperimentConfig()
    config.lightning_config.callbacks = [ModelCheckpoint(monitor="mean_loss_train", mode="min", save_top_k=5)]
    config.lightning_config.max_epochs = 100
    return config


class SupervisedExperiment:

    def __init__(self):
        self._config: ExperimentConfig = get_experiment_config()

    def _get_dataset_generator(self):
        return DataGenerator(config=self._config.data_generation_config,
                             dataloader_config=self._config.dataloader_config)

    def build_model(self):
        dummy_dataloader, _ = self._get_dataset_generator().get_train_val_dataloaders()
        input_dim = next(iter(dummy_dataloader))["public_obs"].shape[-1]
        return get_model(input_dim=input_dim, config=self._config.net_config)

    def run(self):
        train_dataloader, val_dataloader = self._get_dataset_generator().get_train_val_dataloaders()
        model = self.build_model()
        trainer = pl.Trainer(**self._config.lightning_config.to_dict())
        trainer.fit(model, train_dataloader, val_dataloader)

        if self._config.net_config.path_to_model is not None:
            model.save()


if __name__ == "__main__":
    SupervisedExperiment().run()
