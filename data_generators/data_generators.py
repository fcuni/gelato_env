from pathlib import Path
from typing import Tuple, Optional
import os

import numpy as np
import torch
import pandas as pd

from torch.utils.data import Dataset, DataLoader, random_split

from utils.config import DataGenerationConfig, DataLoaderConfig
from utils.enums import Flavour


class PandasDataset(Dataset):

    def __init__(self, data: pd.DataFrame, target_name: str):
        self._target_name = target_name
        self._n_flavours = len(Flavour.get_all_flavours())
        self._flavour_encoding = Flavour.get_flavour_encoding()
        self._features, self._labels = self._prep_data(data)

    def _prep_data(self, data: pd.DataFrame):
        log_labels = torch.log(1 + torch.from_numpy(data[[self._target_name]].to_numpy()))
        days = torch.from_numpy(data[["day"]].to_numpy()) / 365
        stock = torch.from_numpy(data[["stock"]].to_numpy() / data["stock"].max())
        flavours = torch.from_numpy(data["flavour"].map(self._flavour_encoding).to_numpy())
        flavours_one_hot = torch.nn.functional.one_hot(flavours, self._n_flavours)
        num_features = torch.from_numpy(data[["price", "markdown"]].to_numpy())
        features = torch.hstack([days, stock, num_features, flavours_one_hot])
        return features.float(), log_labels.float()

    def __len__(self):
        return len(self._features)

    def __getitem__(self, idx):
        return self._features[idx], self._labels[idx]


class DataGenerator:

    def __init__(self, config: DataGenerationConfig, dataloader_config: DataLoaderConfig):
        self._config = config
        self._dataloader_config = dataloader_config
        self._data: Optional[pd.DataFrame] = None

    @property
    def config(self):
        return self._config

    @property
    def dataloader_config(self):
        return self._dataloader_config

    def load(self, path: Path):
        self._data = pd.read_csv(path)

    def _generate(self):
        # generate features: [day, flavour, price, stock, markdown, sales]
        init_state = self.config.init_state
        n_days = self.config.time_period_in_days
        flavours = [(product.flavour, product) for product in init_state.products.values()]
        dfs = []
        for flavour, product in flavours:
            df = pd.DataFrame()
            df["day"] = np.arange(0, n_days)
            df["flavour"] = flavour.value
            df["price"] = product.base_price
            df["stock"] = product.stock
            reds = torch.randint(low=0, high=71, size=(n_days,)) / 100
            df["markdown"] = reds.numpy()
            expected_sales = self.config.expected_sales_generator.sample((n_days,)).numpy()
            df["_expected_sales"] = expected_sales
            probs_sales = self.config.uplift_generator.prob_sales_at_reduction(reds).numpy()
            df["_prob_sales"] = probs_sales
            df["sales"] = expected_sales * probs_sales
            dfs += [df]
        dfs = pd.concat(dfs)
        dfs.sort_values([], axis="columns", inplace=True)

        if self.config.cache_data:
            os.makedirs(name=self.config.dir_path, exist_ok=True)
            dfs.to_csv(self.config.dir_path / self.config.data_filename, index=False)
        return dfs

    def get_train_val_dataloaders(self) -> Tuple[DataLoader, DataLoader]:
        df = self._generate() if self._data is None else self._data
        dataset = PandasDataset(df, self.config.target_name)
        train_size = int(self.dataloader_config.train_val_split * len(dataset))
        val_size = len(dataset) - train_size
        train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
        train_dataloader = DataLoader(train_dataset,
                                      batch_size=self.dataloader_config.batch_size,
                                      shuffle=self.dataloader_config.shuffle,
                                      collate_fn=self.dataloader_config.collate_fn)
        val_dataloader = DataLoader(val_dataset,
                                    batch_size=self.dataloader_config.batch_size,
                                    shuffle=False,
                                    collate_fn=self.dataloader_config.collate_fn)
        return train_dataloader, val_dataloader
