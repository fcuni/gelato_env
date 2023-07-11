from dataclasses import dataclass, field
from pathlib import Path
from typing import (
    Any,
    Callable,
    Dict,
    Iterable,
    List,
    Optional,
    Sequence,
    TypeVar,
    Union,
)

import numpy as np
import torch
from pytorch_lightning.accelerators import Accelerator
from pytorch_lightning.callbacks import Callback
from torch.utils.data import Sampler

from data_generators.gaussian_generators import StrongSeasonalGaussian
from data_generators.generator import Generator
from data_generators.sigmoid_generators import SigmoidGaussian
from env.gelateria import GelateriaState, default_init_state
from utils.misc import custom_collate_fn, get_root_dir

ROOT_DIR = get_root_dir()

T_co = TypeVar("T_co", covariant=True)
T = TypeVar("T")


@dataclass
class BaseConfig:
    def to_dict(self):
        return {k: v for k, v in self.__dict__.items() if not k.startswith("_")}


@dataclass
class NetConfig(BaseConfig):
    input_key: str = "public_obs"
    target: str = "target"
    embedding_dims: List[int] = field(default_factory=lambda: [32, 32, 32])
    activation: str = "ReLU"
    lr: float = 1e-3
    lr_scheduler: torch.optim.lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau
    scheduler_config: Optional[Dict] = None
    optim: torch.optim = torch.optim.Adam
    path_to_model: Path = ROOT_DIR / "experiment_data/trained_models"


@dataclass
class LightningConfig(BaseConfig):
    default_root_dir: Optional[str] = ROOT_DIR / "experiment_data/training_logs"
    callbacks: Optional[Union[List[Callback], Callback]] = None
    enable_progress_bar: bool = True
    max_epochs: Optional[int] = 10
    accelerator: Optional[Union[str, Accelerator]] = "auto"


@dataclass
class DataGenerationConfig(BaseConfig):
    dir_path: Path = ROOT_DIR / "experiment_data"
    data_filename: str = "experiment_data.csv"
    target_name: str = "sales"
    time_period_in_days: int = 365
    expected_sales_generator: Generator = StrongSeasonalGaussian()
    uplift_generator: Generator = SigmoidGaussian()
    cache_data: bool = True
    init_state: GelateriaState = field(default_factory=default_init_state)


@dataclass
class DataLoaderConfig(BaseConfig):
    batch_size: Optional[int] = 64
    train_val_split: float = 0.8
    shuffle: Optional[bool] = True
    sampler: Union[Sampler, Iterable, None] = None
    batch_sampler: Union[Sampler[Sequence], Iterable[Sequence], None] = None
    collate_fn: Optional[Callable[[List[T]], Any]] = custom_collate_fn
    pin_memory: bool = False
    drop_last: bool = False
    timeout: float = 0
    worker_init_fn: Optional[Callable[[int], None]] = None
    multiprocessing_context = None
    generator = None
    prefetch_factor: int = 2
    persistent_workers: bool = False
    pin_memory_device: str = ""


@dataclass
class OptimiserConfig(BaseConfig):
    n_episodes: int = 100
    horizon_steps: int = 10
    epsilon: float = 0.9
    gamma: float = 0.9
    alpha: float = 0.9
    warm_start: Optional[int] = None
    q_init: Optional[np.array] = None
    path_to_model: Path = ROOT_DIR / "experiment_data/trained_models"
    seed: int = 42


@dataclass
class SACConfig(BaseConfig):
    n_episodes: int = 1000
    gamma: float = 0.99
    tau: float = 1e-2
    learning_rate: float = 5e-4
    buffer_size: int = 10000
    batch_size: int = 1
    initial_random_steps: int = 1000


@dataclass
class WandbConfig(BaseConfig):
    project: str = "msc_project"
    entity: str = "timc"
    mode: str = "offline"


@dataclass
class ExperimentConfig(BaseConfig):
    net_config: NetConfig = field(default_factory=NetConfig)
    lightning_config: LightningConfig = field(default_factory=LightningConfig)
    data_generation_config: DataGenerationConfig = field(default_factory=DataGenerationConfig)
    dataloader_config: DataLoaderConfig = field(default_factory=DataLoaderConfig)
    optimiser_config: Optional[OptimiserConfig] = field(default_factory=OptimiserConfig)


@dataclass
class SACExperimentConfig(BaseConfig):
    net_config: NetConfig = field(default_factory=NetConfig)
    data_generation_config: DataGenerationConfig = field(default_factory=DataGenerationConfig)
    dataloader_config: DataLoaderConfig = field(default_factory=DataLoaderConfig)
    wandb_config: WandbConfig = field(default_factory=WandbConfig)
    sac_config: SACConfig = field(default_factory=SACConfig)




