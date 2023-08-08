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
from env.gelateria import GelateriaState, default_init_state, default_init_state_new
from env.markdown_trigger.base_trigger import BaseTrigger
from env.markdown_trigger.triggers import DelayTrigger, DefaultTrigger
from env.mask.simple_masks import BooleanMonotonicMarkdownsMask, NoRestrictionBooleanMask
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
    accelerator: Optional[Union[str, Accelerator]] = "cpu"


@dataclass
class DataGenerationConfig(BaseConfig):
    dir_path: Path = ROOT_DIR / "experiment_data"
    data_filename: str = "experiment_data.csv"
    target_name: str = "sales"
    time_period_in_days: int = 365
    expected_sales_generator: Generator = StrongSeasonalGaussian()
    uplift_generator: Generator = SigmoidGaussian()
    cache_data: bool = True
    # init_state: GelateriaState = field(default_factory=default_init_state)

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
class TDZeroConfig(BaseConfig):
    n_episodes: int = 10000
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
    seed: int = 42
    torch_deterministic: bool = True
    n_episodes: int = 500 #1000
    gamma: float = 0.99
    tau: float = 1e-2  #1.0
    learning_rate: float = 5e-4
    auto_entropy_tuning: bool = True
    alpha: float = 0.2  # alpha for entropy (when automatic entropy tuning is off)
    buffer_size: int = 200000
    batch_size: int = 64
    initial_random_steps: int = 20000   # 50000
    target_network_frequency: int = 8000  # 8000  # how often to update the target network (in steps)
    replay_buffer_path: Optional[Path] = ROOT_DIR / "experiment_data/buffers/sac_buffer.pkl"  # if path is provided, no new experience buffer is generated
    save_replay_buffer: bool = True
    target_entropy_scale: float = 0.7#0.89
    markdown_penalty: float = 1.0
    waste_penalty: float = 0.0
    max_markdown_changes: int = 3  # TODO: not implemented yet
    regenerate_buffer: bool = True
    update_frequency: int = 800  # how often to update the actor & critic network (in steps)
    minimum_markdown_duration: Optional[int] = None  # minimum number of days a markdown would last, if None, no minimum duration
    warmup_steps: int = 173  # how many no-discount steps before taking actions from the policy network
    markdown_trigger_fn: BaseTrigger = DelayTrigger(delay=warmup_steps)  # The function to decide if a markdown should be triggered
    actor_network_hidden_layers: Optional[Sequence[int]] = (64,128, 128,64)
    critic_network_hidden_layers: Optional[Sequence[int]] = (64,128, 128,64)
    epsilon_greedy: bool = True
    epsilon_greedy_min_epsilon: float = 2e-3
    epsilon_greedy_epsilon_decay_rate: float = 0.99

@dataclass
class SACConfig_New_Env(BaseConfig):
    seed: int = 42
    torch_deterministic: bool = True
    n_episodes: int = 2500 #1000
    gamma: float = 0.99
    tau: float = 0.89  #1.0 #target smoothing coefficient
    learning_rate: float = 5e-4
    auto_entropy_tuning: bool = True
    alpha: float = 0.2  # alpha for entropy (when automatic entropy tuning is off)
    buffer_size: int = 100000
    batch_size: int = 64
    target_network_frequency: int = 8000  # 8000  # how often to update the target network (in steps)
    # replay_buffer_path: Optional[Path] = ROOT_DIR / "experiment_data/buffers/sac_buffer_new_env.pkl"  # if path is provided, no new experience buffer is generated
    # save_replay_buffer: bool = True
    target_entropy_scale: float = 0.89 #0.7#0.89
    markdown_penalty: float = 1.0
    waste_penalty: float = 0.0
    regenerate_buffer: bool = True
    update_frequency: int = 100  # how often to update the actor & critic network (in steps)
    warmup_episodes: int = 1000  # how many no-discount steps before taking actions from the policy network
    markdown_trigger_fn: BaseTrigger = DefaultTrigger()  # The function to decide if a markdown should be triggered
    actor_network_hidden_layers: Optional[Sequence[int]] = (256,1024,256)#(128, 512, 512, 128) # (64,128, 128,64)
    critic_network_hidden_layers: Optional[Sequence[int]] = (256,1024,256)#(128, 512, 512, 128)


@dataclass
class DQNConfig(BaseConfig):
    seed: int = 42
    torch_deterministic: bool = True
    n_episodes: int = 5000 #1000
    gamma: float = 0.99
    tau: float = 0.89  #1.0 #target smoothing coefficient
    learning_rate: float = 1e-4
    markdown_trigger_fn: BaseTrigger = DefaultTrigger()  # The function to decide if a markdown should be triggered
    buffer_size: int = 100000
    batch_size: int = 64
    target_network_frequency: int = 5000  # 8000  # how often to update the target network (in steps)
    train_frequency: int = 50
    # markdown_penalty: float = 1.0
    # waste_penalty: float = 0.0
    # update_frequency: int = 8  # how often to update the actor & critic network (in steps)
    warmup_episodes: int = 100  # how many no-discount steps before taking actions from the policy network
    markdown_trigger_fn: BaseTrigger = DefaultTrigger()  # The function to decide if a markdown should be triggered
    q_network_hidden_layers: Optional[Sequence[int]] = (256,1024,256)#(128, 512, 512, 128)  # (64,128, 128,64)


@dataclass
class WandbConfig(BaseConfig):
    use_wandb: bool = True
    project: str = "msc_project_v2"
    entity: str = "timc"
    mode: str = "online"


@dataclass
class ExperimentConfig(BaseConfig):
    net_config: NetConfig = field(default_factory=NetConfig)
    data_generation_config: DataGenerationConfig = field(default_factory=DataGenerationConfig)
    dataloader_config: DataLoaderConfig = field(default_factory=DataLoaderConfig)
    wandb_config: WandbConfig = field(default_factory=WandbConfig)

    # Set seeds & deterministic behaviour
    seed: int = 42
    torch_deterministic: bool = True


@dataclass
class TDZeroExperimentConfig(ExperimentConfig):
    td_zero_config: Optional[TDZeroConfig] = field(default_factory=TDZeroConfig)


@dataclass
class SACExperimentConfig(ExperimentConfig):
    # sac_config: SACConfig = field(default_factory=SACConfig)
    sac_config: SACConfig_New_Env = field(default_factory=SACConfig_New_Env)

@dataclass
class DQNExperimentConfig(ExperimentConfig):
    dqn_config: DQNConfig = field(default_factory=DQNConfig)


@dataclass
class SupervisedExperimentConfig(ExperimentConfig):
    lightning_config: LightningConfig = field(default_factory=LightningConfig)




