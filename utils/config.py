from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, TypeVar, Union, Tuple

import numpy as np
import pandas as pd

import torch
# from pytorch_lightning.accelerators import Accelerator
# from pytorch_lightning.callbacks import Callback
from torch.utils.data import Sampler

# from data_generators.gaussian_generators import StrongSeasonalGaussian
# from data_generators.generator import Generator
# from data_generators.sigmoid_generators import SigmoidGaussian
from env.mask.monotonic_markdowns_mask import MonotonicMarkdownsBooleanMask
from env.mask.phased_markdown_mask import PhasedMarkdownMask
from env.reward.multi_objective_reward import MultiObjectiveReward
from utils.misc import custom_collate_fn, get_root_dir

ROOT_DIR = get_root_dir()

T_co = TypeVar("T_co", covariant=True)
T = TypeVar("T")


def get_markdown_schedule(path: str = "markdown_schedule.csv"):
    """
    Load the markdown schedule and returns it as a pandas DataFrame.

    Args:
        path: path to the markdown schedule csv file.

    Returns:
        markdown schedule as a pandas DataFrame.
    """
    schedule_df = pd.read_csv(path)
    schedule_df['start_date'] = pd.to_datetime(schedule_df['start_date'])
    schedule_df['end_date'] = pd.to_datetime(schedule_df['end_date'])
    return schedule_df


@dataclass
class BaseConfig:
    def to_dict(self):
        return {k: v for k, v in self.__dict__.items() if not k.startswith("_")}

@dataclass
class SACConfig(BaseConfig):
    seed: int = 42
    torch_deterministic: bool = True

    gamma: float = 0.99
    tau: float = 0.89  #1.0 #target smoothing coefficient
    learning_rate: float = 1e-5
    auto_entropy_tuning: bool = True
    alpha: float = 0.2  # alpha for entropy (when automatic entropy tuning is off)
    buffer_size: int = 10000
    batch_size: int = 128
    target_network_frequency: int = 5000  # 8000  # how often to update the target network (in steps)
    target_entropy_scale: float = 0.89
    update_frequency: int = 50  # how often to update the actor & critic network (in steps)
    actor_network_hidden_layers: Optional[Sequence[int]] = (64, 256, 64)
    critic_network_hidden_layers: Optional[Sequence[int]] = (64, 256, 64)

    init_exploration_steps: int = 1000
    num_epoch: int = 1000
    epoch_length:int = 100
    min_pool_size: int = 1000


@dataclass
class DQNConfig(BaseConfig):
    seed: int = 42
    torch_deterministic: bool = True

    gamma: float = 0.99
    tau: float = 1.0  # target smoothing coefficient
    learning_rate: float = 1e-5
    buffer_size: int = 100000
    batch_size: int = 128
    target_network_frequency: int = 7000  # how often to update the target network (in steps)
    update_frequency: int = 8  # how often to update the q network (in steps)
    warmup_episodes: int = 1000  # how many no-discount steps before taking actions from the policy network
    q_network_hidden_layers: Optional[Sequence[int]] = (64, 256, 64)
    epsilon_start: float = 1.0
    epsilon_decay: float = 0.99
    epsilon_min: float = 0.01

    init_exploration_steps: int = 10000
    num_epoch: int = 1000
    epoch_length: int = 100
    min_pool_size: int = 1000

@dataclass
class MBPOConfig(BaseConfig):
    num_networks: int = 4
    num_elites: int = 2
    pred_hidden_size: int = 200
    use_decay: bool = True
    replay_size: int = 100000
    rollout_batch_size: int = 400#1000
    model_retain_epochs: int = 1
    max_path_length: int = 3
    # training parameters
    init_exploration_steps: int = 10000  # 5000,
    num_epoch: int = 250
    epoch_length: int = 100#80 #1000
    min_pool_size: int = 1000
    model_train_freq: int = 25#20 #250
    real_ratio: float = 1#0.05
    predict_model_batch_size: int = 256
    predict_model_holdout_ratio: float = 0.2

    train_every_n_steps: int = 1
    max_train_repeat_per_step: int = 5
    num_train_repeat: int = 1
    policy_train_batch_size: int = 256

    rollout_min_length: int = 1
    rollout_max_length: int = 1
    rollout_max_epoch: int = 80 #150
    rollout_min_epoch: int = 15 #20

    # GelateriaEnv_v2 parameters
    days_per_step: int = 7


@dataclass
class RuleBasedAgentConfig(BaseConfig):
    fixed_markdown_schedule: Tuple[int] = (40, 40, 40, 40, 50, 50, 60, 60, 60, 60, 75, 75, 75, 75)
    num_epoch: int = 1000


@dataclass
class WandbConfig(BaseConfig):
    use_wandb: bool = True
    project: str = "msc_project_v4"
    entity: str = "timc"
    mode: str = "offline"


@dataclass
class EnvConfig(BaseConfig):
    action_mask_fn: Optional[Callable] = MonotonicMarkdownsBooleanMask()#PhasedMarkdownMask(get_markdown_schedule(), delta_markdown=10)
    reward_fn: Callable = MultiObjectiveReward(sell_through_coeff=0.2)
    restock_fn: Optional[Callable] = None
    days_per_step: int = 7
    end_date: Optional[datetime] = None
    max_steps: Optional[int] = 3
    single_product: bool = True


@dataclass
class ExperimentConfig(BaseConfig):
    env_config: EnvConfig = field(default_factory=EnvConfig)
    wandb_config: WandbConfig = field(default_factory=WandbConfig)

    # Set seeds & deterministic behaviour
    seed: int = 42
    torch_deterministic: bool = True

@dataclass
class SACExperimentConfig(ExperimentConfig):
    sac_config: SACConfig = field(default_factory=SACConfig)


@dataclass
class DQNExperimentConfig(ExperimentConfig):
    dqn_config: DQNConfig = field(default_factory=DQNConfig)


@dataclass
class MBPOExperimentConfig(ExperimentConfig):
    mbpo_config: MBPOConfig = field(default_factory=MBPOConfig)
    sac_config: SACConfig = field(default_factory=SACConfig)
    dqn_config: DQNConfig = field(default_factory=DQNConfig)


@dataclass
class RuleBasedAgentExperimentConfig(ExperimentConfig):
    rule_based_agent_config: RuleBasedAgentConfig = field(default_factory=RuleBasedAgentConfig)





