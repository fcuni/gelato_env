import uuid
from abc import abstractmethod
import random
from typing import Callable, Optional, Dict, Any

import numpy as np
import pandas as pd

import torch
import torch.nn as nn

from datetime import datetime

import wandb
from wandb.wandb_run import Run

from data_generators.data_generators import DataGenerator
from env.gelateria import GelateriaState, Gelato
from env.gelateria_env_v2 import GelateriaEnv_v2 as GelateriaEnv
from env.mask.action_mask import ActionMask
from env.reward.base_reward import BaseReward
from models.base_rl_agent import RLAgent
from utils.config import WandbConfig, ExperimentConfig, EnvConfig
from utils.enums import Flavour


class BaseExperiment:
    """Base class for experiments."""

    # def get_experiment_config():
    #     config = ExperimentConfig()
    #     return config

    def __init__(self, name: str, config: ExperimentConfig):
        self._name = name
        self._env: Optional[GelateriaEnv] = None
        self._reward: Optional[BaseReward] = None
        self._action_mask_fn: Optional[ActionMask] = None
        self._supervised_model: Optional[nn.Module] = None
        self._init_state: Optional[GelateriaState] = None
        self._config: ExperimentConfig = config

    @property
    def name(self):
        return self._name

    def init_wandb(self, env: Optional[GelateriaEnv] = None, agent_config: Optional[Dict[str, Any]] = None) \
            -> Optional[Run]:

        wandb_config = WandbConfig()

        if not wandb_config.use_wandb:
            return None

        run_config = {"experiment_name": self.name}

        # Append env configs
        if env is not None:
            run_config = {
                **run_config,
                "environment": env.name,
                "sales_model": env.sales_model_name,
                "reward_type": env.reward_type_name,
                "action_mask": env.action_mask_name,
                "seed": self._config.seed,
                "n_products": env.state.n_products,
                "products": [product.flavour.value for product in env.state.products.values()]
            }

        # Append Agent
        if self._reward is not None:
            run_config = {**run_config, **self._reward.configs}

        # Append agent-specific configs
        if agent_config is not None:
            run_config = {**run_config, **agent_config}

        # Init wandb
        current_time = datetime.now()
        run_name = f"{self.name}_{current_time.year:04d}{current_time.month:02d}{current_time.day:02d}" \
                   f"_{current_time.hour:02d}{current_time.minute:02d}{current_time.second:02d}"
        return wandb.init(project=wandb_config.project, entity=wandb_config.entity, config=run_config,
                          mode=wandb_config.mode, name=run_name)

    def _get_dataset_generator(self):
        return DataGenerator(
            config=self._config.data_generation_config, dataloader_config=self._config.dataloader_config
        )

    def get_supervised_model(self, supervised_model: nn.Module) -> nn.Module:
        dummy_dataloader, _ = self._get_dataset_generator().get_train_val_dataloaders()
        input_dim = next(iter(dummy_dataloader))["public_obs"].shape[-1]
        model = supervised_model(input_dim=input_dim, name="mlp_sales", config=self._config.net_config)
        model.load()
        return model

    def build_env(self, init_state: GelateriaState, supervised_model: nn.Module, reward: BaseReward,
                  action_mask_fn: Optional[ActionMask] = None, restock_fct: Optional[Callable] = None):
        if not isinstance(init_state, GelateriaState):
            raise ValueError("init_state must be of type GelateriaState")
        if not callable(supervised_model):
            raise ValueError("supervised_model must be callable")
        if not isinstance(reward, BaseReward):
            raise ValueError("reward must be a subclass of BaseReward")

        self._init_state = init_state
        self._supervised_model = self.get_supervised_model(supervised_model)
        self._reward = reward
        self._action_mask_fn = action_mask_fn

        env = GelateriaEnv(init_state=init_state,
                           sales_model=self._supervised_model,
                           reward=reward,
                           mask_fn=action_mask_fn,
                           restock_fct=restock_fct)

        return env

    @staticmethod
    def build_env_v2(config: EnvConfig):
        from models.sales.sales_prediction_models import GenericSalesPredictionModel
        from models.sales.base_sales_models import CombinedMLPLogBaseSalesModel
        from models.sales.sales_uplift_models import GammaSalesUpliftModel
        from models.sales.dataset import transform_gym_inputs
        from env.gelateria_env_v2 import GelateriaEnv_v2
        from env_wrapper.wrapper import DefaultGelatoEnvWrapper

        sales_model = GenericSalesPredictionModel(
            base_sales_model=CombinedMLPLogBaseSalesModel(
                load_from_dir="experiment_data/trained_models/base_sales_models"),
            uplift_model=GammaSalesUpliftModel(rate=1.2),
            base_sales_input_transform_fn=transform_gym_inputs
        )

        if config.single_product:
            init_state = BaseExperiment._get_experiment_init_state_single_product()
        else:
            init_state = BaseExperiment._get_experiment_init_state()

        env = GelateriaEnv_v2(init_state=init_state,
                              sales_model=sales_model,
                              reward=config.reward_fn,
                              mask_fn=config.action_mask_fn,
                              days_per_step=config.days_per_step,
                              end_date=config.end_date,
                              max_steps=config.max_steps,
                              restock_fct=config.restock_fn)

        env_wrap = DefaultGelatoEnvWrapper(env)
        return env_wrap

    @staticmethod
    def _get_experiment_init_state() -> GelateriaState:

        # load the masked dataset
        df = pd.read_csv("masked_dataset.csv")
        # sort the dataset by date
        df['calendar_date'] = pd.to_datetime(df['calendar_date'])
        df.sort_values(by='calendar_date', inplace=True)
        # query the last date in the dataset
        last_date = df['calendar_date'].max()

        products = []
        current_markdowns = {}

        count = 0
        # Loop through the rows in the filtered DataFrame
        for index, row in df[df['calendar_date'] == last_date].iterrows():

            # Access the values of each column for the current row
            products_id = uuid.uuid4()
            flavour = Flavour(row['flavour'])
            base_price = float(row['full_price_masked'])
            stock = int(row['stock'])
            current_markdowns[products_id] = row['markdown']
            restock_possible = False
            products += [Gelato(flavour=flavour, base_price=base_price, stock=stock, id=products_id,
                                restock_possible=restock_possible)]
            # break # TODO: test with only one product
        return GelateriaState(
            products={product.id: product for product in products},
            current_markdowns=current_markdowns,
            last_markdowns={product.id: None for product in products},
            last_actions={product.id: [] for product in products},
            local_reward={product.id: None for product in products},
            historical_sales={product.id: [] for product in products},
            current_date=last_date,
            end_date=datetime(2023, 10, 9),
            max_steps=None
        )

    @staticmethod
    def _get_experiment_init_state_single_product() -> GelateriaState:

        # load the masked dataset
        df = pd.read_csv("masked_dataset.csv")
        # sort the dataset by date
        df['calendar_date'] = pd.to_datetime(df['calendar_date'])
        df.sort_values(by='calendar_date', inplace=True)
        # query the last date in the dataset
        last_date = df['calendar_date'].max()

        products = []
        current_markdowns = {}

        count = 0
        # Loop through the rows in the filtered DataFrame
        for index, row in df[df['calendar_date'] == last_date].iterrows():
            if row['flavour'] != 'Blood Orange Sorbetto':
                continue
            # Access the values of each column for the current row
            products_id = uuid.uuid4()
            flavour = Flavour(row['flavour'])
            base_price = float(row['full_price_masked'])
            stock = int(row['stock'])
            current_markdowns[products_id] = row['markdown']
            restock_possible = False
            products += [Gelato(flavour=flavour, base_price=base_price, stock=stock, id=products_id,
                                restock_possible=restock_possible)]
            # break # TODO: test with only one product
        return GelateriaState(
            products={product.id: product for product in products},
            current_markdowns=current_markdowns,
            last_markdowns={product.id: None for product in products},
            last_actions={product.id: [] for product in products},
            local_reward={product.id: None for product in products},
            historical_sales={product.id: [] for product in products},
            current_date=last_date,
            end_date=datetime(2023, 10, 9),
            max_steps=None
        )

    @abstractmethod
    def get_rl_model(self) -> RLAgent:
        raise NotImplementedError

    def run(self):

        assert self._env is not None, "Env is not initialised."

        # Set seeds
        random.seed(self._config.seed)
        np.random.seed(self._config.seed)
        torch.manual_seed(self._config.seed)
        torch.backends.cudnn.deterministic = self._config.torch_deterministic

        # Initialise the agent
        agent = self.get_rl_model()

        # Get the wandb Run object for logging, if wandb is enabled (otherwise None)
        wandb_run = self.init_wandb(env=self._env, agent_config=agent.configs)

        # Start training
        agent.train(wandb_run=wandb_run)

        # End of training: save the models
        agent.save()

        if wandb_run is not None:
            wandb_run.finish()


