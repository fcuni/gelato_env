from abc import abstractmethod
from typing import Optional
from wandb.wandb_run import Run

from env.gelateria_env import GelateriaEnv
from utils.config import BaseConfig

from datetime import datetime


class RLAgent:
    def __init__(self, env: GelateriaEnv, name: str, run_name: Optional[str] = None):
        self._env = env
        self._name = name
        if run_name is None:
            start_time = datetime.now()
            datetime_str = f"{start_time.year}{start_time.month:02d}{start_time.day:02d}_" \
                           f"{start_time.hour:02d}{start_time.minute:02d}{start_time.second:02d}"
            self._run_name = f"{self._name}_{datetime_str}"
        else:
            self._run_name = run_name

        # retrieve environment information
        self._state_size: int = tuple(self._env.get_single_observation_space_size())[-1]
        self._action_size: int = 1 if len(self._env.action_space.shape) == 0 else len(self._env.action_space.shape)
        self._action_num: int = self._env.action_space.n
        self._reward_size: int = 1

    @property
    def name(self):
        return self._name

    @property
    def run_name(self):
        return self._run_name

    @property
    def config(self):
        return self._config

    @property
    def configs(self):
        raise NotImplementedError

    @abstractmethod
    def train(self, wandb_run: Optional[Run] = None):
        raise NotImplementedError

    @abstractmethod
    def save(self):
        raise NotImplementedError

    @abstractmethod
    def load(self):
        raise NotImplementedError
