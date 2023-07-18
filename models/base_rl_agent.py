from abc import abstractmethod

from env.gelateria_env import GelateriaEnv
from utils.config import BaseConfig


class RLAgent:
    def __init__(self, env: GelateriaEnv, config: BaseConfig, name: str):
        self._env = env
        self._config = config
        self._name = name

    @property
    def name(self):
        return self._name

    @property
    def config(self):
        return self._config

    @abstractmethod
    def train(self):
        raise NotImplementedError

    @abstractmethod
    def save(self):
        raise NotImplementedError

    @abstractmethod
    def load(self):
        raise NotImplementedError
