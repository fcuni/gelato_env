import random
from pathlib import Path
from typing import Optional, List, Union

import pandas as pd
from env.gelateria_env import GelateriaEnv
from models.td_zero import StateQuad


class ReplayBuffer:

    def __init__(self,
                 env: GelateriaEnv,
                 capacity: int = 100_000,
                 data: Optional[Union[Path, pd.DataFrame]] = None):
        self._env = env
        self._capacity = capacity
        self._trajectories: Optional[List[StateQuad]] = None
        self._data = self._load_data(data) if data is not None else None

    @staticmethod
    def _load_data(data: Union[Path, pd.DataFrame]) -> pd.DataFrame:
        if isinstance(data, Path):
            data = pd.read_csv(data)
        return data

    def reset(self):
        ...

    def append(self, trajectory: StateQuad):
        self._trajectories.append(trajectory)
        if len(self._trajectories) > self._capacity:
            self._trajectories.pop(0)

    def sample(self, batch_size: int) -> List[StateQuad]:
        return random.sample(self._trajectories, batch_size)

    def __len__(self):
        return len(self._trajectories)

    def __getitem__(self, item):
        return self._trajectories[item]

    def __iter__(self):
        return iter(self._trajectories)
