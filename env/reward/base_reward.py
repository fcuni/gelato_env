from abc import abstractmethod
from typing import Dict

from env.gelateria import GelateriaState


class BaseReward:

    @abstractmethod
    def __call__(self, sales: Dict[str, float], state: GelateriaState) -> Dict[str, float]:
        raise NotImplementedError

    @abstractmethod
    def get_terminal_penalty(self, state: GelateriaState) -> float:
        raise NotImplementedError
