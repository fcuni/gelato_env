from abc import abstractmethod
from typing import Dict, Optional

from env.gelateria import GelateriaState


class BaseReward:

    def __init__(self, name="BaseReward"):
        self._name: str = name

    @property
    def name(self) -> str:
        return self._name

    @abstractmethod
    def __call__(self, sales: Dict[str, float], state: GelateriaState,
                 previous_state: Optional[GelateriaState] = None) -> Dict[str, float]:
        raise NotImplementedError

    @abstractmethod
    def get_terminal_penalty(self, state: GelateriaState) -> Dict[str, float]:
        raise NotImplementedError
