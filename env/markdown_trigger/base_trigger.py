from abc import abstractmethod


class BaseTrigger:
    def __init__(self, name):
        self._name = name
        self._triggered: bool = False

    @property
    def name(self):
        return self._name

    @abstractmethod
    def __call__(self, *args, **kwargs) -> bool:
        raise NotImplementedError

    def reset(self):
        self._triggered = False

