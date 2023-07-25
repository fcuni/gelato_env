from typing import Tuple

from env.gelateria import GelateriaState
from env.markdown_trigger.base_trigger import BaseTrigger


class DefaultTrigger(BaseTrigger):
    """Default trigger that always returns True."""

    def __init__(self):
        name = "DefaultTrigger"
        super().__init__(name)

    def __call__(self):
        if not self._triggered:
            self._triggered = True
        # if self._triggered:
        #     return 0.0, 1.0
        return self._triggered




class DelayTrigger(BaseTrigger):
    """Trigger that returns True if the delay is reached.

    Args:
        delay: Delay in steps.
    """

    def __init__(self, delay: int):
        name = f"DelayTrigger({delay})"
        super().__init__(name)
        self._delay: int = delay

    @property
    def delay(self) -> int:
        """Return the delay."""
        return self._delay

    def __call__(self, state: GelateriaState) -> bool:  #Tuple[float, float]:
        """Return markdown range if the delay is reached.

        Args:
            state: Current state.
        """
        if not self._triggered and state.step >= self._delay:
            self._triggered = True
        # if self._triggered:
        #     return 0.0, 1.0
        # else:
        #     return 0.0, 0.0
        return self._triggered


class SalesGradientTrigger(BaseTrigger):
    """Trigger that returns True if the sales gradient turns negative."""

    def __init__(self):
        name = "SalesDownwardTrendTrigger"
        super().__init__(name)
        # sales

    def check_sales_gradient(self, state: GelateriaState) -> bool:
        pass

    def __call__(self, *args, **kwargs):
        # TODO: implement this
        sales_gradient = kwargs["sales_gradient"]
        return sales_gradient > 0.0

