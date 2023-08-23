from abc import abstractmethod
from typing import Dict, Optional, Any

from env.gelateria import GelateriaState


class BaseReward:

    def __init__(self, name="BaseReward"):
        self._name: str = name
        self._info: Dict[str, Any] = {}

    @property
    def name(self) -> str:
        return self._name

    @property
    def configs(self) -> Dict[str, Any]:
        return {}

    @property
    def info(self):
        return self._info

    @abstractmethod
    def __call__(self, sales: Dict[str, int], state: GelateriaState,
                 previous_state: Optional[GelateriaState] = None) -> Dict[str, float]:
        raise NotImplementedError

    @staticmethod
    def get_reduced_price(state: GelateriaState) -> Dict[str, float]:
        """Shorthand to compute the reduced price (after markdown) for products in the state."""
        return {
            product_id: product.current_price(markdown=state.current_markdowns[product_id])
            for product_id, product in state.products.items()
        }

    @staticmethod
    def get_base_price(state: GelateriaState) -> Dict[str, float]:
        """Shorthand to get the base price for products in the state."""
        return {
            product_id: product.base_price
            for product_id, product in state.products.items()
        }





