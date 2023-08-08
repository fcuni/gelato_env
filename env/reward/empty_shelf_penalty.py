from typing import Dict, Optional

from env.gelateria import GelateriaState
from env.reward.base_reward import BaseReward


class EmptyShelfPenalty(BaseReward):
    """Reward function that penalises empty shelves from previous state."""
    def __init__(self, empty_shelf_penalty: float = 1.0):
        """
        Args:
            empty_shelf_penalty: penalty for empty shelf (should be negative value)
        """
        super().__init__(name="EmptyShelfPenalty")
        self._empty_shelf_penalty = empty_shelf_penalty

    def configs(self):
        """
        Returns the reward configuration.

        Returns:
            Dict[str, Any]: reward configuration
        """
        return {"rewards/empty_shelf_penalty": self._empty_shelf_penalty}

    def __call__(self, sales: Dict[str, int], state: GelateriaState, previous_state: Optional[GelateriaState] = None) \
            -> Dict[str, float]:
        """
        Get rewards that penalises empty shelves from previous state.

        Args:
            sales: sales from current state
            state: current state
            previous_state: previous state

        Returns:
            Dict[str, float]: rewards
        """

        empty_shelf_penalty = {}

        for product_id, product in state.products.items():
            if previous_state.products[product_id].stock == 0 and product.stock == 0:
                empty_shelf_penalty[product_id] = self._empty_shelf_penalty
            else:
                empty_shelf_penalty[product_id] = 0.0

        return empty_shelf_penalty
