from typing import Dict, Optional

from env.gelateria import GelateriaState
from env.reward.base_reward import BaseReward


class SellThroughReward(BaseReward):
    """Reward function that rewards sell through rate."""
    def __init__(self):
        super().__init__(name="SellThroughReward")

    def __call__(self, sales: Dict[str, int], state: GelateriaState,
                 previous_state: Optional[GelateriaState] = None) -> Dict[str, float]:
        """
        Get rewards that rewards sell through rate.

        Args:
            sales: sales from current state
            state: current state
            previous_state: previous state (not used, default: None)

        Returns:
            Dict[str, float]: sell through reward
        """

        sell_through_reward = {}

        for product_id, product in state.products.items():
            original_stock = state.original_stock[product_id]
            sold_units = sales[product_id]

            sell_through_reward[product_id] = sold_units / original_stock

        return sell_through_reward
