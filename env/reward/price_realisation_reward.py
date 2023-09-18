from typing import Dict, Optional

from env.gelateria import GelateriaState
from env.reward.base_reward import BaseReward


class PriceRealisationReward(BaseReward):
    """Reward function that rewards price realisation."""
    def __init__(self):
        super().__init__(name="PriceRealisationReward")

    def __call__(self, sales: Dict[str, int], state: GelateriaState,
                 previous_state: Optional[GelateriaState] = None) -> Dict[str, float]:
        """
        Get rewards that rewards price realisation.

        Args:
            sales: sales from current state
            state: current state
            previous_state: previous state (not used, default: None)
        """

        reduced_prices = self.get_reduced_price(state)
        base_prices = self.get_base_price(state)

        pr_reward = {}

        for product_id, product in state.products.items():

            sold_units = sales[product_id]

            # Maximum realisation is the maximum possible revenue where all units are sold at the base price
            max_realisation = base_prices[product_id] * state.original_stock[product_id]

            # Realised value is the revenue (sold price * units sold)
            realised_value = reduced_prices[product_id] * sold_units

            pr_reward[product_id] = realised_value / max_realisation

        return pr_reward
