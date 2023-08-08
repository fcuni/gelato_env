from typing import Dict, Optional

from env.gelateria import GelateriaState
from env.reward.base_reward import BaseReward


class SimpleReward(BaseReward):
    """Reward function that assign simple rewards from sales revenue.
    It also penalises unsold stock at termination if `waste_penalty` is specified.
    """

    def __init__(self):
        super().__init__(name="SimpleReward")

    def __call__(self, sales: Dict[str, int], state: GelateriaState,
                 previous_state: Optional[GelateriaState] = None) -> Dict[str, float]:
        """
        Get rewards that rewards sales revenue.

        Args:
            sales: sales from current state
            state: current state
            previous_state: previous state (not used, default: None)
        """

        reduced_prices = self.get_reduced_price(state)

        # sales revenue
        sales_revenue = {product_id: reduced_prices[product_id] * sales[product_id] for product_id in state.products}

        return sales_revenue
