from typing import Dict, Optional

from env.gelateria import GelateriaState
from env.reward.base_reward import BaseReward


def get_reduced_price(state: GelateriaState) -> Dict[str, float]:
    """Shorthand to compute the reduced price for the markdown products in the state."""
    return {
        product_id: product.base_price * (1 - state.current_markdowns[product_id])
        for product_id, product in state.products.items()
    }


class SimpleRewardEarlyTerminationPenalty(BaseReward):
    """Reward function that assign simple rewards from sales revenue.
    It also penalises unsold stock at termination and early termination.

    Args:
        waste_penalty: penalty for unsold stock at termination
        early_term_penalty: penalty for early episode termination (episode ends before specified number of sales days.)
    """

    def __init__(self, waste_penalty: float = 0.0, early_term_penalty: float = 0.0):
        super().__init__(name="SimpleRewardEarlyTerminationPenalty")
        self._waste_penalty = waste_penalty
        self._early_term_penalty = early_term_penalty

    @property
    def configs(self):
        """
        Returns the reward configuration.

        Returns:
            Dict[str, Any]: reward configuration
        """
        return {"rewards/waste_penalty": self._waste_penalty,
                "rewards/early_term_penalty": self._early_term_penalty}

    def __call__(self, sales: Dict[str, float], state: GelateriaState,
                 previous_state: Optional[GelateriaState] = None) -> Dict[str, float]:
        remaining_stock = {product_id: product.stock for product_id, product in state.products.items()}
        reduced_prices = get_reduced_price(state)

        # sales revenue
        sold_units = {product_id: min(remaining_stock[product_id], sales[product_id])
                      for product_id in state.products}
        reward = {product_id: reduced_prices[product_id] * sold_units[product_id] for product_id in state.products}

        return reward

    def get_terminal_penalty(self, state: GelateriaState) -> Dict[str, float]:
        # TODO: confirm the way how to track sales_days
        early_termination_penalty_factor: float = max(0, state.sales_days - state.step) * self._early_term_penalty
        early_termination_penalty = {product_id: early_termination_penalty_factor * state.local_reward[product_id]
                                     for product_id in state.products}
        remaining_stock_penalty = {product_id: self._waste_penalty * product.stock
                                   for product_id, product in state.products.items()}
        return {product_id: remaining_stock_penalty[product_id] + early_termination_penalty[product_id]
                for product_id in state.products}
