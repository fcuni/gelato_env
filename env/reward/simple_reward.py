from typing import Dict

from env.gelateria import GelateriaState
from env.reward.base_reward import BaseReward


def get_reduced_price(state: GelateriaState) -> Dict[str, float]:
    """Shorthand to compute the reduced price for the markdown products in the state."""
    return {
        product_id: product.base_price * (1 - state.current_markdowns[product_id])
        for product_id, product in state.products.items()
    }


class SimpleReward(BaseReward):

    def __init__(self, waste_penalty: float = 0.0):
        super().__init__()
        self._waste_penalty = waste_penalty

    def __call__(self, sales: Dict[str, float], state: GelateriaState) -> Dict[str, float]:
        remaining_stock = {product_id: product.stock for product_id, product in state.products.items()}
        reduced_prices = get_reduced_price(state)

        # sales revenue
        sold_units = {product_id: min(remaining_stock[product_id], sales[product_id])
                      for product_id in state.products}
        reward = {product_id: reduced_prices[product_id] * sold_units[product_id] for product_id in state.products}

        return reward

    def get_terminal_penalty(self, state: GelateriaState) -> float:
        total_remaining_stock = sum(product.stock for product in state.products.values())
        return self._waste_penalty * total_remaining_stock
