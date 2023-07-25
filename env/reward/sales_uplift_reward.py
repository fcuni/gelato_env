from typing import Dict, Any, Optional

import torch

from env.gelateria import GelateriaState
from env.reward.base_reward import BaseReward


def get_reduced_price(state: GelateriaState) -> Dict[str, float]:
    """Shorthand to compute the reduced price for the markdown products in the state."""
    return {
        product_id: product.base_price * (1 - state.current_markdowns[product_id])
        for product_id, product in state.products.items()
    }


def get_sales_revenue(sales: Dict[str, float], state: GelateriaState):
    """Shorthand to compute the sales revenue for the markdown products in the state."""
    remaining_stock = {product_id: product.stock for product_id, product in state.products.items()}
    reduced_prices = get_reduced_price(state)
    sold_units = {product_id: min(remaining_stock[product_id], round(sales[product_id])) for product_id in
                  state.products}
    sales_revenue = {product_id: reduced_prices[product_id] * sold_units[product_id] for product_id in
                     state.products}
    return sales_revenue


class SalesUpliftReward(BaseReward):

    def __init__(self, sales_model: Any, markdown_penalty: Optional[float] = None,
                 waste_penalty: Optional[float] = None):
        """
        Reward function that rewards the sales uplift compared to the sales predictions of with the original markdown.

        Args:
            sales_model: Sales model that predicts the sales given state observations.
            markdown_penalty: Penalty/cost for changing the markdown level for each product.
            waste_penalty: Penalty/cost for each unsold unit at termination.
        """

        super().__init__(name="SalesUpliftReward")

        if markdown_penalty is None:
            markdown_penalty = 0.0
        self._markdown_penalty = markdown_penalty

        if waste_penalty is None:
            waste_penalty = 0.0
        self._waste_penalty = waste_penalty

        if not callable(sales_model):
            raise ValueError("The sales model must be callable.")
        self._sales_model = sales_model

    @property
    def configs(self):
        """
        Returns the reward configuration.

        Returns:
            Dict[str, Any]: reward configuration
        """
        return {"rewards/markdown_penalty": self._markdown_penalty, "rewards/waste_penalty": self._waste_penalty}

    def __call__(self, sales: Dict[str, float], state: GelateriaState,
                 previous_state: Optional[GelateriaState] = None) -> Dict[str, float]:

        if previous_state is None:
            raise ValueError(f'The previous state must be provided for using {self.name}.')

        # get the could-have-been sales from the sales model if the markdown remains the same
        could_have_been_observation = previous_state.get_public_observations()
        if isinstance(self._sales_model, torch.nn.Module):
            could_have_been_observation = could_have_been_observation.to(self._sales_model.device)
        could_have_been_sales_predictions = self._sales_model.get_sales(could_have_been_observation)
        could_have_been_sales = {
            product_id: max(0.0, alt_sales.item() if isinstance(could_have_been_sales_predictions,
                                                                torch.Tensor) else alt_sales)
            for product_id, alt_sales in zip(state.products, could_have_been_sales_predictions)}

        # compute the sales difference between the actual sales and the could-have-been sales
        revenue = get_sales_revenue(sales, state)
        could_have_been_revenue = get_sales_revenue(could_have_been_sales, previous_state)
        revenue_diff = {
            product_id: revenue[product_id] * sales[product_id] - could_have_been_revenue[product_id] *
                        could_have_been_sales[product_id]
            if (state.last_markdowns[product_id] is not None) and (
                        state.current_markdowns[product_id] != state.last_markdowns[product_id]) else 0.0
            for product_id in state.products}

        # check if the markdowns have changed, assign markdown penalty if so
        price_change_penalty = {
            product_id: (-1.0 * self._markdown_penalty)
            if (state.last_markdowns[product_id] is not None) and (
                        state.current_markdowns[product_id] != state.last_markdowns[product_id]) else 0.0
            for product_id in state.products}

        # compute the reward by adding the sales difference and the markdown penalty
        reward = {product_id: revenue_diff[product_id] + price_change_penalty[product_id] for product_id in
                  state.products}

        return reward

    def get_terminal_penalty(self, state: GelateriaState) -> Dict[str, float]:
        remaining_stock_penalty = {product_id: self._waste_penalty * product.stock
                                   for product_id, product in state.products.items()}
        return remaining_stock_penalty
