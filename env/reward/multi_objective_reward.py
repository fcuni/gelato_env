from typing import Dict, Optional

from env.gelateria import GelateriaState
from env.reward.base_reward import BaseReward
from env.reward.empty_shelf_penalty import EmptyShelfPenalty
from env.reward.price_realisation_reward import PriceRealisationReward
from env.reward.sell_through_reward import SellThroughReward


class MultiObjectiveReward(BaseReward):
    """Multi-objective reward function that combines sell through and price realisation rewards, as well as,
    the empty shelf penalty. It gives a scalar reward that is the weighted sum of the three rewards.

    Reward = price_realisation + sell_through_coeff * sell_through + empty_shelf_penalty_coeff * empty_shelf_penalty
    """

    def __init__(self, sell_through_coeff: float = 1.0,
                 empty_shelf_penalty_coeff: Optional[float] = 0.0,
                 empty_shelf_penalty: Optional[float] = 1.0):
        """
        Args:
            sell_through_coeff: coefficient for sell-through reward
            empty_shelf_penalty_coeff: coefficient for empty shelf penalty
            empty_shelf_penalty: penalty for empty shelf (should be negative value)
        """

        super().__init__(name="MultiObjectiveReward")

        self._sell_through_coeff = sell_through_coeff
        self._empty_shelf_penalty_coeff = empty_shelf_penalty_coeff

        self._sell_through_reward = SellThroughReward()
        self._pr_reward = PriceRealisationReward()

        self._empty_shelf_penalty = EmptyShelfPenalty(empty_shelf_penalty) if empty_shelf_penalty_coeff != 0.0 else None

    def configs(self):
        """
        Returns the reward configuration.

        Returns:
            Dict[str, Any]: reward configuration
        """
        if self._empty_shelf_penalty_coeff != 0.0:
            return {"rewards/sell_through_coeff": self._sell_through_coeff,
                    "rewards/empty_shelf_penalty_coeff": self._empty_shelf_penalty_coeff,
                    **self._sell_through_reward.configs, **self._pr_reward.configs, **self._empty_shelf_penalty.configs}
        return {"rewards/sell_through_coeff": self._sell_through_coeff,
                **self._sell_through_reward.configs, **self._pr_reward.configs}

    def __call__(self, sales: Dict[str, int], state: GelateriaState, previous_state: Optional[GelateriaState] = None) \
            -> Dict[str, float]:
        """
        Get rewards that combines sell through and price realisation reward (and the empty shelf penalty).

        Args:
            sales: sales from current state
            state: current state
            previous_state: previous state

        Returns:
            Dict[str, float]: rewards
        """

        sales_through_reward_dict = self._sell_through_reward(sales, state, previous_state)
        pr_reward_dict = self._pr_reward(sales, state, previous_state)

        reward = {}
        info = {"sell_through": [], "price_realisation": []}

        for product_id in state.products:
            reward[product_id] = self._sell_through_coeff * sales_through_reward_dict[product_id] \
                                 + pr_reward_dict[product_id]
            info["sell_through"].append(sales_through_reward_dict[product_id])
            info["price_realisation"].append(pr_reward_dict[product_id])

        if self._empty_shelf_penalty_coeff != 0.0:
            empty_shelf_penalty_dict = self._empty_shelf_penalty(sales, state, previous_state)
            info["empty_shelf_penalty"] = []
            for product_id in state.products:
                reward[product_id] += self._empty_shelf_penalty_coeff * empty_shelf_penalty_dict[product_id]
                info["empty_shelf_penalty"].append(empty_shelf_penalty_dict[product_id])
        return reward
