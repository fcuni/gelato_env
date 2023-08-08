from typing import Dict, Optional

from env.gelateria import GelateriaState
from env.reward.base_reward import BaseReward


class TerminationWastePenalty(BaseReward):
    """Reward function that penalises unsold stock at termination."""

    def __init__(self, waste_penalty: float = 1.0):
        """
        Args:
            waste_penalty: penalty for empty shelf (should be negative value)
        """
        super().__init__(name="TerminationWastePenalty")
        self._waste_penalty = waste_penalty

    def configs(self):
        """
        Returns the reward configuration.

        Returns:
            Dict[str, Any]: reward configuration
        """
        return {"rewards/waste_penalty": self._waste_penalty}

    def __call__(self, sales: Dict[str, int], state: GelateriaState, previous_state: Optional[GelateriaState] = None) \
            -> Dict[str, float]:
        """
        Get rewards that penalises empty shelves from previous state.

        Args:
            sales: sales from current state
            state: current state
            previous_state: previous state (not used, default: None)

        Returns:
            Dict[str, float]: rewards
        """

        if state.is_terminal:
            return {product_id: self._waste_penalty * product.stock for product_id, product in state.products.items()}

        # no penalty if state is not terminal
        return {product_id: 0.0 for product_id, product in state.products.items()}
