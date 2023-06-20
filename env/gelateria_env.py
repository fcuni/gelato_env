from copy import deepcopy
from typing import List, Optional, Callable, Dict, Any
import logging

import gym
import numpy as np
from ray.rllib.utils import override

from env.gelateria import Gelato, GelateriaState
from env.reward.base_reward import BaseReward
from env.mask.action_mask import ActionMask
from env.mask.simple_masks import MonotonicMarkdownsMask
from utils.misc import first_not_none

from env.mask.simple_masks import IdentityMask

logger = logging.getLogger(name=__name__)


class GelateriaEnv(gym.Env):

    def __init__(self,
                 init_state: GelateriaState,
                 sales_model: Any,
                 reward: BaseReward,
                 mask_fn: Callable[[], ActionMask] = MonotonicMarkdownsMask,
                 restock_fct: Optional[Callable[[Gelato], int]] = None,
                 max_stock: int = 100,
                 max_steps: int = 1e8,
                 ):
        """
        Initialize the Gelateria environment.

        Args:
            init_state: Initial state of the environment.
            sales_model: Sales model to use for the environment.
            reward: Reward function to use for the environment.
            restock_fct: Function to use for restocking the products. If None, the initial stock is used.
            max_stock: Maximum stock level for each product.
            max_steps: Maximum number of steps before the environment is reset.
        """
        self._sales_model = sales_model
        self._reward = reward
        self._restock_fct = first_not_none(restock_fct, {product_id: product.stock
                                                         for product_id, product in init_state.products.items()})
        self._max_stock = max_stock

        self._state: Optional[GelateriaState] = None
        self._init_state = init_state

        self._is_reset = False
        self._global_step = 0
        self._max_steps = max_steps
        self._mask = first_not_none(mask_fn, IdentityMask)()

        self.reset()

    @property
    def state(self):
        """Return the current state of the environment."""
        return self._state

    @property
    def state_space_size(self):
        """Return the size of the state space."""
        n_flavour = len(self._state.products)
        return n_flavour, self._max_stock + 1, 101

    def mask_actions(self, state: Optional[GelateriaState] = None) -> np.array:
        """Allow only increasing markdowns."""
        if state is None:
            state = self._state
        return self._mask(state)

    def _restock(self):
        """Restock the products in the environment."""
        self._state.restock(self._restock_fct)

    @staticmethod
    def _update_stock(sales: Dict[str, float], state: GelateriaState):
        """Update the stock levels of the products in the environment.

        Args:
            sales: The sales of each product.
            state: The current state of the environment.
        """
        is_terminal = True
        for product_id, product in state.products.items():
            new_stock_level = round(max(0.0, state.products[product_id].stock - sales[product_id]))
            state.products[product_id].stock = new_stock_level
            if new_stock_level > 0:
                is_terminal = False
        state.is_terminal = is_terminal

    @staticmethod
    def _update_markdowns(action: List[float], state: GelateriaState):
        """Update the markdowns of the products in the environment.

        Args:
            action: The markdowns of each product.
            state: The current state of the environment.
        """
        for product_id, markdown in zip(state.products, action):
            state.last_markdowns[product_id] = state.current_markdowns[product_id]
            state.current_markdowns[product_id] = markdown
            state.last_action[product_id] = markdown

    def _update_internal(self,
                         observations: Dict[str, Dict],
                         from_state: Optional[GelateriaState] = None, ):
        """Update the internal state of the environment.

        Args:
            observations: The observations of the environment.
            from_state: The previous state of the environment. If None, the current state is used.
        """
        state = first_not_none(from_state, self._state)
        sales = {product_id: max(0, sales.item())
                 for product_id, sales in zip(state.products, observations["private_obs"]["sales"])}
        local_reward = self._reward(sales, state)
        self._update_stock(sales, state)

        state.local_reward = local_reward
        state.global_reward += sum(local_reward.values())
        state.step += 1
        state.day_number = (state.day_number + 1) % 365
        if state.step % state.restock_period == 0:
            self._restock()

        if from_state is None:
            self._global_step += 1

    def get_observations(self, state: GelateriaState) -> Dict[str, Dict]:
        """
        Return the observations of the environment.

        Args:
            state: The current state of the environment.

        Returns:
            The observations of the environment.
        """
        public_obs = state.get_public_observations()
        return {"public_obs": public_obs,
                "private_obs": {"sales": self._sales_model.get_sales(public_obs), }
                }

    def get_info(self):
        """Return the info of the environment."""
        return {
            "global_reward": self._state.global_reward,
        }

    def step(self, action: List[float]):
        """
        Perform an action in the environment.

        Args:
            action: The markdowns for each product.

        Returns:
            observations: The observations of the environment.
            reward: The reward of the environment.
            is_terminal: Whether the episode has terminated.
            info: The info of the environment.
        """
        self._update_markdowns(action, self._state)
        observations = self.get_observations(self._state)
        self._update_internal(observations)

        if self._global_step >= self._max_steps:
            self._state.is_terminal = True
            logger.info(f"The episode has terminated after reaching the max number of "
                        f"steps.")

        if self._state.is_terminal:
            # logger.info(f"The episode has terminated after {self._global_step} steps.")
            try:
                self._reward.get_terminal_penalty(self._state)
            except NotImplementedError:
                pass

        return observations, self._state.local_reward, self._state.is_terminal, self.get_info()

    @override(gym.Env)
    def reset(self):
        """Reset the environment."""
        self._state = deepcopy(self._init_state)

        self._is_reset = True

        return self.get_observations(self._state), self._state.local_reward, self._state.is_terminal, self.get_info()
