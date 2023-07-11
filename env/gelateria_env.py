from copy import deepcopy
from typing import List, Optional, Callable, Dict, Any, Union
import logging

import gym
from gym.spaces import Box, Discrete
#import gymnasium as gym
import numpy as np
from ray.rllib.utils import override
import torch

from env.gelateria import Gelato, GelateriaState
from env.reward.base_reward import BaseReward
from env.mask.action_mask import ActionMask
from env.mask.simple_masks import MonotonicMarkdownsMask
from utils.enums import Flavour
from utils.misc import first_not_none

from env.mask.simple_masks import IdentityMask

logger = logging.getLogger(name=__name__)


class OneHotEncoding(gym.Space):
    """
    {0,...,1,...,0}

    Example usage:
    self.observation_space = OneHotEncoding(size=4)
    """
    def __init__(self, size=None):
        assert isinstance(size, int) and size > 0
        self.size = size
        gym.Space.__init__(self, (), np.int64)

    def sample(self):
        one_hot_vector = np.zeros(self.size)
        one_hot_vector[np.random.randint(self.size)] = 1
        return one_hot_vector

    def contains(self, x):
        if isinstance(x, (list, tuple, np.ndarray)):
            number_of_zeros = list(x).contains(0)
            number_of_ones = list(x).contains(1)
            return (number_of_zeros == (self.size - 1)) and (number_of_ones == 1)
        else:
            return False

    def __repr__(self):
        return "OneHotEncoding(%d)" % self.size

    def __eq__(self, other):
        return self.size == other.size
    
    

class GelateriaEnv(gym.Env):

    def __init__(self,
                 init_state: GelateriaState,
                 sales_model: Any,
                 reward: BaseReward,
                 mask_fn: Callable[[], ActionMask] = MonotonicMarkdownsMask,
                 restock_fct: Optional[Callable[[Gelato], int]] = None,
                 max_stock: int = 100,
                 max_steps: int = int(1e8),
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
        self._name = "GelateriaEnv"
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

        
        spaces = {
            'day_of_year': Box(low=0, high=1, dtype=np.float32), # current day of the date / # of days in the year
            'stock_level': Box(low=0, high=np.inf, dtype=int),
            'current_markdowns': Box(low=0, high=1, dtype=np.float32),
            'base_price': Box(low=0, high=np.inf, dtype=np.float32),
            'flavour': OneHotEncoding(size=len(Flavour.get_all_flavours()))
        }
        self.observation_space = gym.spaces.Dict(spaces)
        self.action_space = Discrete(101) # define the action space as discrete

        self.reset()

    @property
    def name(self):
        return self._name

    @property
    def sales_model_name(self):
        return self._sales_model.name

    @property
    def state(self):
        """Return the current state of the environment."""
        return self._state

    @property
    def state_space_size(self):
        """Return the size of the state space."""
        assert self._state is not None

        n_flavour = len(self._state.products)
        return n_flavour, self._max_stock + 1, 101

    def mask_actions(self, state: Optional[GelateriaState] = None) -> np.ndarray:
        """Allow only increasing markdowns."""
        if state is None:
            state = self._state
        return self._mask(state)

    def _restock(self):
        """Restock the products in the environment."""
        assert self._state is not None
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
    def _update_markdowns(action: Union[List[float], List[int]], state: GelateriaState):
        """Update the markdowns of the products in the environment.

        Args:
            action: The markdowns of each product.
            state: The current state of the environment.
        """

        assert (state.last_markdowns is not None) and (state.current_markdowns is not None) and (state.last_action is not None)

        for product_id, markdown in zip(state.products, action):
            if isinstance(markdown, int):
                markdown = round(markdown / 100, 2)
            elif isinstance(markdown, float):
                markdown = round(markdown, 2)
            state.last_markdowns[product_id] = state.current_markdowns[product_id]
            state.current_markdowns[product_id] = markdown
            state.last_action[product_id] = markdown

    def _update_internal(self,
                         observations: Dict[str, Union[torch.Tensor, Dict]],
                         from_state: Optional[GelateriaState] = None):
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

    def get_observations(self, state: GelateriaState) -> Dict[str, Union[torch.Tensor, Dict]]:
        """
        Return the observations of the environment.

        Args:
            state: The current state of the environment.

        Returns:
            The observations of the environment.
        """

        assert state is not None

        public_obs = state.get_public_observations()
        return {"public_obs": public_obs,
                "private_obs": {"sales": self._sales_model.get_sales(public_obs), }
                }

    def get_info(self):
        """Return the info of the environment."""

        assert self._state is not None

        return {
            "global_reward": self._state.global_reward,
        }

    def step(self, action: Union[List[float], List[int]]):
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

        assert self._state is not None, "The environment must be reset before stepping it."

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

    def get_single_observation_space_size(self):
        """Return the observation space size of a single agent. Public observations only."""
        assert self._state is not None
        return self._state.get_public_observations().shape

    @property
    def product_stocks(self)->List[int]:
        assert self._state is not None
        return [product.stock for product in self._state.products.values()]

    @property
    def per_product_done_signal(self)->List[bool]:
        assert self._state is not None
        return [product.stock == 0 for product in self._state.products.values()]

    @override(gym.Env)
    def reset(self):

        """Reset the environment."""
        self._state = deepcopy(self._init_state)

        self._is_reset = True

        return self.get_observations(self._state), self._state.local_reward, self._state.is_terminal, self.get_info()
