from copy import deepcopy
from datetime import timedelta
from typing import List, Optional, Callable, Dict, Any, Union, Sequence, Tuple
import logging

import gym
from gym.spaces import Box, Discrete, MultiBinary
# import gymnasium as gym
import numpy as np
from ray.rllib.utils import override
import torch

from env.gelateria import Gelato, GelateriaState
from env.markdown_trigger.base_trigger import BaseTrigger
from env.markdown_trigger.triggers import DefaultTrigger
from env.reward.base_reward import BaseReward
from env.mask.action_mask import ActionMask
from env.mask.simple_masks import MonotonicMarkdownsMask
# from models.sales.dataset import transform_gym_inputs
from utils.enums import Flavour
from utils.misc import first_not_none

from env.mask.simple_masks import IdentityMask

logger = logging.getLogger(name=__name__)


class OneHotEncoding(MultiBinary):
    def __init__(self, n: Union[np.ndarray, Sequence[int], int],
                 seed: Optional[Union[int, np.random.Generator]] = None):
        assert isinstance(n, int), "n must be an int"
        super().__init__(n=n, seed=seed)

    def sample(self, mask: Optional[np.ndarray] = None) -> np.ndarray:
        one_hot_arr = np.zeros(self.n, dtype=self.dtype)
        index = self.np_random.integers(low=0, high=self.n, dtype=int)
        one_hot_arr[index] = 1
        return one_hot_arr

    def contains(self, x) -> bool:
        """Return boolean specifying if x is a valid member of this space."""
        if isinstance(x, Sequence):
            x = np.array(x)  # Promote list to array for contains check

        return bool(
            isinstance(x, np.ndarray)
            and self.shape == x.shape
            and np.all((x == 0) | (x == 1))
            and np.sum(x) == 1
        )

    def __repr__(self):
        return "OneHotEncoding(%d)" % self.n

    def __eq__(self, other):
        return self.n == other.n


class GelateriaEnv_v2(gym.Env):

    def __init__(self,
                 init_state: GelateriaState,
                 sales_model: Any,
                 reward: BaseReward,
                 mask_fn: Callable[[], ActionMask] = MonotonicMarkdownsMask,
                 restock_fct: Optional[Callable[[Gelato], int]] = None,
                 max_stock: Optional[int] = None,
                 max_steps: int = 10,  # int(1e8)
                 days_per_step: int = 7,
                 obs_transform_fn: Optional[Callable[[torch.Tensor], torch.Tensor]] = None
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

        super(GelateriaEnv_v2, self).__init__()

        self._name = "GelateriaEnv_v2"
        self._sales_model = sales_model
        self._reward = reward
        self._restock_fct = first_not_none(restock_fct, {product_id: product.stock
                                                         for product_id, product in init_state.products.items()})
        # self._max_stock = max_stock

        self._state: Optional[GelateriaState] = None
        self._init_state = init_state

        self._is_reset = False
        self._global_step = 0
        self._max_steps = max_steps
        self._days_per_step = days_per_step
        # self._obs_transform_fn = transform_gym_inputs if obs_transform_fn is not None else lambda x: x

        self._mask = first_not_none(mask_fn, IdentityMask)()
        # Define the observation and action spaces
        observation_spaces = {
            'current_markdown': Box(low=0, high=1, dtype=np.float32),
            'day_of_year': Box(low=0, high=365, dtype=int),  # current day of the year
            'available_stock': Box(low=0, high=np.inf, dtype=int),  # 'stock_level': Box(low=0, high=np.inf, dtype=int),
            'normalised_base_price': Box(low=0, high=1, dtype=np.float32),  # normalised
            'last_sales': Box(low=0, high=np.inf, dtype=np.float32),
            'flavour': OneHotEncoding(n=len(Flavour.get_all_flavours()))
        }

        self.observation_space = gym.spaces.Dict(observation_spaces)
        self.action_space = Discrete(101)  # define the action space as discrete
        self.reset()

    @property
    def name(self):
        """Return the name of the environment."""
        return self._name

    @property
    def sales_model_name(self):
        """Return the name of the sales model."""
        return self._sales_model.name

    @property
    def reward_type_name(self):
        """Return the name of the reward type."""
        return self._reward.name

    @property
    def action_mask_name(self):
        return self._mask.name

    @property
    def state(self):
        """Return the current state of the environment."""
        return self._state

    # @property
    # def state_space_size(self):
    #     """Return the size of the state space."""
    #     assert self._state is not None
    #
    #     n_flavour = len(self._state.products)
    #     return n_flavour, self._max_stock + 1, 101

    def set_state(self, state: GelateriaState):
        """Load the input state as the state of the environment.
        """
        self._state = state

    def mask_actions(self, state: Union[GelateriaState, torch.Tensor] = None) -> np.ndarray:
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
            # if abs(sales[product_id]) >= 1:
            #     print("stock_level change")
            new_stock_level = round(max(0, state.products[product_id].stock - sales[product_id]))
            state.products[product_id].stock = new_stock_level
            # The episode only terminates if all products are out of stock
            if new_stock_level > 0:
                is_terminal = False

        state.is_terminal = is_terminal

    @staticmethod
    def _update_markdowns(action: Union[Sequence[float], Sequence[int]], state: GelateriaState):
        """Update the markdowns of the products in the environment.

        Args:
            action: The markdowns of each product. If the markdown is an integer, it is divided by 100.
            state: The current state of the environment.
        """

        assert (state.last_markdowns is not None) and (state.current_markdowns is not None) and (
                state.last_actions is not None)

        for product_id, markdown in zip(state.products, action):
            if isinstance(markdown, int):
                markdown = round(markdown / 100, 2)
            elif isinstance(markdown, float):
                markdown = round(markdown, 2)

            # TODO: remove the markdown action count restriction (only for testing)
            # if state.historical_actions_count(product_id)[product_id] <= 3:
            if state.current_markdowns is not None:
                if not state.current_markdowns == markdown:
                    state.last_markdowns[product_id] = state.current_markdowns[product_id]
                    state.current_markdowns[product_id] = markdown

            # Add current markdown to last_actions (list of all previous markdowns)
            state.last_actions[product_id].append(markdown)

    def _update_terminal_reward(self, state: GelateriaState, terminal_reward: Dict[str, float]):
        """Update the reward of the terminal state.

        Args:
            state: The current state of the environment.
            terminal_reward: The reward of the terminal state.
        """
        assert state.is_terminal

        for product_id in state.products:
            state.local_reward[product_id] -= terminal_reward[product_id]
            state.global_reward -= state.local_reward[product_id]

    def _update_internal(self,
                         observations: Dict[str, Union[torch.Tensor, Dict]],
                         from_state: Optional[GelateriaState] = None,
                         prev_state: Optional[GelateriaState] = None):
        """Update the internal state of the environment.

        Args:
            observations: The observations of the environment.
            from_state: The previous state of the environment. If None, the current state is used.
        """

        state = first_not_none(from_state, self._state)
        sales = observations['private_obs']['sales']

        # calculate clipped sales and update historical sales in state
        for product_id in state.products:
            state.historical_sales[product_id].append(sales[product_id])

        # calculate local and global reward
        local_reward = self._reward(sales=sales, state=state, previous_state=prev_state)
        state.local_reward = local_reward
        state.global_reward += sum(local_reward.values())

        # update stock levels
        self._update_stock(sales, state)

        # update step and date
        state.step += 1
        state.current_date = state.current_date + timedelta(days=self._days_per_step)

        # restock if necessary
        if state.step % state.restock_period == 0:
            self._restock()

        # update global step (only if no from_state is specified)
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

        # public_obs = state.get_public_observations()

        # TODO: this should move to get_public_observations in gelateria.py
        public_obs_tensor = []
        all_flavour_encoding = Flavour.get_flavour_encoding()
        n_flavours = len(Flavour.get_all_flavours())
        for product_id, product in state.products.items():
            flavour_encoding = all_flavour_encoding[product.flavour.value]

            # get the observations for each product
            current_markdown = torch.tensor(state.current_markdowns[product_id])
            day_of_year = torch.tensor(state.current_date.timetuple().tm_yday) - 1  # the -1 is to make it 0-indexed
            available_stock = torch.tensor(product.stock)
            base_price = torch.tensor(product.base_price)
            last_sales = torch.tensor(0.0) if len(state.historical_sales[product_id]) == 0 else \
                torch.tensor(state.historical_sales[product_id][-1])
            flavour_one_hot = torch.nn.functional.one_hot(torch.tensor(flavour_encoding), n_flavours)
            public_obs_tensor.append(torch.hstack(
                [current_markdown, day_of_year, available_stock, base_price, last_sales, flavour_one_hot]).float())
            public_obs = torch.vstack(public_obs_tensor)

        sales, sales_info = self._sales_model.get_sales(public_obs, output_info=True)
        combined_sales_info = {
            "sales_not_clipped": {product_id: sales[i] for i, product_id in enumerate(state.products)},
            "sales": {product_id: max(0, min(state.products[product_id].stock, int(sales[i]))) for i, product_id in
                      enumerate(state.products)}}
        for info_key, info_val in sales_info.items():
            if isinstance(info_val, list):
                combined_sales_info[info_key] = {product_id: info_val[i] for i, product_id in enumerate(state.products)}
            else:
                combined_sales_info[info_key] = info_val

        return {"public_obs": public_obs, "private_obs": combined_sales_info}

    def get_info(self):
        """Return the info of the environment."""

        assert self._state is not None

        return {
            "global_reward": self._state.global_reward,
        }

    def step(self, action: Union[List[float], List[int], np.ndarray, torch.Tensor], action_dtype: Optional[str] = None):
        """
        Perform an action in the environment.

        Args:
            action: The markdowns for each product.
            action_dtype: [Optional] The date type of the actions. For example, if action_dtype == int, they will be
                automatically divided by 100 to fit in the range of [0,1].

        Returns:
            observations: The observations of the environment.
            reward: The reward of the environment.
            is_terminal: Whether the episode has terminated.
            info: The info of the environment.
        """

        assert self._state is not None, "The environment must be reset before stepping it."
        if action_dtype is not None:
            if action_dtype == "int":
                if isinstance(action, List):
                    for i in range(len(action)):
                        action[i] = round(action[i] / 100, 2)
                elif isinstance(action, np.ndarray):
                    action = np.round(action.astype(int) / 100, 2)
                elif isinstance(action, torch.Tensor):
                    action = np.round(action.int().numpy() / 100, 2)
                else:
                    raise ValueError("The action must be a list, a numpy array or a torch tensor.")
        prev_state = deepcopy(self._state)
        self._update_markdowns(action, self._state)
        observations = self.get_observations(self._state)
        self._update_internal(observations, prev_state=prev_state)
        observations['public_obs'] = self.get_observations(self._state)['public_obs']

        if self._global_step >= self._max_steps:
            self._state.is_terminal = True
            logger.info(f"The episode has terminated after reaching the max number of "
                        f"steps.")

        # TODO: check if this is the right way to incorporate the terminal penalty
        if self._state.is_terminal:
            # logger.info(f"The episode has terminated after {self._global_step} steps.")
            try:
                terminal_penalty: Dict[str, float] = self._reward.get_terminal_penalty(self._state)
                self._update_terminal_reward(self._state, terminal_penalty)
            except NotImplementedError:
                pass

        info = {**(observations['private_obs']), **(self.get_info())}

        return observations, self._state.local_reward, self._state.is_terminal, info

    # def get_single_observation_space_size(self):
    #     """Return the observation space size of a single agent. Public observations only."""
    #     assert self._state is not None
    #     return self._state.get_public_observations().shape

    def sample(self, size: Optional[int] = None) -> np.ndarray:
        """Sample observations from the environment.

        Args:
            size: The number of observations to sample. If None, a single observation is sampled.
        """

        def sample_one_obs():
            sampled_obs = self.observation_space.sample()
            # TODO: check if we want to keep 365 in config
            day_of_year = np.clip(np.round(sampled_obs['day_of_year'] * 365), a_min=0, a_max=365) / 365
            md = np.round(sampled_obs['current_markdowns'], 2)
            # [RANDOM PRODUCT APPROACH]: sample a product from the flavour encoding index of each product available in
            #   the Gelato shop
            random_product = np.random.choice([Flavour.get_flavour_encoding()[self._state.products[
                list(self._state.products.keys())[i]].flavour.value] for i in range(self._state.n_products)])
            flavour = np.zeros(len(Flavour.get_all_flavours()), dtype=np.float32)
            flavour[random_product] = 1.0
            stock_level = sampled_obs['stock_level']
            base_price = sampled_obs['base_price']
            return np.concatenate([day_of_year, stock_level, base_price, md, flavour])

        if size is None:
            return sample_one_obs()
        else:
            return np.array([sample_one_obs() for _ in range(size)])

    def sample_from_current_store(self) -> Tuple[GelateriaState, np.ndarray]:
        """Sample observations from the current store (based on the products in the store)

        Returns:
            sampled_state: A sampled state of the environment.
            sampled_observation: The public observation of the sampled observation.
        """
        assert self._state is not None, "The environment must be reset before sampling from it."
        current_obs = self._state.get_public_observations()
        sampled_obs = self.sample(size=self._state.n_products)
        day_of_year = (np.array([sampled_obs[0, 0]] * self._state.n_products)).reshape(-1, 1)
        stock_level = (np.round(self._state.max_stock * sampled_obs[:, 1]) / self._state.max_stock).reshape(-1, 1)
        base_price = current_obs[:, 2].reshape(-1, 1)
        md = sampled_obs[:, 3].reshape(-1, 1)
        flavour = current_obs[:, -len(Flavour.get_all_flavours()):]
        sampled_observation = np.concatenate([day_of_year, stock_level, base_price, md, flavour], axis=1)

        sampled_state = deepcopy(self._init_state)
        sampled_state.day_number = round(day_of_year[0, 0] * 365)

        for i, key in enumerate(sampled_state.products.keys()):
            sampled_state.products[key].stock = round(stock_level[i, 0] * sampled_state.max_stock)
            sampled_state.current_markdowns[key] = md[i, 0]
            # sampled_state.last_markdowns[key] = md[i, 0]
            # sampled_state.last_actions[key].append(md[i, 0])

        return sampled_state, sampled_observation

    # def hallucinate(self, size: Optional[int] = None) -> np.ndarray:
    #         """Sample observations from the environment.
    #
    #         Args:
    #             size: The number of observations to sample. If None, a single observation is sampled.
    #         """
    #         sample_obs = self.sample()
    #         sample_state = deepcopy(self._state)
    #         sample_state.day_number = np.clip(np.round(sample_obs[0]*365), a_min=0, a_max=365)
    #         sample_state.stock_level = sample_obs[1]

    @override(gym.Env)
    def reset(self):

        """Reset the environment."""
        self._state = deepcopy(self._init_state)

        self._is_reset = True
        self._global_step = 0

        return self.get_observations(self._state), self._state.local_reward, self._state.is_terminal, self.get_info()
