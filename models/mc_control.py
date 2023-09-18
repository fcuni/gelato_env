import logging
import os
import pickle
from copy import deepcopy
from dataclasses import dataclass
from typing import List, Tuple, Union

import numpy as np
import tqdm

from env.gelateria_env_old import GelateriaEnv
from models.base_rl_agent import RLAgent
from utils.config import OptimiserConfig

logger = logging.getLogger(__name__)


@dataclass
class StateTriplet:
    """
    Dataclass for storing the state triplets (stock, actions, reward) of the environment.

    Attributes:
        stock: the stock levels of the products
        actions: the actions taken by the agent
        reward: the reward received by the agent
    """

    stock: Union[List[int], np.array]
    actions: Union[List[float], np.array]
    reward: Union[List[float], np.array]

    def __post_init__(self):
        """
        Convert the lists to numpy arrays and round the actions to two decimal places.
        """
        self.stock = np.array(self.stock)
        self.actions = np.round(100 * np.array(self.actions), 2).astype(int)
        self.reward = np.array(self.reward)

    def triplet(self) -> Tuple[np.array, np.array, np.array]:
        """
        Return the state triplet as a tuple.
        """
        return self.stock, self.actions, self.reward


@dataclass
class Trajectory:
    """
    Dataclass for storing the trajectories of the environment.

    Attributes:
        states: the states of the environment
    """

    states: List[StateTriplet]

    def append(self, state: StateTriplet):
        self.states += [state]

    def get_triplets_reversed(self):
        for state in reversed(self.states):
            yield state.triplet()

    def __len__(self):
        return len(self.states)

    def __getitem__(self, item):
        return self.states[item]

    def __iter__(self):
        return iter(self.states)


class MCControl(RLAgent):
    def __init__(
        self,
        env: GelateriaEnv,
        config: OptimiserConfig,
        name: str = "mc_control",
    ):
        """
        Initialize the MC Control algorithm.

        Args:
            env(GelateriaEnv): the environment to train on
            n_episodes(int): the number of episodes to train for
            horizon_steps(int): the number of steps to take in each episode
            name(str): the name of the algorithm
            epsilon(float): the epsilon parameter for the epsilon-greedy policy
            gamma(float): the discount factor
            warm_start(int): the number of episodes to train for before starting to update the policy
        """

        self._env = env
        self._n_episodes = config.n_episodes
        self._horizon_steps = config.horizon_steps
        self._name = name
        self._gamma = config.gamma
        self._epsilon = config.epsilon
        self._warm_start = config.warm_start

        # dims: (n_flavours, stock, reductions)
        # usually taken to be (n_flavours, 101, 101)
        self._dims = env.state_space_size
        self._dims = (self._dims[0], 101, 101)
        self._n = np.zeros(self._dims, dtype=np.float16)
        self._Q = np.random.normal(size=self._dims)
        self._G = np.zeros(self._dims[0], dtype=np.float16)
        self._policy = np.zeros(self._dims[:-1], dtype=np.float16)

        self._rng = np.random.default_rng(seed=42)

        self._rewards = []
        self._discounted_rewards = []
        self._path_to_model = config.path_to_model

    @property
    def policy(self):
        return self._policy.squeeze(axis=0)

    @property
    def q_values(self):
        return self._Q.squeeze(axis=0)

    @property
    def q_values_mean_normalised(self):
        """
        Returns the Q values with the mean subtracted from each row.
        """
        means = []
        for idx in range(self._Q.shape[0]):
            means.append(np.mean(self._Q[idx]))
        return (self._Q - np.array(means)).squeeze(axis=0)

    def _select_action(self, current_stock: List[int], mask: np.array):
        """
        Selects an action from the masked action space.
        Args:
            current_stock: The current stock of each flavour.
            mask: Mask invalid actions.

        Returns:
            The action to take.
        """
        if self._rng.random() <= self._epsilon:
            return self._select_greedy_action(current_stock=current_stock, mask=mask)
        else:
            return self._select_random_action(mask=mask)

    def _select_greedy_action(self, current_stock: List[int], mask: np.array):
        """
        Selects the greedy action from the action space.

        Args:
            current_stock: The current stock of each flavour.
            mask: Masked actions in the current state. By default, lower reductions are masked out.

        Returns:
            The greedy actions to take.
        """
        if len(mask.shape) < len(self._Q.shape):
            actions = np.take_along_axis(np.argmax(self._Q + mask.reshape(*(mask.shape), 1), axis=-1) / 100,
                                         (np.array(current_stock)[:, None]), axis=1).squeeze(-1)
        else:
            actions = (np.argmax(self._Q + mask, axis=-1) / 100)[:, current_stock]
        return actions

    def _select_random_action(self, mask: np.array):
        """Selects a random action from the masked action space with uniform probability."""
        lower_bounds = np.argmax(mask == 0, axis=-1)
        masked_actions = self._rng.integers(low=lower_bounds, high=101) / 100
        if isinstance(masked_actions, float):
            masked_actions = [masked_actions]
        return masked_actions

    def _train_step(self, trajectory: Trajectory):
        """
        Performs a single training step.

        Args:
            trajectory: The trajectory to train on.
        """
        for st, at, rt in trajectory.get_triplets_reversed():
            # dims: (n_flavours, stock, reductions)
            self._G = rt + self._gamma * self._G
            np.add.at(self._n, (slice(None), st, at), 1)
            self._Q[:, st, at] += (self._G - self._Q[:, st, at]) / self._n[:, st, at]

    def train(self):
        """
        Train the agent.
        """

        self._env.reset()
        if self._warm_start is not None:
            logger.info(f"Warm starting for {self._warm_start} steps.")
            for _ in range(self._warm_start):
                self._env.reset()
                no_op = [0] * self._env.state_space_size[0]
                self._env.step(no_op)

        for epi in tqdm.tqdm(range(self._n_episodes)):
            if epi % 100 == 0:
                # logger.info(f"Episode {epi + 1}/{self._n_episodes}")
                pass

            env = deepcopy(self._env)
            st_0 = [product.stock for product in env.state.products.values()]
            trajectory: Trajectory = Trajectory(states=[])
            is_terminal = False
            self._G = np.zeros(self._dims[0], dtype=np.float16)

            for step in range(self._horizon_steps):
                if is_terminal:
                    logger.info("Reached terminal state, ending episode.")
                    break
                mask = env.mask_actions()
                a_i = self._select_action(current_stock=st_0, mask=mask)
                _, r_i, is_terminal, _ = env.step(a_i)
                st_i = [product.stock for product in env.state.products.values()]
                trajectory.append(StateTriplet(stock=st_0, actions=a_i, reward=list(r_i.values())))
                st_0 = st_i

            self._train_step(trajectory)

            self._policy = np.round(np.argmax(self._Q, axis=-1) / 100, 2)
            self._rewards += [env.state.global_reward]
            self._discounted_rewards += [self._G.tolist()]

    def save(self):
        """Saves the model to disk."""
        os.makedirs(self._path_to_model, exist_ok=True)
        path = self._path_to_model / f"{self.name}.pkl"
        with open(path, "wb") as f:
            pickle.dump(self, f)

    def load(self):
        """Loads the model from disk."""
        with open(self._path_to_model, "rb") as f:
            base_model = pickle.load(f)
        self._Q = base_model._Q
        self._policy = base_model._policy
