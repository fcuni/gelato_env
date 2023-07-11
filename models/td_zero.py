from collections import deque
import logging
import pickle
from copy import deepcopy
from dataclasses import dataclass
from typing import List, Union

import numpy as np
import tqdm
import os

from env.gelateria_env import GelateriaEnv
from models.base_rl_agent import RLAgent
from utils.config import OptimiserConfig
from utils.misc import first_not_none
import wandb

logger = logging.getLogger(__name__)


@dataclass
class StateQuad:
    stock: Union[List[int], np.ndarray] # can be a list of integers or a numpy array
    actions: Union[List[float], np.ndarray] # can be a list of floats or a numpy array
    reward: Union[List[float], np.ndarray] # can be a list of floats or a numpy array
    next_stock: Union[List[int], np.ndarray] # can be a list of integers or a numpy array

    def __post_init__(self):
        self.stock = np.array(self.stock)
        self.actions = np.round(100 * np.array(self.actions), 2).astype(int)
        self.reward = np.array(self.reward)
        self.next_stock = np.array(self.next_stock)

    def quad(self):
        return self.stock, self.actions, self.reward, self.next_stock


class TDZero(RLAgent):
    def __init__(
        self,
        env: GelateriaEnv,
        config: OptimiserConfig,
        name: str = "TDZero",
        
    ):
        super().__init__(env=env, config=config, name=name)

        # TODO(cunillera): get rid of this and use base class self.config to access config
        self._n_episodes = config.n_episodes
        self._horizon_steps = config.horizon_steps
        self._gamma = config.gamma
        self._epsilon = config.epsilon
        self._alpha = config.alpha
        self._warm_start = config.warm_start
        self._path_to_model = config.path_to_model

        # dims: (n_flavours, stock, reductions)
        # usually taken to be (n_flavours, 101, 101)
        self._dims = env.state_space_size
        self._Q = first_not_none(config.q_init, np.random.normal(size=self._dims))
        self._G = np.zeros(self._dims[0], dtype=np.float16)
        self._policy = np.zeros(self._dims[:-1], dtype=np.float16)

        self._rng = np.random.default_rng(seed=self.config.seed)

        self._rewards = []
        self._discounted_rewards = []

    @property
    def policy(self):
        return self._policy.squeeze(axis=0)

    @property
    def q_values(self):
        return self._Q.squeeze(axis=0)

    @property
    def q_values_mean_normalised(self):
        means = []
        for idx in range(self._Q.shape[0]):
            means.append(np.mean(self._Q[idx]))
        return (self._Q - np.array(means)).squeeze(axis=0)

    def _select_action(self, current_stock: List[int], mask: np.ndarray):
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

    def _select_greedy_action(self, current_stock: List[int], mask: np.ndarray):
        """
        Selects the greedy action from the action space.

        Args:
            current_stock: The current stock of each flavour.
            mask: Masked actions in the current state. By default, lower reductions are masked out.

        Returns:
            The greedy actions to take.
        """
        # TODO(cunillera): sub np.argmax for unbiased argmax
        
        # need to delete later
        # print(f"current stock: {current_stock}, shape: ({len(current_stock)},)")
        # print(f"mask: {mask.shape}, values: {mask}")
        # print(f"Q: {self._Q.shape}")
        # print(f"{}")
        if len(mask.shape)< len(self._Q.shape):
            # print(f"np.argmax(self._Q + mask.reshape(*(mask.shape), 1), axis=-1)/100 shape {(np.argmax(self._Q + mask.reshape(*(mask.shape), 1), axis=-1)/100).shape}")
            # print(f"np.argmax(self._Q + mask.reshape(*(mask.shape), 1))/100 {np.argmax(self._Q + mask.reshape(*(mask.shape), 1), axis=-1)/100}")
            # print(f"np.array(current_stock)[:,None] {np.array(current_stock)[:,None]}")
            actions = np.take_along_axis(np.argmax(self._Q + mask.reshape(*(mask.shape), 1), axis=-1)/100, (np.array(current_stock)[:,None]), axis=1).squeeze(-1)
        else:
            actions = (np.argmax(self._Q + mask, axis=-1) / 100)[:, current_stock]
        # print(f"actions:{actions}")
        return actions

    def _select_random_action(self, mask: np.ndarray):
        """Selects a random action from the masked action space with uniform probability."""
        # TODO(cunillera): sub np.argmax for unbiased argmax
        lower_bounds = np.argmax(mask == 0, axis=-1)
        masked_actions = self._rng.integers(low=lower_bounds, high=101) / 100
        if isinstance(masked_actions, float):
            masked_actions = [masked_actions]
        return masked_actions

    def _train_step(self, step: StateQuad):
        """
        Performs a single training step.

        Args:
            step: The step to train on.
        """
        st, at, rt, st_next = step.quad()
        # dims: (n_flavours, stock, reductions)
        self._G = rt + self._gamma * np.max(self._Q, axis=-1)[:, st_next]
        self._Q[:, st, at] += self._alpha * (self._G - self._Q[:, st, at])

    def train(self):
        """
        Trains the agent.
        """
        average10 = deque(maxlen=10)
        total_steps = 0
        episodes = 1000
        wandb_config = {
            "algorithm": "TD_Zero",
            "episodes": episodes,
            "environment": "gelateria"

        }
        with wandb.init(project="msc_project", entity="timc", config=wandb_config, mode="offline"):

            self._env.reset()
            if self._warm_start is not None:
                logger.info(f"Warm starting for {self._warm_start} steps.")
                for _ in range(self._warm_start):
                    self._env.reset()
                    no_op = [0] * self._env.state_space_size[0]
                    self._env.step(no_op)

            

            
            for epi in tqdm.tqdm(range(episodes)):
            # for epi in tqdm.tqdm(range(self._n_episodes)):
                if epi % 100 == 0:
                    # logger.info(f"Episode {epi + 1}/{self._n_episodes}")
                    pass

                env = deepcopy(self._env)
                st_0 = [product.stock for product in env.state.products.values()]
                is_terminal = False
                self._G = np.zeros(self._dims[0], dtype=np.float16)
                episode_steps = 0

                for step in range(self._horizon_steps):
                    if is_terminal:
                        # logger.info("Reached terminal state, ending episode.")
                        break
                    mask = env.mask_actions()
                    a_i = self._select_action(current_stock=st_0, mask=mask)
                    _, r_i, is_terminal, _ = env.step(a_i)
                    r_i = [r for r in r_i.values()]
                    st_i = [product.stock for product in env.state.products.values()]
                    step = StateQuad(st_0, a_i, r_i, st_i)
                    self._train_step(step)
                    st_0 = st_i
                    episode_steps += 1

                self._policy = np.round(np.argmax(self._Q, axis=-1) / 100, 2)
                self._rewards += [env.state.global_reward]
                self._discounted_rewards += [self._G.tolist()]

                average10.append(env.state.global_reward)
                total_steps += episode_steps

                wandb.log({"Reward": env.state.global_reward,
                        "Average10": np.mean(average10),
                        "Episodic steps": episode_steps,
                        "Steps": total_steps,
                        # "Policy Loss": policy_loss,
                        # "Alpha Loss": alpha_loss,
                        # "Bellmann error 1": bellmann_error1,
                        # "Bellmann error 2": bellmann_error2,
                        # "Alpha": current_alpha,
                        "Episode": epi,
                        })


    def save(self):
        """Saves the model to disk."""
        os.makedirs(self.config.path_to_model, exist_ok=True)
        path = self.config.path_to_model / f"{self.name}.pkl"
        with open(path, "wb") as f:
            pickle.dump(self, f)

    def load(self):
        """Loads the model from disk."""
        path = self.config.path_to_model / f"{self.name}.pkl"
        with open(path, "rb") as f:
            base_model = pickle.load(f)
        self._Q = base_model._Q
        self._policy = base_model._policy
