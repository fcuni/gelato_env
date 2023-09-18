from typing import Tuple

import numpy as np

from env.gelateria_env_old import GelateriaEnv


class UCB:
    # TODO extend to multi-dimensional action space

    def __init__(self,
                 env: GelateriaEnv,
                 horizon_steps: int,
                 n_arms: int = 101,
                 delta: float = 0.1,
                 name: str = "UCB",
                 dims: Tuple[int, ...] = None
                 ):
        assert 0 < delta < 1, "delta must be in (0,1)"
        assert env.state.n_products == 1, "UCB only works for one product"
        self._env = env
        self._H = horizon_steps
        self._c = np.log(horizon_steps * n_arms / delta)
        self._n_arms = n_arms
        self._name = name

        self._dims = dims if dims is not None else (n_arms,)

        self._t = 1
        self._n = np.zeros(self._dims[1:])
        self._Q = np.zeros(self._dims[1:])

        self._rewards = []
        self._actions = []

    @property
    def name(self):
        return self._name

    def reset(self):
        self._t = 1
        self._n = np.zeros(self._n_arms)
        self._Q = np.zeros(self._n_arms)

    def get_action(self):
        if self._t <= self._n_arms:
            return [self._t - 1]
        else:
            return [np.argmax(self._Q + np.sqrt(self._c / self._n))]

    def train_step(self, action, reward):
        self._n[action] += 1
        self._Q[action] += (reward - self._Q[action]) / self._n[action]
        self._t += 1
        return self.get_action()

    def train(self):
        is_terminal = False
        action = None
        self.reset()

        for i in range(self._H):
            if is_terminal or action is None:
                obs, reward, is_terminal, _ = self._env.reset()
                action = self.get_action()
            else:
                obs, reward, is_terminal, _ = self._env.step(action)

            action = self.train_step(action, reward)

            self._rewards.append(reward)
            self._actions.append(action)
            if i > self._n_arms:
                pass

    def get_reward(self, action):
        return self._Q[action]
