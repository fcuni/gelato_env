from collections import deque

from pathlib import Path
from typing import Deque, Optional, Dict, Any, Union

import gym
import numpy as np
import torch

EnvType = Union[gym.Env, gym.core.Env]
from models.base_rl_agent import RLAgent
from wandb.wandb_run import Run

from utils.config import RuleBasedAgentConfig

from utils.types import TensorType
from utils.logging import EpisodeLogger


class RuleBasedAgent(RLAgent):

    def __init__(self, env: EnvType,
                 config: RuleBasedAgentConfig,
                 name: str = "RuleBasedAgent",
                 device: Optional[torch.device] = None,
                 run_name: Optional[str] = None):

        super().__init__(env, name, run_name=run_name)
        self._config: RuleBasedAgentConfig = config

        assert isinstance(self._env.action_space, gym.spaces.Discrete), "only discrete action space is supported"
        if device is None:
            self._device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        # Initialise the models
        self.initialise_models()

    def save(self, path: Optional[Path] = None):
        pass

    def load(self, model_dir: Optional[Path] = None):
        pass

    def select_action(self, obs: Union[Dict[str, Any], TensorType], evaluation: Optional[bool] = False,
                      mask: Optional[TensorType] = None) -> np.ndarray:
        raise NotImplementedError

    @property
    def configs(self) -> Dict[str, Any]:
        """Return the configurations of the agent."""
        return {
            "num_epoch": self._config.num_epoch
        }

    def initialise_models(self):
        pass

    def train(self, wandb_run: Optional[Run] = None):
        """Train the agent."""

        average_10_episode_reward: Deque = deque(maxlen=10)

        # Loop over epochs
        for epoch_step in range(self._config.num_epoch):

            # Initialise variables
            obs, init_state_info = self._env.reset(get_info=True)
            logger = EpisodeLogger()
            episode_reward = 0.0
            is_terminated = False
            test_step = 0
            logger.log_info(init_state_info)

            while not is_terminated:
                actions = np.array([self._config.fixed_markdown_schedule[test_step]]* 33)

                # Execute action in environment
                next_obs, rewards, is_terminated, infos = self._env.step(actions)
                obs = next_obs

                # Accumulate episode reward
                episode_reward += rewards.sum()

                test_step += 1
                logger.log_info(infos)

            average_10_episode_reward.append(episode_reward)

            print(f"Epoch {epoch_step:04d} - Episode Reward: {episode_reward}")

            episode_summary = logger.get_episode_summary()

            # Use wandb to record rewards per episode
            if wandb_run is not None:
                fig = logger.plot_episode_summary(title=f"Epoch {epoch_step}")

                wandb_log = {
                    "episode_reward": episode_reward,
                    "average_10_episode_reward": 0.0 if len(average_10_episode_reward) == 0 else np.mean(
                        average_10_episode_reward),
                    "episode_step": test_step,
                    "epoch": epoch_step,
                    "summary_plots": fig,
                    "total_revenue": episode_summary["total_revenue"],
                    **{f"rewards/average_{k}": v for k, v in
                       episode_summary["mean_product_reward_per_type"].items()}
                }

                wandb_run.log(wandb_log)

    def update_parameters(self, memory, current_step):
        raise NotImplementedError
