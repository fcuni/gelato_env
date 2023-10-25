# Code adopted from ClearRL Github https://docs.cleanrl.dev/rl-algorithms/dqn/#dqnpy under MIT License
# Citation:
# @article{huang2022cleanrl,
#   author  = {Shengyi Huang and Rousslan Fernand Julien Dossa and Chang Ye and Jeff Braga and Dipam Chakraborty and Kinal Mehta and João G.M. Araújo},
#   title   = {CleanRL: High-quality Single-file Implementations of Deep Reinforcement Learning Algorithms},
#   journal = {Journal of Machine Learning Research},
#   year    = {2022},
#   volume  = {23},
#   number  = {274},
#   pages   = {1--18},
#   url     = {http://jmlr.org/papers/v23/21-1342.html}
# }
import pickle
from collections import deque
from itertools import count
from pathlib import Path
from typing import Deque, Optional, Dict, Any, Union

import gym
import numpy as np
import torch

import torch.optim as optim
import torch.nn.functional as F

from models.ddqn.model import QNetwork

EnvType = Union[gym.Env, gym.core.Env]
from models.base_rl_agent import RLAgent
from utils.buffer import ReplayBuffer as ReplayBuffer
from utils.config import DDQNConfig
from wandb.wandb_run import Run

from utils.types import TensorType
from utils.logging import EpisodeLogger


def linear_schedule(start_e: float, end_e: float, duration: int, t: int):
    slope = (end_e - start_e) / duration
    return max(slope * t + start_e, end_e)


class DDQN(RLAgent):

    def __init__(self, env: EnvType,
                 config: DDQNConfig,
                 name: str = "DQN",
                 device: Optional[torch.device] = None,
                 run_name: Optional[str] = None):

        super().__init__(env, name, run_name=run_name)
        self._config: DDQNConfig = config

        assert isinstance(self._env.action_space, gym.spaces.Discrete), "only discrete action space is supported"
        if device is None:
            self._device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        # Initialise the models
        self.initialise_models()

        # Initialise the Replay Buffer
        self.replay_buffer = ReplayBuffer(
            buffer_size=self._config.buffer_size,
            batch_size=self._config.batch_size,
            device=self._device
        )

        self._epsilon: float = self._config.epsilon_start

    def save(self, path: Optional[Path] = None):
        """Save the trained models.

        Args:
            path (Optional[Path]): Path to save the models to. Defaults to None.
        """

        # Set default path if none is provided
        if path is None:
            path = Path.cwd() / "experiment_data" / self.run_name/ "trained_models"

        # Create directory if it does not exist
        if not path.exists():
            path.mkdir(parents=True)

        torch.save(self._q_network.state_dict(), path / "dqn_q_network.pt")
        torch.save(self._target_network.state_dict(), path / "dqn_target_q_network.pt")


    def load(self, model_dir: Optional[Path] = None):
        """Load the trained models.

        Args:
            model_dir (Path): Path to the directory containing the models.

        Raises:
            FileNotFoundError: If the path does not exist.
        """
        if model_dir is None:
            model_dir = self._config.path_to_model
        if not model_dir.exists():
            raise FileNotFoundError(f"Path {model_dir} does not exist.")
        raise NotImplementedError
        self._q_network.load_state_dict(torch.load(model_dir / "dqn_q_network.pt"))
        self._target_network.load_state_dict(torch.load(model_dir / "dqn_target_q_network.pt"))
        self._q_network.eval()
        self._target_network.eval()

    def select_action(self, obs: Union[Dict[str, Any], TensorType], evaluation: Optional[bool] = False,
                      mask: Optional[TensorType] = None) -> np.ndarray:
        """Act based on the observation.

        Args:
            obs (Dict[str, Any]): Observation from the environment.
            evaluation (bool, optional): Whether to act deterministically. Defaults to False.
            mask (Optional[TensorType], optional): Mask for the actions. Defaults to None.

        Returns:
            np.ndarray: Action to take in the environment.
        """

        if (not evaluation) and np.random.uniform() < self._epsilon:
            mask = mask.astype(np.int8)
            actions = np.array(
                [self._env.action_space.sample(mask=mask[i]) for i in range(obs.shape[0])])
        else:
            # Get action from actor
            obs = obs.to(self._device)
            actions = self._q_network.get_action(obs, mask=mask)
        return actions

    @property
    def configs(self) -> Dict[str, Any]:
        """Return the configurations of the agent."""
        return {

            "init_exploration_steps": self._config.init_exploration_steps,
            "min_pool_size": self._config.min_pool_size,
            "num_epoch": self._config.num_epoch,
            "epoch_length": self._config.epoch_length,

            "ddqn/buffer_size": self._config.buffer_size,
            "ddqn/batch_size": self._config.batch_size,
            "ddqn/learning_rate": self._config.learning_rate,
            "ddqn/gamma": self._config.gamma,
            "ddqn/q_network/hidden_layers": self._config.q_network_hidden_layers,
            "ddqn/update_frequency": self._config.update_frequency,
            "ddqn/target_network_frequency": self._config.target_network_frequency,
            "ddqn/tau": self._config.tau,
            "ddqn/epsilon_start": self._config.epsilon_start,
            "ddqn/epsilon_decay": self._config.epsilon_decay,
            "ddqn/epsilon_min": self._config.epsilon_min

        }

    def initialise_models(self):
        """Initialize the Actor and Critic networks."""

        self._q_network = QNetwork(self._state_size, self._action_num,
                                   hidden_layers=self._config.q_network_hidden_layers).to(self._device)
        self._optimizer = optim.Adam(self._q_network.parameters(), lr=self._config.learning_rate)
        self._target_network = QNetwork(self._state_size, self._action_num,
                                        hidden_layers=self._config.q_network_hidden_layers).to(self._device)
        self._target_network.load_state_dict(self._q_network.state_dict())


    def train(self, wandb_run: Optional[Run] = None):
        """Train the agent."""

        # Initialise variables
        global_step: int = 0
        is_terminated = True

        # Warmup phase: collect random actions
        while global_step < self._config.init_exploration_steps:

            if is_terminated:
                # Reset environment
                obs = self._env.reset(get_info=False)

            # Sample random action
            action_mask = self._env.mask_actions().astype(np.int8)
            actions = np.array(
                [self._env.action_space.sample(mask=action_mask[i]) for i in range(self._env.state.n_products)])

            # Execute action in environment
            orig_dones = self._env.state.per_product_done_signal
            next_obs, rewards, is_terminated, infos = self._env.step(actions)
            dones = self._env.state.per_product_done_signal

            # Add transition into the replay buffer
            self.replay_buffer.add(state=obs[~orig_dones],
                                   action=actions[~orig_dones],
                                   reward=rewards[~orig_dones],
                                   next_state=next_obs[~orig_dones],
                                   terminated=dones[~orig_dones],
                                   action_mask=action_mask[~orig_dones])
            obs = next_obs
            global_step += 1

        if np.random.uniform() < self._epsilon:
            # Get random action
            mask = action_mask.astype(np.int8)
            actions = np.array(
                [self._env.action_space.sample(mask=mask[i]) for i in range(self._env.state.n_products)])

        else:
            # Get action from actor
            obs_tensor = torch.from_numpy(obs).to(self._device)
            actions = self._q_network.get_action(obs_tensor, mask=action_mask)

        average_10_episode_reward: Deque = deque(maxlen=10)
        is_terminated = True

        # Loop over epochs
        for epoch_step in range(self._config.num_epoch):

            start_step = global_step

            for i in count():
                cur_step = global_step - start_step
                if cur_step >= self._config.epoch_length and len(self.replay_buffer) > self._config.min_pool_size:
                    break

                if is_terminated:
                    # Reset environment
                    obs = self._env.reset(get_info=False)

                action_mask = self._env.mask_actions()

                if np.random.uniform() < self._epsilon:
                    # Get random action
                    mask = action_mask.astype(np.int8)
                    actions = np.array(
                        [self._env.action_space.sample(mask=mask[i]) for i in range(self._env.state.n_products)])
                else:
                    # Get action from actor
                    obs_tensor = torch.from_numpy(obs).to(self._device)
                    actions = self._q_network.get_action(obs_tensor, mask=action_mask)

                # Execute action in environment
                orig_dones = self._env.state.per_product_done_signal
                next_obs, rewards, is_terminated, infos = self._env.step(actions)
                dones = self._env.state.per_product_done_signal

                # Add transition into the replay buffer
                self.replay_buffer.add(state=obs[~orig_dones],
                                       action=actions[~orig_dones],
                                       reward=rewards[~orig_dones],
                                       next_state=next_obs[~orig_dones],
                                       terminated=dones[~orig_dones],
                                       action_mask=action_mask[~orig_dones])
                obs = next_obs

                if len(self.replay_buffer) > self._config.min_pool_size:
                    if global_step % self._config.update_frequency == 0:
                        # Sample a batch from the replay buffer
                        memory = self.replay_buffer.sample(self._config.batch_size)
                        # Update the parameters of the networks every few steps (as configured)
                        training_log = self.update_parameters(memory, global_step)

                        # Append additional information to the training log
                        training_log = {
                            **{f"losses/{k}": v for k, v in training_log.items()},
                            "global_step": global_step,
                            "epsilon": self._epsilon
                        }

                        # Log training information
                        if wandb_run is not None:
                            wandb_run.log(training_log)

                # Increment global step
                global_step += 1

                # Evaluate the agent at the end of each epoch
                if global_step % self._config.epoch_length == 0:

                    # Initialise variables
                    obs, init_state_info = self._env.reset(get_info=True)
                    logger = EpisodeLogger()
                    episode_reward = 0.0
                    is_terminated = False
                    test_step = 0
                    logger.log_info(init_state_info)

                    while not is_terminated:
                        action_mask = self._env.mask_actions()
                        # Get action from actor
                        obs_tensor = torch.from_numpy(obs).to(self._device)
                        actions = self._q_network.get_action(obs_tensor, mask=action_mask)

                        # Execute action in environment
                        next_obs, rewards, is_terminated, infos = self._env.step(actions)
                        obs = next_obs

                        # Accumulate episode reward
                        episode_reward += rewards.sum()

                        test_step += 1
                        logger.log_info(infos)

                    average_10_episode_reward.append(episode_reward)

                    print(f"Epoch {epoch_step:04d} - Step Reward: {global_step} {episode_reward}")

                    episode_summary = logger.get_episode_summary()

                    # Use wandb to record rewards per episode
                    if wandb_run is not None:
                        fig = logger.plot_episode_summary(title=f"Epoch {epoch_step}")

                        wandb_log = {
                            "env_buffer_usage": len(self.replay_buffer),
                            "episode_reward": episode_reward,
                            "average_10_episode_reward": 0.0 if len(average_10_episode_reward) == 0 else np.mean(
                                average_10_episode_reward),
                            "episode_step": test_step,
                            "epoch": epoch_step,
                            "global_step": global_step,
                            "summary_plots": fig,
                            "total_revenue": episode_summary["total_revenue"],
                            **{f"rewards/average_{k}": v for k, v in
                               episode_summary["mean_product_reward_per_type"].items()}
                        }

                        wandb_run.log(wandb_log)

                    # save checkpoint (model and episode summary)
                    checkpoint_path = Path.cwd() / "experiment_data" / self.run_name / "checkpoints" \
                                      / f"epoch_{epoch_step:04d}"
                    # save model
                    self.save(checkpoint_path / "saved_models")
                    # save episode summary
                    with open(checkpoint_path / "episode_summary.pkl", "wb") as f:
                        pickle.dump(episode_summary, f)

    def update_parameters(self, memory, current_step):

        self._epsilon = max(self._epsilon * self._config.epsilon_decay, self._config.epsilon_min)

        # Update the networks every few steps (as configured)
        if current_step % self._config.update_frequency == 0:

            # Sample a batch from the replay buffer
            # sample_obs, sample_actions, sample_next_obs, sample_rewards, sample_dones, sample_mask \
            #     = self.replay_buffer.sample(self._config.batch_size)

            sample_obs, sample_actions, sample_next_obs, sample_rewards, sample_dones, sample_mask = memory

            sample_obs_tensor = sample_obs.to(self._device)
            sample_next_obs_tensor = sample_next_obs.to(self._device)

            with torch.no_grad():
                target_max = self._target_network.get_action(sample_next_obs_tensor, mask=sample_mask)
                td_target = sample_rewards.flatten() + self._config.gamma * torch.from_numpy(target_max) * (
                            1 - sample_dones.flatten())
            old_val = self._q_network(sample_obs_tensor).gather(1, sample_actions.type(torch.int64)).squeeze()
            loss = F.mse_loss(td_target, old_val)

            # optimize the model
            self._optimizer.zero_grad()
            loss.backward()
            self._optimizer.step()

        # update target network
        if current_step % self._config.target_network_frequency == 0:
            for target_network_param, q_network_param in zip(self._target_network.parameters(),
                                                             self._q_network.parameters()):
                target_network_param.data.copy_(
                    self._config.tau * q_network_param.data + (1.0 - self._config.tau) * target_network_param.data
                )

        return {
            "td_loss": loss,
            "q_values": old_val.mean().item(),
        }

