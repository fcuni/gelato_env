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

from collections import deque
from pathlib import Path
from typing import Deque, Optional, Dict, Any, Union

import gym
import numpy as np
import torch

import torch.optim as optim
import torch.nn.functional as F

from models.dqn.model import QNetwork

EnvType = Union[gym.Env, gym.core.Env]
from models.base_rl_agent import RLAgent
from utils.buffer import ReplayBuffer_v2 as ReplayBuffer
from utils.config import DQNConfig
from wandb.wandb_run import Run

from utils.types import TensorType
from utils.logging import EpisodeLogger


def linear_schedule(start_e: float, end_e: float, duration: int, t: int):
    slope = (end_e - start_e) / duration
    return max(slope * t + start_e, end_e)


class DQN(RLAgent):

    def __init__(self, env: EnvType,
                 config: DQNConfig,
                 name: str = "DQN",
                 device: Optional[torch.device] = None,
                 run_name: Optional[str] = None):

        super().__init__(env, name, run_name=run_name)
        self._config: DQNConfig = config

        assert isinstance(self._env.action_space, gym.spaces.Discrete), "only discrete action space is supported"
        if device is None:
            self._device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        # Assign additional attributes for DQN
        self._state_size: int = tuple(self._env.get_single_observation_space_size())[-1]
        self._action_size: int = self._env.action_space.n
        # self._markdown_trigger_fn = self._config.markdown_trigger_fn

        # Initialise the models
        self.initialise_models()

        # Initialise the Replay Buffer
        self.replay_buffer = ReplayBuffer(
            buffer_size=self._config.buffer_size,
            batch_size=self._config.batch_size,
            device=self._device
        )

    def save(self, path: Optional[Path] = None):
        """Save the trained models.

        Args:
            path (Optional[Path]): Path to save the models to. Defaults to None.
        """

        # Set default path if none is provided
        if path is None:
            path = Path.cwd() / "experiment_data" / "trained_models" / self.run_name

        # Create directory if it does not exist
        if not path.exists():
            path.mkdir(parents=True)

        torch.save(self._q_network.state_dict(), path / "dqn_q_network.pt")
        torch.save(self._target_network.state_dict(), path / "dqn_target_q_network.pt")
        # raise NotImplementedError


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


    def act(self, obs: Union[Dict[str, Any], TensorType]) -> np.ndarray:
        """Act based on the observation.

        Args:
            obs (Dict[str, Any]): Observation from the environment.

        Returns:
            np.ndarray: Action to take in the environment.
        """
        obs = obs.to(self._device)
        raise NotImplementedError
        return self._q_network.get_action(obs, mask=self._env.mask_actions(obs))  # TODO: check if we need to use stochastic action

    @property
    def configs(self) -> Dict[str, Any]:
        """Return the configurations of the agent."""
        return {
            "episodes": self._config.n_episodes,
            "buffer_size": self._config.buffer_size,
            "batch_size": self._config.batch_size,
            # "markdown_trigger_fn": self._markdown_trigger_fn.name,
            "dqn/warmup_episodes": self._config.warmup_episodes,
            "dqn/learning_rate": self._config.learning_rate,
            "dqn/gamma": self._config.gamma,
            "dqn/q_network/hidden_layers": self._config.q_network_hidden_layers,
            "dqn/train_frequency": self._config.train_frequency,
            "dqn/target_network_frequency": self._config.target_network_frequency,
            "dqn/tau": self._config.tau
        }

    def initialise_models(self):
        """Initialize the Actor and Critic networks."""

        self._q_network = QNetwork(self._state_size, self._action_size,
                                   hidden_layers=self._config.q_network_hidden_layers).to(self._device)
        self._optimizer = optim.Adam(self._q_network.parameters(), lr=self._config.learning_rate)
        self._target_network = QNetwork(self._state_size, self._action_size,
                                        hidden_layers=self._config.q_network_hidden_layers).to(self._device)
        self._target_network.load_state_dict(self._q_network.state_dict())


    def train(self, wandb_run: Optional[Run] = None):
        """Train the agent."""


        # Initialise variables
        global_step: int = 0
        episode_i: int = 0
        cumulative_reward: float = 0.0
        average_10_episode_reward: Deque = deque(maxlen=10)

        # Loop over episodes
        while episode_i < self._config.n_episodes:

            print(f"Starting Episode {episode_i}...")

            # Initialise variables for episode
            episode_reward: float = 0.0
            episode_step: int = 0
            # self._markdown_trigger_fn.reset()
            logger = EpisodeLogger()

            # Reset environment
            obs, init_info = self._env.reset(get_info=True)
            is_terminated = False

            logger.log_info(init_info)

            # Training Loop for single episode
            while not is_terminated:

                learning_started = episode_i >= self._config.warmup_episodes

                # Epsilon greedy
                # epsilon = linear_schedule(1, 0.001, 0.5 * self._config.n_episodes, global_step-episode_i*self._config.episode_length)
                epsilon = 5/np.sqrt(max(0, episode_i - self._config.warmup_episodes) + 1)
                # epsilon = np.max([epsilon, 0.001])
                if np.random.uniform() < epsilon or not learning_started:
                    action_mask = self._env.mask_actions().astype(np.int8)
                    actions = np.array(
                        [self._env.action_space.sample(mask=action_mask[i]) for i in range(self._env.state.n_products)])

                # Normal training phase
                else:
                    action_mask = self._env.mask_actions()
                    # Get action from actor
                    obs_tensor = torch.from_numpy(obs).to(self._device)
                    actions = self._q_network.get_action(obs_tensor, mask=action_mask)

                # Execute action in environment
                orig_dones = self._env.state.per_product_done_signal
                next_obs, rewards, is_terminated, infos = self._env.step(actions)
                # next_obs_tensor = torch.from_numpy(next_obs).to(self._device)
                dones = self._env.state.per_product_done_signal

                # Accumulate episode reward
                episode_reward += rewards.sum()

                # Logging
                logger.log_info(infos)

                # Add trajectory into the replay buffer
                self.replay_buffer.add(state=obs[~orig_dones],
                                       action=actions[~orig_dones],
                                       reward=rewards[~orig_dones],
                                       next_state=next_obs[~orig_dones],
                                       terminated=dones[~orig_dones],
                                       action_mask=action_mask[~orig_dones])

                obs = next_obs

                if learning_started:
                    # Update the networks every few steps (as configured)
                    if global_step % self._config.train_frequency == 0:

                        # Sample a batch from the replay buffer
                        sample_obs, sample_actions, sample_next_obs, sample_rewards, sample_dones, sample_mask \
                            = self.replay_buffer.sample(self._config.batch_size)

                        sample_obs_tensor = sample_obs.to(self._device)
                        sample_next_obs_tensor = sample_next_obs.to(self._device)

                        # data = rb.sample(args.batch_size)
                        with torch.no_grad():
                            target_max = self._target_network.get_action(sample_next_obs_tensor, mask=sample_mask)
                            td_target = sample_rewards.flatten() + self._config.gamma * torch.from_numpy(target_max) * (1 - sample_dones.flatten())
                        old_val = self._q_network(sample_obs_tensor).gather(1, sample_actions.type(torch.int64)).squeeze()
                        loss = F.mse_loss(td_target, old_val)


                        # Log the losses to wandb
                        if wandb_run is not None:
                            try:
                                wandb_run.log({
                                    "losses/td_loss": loss,
                                    "losses/q_values": old_val.mean().item(),
                                    "epsilon": epsilon,
                                    "global_step": global_step,
                                })
                            except NameError:
                                print(f"[Unable to log to wandb] Step {global_step}: losses not defined")


                        # optimize the model
                        self._optimizer.zero_grad()
                        loss.backward()
                        self._optimizer.step()

                    # update target network
                    if global_step % self._config.target_network_frequency == 0:
                        for target_network_param, q_network_param in zip(self._target_network.parameters(),
                                                                         self._q_network.parameters()):
                            target_network_param.data.copy_(
                                self._config.tau * q_network_param.data + (1.0 - self._config.tau) * target_network_param.data
                            )

                # Increment the step counters
                global_step += 1
                episode_step += 1

            # End of episode: log the episode reward and reset the environment
            cumulative_reward += episode_reward
            average_10_episode_reward.append(episode_reward)
            print(f"[Episode {episode_i}] Episode reward: {episode_reward} Episode steps: {episode_step}")

            # Use wandb to record rewards per episode
            if wandb_run is not None:

                if episode_i % 10 == 0 or episode_i == self._config.n_episodes - 1:
                    fig = logger.plot_episode_summary(title=f"Episode {episode_i}")

                    wandb_log = {
                        "buffer_usage": len(self.replay_buffer),
                        "episode_reward": episode_reward,
                        "average_10_episode_reward": 0.0 if len(average_10_episode_reward) == 0 else np.mean(
                            average_10_episode_reward),
                        "cumulative_reward": cumulative_reward,
                        "episode_step": episode_step,
                        "episode": episode_i,
                        "global_step": global_step,
                        "summary_plots": fig
                    }
                else:
                    wandb_log = {
                        "buffer_usage": len(self.replay_buffer),
                        "episode_reward": episode_reward,
                        "average_10_episode_reward": 0.0 if len(average_10_episode_reward) == 0 else np.mean(
                            average_10_episode_reward),
                        "cumulative_reward": cumulative_reward,
                        "episode_step": episode_step,
                        "episode": episode_i,
                        "global_step": global_step
                    }
                wandb_run.log(wandb_log)

            # Increment episode counter
            episode_i += 1


