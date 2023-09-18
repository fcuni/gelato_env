# Code adopted from ClearRL Github https://github.com/vwxyzjn/cleanrl/blob/master/cleanrl/sac_atari.py under MIT License
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

EnvType = Union[gym.Env, gym.core.Env]
from models.base_rl_agent import RLAgent
from models.sac.networks import ActorNetwork, SoftQNetwork
from utils.buffer import ReplayBuffer
from utils.config import SACConfig
from wandb.wandb_run import Run

from utils.types import TensorType
from utils.logging import EpisodeLogger


class SACDiscrete(RLAgent):

    def __init__(self, env: EnvType,
                 config: SACConfig,
                 name: str = "SAC_Discrete",
                 device: Optional[torch.device] = None,
                 run_name: Optional[str] = None):

        super().__init__(env, name, run_name=run_name)
        self._config: SACConfig = config

        assert isinstance(self._env.action_space, gym.spaces.Discrete), "only discrete action space is supported"
        if device is None:
            self._device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        else:
            self._device = device

        # Assign additional attributes for SAC
        self._alpha: Optional[float] = None

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
            path = Path.cwd() / "experiment_data" / self.run_name/ "trained_models"

        # Create directory if it does not exist
        if not path.exists():
            path.mkdir(parents=True)

        torch.save(self._actor.state_dict(), path / "actor.pt")
        torch.save(self._qf1.state_dict(), path / "qf1.pt")
        torch.save(self._qf2.state_dict(), path / "qf2.pt")
        torch.save(self._qf1_target.state_dict(), path / "qf1_target.pt")
        torch.save(self._qf2_target.state_dict(), path / "qf2_target.pt")

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
        self._actor.load_state_dict(torch.load(model_dir / "actor.pt"))
        self._qf1.load_state_dict(torch.load(model_dir / "qf1.pt"))
        self._qf2.load_state_dict(torch.load(model_dir / "qf2.pt"))
        self._qf1_target.load_state_dict(torch.load(model_dir / "qf1_target.pt"))
        self._qf2_target.load_state_dict(torch.load(model_dir / "qf2_target.pt"))
        self._actor.eval()
        self._qf1.eval()
        self._qf2.eval()
        self._qf1_target.eval()
        self._qf2_target.eval()

    def select_action(self, obs: Union[Dict[str, Any], TensorType], evaluation: bool = False,
                      mask: Optional[TensorType] = None) -> np.ndarray:
        """Act based on the observation.

        Args:
            obs (Dict[str, Any]): Observation from the environment.
            evaluation (bool, optional): Whether to act deterministically or stochastically. Defaults to False.

        Returns:
            np.ndarray: Action to take in the environment.
        """
        obs = obs.to(self._device)
        if evaluation:
            return self._actor.evaluate(obs, mask=mask)[0]
        return self._actor.get_det_action(obs, mask=mask)

    @property
    def configs(self) -> Dict[str, Any]:
        """Return the configurations of the agent."""
        return {

            "init_exploration_steps": self._config.init_exploration_steps,
            "min_pool_size": self._config.min_pool_size,
            "num_epoch": self._config.num_epoch,
            "epoch_length": self._config.epoch_length,



            "sac/buffer_size": self._config.buffer_size,
            "sac/batch_size": self._config.batch_size,
            "sac/learning_rate": self._config.learning_rate,
            "sac/gamma": self._config.gamma,
            "sac/tau": self._config.tau,
            "sac/auto_entropy_tuning": self._config.auto_entropy_tuning,
            "sac/alpha": self._config.alpha,
            "sac/update_frequency": self._config.update_frequency,
            "sac/target_network_frequency": self._config.target_network_frequency,
            "sac/actor/hidden_layers": self._config.actor_network_hidden_layers,
            "sac/critic/hidden_layers": self._config.critic_network_hidden_layers,
        }

    def initialise_models(self):
        """Initialize the Actor and Critic networks."""
        self._actor = ActorNetwork(self._state_size, self._action_num,
                                   hidden_layers=self._config.actor_network_hidden_layers).to(self._device)
        self._qf1 = SoftQNetwork(self._state_size, self._action_num,
                                 hidden_layers=self._config.critic_network_hidden_layers,
                                 seed=self._config.seed).to(self._device)
        self._qf2 = SoftQNetwork(self._state_size, self._action_num,
                                 hidden_layers=self._config.critic_network_hidden_layers,
                                 seed=self._config.seed + 1).to(self._device)
        self._qf1_target = SoftQNetwork(self._state_size, self._action_num,
                                        hidden_layers=self._config.critic_network_hidden_layers).to(
            self._device)
        self._qf2_target = SoftQNetwork(self._state_size, self._action_num,
                                        hidden_layers=self._config.critic_network_hidden_layers).to(
            self._device)
        self._qf1_target.load_state_dict(self._qf1.state_dict())
        self._qf2_target.load_state_dict(self._qf2.state_dict())

        # TRY NOT TO MODIFY: eps=1e-4 increases numerical stability
        self._q_optimizer = optim.Adam(list(self._qf1.parameters()) + list(self._qf2.parameters()),
                                       lr=self._config.learning_rate, eps=1e-4)
        self._actor_optimizer = optim.Adam(list(self._actor.parameters()), lr=self._config.learning_rate,
                                           eps=1e-4)  # TODO: might want to set different learning rate for actor

        # Set up optimiser for automatic entropy tuning
        if self._config.auto_entropy_tuning:
            self._target_entropy = - self._config.target_entropy_scale * torch.log(
                1 / torch.tensor(self._action_num))
            self._log_alpha = torch.zeros(1, requires_grad=True, device=self._device)
            self._alpha = self._log_alpha.exp().item()
            self._a_optimizer = optim.Adam([self._log_alpha], lr=self._config.learning_rate, eps=1e-4)
        else:
            self._alpha = self._config.alpha

    def update_parameters(self, memory, current_step):

        if current_step % self._config.update_frequency == 0:

            sample_obs, sample_actions, sample_next_obs, sample_rewards, sample_dones, sample_mask = memory

            sample_obs_tensor = sample_obs.to(self._device)
            sample_next_obs_tensor = sample_next_obs.to(self._device)
            if isinstance(sample_mask, np.ndarray):
                sample_mask = torch.from_numpy(sample_mask)
            sample_mask = sample_mask.to(self._device)
            # CRITIC training
            with torch.no_grad():
                _, next_state_action_probs, next_state_log_pi = self._actor.evaluate(
                    sample_next_obs_tensor, mask=sample_mask)
                qf1_next_target = self._qf1_target(sample_next_obs_tensor)
                qf2_next_target = self._qf2_target(sample_next_obs_tensor)
                # we can use the action probabilities instead of MC sampling to estimate the expectation
                min_qf_next_target = next_state_action_probs * (
                        torch.min(qf1_next_target, qf2_next_target) - self._alpha * next_state_log_pi
                )

                # adapt Q-target for discrete Q-function
                min_qf_next_target = min_qf_next_target.sum(dim=1)
                next_q_value = sample_rewards.flatten() + (
                        1 - sample_dones.flatten()) * self._config.gamma * min_qf_next_target

            # use Q-values only for the taken actions
            qf1_values = self._qf1(sample_obs_tensor)
            qf2_values = self._qf2(sample_obs_tensor)
            qf1_a_values = qf1_values.gather(1, sample_actions.long()).view(-1)
            qf2_a_values = qf2_values.gather(1, sample_actions.long()).view(-1)
            qf1_loss = F.mse_loss(qf1_a_values, next_q_value) # JQ = 𝔼(st,at)~D[0.5(Q1(st,at) - r(st,at) - γ(𝔼st+1~p[V(st+1)]))^2]
            qf2_loss = F.mse_loss(qf2_a_values, next_q_value)
            qf_loss = qf1_loss + qf2_loss

            self._q_optimizer.zero_grad()
            qf_loss.backward()
            self._q_optimizer.step()

            # ACTOR training
            _, action_probs, log_pi = self._actor.evaluate(sample_obs_tensor, mask=sample_mask)
            # mask=self._env.mask_actions(sample_obs_tensor))
            with torch.no_grad():
                qf1_values = self._qf1(sample_obs_tensor)
                qf2_values = self._qf2(sample_obs_tensor)
                min_qf_values = torch.min(qf1_values, qf2_values)
            # Re-parameterization is not needed, as the expectation can be calculated for discrete actions
            actor_loss = (action_probs * ((self._alpha * log_pi) - min_qf_values)).mean()

            self._actor_optimizer.zero_grad()
            actor_loss.backward()
            self._actor_optimizer.step()

            # Update the temperature parameter (if auto entropy tuning is on)
            if self._config.auto_entropy_tuning:
                # re-use action probabilities for temperature loss
                alpha_loss = (action_probs.detach() * (
                        -self._log_alpha * (log_pi + self._target_entropy).detach())).mean()

                self._a_optimizer.zero_grad()
                alpha_loss.backward()
                self._a_optimizer.step()
                self._alpha = self._log_alpha.exp().item()

        # Update the target networks
        if current_step % self._config.target_network_frequency == 0:
            for param, target_param in zip(self._qf1.parameters(), self._qf1_target.parameters()):
                target_param.data.copy_(self._config.tau * param.data + (
                        1 - self._config.tau) * target_param.data)
            for param, target_param in zip(self._qf2.parameters(), self._qf2_target.parameters()):
                target_param.data.copy_(self._config.tau * param.data + (
                        1 - self._config.tau) * target_param.data)

        return {"qf1_loss": qf1_loss.item(),
                "qf2_loss": qf2_loss.item(),
                "qf_loss": qf_loss.item() / 2.0,  # divide by 2 to match the scale of the other losses
                "actor_loss": actor_loss.item(),
                "alpha_loss": alpha_loss.item(),
                "qf1_values": qf1_a_values.mean().item(),
                "qf2_values": qf2_a_values.mean().item(),
                "alpha": self._alpha}

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
                # Get action from actor
                obs_tensor = torch.from_numpy(obs).to(self._device)
                actions = self._actor.evaluate(obs_tensor, mask=action_mask)[0]

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
                        parameters_update_log = self.update_parameters(memory, global_step)

                        # Log the losses to wandb
                        if wandb_run is not None:
                            wandb_run.log({
                                **{f"losses/{k}": v for k, v in parameters_update_log.items()},
                                "global_step": global_step,
                            })

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
                        actions = self._actor.get_det_action(obs_tensor, mask=action_mask)

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
                            **{f"rewards/average_{k}": v for k, v in episode_summary["mean_product_reward_per_type"].items()}
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
