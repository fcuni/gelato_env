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

import random
from collections import deque
from typing import Deque, Optional
import wandb
import gym
import numpy as np
import torch

import torch.optim as optim
import torch.nn.functional as F

from env.gelateria_env import GelateriaEnv
from models.base_rl_agent import RLAgent
from models.sac.networks import ActorNetwork, SoftQNetwork
from utils.buffer import ReplayBuffer
from models.sac.utils import collect_random_v2
from utils.misc import get_flatten_observation_from_state, convert_dict_to_numpy
from utils.config import BaseConfig


class SACDiscrete(RLAgent):

    def __init__(self, env: GelateriaEnv,
                 config: BaseConfig,
                 name: str = "SAC_Discrete_v2",
                 device: Optional[torch.device] = None):

        super().__init__(env, config, name)
        assert isinstance(self._env.action_space, gym.spaces.Discrete), "only discrete action space is supported"
        if device is None:
            self._device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        # Assign additional attributes for SAC
        self._state_size: int = tuple(self._env.get_single_observation_space_size())[-1]
        self._action_size: int = self._env.action_space.n
        self._alpha: Optional[float] = None

        # Seeding
        random.seed(self._config.sac_config.seed)
        np.random.seed(self._config.sac_config.seed)
        torch.manual_seed(self._config.sac_config.seed)
        torch.backends.cudnn.deterministic = self._config.sac_config.torch_deterministic

        # Initialise wandb (if enabled)
        if self._config.wandb_config.use_wandb:
            self.init_wandb()

        # Initialise the models
        self.initialise_models()

        # Initialise the Replay Buffer
        self.replay_buffer = ReplayBuffer(
            buffer_size=self._config.sac_config.buffer_size,
            batch_size=self._config.sac_config.batch_size,
            seed=self._config.sac_config.seed,
            device=self._device
        )

    def save(self):
        raise NotImplementedError

    def load(self):
        raise NotImplementedError

    def init_wandb(self):
        """Initialise wandb."""
        run_config = {
            "algorithm": self.name,
            "episodes": self._config.sac_config.n_episodes,
            "buffer_size": self._config.sac_config.buffer_size,
            "batch_size": self._config.sac_config.batch_size,
            "environment": self._env.name,
            "sales_model": self._env.sales_model_name,
            "reward_type": self._env.reward_type_name,
            "action_mask": self._env.action_mask_name
        }
        wandb.init(
            project=self._config.wandb_config.project,
            entity=self._config.wandb_config.entity,
            config=run_config,
            mode=self._config.wandb_config.mode,
            # name=run_name
        )

    def initialise_models(self):
        """Initialize the Actor and Critic networks."""
        self._actor = ActorNetwork(self._state_size, self._action_size).to(self._device)
        self._qf1 = SoftQNetwork(self._state_size, self._action_size, seed=self._config.sac_config.seed).to(
            self._device)
        self._qf2 = SoftQNetwork(self._state_size, self._action_size, seed=self._config.sac_config.seed + 1).to(
            self._device)
        self._qf1_target = SoftQNetwork(self._state_size, self._action_size).to(self._device)
        self._qf2_target = SoftQNetwork(self._state_size, self._action_size).to(self._device)
        self._qf1_target.load_state_dict(self._qf1.state_dict())
        self._qf2_target.load_state_dict(self._qf2.state_dict())

        # TRY NOT TO MODIFY: eps=1e-4 increases numerical stability
        self._q_optimizer = optim.Adam(list(self._qf1.parameters()) + list(self._qf2.parameters()),
                                       lr=self._config.sac_config.learning_rate, eps=1e-4)
        self._actor_optimizer = optim.Adam(list(self._actor.parameters()), lr=self._config.sac_config.learning_rate,
                                           eps=1e-4)  # TODO: might want to set different learning rate for actor

        # Set up optimiser for automatic entropy tuning
        if self._config.sac_config.auto_entropy_tuning:
            self._target_entropy = - self._config.sac_config.target_entropy_scale * torch.log(
                1 / torch.tensor(self._action_size))
            self._log_alpha = torch.zeros(1, requires_grad=True, device=self._device)
            self._alpha = self._log_alpha.exp().item()
            self._a_optimizer = optim.Adam([self._log_alpha], lr=self._config.sac_config.learning_rate, eps=1e-4)
        else:
            self._alpha = self._config.sac_config.alpha

    def train(self):
        """Train the agent."""

        # Load Replay Buffer from file / generate buffer
        if not self._config.sac_config.regenerate_buffer:
            self.replay_buffer.load(self._config.sac_config.replay_buffer_path)
        if self._config.sac_config.regenerate_buffer or len(self.replay_buffer) == 0:
            collect_random_v2(self._env, self.replay_buffer, self._config.sac_config.initial_random_steps,
                              state_transform_fn=get_flatten_observation_from_state)
            # save buffer to file after regenerating buffer
            if self._config.sac_config.save_replay_buffer:
                self.replay_buffer.save(self._config.sac_config.replay_buffer_path)

        # Initialise variables
        pre_training_step: int = 0
        global_step: int = 0
        episode_i: int = 0
        episodic_reward: float = 0.0
        cumulative_reward: float = 0.0
        average_10_episode_reward: Deque = deque(maxlen=10)

        # Loop over episodes
        while episode_i < self._config.sac_config.n_episodes:

            print(f"Starting Episode {episode_i}...")

            # Initialise variables for episode
            episode_reward = 0.0
            episode_step = 0

            # Reset environment
            obs, _, is_terminated, _ = self._env.reset()

            # Training Loop for single episode
            while not is_terminated:

                # Get observation in tensor format, since that is the shape and format what the actor expects
                obs_tensor = torch.Tensor(get_flatten_observation_from_state(obs)).to(self._device)

                # Get action from actor
                actions = self._actor.get_det_action(obs_tensor, mask=self._env.mask_actions())

                # Execute action in environment
                orig_dones = self._env.state.per_product_done_signal
                next_obs, rewards, is_terminated, infos = self._env.step(actions, action_dtype="int")
                next_obs_tensor = torch.Tensor(get_flatten_observation_from_state(next_obs)).to(self._device)
                dones = self._env.state.per_product_done_signal

                # Accumulate episode reward
                episode_reward += np.sum(convert_dict_to_numpy(rewards))

                # TODO: for debugging only (remove later)
                sales = self._env.get_observations(self._env.state)["private_obs"]["sales"]
                print(f"sales: {sales}\nrewards: {convert_dict_to_numpy(rewards)}\nactions: {actions}")
                if np.any(convert_dict_to_numpy(rewards) >= 1):
                    print(f"Found a reward of {rewards} at step {episode_step} of episode {episode_i}")

                # Add trajectory into the replay buffer
                self.replay_buffer.add(obs_tensor[~orig_dones], actions[~orig_dones],
                                       convert_dict_to_numpy(rewards)[~orig_dones], next_obs_tensor[~orig_dones],
                                       dones[~orig_dones])

                obs = next_obs

                # Update the networks every few steps (as configured)
                if global_step % self._config.sac_config.update_frequency == 0:

                    # Sample a batch from the replay buffer
                    sample_obs_tensor, sample_actions, sample_rewards, sample_next_obs_tensor, sample_dones \
                        = self.replay_buffer.sample(self._config.sac_config.batch_size)

                    # CRITIC training
                    with torch.no_grad():
                        _, next_state_action_probs, next_state_log_pi = self._actor.evaluate(
                            sample_next_obs_tensor, mask=self._env.mask_actions(sample_next_obs_tensor))
                        qf1_next_target = self._qf1_target(sample_next_obs_tensor)
                        qf2_next_target = self._qf2_target(sample_next_obs_tensor)
                        # we can use the action probabilities instead of MC sampling to estimate the expectation
                        min_qf_next_target = next_state_action_probs * (
                                torch.min(qf1_next_target, qf2_next_target) - self._alpha * next_state_log_pi
                        )
                        # adapt Q-target for discrete Q-function
                        min_qf_next_target = min_qf_next_target.sum(dim=1)
                        next_q_value = sample_rewards.flatten() + (
                                1 - sample_dones.flatten()) * self._config.sac_config.gamma * min_qf_next_target

                    # use Q-values only for the taken actions
                    qf1_values = self._qf1(sample_obs_tensor)
                    qf2_values = self._qf2(sample_obs_tensor)
                    qf1_a_values = qf1_values.gather(1, sample_actions.long()).view(-1)
                    qf2_a_values = qf2_values.gather(1, sample_actions.long()).view(-1)
                    qf1_loss = F.mse_loss(qf1_a_values, next_q_value)
                    qf2_loss = F.mse_loss(qf2_a_values, next_q_value)
                    qf_loss = qf1_loss + qf2_loss

                    self._q_optimizer.zero_grad()
                    qf_loss.backward()
                    self._q_optimizer.step()

                    # ACTOR training
                    _, action_probs, log_pi = self._actor.evaluate(sample_obs_tensor,
                                                                   mask=self._env.mask_actions(sample_obs_tensor))
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
                    if self._config.sac_config.auto_entropy_tuning:
                        # re-use action probabilities for temperature loss
                        alpha_loss = (action_probs.detach() * (
                                -self._log_alpha * (log_pi + self._target_entropy).detach())).mean()

                        self._a_optimizer.zero_grad()
                        alpha_loss.backward()
                        self._a_optimizer.step()
                        self._alpha = self._log_alpha.exp().item()

                # Update the target networks
                if global_step % self._config.sac_config.target_network_frequency == 0:
                    for param, target_param in zip(self._qf1.parameters(), self._qf1_target.parameters()):
                        target_param.data.copy_(self._config.sac_config.tau * param.data + (
                                    1 - self._config.sac_config.tau) * target_param.data)
                    for param, target_param in zip(self._qf2.parameters(), self._qf2_target.parameters()):
                        target_param.data.copy_(self._config.sac_config.tau * param.data + (
                                    1 - self._config.sac_config.tau) * target_param.data)

                # Log the losses to wandb
                if self._config.wandb_config.use_wandb:
                    try:
                        wandb.log({
                            "losses/qf1_values": qf1_a_values.mean().item(),
                            "losses/qf2_values": qf2_a_values.mean().item(),
                            "losses/qf1_loss": qf1_loss.item(),
                            "losses/qf2_loss": qf2_loss.item(),
                            "losses/qf_loss": qf_loss.item() / 2.0,
                            "losses/actor_loss": actor_loss.item(),
                            "losses/alpha": self._alpha,
                            "global_step": global_step
                        })
                    except NameError:
                        print(f"[Unable to log to wandb] Step {global_step}: qf1_a_values not defined")

                # Increment the step counters
                global_step += 1
                episode_step += 1

            # End of episode: log the episode reward and reset the environment
            cumulative_reward += episode_reward
            average_10_episode_reward.append(episode_reward)
            print(f"Actions in the episode: {self._env.state.last_actions}") # TODO: for debugging
            print(f"[Episode {episode_i}] Episode reward: {episode_reward} Episode steps: {episode_step}")

            # Use wandb to record rewards per episode
            if self._config.wandb_config.use_wandb:
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
                wandb.log(wandb_log)

            # Increment episode counter
            episode_i += 1


# if __name__ == "__main__":
#     config = SACExperimentConfig()
#     run_name = f"{args.env_id}__{args.exp_name}__{args.seed}__{int(time.time())}"
#
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#
#     # env setup
#     env = SACExperiment().build_env()
#     assert isinstance(env.action_space, gym.spaces.Discrete), "only discrete action space is supported"
#
#     state_size = tuple(env.get_single_observation_space_size())[-1]
#     action_size = env.action_space.n
#
#     if config.wandb_config.use_wandb:
#         import wandb
#
#         wandb_config = {
#             "algorithm": "SAC",
#             "episodes": config.sac_config.n_episodes,
#             "buffer_size": config.sac_config.buffer_size,
#             "batch_size": config.sac_config.batch_size,
#             "environment": env.name,
#             "sales_model": env.sales_model_name,
#             "reward_type": env.reward_type_name,
#             "action_mask": env.action_mask_name
#         }
#
#         wandb.init(
#             project=config.wandb_config.project,
#             entity=config.wandb_config.entity,
#             # sync_tensorboard=True,
#             config=wandb_config,
#             mode=config.wandb_config.mode,
#             name=run_name
#             # monitor_gym=True,
#             # save_code=True,
#         )
#     # writer = SummaryWriter(f"runs/{run_name}")
#     # writer.add_text(
#     #     "hyperparameters",
#     #     "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
#     # )
#
#     actor = ActorNetwork(state_size, action_size).to(device)
#     qf1 = SoftQNetwork(state_size, action_size, seed=1).to(device)
#     qf2 = SoftQNetwork(state_size, action_size, seed=2).to(device)
#     qf1_target = SoftQNetwork(state_size, action_size).to(device)
#     qf2_target = SoftQNetwork(state_size, action_size).to(device)
#     qf1_target.load_state_dict(qf1.state_dict())
#     qf2_target.load_state_dict(qf2.state_dict())
#     # TRY NOT TO MODIFY: eps=1e-4 increases numerical stability
#     q_optimizer = optim.Adam(list(qf1.parameters()) + list(qf2.parameters()), lr=config.sac_config.learning_rate,
#                              eps=1e-4)
#     actor_optimizer = optim.Adam(list(actor.parameters()), lr=config.sac_config.learning_rate,
#                                  eps=1e-4)  # TODO: might want to set different learning rate for policy
#
#     # Automatic entropy tuning
#     if config.sac_config.auto_entropy_tuning:
#         target_entropy = - config.sac_config.target_entropy_scale * torch.log(1 / torch.tensor(env.action_space.n))
#         log_alpha = torch.zeros(1, requires_grad=True, device=device)
#         alpha = log_alpha.exp().item()
#         a_optimizer = optim.Adam([log_alpha], lr=config.sac_config.learning_rate, eps=1e-4)
#     else:
#         alpha = config.sac_config.alpha
#
#     rb = ReplayBuffer(
#         buffer_size=config.sac_config.buffer_size,
#         batch_size=config.sac_config.batch_size,
#         seed=config.sac_config.seed,
#         device=device
#     )
#
#     # load existing buffer by default
#     if not config.sac_config.regenerate_buffer:
#         rb.load(config.sac_config.replay_buffer_path)
#     if config.sac_config.regenerate_buffer or len(rb) == 0:
#         collect_random_v2(env, rb, config.sac_config.initial_random_steps,
#                           state_transform_fn=get_flatten_observation_from_state)
#         # save buffer to file after regenerating buffer
#         if config.sac_config.save_replay_buffer:
#             rb.save(config.sac_config.replay_buffer_path)
#
#     start_time = time.time()
#
#     # Start the game
#     obs, _, is_terminated, _ = env.reset()
#
#     pre_training_step: int = 0
#     global_step: int = 0
#     episode_i: int = 0
#     episodic_reward: float = 0.0
#     cumulative_reward: float = 0.0
#     average_10_episode_reward: Deque = deque(maxlen=10)
#
#     while episode_i < config.sac_config.n_episodes:
#
#         # ALGO LOGIC: put action logic here
#         print(f"Starting Episode {episode_i}...")
#         episode_reward = 0.0
#         episode_step = 0
#
#         obs, _, is_terminated, _ = env.reset()
#
#         while not is_terminated:
#
#             # get observation in tensor format, since that is the shape and format what the actor expects
#             obs_tensor = torch.Tensor(get_flatten_observation_from_state(obs)).to(device)
#
#             actions = actor.get_det_action(obs_tensor, mask=env.mask_actions())
#
#             # TRY NOT TO MODIFY: execute the game and log data.
#             orig_dones = env.state.per_product_done_signal
#             next_obs, rewards, is_terminated, infos = env.step(actions, action_dtype="int")
#             next_obs_tensor = torch.Tensor(get_flatten_observation_from_state(next_obs)).to(device)
#             dones = env.state.per_product_done_signal
#
#             # accumulate episode reward
#             episode_reward += np.sum(convert_dict_to_numpy(rewards))
#
#             sales = env.get_observations(env.state)["private_obs"]["sales"]
#
#             print(f"sales: {sales}\nrewards: {convert_dict_to_numpy(rewards)}\nactions: {actions}")
#
#             if np.any(convert_dict_to_numpy(rewards) >= 1):
#                 print(f"Found a reward of {rewards} at step {episode_step} of episode {episode_i}")
#
#             # add trajectory into the replay buffer
#             rb.add(obs_tensor, actions, convert_dict_to_numpy(rewards), next_obs_tensor, dones)
#
#             obs = next_obs
#
#             # ALGO LOGIC: training.
#             update_frequency: int = 4  # TODO: add to config: args.update_frequency = 4
#             if global_step % update_frequency == 0:
#                 sample_obs_tensor, sample_actions, sample_rewards, sample_next_obs_tensor, sample_dones = rb.sample(
#                     config.sac_config.batch_size)
#                 # CRITIC training
#                 with torch.no_grad():
#                     _, next_state_action_probs, next_state_log_pi = actor.evaluate(sample_next_obs_tensor,
#                                                                                    mask=env.mask_actions(
#                                                                                        sample_next_obs_tensor))
#                     qf1_next_target = qf1_target(sample_next_obs_tensor)
#                     qf2_next_target = qf2_target(sample_next_obs_tensor)
#                     # we can use the action probabilities instead of MC sampling to estimate the expectation
#                     min_qf_next_target = next_state_action_probs * (
#                             torch.min(qf1_next_target, qf2_next_target) - alpha * next_state_log_pi
#                     )
#                     # adapt Q-target for discrete Q-function
#                     min_qf_next_target = min_qf_next_target.sum(dim=1)
#                     next_q_value = sample_rewards.flatten() + (
#                             1 - sample_dones.flatten()) * config.sac_config.gamma * min_qf_next_target
#
#                 # use Q-values only for the taken actions
#                 qf1_values = qf1(sample_obs_tensor)
#                 qf2_values = qf2(sample_obs_tensor)
#                 qf1_a_values = qf1_values.gather(1, sample_actions.long()).view(-1)
#                 qf2_a_values = qf2_values.gather(1, sample_actions.long()).view(-1)
#                 qf1_loss = F.mse_loss(qf1_a_values, next_q_value)
#                 qf2_loss = F.mse_loss(qf2_a_values, next_q_value)
#                 qf_loss = qf1_loss + qf2_loss
#
#                 q_optimizer.zero_grad()
#                 qf_loss.backward()
#                 q_optimizer.step()
#
#                 # ACTOR training
#                 _, action_probs, log_pi = actor.evaluate(sample_obs_tensor,
#                                                          mask=env.mask_actions(sample_obs_tensor))
#                 with torch.no_grad():
#                     qf1_values = qf1(sample_obs_tensor)
#                     qf2_values = qf2(sample_obs_tensor)
#                     min_qf_values = torch.min(qf1_values, qf2_values)
#                 # no need for reparameterization, the expectation can be calculated for discrete actions
#                 actor_loss = (action_probs * ((alpha * log_pi) - min_qf_values)).mean()
#
#                 actor_optimizer.zero_grad()
#                 actor_loss.backward()
#                 actor_optimizer.step()
#
#                 if config.sac_config.auto_entropy_tuning:
#                     # re-use action probabilities for temperature loss
#                     alpha_loss = (action_probs.detach() * (-log_alpha * (log_pi + target_entropy).detach())).mean()
#
#                     a_optimizer.zero_grad()
#                     alpha_loss.backward()
#                     a_optimizer.step()
#                     alpha = log_alpha.exp().item()
#
#             # update the target networks
#             if global_step % config.sac_config.target_network_frequency == 0:
#                 for param, target_param in zip(qf1.parameters(), qf1_target.parameters()):
#                     target_param.data.copy_(
#                         config.sac_config.tau * param.data + (1 - config.sac_config.tau) * target_param.data)
#                 for param, target_param in zip(qf2.parameters(), qf2_target.parameters()):
#                     target_param.data.copy_(
#                         config.sac_config.tau * param.data + (1 - config.sac_config.tau) * target_param.data)
#
#             if config.wandb_config.use_wandb:
#                 try:
#                     wandb.log({
#                         "losses/qf1_values": qf1_a_values.mean().item(),
#                         "losses/qf2_values": qf2_a_values.mean().item(),
#                         "losses/qf1_loss": qf1_loss.item(),
#                         "losses/qf2_loss": qf2_loss.item(),
#                         "losses/qf_loss": qf_loss.item() / 2.0,
#                         "losses/actor_loss": actor_loss.item(),
#                         "losses/alpha": alpha,
#                         "global_step": global_step
#                     })
#                 except NameError:
#                     print(f"[Unable to log to wandb] Step {global_step}: qf1_a_values not defined")
#
#             global_step += 1
#             episode_step += 1
#
#         cumulative_reward += episode_reward
#         average_10_episode_reward.append(episode_reward)
#         print(f"Actions in the episode: {env.state.last_actions}")
#
#         print(f"[Episode {episode_i}] Episode reward: {episode_reward} Episode steps: {episode_step}")
#         # Use wandb to record rewards per episode
#         if config.wandb_config.use_wandb:
#             wandb_log = {
#                 "buffer_usage": len(rb),
#                 "episode_reward": episode_reward,
#                 "average_10_episode_reward": 0.0 if len(average_10_episode_reward) == 0 else np.mean(
#                     average_10_episode_reward),
#                 "cumulative_reward": cumulative_reward,
#                 "episode_step": episode_step,
#                 "episode": episode_i,
#                 "global_step": global_step
#             }
#             wandb.log(wandb_log)
#         episode_i += 1

    # plot_data = [[x, y] for (x, y) in zip(info["episode"], info["episode_reward"])]
    # table = wandb.Table(data=plot_data, columns=["x", "y"])
    # wandb.log(
    #     {"episode_reward_plot": wandb.plot.line(table, "x", "y", title="Episode-Reward Plot")})

    # TODO: move this buffer saving logic to elsewhere


