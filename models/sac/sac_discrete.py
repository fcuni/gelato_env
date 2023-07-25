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

from collections import deque
from pathlib import Path
from typing import Deque, Optional, Dict, Any, Union

import pandas as pd
import gym
import numpy as np
import torch

import torch.optim as optim
import torch.nn.functional as F
from plotly.subplots import make_subplots

from env.gelateria_env import GelateriaEnv
from env.mask.simple_masks import OnlyCurrentActionBooleanMask
from models.base_rl_agent import RLAgent
from models.sac.networks import ActorNetwork, SoftQNetwork
from utils.buffer import ReplayBuffer
from models.sac.utils import collect_random_v2
from utils.misc import get_flatten_observation_from_state, convert_dict_to_numpy
from utils.config import SACConfig
import plotly.express as px
from wandb.wandb_run import Run

from utils.types import TensorType


class SACDiscrete(RLAgent):

    def __init__(self, env: GelateriaEnv,
                 config: SACConfig,
                 name: str = "SAC_Discrete_v2",
                 device: Optional[torch.device] = None,
                 run_name: Optional[str] = None):

        super().__init__(env, name, run_name=run_name)
        self._config: SACConfig = config

        assert isinstance(self._env.action_space, gym.spaces.Discrete), "only discrete action space is supported"
        if device is None:
            self._device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        # Assign additional attributes for SAC
        self._state_size: int = tuple(self._env.get_single_observation_space_size())[-1]
        self._action_size: int = self._env.action_space.n
        self._alpha: Optional[float] = None
        self._markdown_trigger_fn = self._config.markdown_trigger_fn

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

    def act(self, obs: Union[Dict[str, Any], TensorType]) -> np.ndarray:
        """Act based on the observation.

        Args:
            obs (Dict[str, Any]): Observation from the environment.

        Returns:
            np.ndarray: Action to take in the environment.
        """
        if isinstance(obs, dict):
            obs = get_flatten_observation_from_state(obs)
        return self._actor.get_det_action(obs, mask=self._env.mask_actions(obs))

    @property
    def configs(self) -> Dict[str, Any]:
        """Return the configurations of the agent."""
        return {
            "sac/episodes": self._config.n_episodes,
            "sac/buffer_size": self._config.buffer_size,
            "sac/batch_size": self._config.batch_size,
            "sac/initial_random_steps": self._config.initial_random_steps,
            "sac/learning_rate": self._config.learning_rate,
            "sac/gamma": self._config.gamma,
            "sac/tau": self._config.tau,
            "sac/auto_entropy_tuning": self._config.auto_entropy_tuning,
            "sac/alpha": self._config.alpha,
            "sac/update_frequency": self._config.update_frequency,
            "sac/target_network_frequency": self._config.target_network_frequency,
            "sac/minimum_markdown_duration": self._config.minimum_markdown_duration,
            "sac/markdown_trigger_fn": self._markdown_trigger_fn.name,
            "sac/warmup_steps": self._config.warmup_steps,
            "sac/actor/hidden_layers": self._config.actor_network_hidden_layers,
            "sac/critic/hidden_layers": self._config.critic_network_hidden_layers,
            "sac/epsilon_greedy": self._config.epsilon_greedy,
            "sac/epsilon_greedy_min_epsilon": self._config.epsilon_greedy_min_epsilon,
            "sac/epsilon_greedy_epsilon_decay_rate": self._config.epsilon_greedy_epsilon_decay_rate
        }

    def initialise_models(self):
        """Initialize the Actor and Critic networks."""
        self._actor = ActorNetwork(self._state_size, self._action_size,
                                   hidden_layers=self._config.actor_network_hidden_layers).to(self._device)
        self._qf1 = SoftQNetwork(self._state_size, self._action_size,
                                 hidden_layers=self._config.critic_network_hidden_layers,
                                 seed=self._config.seed).to(self._device)
        self._qf2 = SoftQNetwork(self._state_size, self._action_size,
                                 hidden_layers=self._config.critic_network_hidden_layers,
                                 seed=self._config.seed + 1).to(self._device)
        self._qf1_target = SoftQNetwork(self._state_size, self._action_size,
                                        hidden_layers=self._config.critic_network_hidden_layers).to(
            self._device)
        self._qf2_target = SoftQNetwork(self._state_size, self._action_size,
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
                1 / torch.tensor(self._action_size))
            self._log_alpha = torch.zeros(1, requires_grad=True, device=self._device)
            self._alpha = self._log_alpha.exp().item()
            self._a_optimizer = optim.Adam([self._log_alpha], lr=self._config.learning_rate, eps=1e-4)
        else:
            self._alpha = self._config.alpha

    def train(self, wandb_run: Optional[Run] = None):
        """Train the agent."""

        # Load Replay Buffer from file / generate buffer
        if not self._config.regenerate_buffer:
            self.replay_buffer.load(self._config.replay_buffer_path)
        if self._config.regenerate_buffer or len(self.replay_buffer) == 0:
            collect_random_v2(self._env, self.replay_buffer, self._config.initial_random_steps,
                              state_transform_fn=get_flatten_observation_from_state)
            # save buffer to file after regenerating buffer
            if self._config.save_replay_buffer:
                self.replay_buffer.save(self._config.replay_buffer_path)

        # Initialise variables
        global_step: int = 0
        episode_i: int = 0
        cumulative_reward: float = 0.0
        average_10_episode_reward: Deque = deque(maxlen=10)
        episode_step_action_df = pd.DataFrame(data=[], columns=['Episode', 'Episode step',
                                                                *(self._env.state.get_product_labels())])
        episode_step_stock_df = pd.DataFrame(data=[], columns=['Episode', 'Episode step',
                                                               *(self._env.state.get_product_labels())])
        episode_step_sales_df = pd.DataFrame(data=[], columns=['Episode', 'Episode step',
                                                               *(self._env.state.get_product_labels())])
        episode_step_revenue_df = pd.DataFrame(data=[], columns=['Episode', 'Episode step',
                                                                 *(self._env.state.get_product_labels())])

        # Loop over episodes
        while episode_i < self._config.n_episodes:

            print(f"Starting Episode {episode_i}...")

            # Initialise variables for episode
            episode_reward: float = 0.0
            episode_step: int = 0
            current_markdown_duration: int = 0
            self._markdown_trigger_fn.reset()

            # Reset environment
            obs, _, is_terminated, _ = self._env.reset()

            # Initialise trackers for single episode
            single_episode_stock_per_product = {
                product_id: [] for i, product_id in enumerate(self._env.state.products)
            }
            single_episode_sales_per_product = {
                product_id: [] for i, product_id in enumerate(self._env.state.products)
            }

            single_episode_revenue_per_product = {
                product_id: [] for i, product_id in enumerate(self._env.state.products)
            }

            # Training Loop for single episode
            while not is_terminated:

                # Warmup steps
                if not self._markdown_trigger_fn(state=self._env.state):
                    actions = np.array([0] * self._env.state.n_products).astype(int)

                    pre_step_stocks = self._env.state.product_stocks

                    next_obs, rewards, is_terminated, _ = self._env.step(actions, action_dtype="int")

                    # Accumulate episode reward
                    episode_reward += np.sum(convert_dict_to_numpy(rewards))

                    # Store historical sales and stock per product
                    for i, product_id in enumerate(self._env.state.products):
                        single_episode_sales_per_product[product_id].append(
                            max(0.0, next_obs['private_obs']['sales'][i].item()))
                        current_price = self._env.state.products[product_id].current_price(
                            markdown=self._env.state.last_actions[product_id][-1])
                        single_episode_revenue_per_product[product_id].append(current_price * round(
                            min(pre_step_stocks[i], max(0.0, next_obs['private_obs']['sales'][i].item()))))
                        single_episode_stock_per_product[product_id].append(self._env.state.products[product_id].stock)

                    # TODO: debug
                    # if sum([single_episode_stock_per_product[product_id][-1] for product_id in
                    #         self._env.state.products]) < 300:
                    #     print(f"Mismatch")

                    obs = next_obs
                    current_markdown_duration += 1

                # Normal training steps
                else:
                    # Get observation in tensor format, since that is the shape and format what the actor expects
                    obs_tensor = torch.Tensor(get_flatten_observation_from_state(obs)).to(self._device)

                    if self._config.minimum_markdown_duration is not None \
                            and current_markdown_duration < self._config.minimum_markdown_duration:
                        action_mask = OnlyCurrentActionBooleanMask()(self._env.state)
                    else:
                        action_mask = self._env.mask_actions()

                    # Get action from actor

                    # Epsilon greedy (\epsilon = \sqrt{t})
                    if self._config.epsilon_greedy:
                        epsilon = np.random.rand()
                        if epsilon < max(self._config.epsilon_greedy_min_epsilon,
                                         self._config.epsilon_greedy_epsilon_decay_rate ** global_step):
                            print("Random action")
                            actions = np.array(
                                [self._env.action_space.sample(mask=action_mask.astype(np.int8)[i]) for i in
                                 range(self._env.state.n_products)])
                        else:
                            actions = self._actor.get_det_action(obs_tensor, mask=action_mask)
                    else:
                        actions = self._actor.get_det_action(obs_tensor, mask=action_mask)
                        # actions = self._actor.evaluate(obs_tensor, mask=action_mask)[0]  # TODO: test stochastic action
                    try:
                        last_actions = [round(self._env.state.last_actions[product_id][-1] * 100) for product_id in
                                        self._env.state.products]
                    except:
                        last_actions = [0.0 for _ in self._env.state.products]
                    if not np.all(actions == np.array(last_actions)):
                        current_markdown_duration = 0
                    else:
                        current_markdown_duration += 1

                    # Execute action in environment
                    orig_dones = self._env.state.per_product_done_signal
                    pre_step_stocks = self._env.state.product_stocks
                    next_obs, rewards, is_terminated, infos = self._env.step(actions, action_dtype="int")
                    next_obs_tensor = torch.Tensor(get_flatten_observation_from_state(next_obs)).to(self._device)
                    dones = self._env.state.per_product_done_signal

                    # Accumulate episode reward
                    episode_reward += np.sum(convert_dict_to_numpy(rewards))

                    # Store historical sales and stock per product
                    for i, product_id in enumerate(self._env.state.products):
                        single_episode_sales_per_product[product_id].append(
                            max(0.0, next_obs['private_obs']['sales'][i].item()))
                        current_price = self._env.state.products[product_id].current_price(
                            markdown=self._env.state.last_actions[product_id][-1])
                        single_episode_revenue_per_product[product_id].append(current_price * round(
                            min(pre_step_stocks[i], max(0.0, next_obs['private_obs']['sales'][i].item()))))
                        single_episode_stock_per_product[product_id].append(self._env.state.products[product_id].stock)

                    # Add trajectory into the replay buffer
                    self.replay_buffer.add(state=obs_tensor[~orig_dones], action=actions[~orig_dones],
                                           reward=convert_dict_to_numpy(rewards)[~orig_dones],
                                           next_state=next_obs_tensor[~orig_dones],
                                           terminated=dones[~orig_dones])

                    obs = next_obs

                    # Update the networks every few steps (as configured)
                    if global_step % self._config.update_frequency == 0:

                        # Sample a batch from the replay buffer
                        sample_obs_tensor, sample_actions, sample_next_obs_tensor, sample_rewards, sample_dones \
                            = self.replay_buffer.sample(self._config.batch_size)

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
                                    1 - sample_dones.flatten()) * self._config.gamma * min_qf_next_target

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
                    if global_step % self._config.target_network_frequency == 0:
                        for param, target_param in zip(self._qf1.parameters(), self._qf1_target.parameters()):
                            target_param.data.copy_(self._config.tau * param.data + (
                                    1 - self._config.tau) * target_param.data)
                        for param, target_param in zip(self._qf2.parameters(), self._qf2_target.parameters()):
                            target_param.data.copy_(self._config.tau * param.data + (
                                    1 - self._config.tau) * target_param.data)

                    # Log the losses to wandb
                    if wandb_run is not None:
                        try:
                            wandb_run.log({
                                "losses/qf1_values": qf1_a_values.mean().item(),
                                "losses/qf2_values": qf2_a_values.mean().item(),
                                "losses/qf1_loss": qf1_loss.item(),
                                "losses/qf2_loss": qf2_loss.item(),
                                "losses/qf_loss": qf_loss.item() / 2.0,
                                "losses/actor_loss": actor_loss.item(),
                                "losses/alpha": self._alpha,
                                "global_step": global_step,
                                # **{f"stock/{self._env.state.get_product_labels()[i]}": product.stock for i, product in
                                #    enumerate(self._env.state.products.values())}
                            })
                        except NameError:
                            print(f"[Unable to log to wandb] Step {global_step}: qf1_a_values not defined")

                # for i, product in enumerate(self._env.state.products.values()):
                #     wandb.log({
                #         "action": actions[i],
                #         "product": str(product),
                #         "episode_step": episode_step,
                #         "episode": episode_i
                #     })
                #
                # for product_id in self._env.state.products:
                # actions_per_product = {f"actions/{str(self._env.state.products[product_id])}_{product_id}":  round(actions[i]/100, 2)
                #                        for i, product_id in enumerate(self._env.state.products.keys())}
                # wandb.log({
                #     "actions/episode_step": episode_step,
                #     "actions/episode": episode_i,
                #     "actions/global_step": global_step,
                #     **actions_per_product
                # })

                # Increment the step counters
                global_step += 1
                episode_step += 1

            # End of episode: log the episode reward and reset the environment
            cumulative_reward += episode_reward
            average_10_episode_reward.append(episode_reward)
            print(f"[Episode {episode_i}] Episode reward: {episode_reward} Episode steps: {episode_step}")

            # Use wandb to record rewards per episode
            if wandb_run is not None:
                product_labels = self._env.state.get_product_labels()
                actions_per_product = {
                    f"{product_labels[i]}": self._env.state.last_actions[product_id]
                    for i, product_id in enumerate(self._env.state.products)}

                episode_step_action_df = pd.concat(
                    [episode_step_action_df, pd.DataFrame({'Episode': [episode_i for _ in range(episode_step)],
                                                           'Episode step': [k for k in range(episode_step)],
                                                           **actions_per_product})])
                action_fig = px.line(episode_step_action_df.loc[episode_step_action_df["Episode"] == episode_i],
                                     x='Episode step', y=product_labels)
                action_fig.update_layout(showlegend=False)

                stock_per_product = {
                    f"{product_labels[i]}": single_episode_stock_per_product[product_id]
                    for i, product_id in enumerate(self._env.state.products)}
                episode_step_stock_df = pd.concat(
                    [episode_step_stock_df, pd.DataFrame({'Episode': [episode_i for _ in range(episode_step)],
                                                          'Episode step': [k for k in range(episode_step)],
                                                          **stock_per_product})])

                stock_fig = px.line(episode_step_stock_df.loc[episode_step_stock_df["Episode"] == episode_i],
                                    x='Episode step', y=product_labels)
                stock_fig.update_layout(showlegend=False)

                sales_per_product = {
                    f"{product_labels[i]}": single_episode_sales_per_product[product_id]
                    for i, product_id in enumerate(self._env.state.products)}

                episode_step_sales_df = pd.concat(
                    [episode_step_sales_df, pd.DataFrame({'Episode': [episode_i for _ in range(episode_step)],
                                                          'Episode step': [k for k in range(episode_step)],
                                                          **sales_per_product})])

                sales_fig = px.line(episode_step_sales_df.loc[episode_step_sales_df["Episode"] == episode_i],
                                    x='Episode step', y=product_labels)
                sales_fig.update_layout(showlegend=False)

                revenue_per_product = {
                    f"{product_labels[i]}": single_episode_revenue_per_product[product_id]
                    for i, product_id in enumerate(self._env.state.products)}

                episode_step_revenue_df = pd.concat(
                    [episode_step_revenue_df, pd.DataFrame({'Episode': [episode_i for _ in range(episode_step)],
                                                            'Episode step': [k for k in range(episode_step)],
                                                            **revenue_per_product})])

                revenue_fig = px.line(episode_step_revenue_df.loc[episode_step_revenue_df["Episode"] == episode_i],
                                      x='Episode step', y=product_labels)
                revenue_fig.update_layout(showlegend=False)

                # Create a 2x2 subplot layout
                fig = make_subplots(rows=2, cols=2, shared_xaxes="all", vertical_spacing=0.15, horizontal_spacing=0.05,
                                    subplot_titles=(f"Actions taken (ep. {episode_i})", f"Revenue (ep. {episode_i})",
                                                    f"Stock (ep. {episode_i})", f"Sales (ep. {episode_i})"),
                                    x_title="Episode step")

                # Add each of the original figures to the subplots
                for i, subplot_fig in enumerate([action_fig, revenue_fig, stock_fig, sales_fig]):
                    for trace in subplot_fig['data']:
                        fig.add_trace(trace, row=(i // 2) + 1, col=(i % 2) + 1)

                # Update the layout to have a shared legend
                # fig.update_layout(
                #     legend=dict(
                #         orientation="h",
                #         yanchor="bottom",
                #         y=1.02,
                #         xanchor="right",
                #         x=1
                #     )
                # )
                fig.update_layout(showlegend=False)

                wandb_log = {
                    "buffer_usage": len(self.replay_buffer),
                    "episode_reward": episode_reward,
                    "average_10_episode_reward": 0.0 if len(average_10_episode_reward) == 0 else np.mean(
                        average_10_episode_reward),
                    "cumulative_reward": cumulative_reward,
                    "episode_step": episode_step,
                    "episode": episode_i,
                    "global_step": global_step,
                    # "actions_taken_plot": action_fig,
                    # "stock_plot": stock_fig,
                    # "sales_plot": sales_fig,
                    # "revenue_plot": revenue_fig
                    "per_step_plots": fig
                }
                wandb_run.log(wandb_log)

            # Increment episode counter
            episode_i += 1
