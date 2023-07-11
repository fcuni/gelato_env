from collections import deque
from typing import List, Optional, Union
import numpy as np
import torch
import torch.optim as optim
import torch.nn.functional as F
import torch.nn as nn
from torch.nn.utils import clip_grad_norm_
import wandb
from env.gelateria import GelateriaState
from models.sac.buffer import ReplayBuffer
from models.sac.networks import CriticNetwork, ActorNetwork
import copy

from utils.misc import first_not_none, get_flatten_observation_from_state

from models.sac.utils import collect_random


class SACAgent(nn.Module):
    """Interacts with and learns from the environment."""

    # def __init__(self,
    #                     state_size,
    #                     action_size,
    #                     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #             ):
    def __init__(self,
                 env,
                 config,
                 name: str = "SAC_Discrete",
                 device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
                 ):
        """Initialize an Agent object.
        
        Params
        ======
            state_size (int): dimension of each state
            action_size (int): dimension of each action
            random_seed (int): random seed
        """
        super(SACAgent, self).__init__()
        self._name = name
        self._env = env
        self._dims = tuple(env.state_space_size)
        self.state_size = tuple(env.get_single_observation_space_size())[-1]
        self.action_size = self._env.action_space.n

        self.device = device

        self._config = config

        self.state_transform_fn = get_flatten_observation_from_state

        learning_rate = config.sac_config.learning_rate
        self.clip_grad_param = 1

        self.target_entropy = -self.action_size  # -dim(A)

        self.log_alpha = torch.tensor([0.0], requires_grad=True)
        self.alpha = self.log_alpha.exp().detach()
        self.alpha_optimizer = optim.Adam(params=[self.log_alpha], lr=learning_rate)

        # Actor Network 

        self.actor_local = ActorNetwork(self.state_size, self.action_size).to(device)
        self.actor_optimizer = optim.Adam(self.actor_local.parameters(), lr=learning_rate)

        # Critic Network (w/ Target Network)

        self.critic1 = CriticNetwork(self.state_size, self.action_size, seed=2).to(device)
        self.critic2 = CriticNetwork(self.state_size, self.action_size, seed=1).to(device)

        assert self.critic1.parameters() != self.critic2.parameters()

        self.critic1_target = CriticNetwork(self.state_size, self.action_size).to(device)
        self.critic1_target.load_state_dict(self.critic1.state_dict())

        self.critic2_target = CriticNetwork(self.state_size, self.action_size).to(device)
        self.critic2_target.load_state_dict(self.critic2.state_dict())

        self.critic1_optimizer = optim.Adam(self.critic1.parameters(), lr=learning_rate)
        self.critic2_optimizer = optim.Adam(self.critic2.parameters(), lr=learning_rate)

    @property
    def name(self):
        return self._name

    def get_action(self, from_state: Optional[GelateriaState]) -> Optional[List[Optional[Union[float, int]]]]:
        """Returns actions for given state as per current policy."""
        # use current state from env if not provided
        state = first_not_none(from_state, self._env.state)
        mask = self._env.mask_actions()
        state_obs = torch.from_numpy(self.state_transform_fn(state)).float().to(self.device)

        with torch.no_grad():
            action = self.actor_local.get_det_action(state_obs, mask=mask)
        return action

    def calc_policy_loss(self, states, alpha):
        mask = self._env.mask_actions(states)
        _, action_probs, log_pis = self.actor_local.evaluate(states, mask=mask)

        q1 = self.critic1(states)
        q2 = self.critic2(states)
        min_Q = torch.min(q1, q2)
        actor_loss = (action_probs * (alpha * log_pis - min_Q)).sum(1).mean()
        log_action_pi = torch.sum(log_pis * action_probs, dim=1)
        return actor_loss, log_action_pi

    def learn(self, experiences, gamma):
        """Updates actor, critics and entropy_alpha parameters using given batch of experience tuples.
        Q_targets = r + γ * (min_critic_target(next_state, actor_target(next_state)) - α *log_pi(next_action|next_state))
        Critic_loss = MSE(Q, Q_target)
        Actor_loss = α * log_pi(a|s) - Q(s,a)
        where:
            actor_target(state) -> action
            critic_target(state, action) -> Q-value
        Params
        ======
            experiences (Tuple[torch.Tensor]): tuple of (s, a, r, s', done) tuples 
            gamma (float): discount factor
        """
        states, actions, rewards, next_states, dones = experiences

        # torch.autograd.set_detect_anomaly(True)

        # ---------------------------- update actor ---------------------------- #
        current_alpha = copy.deepcopy(self.alpha)
        actor_loss, log_pis = self.calc_policy_loss(states, current_alpha.to(self.device))
        self.actor_optimizer.zero_grad()
        actor_loss.backward()

        self.actor_optimizer.step()

        # Compute alpha loss
        alpha_loss = - (self.log_alpha.exp() * (log_pis.cpu() + self.target_entropy).detach().cpu()).mean()
        self.alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.alpha_optimizer.step()
        self.alpha = self.log_alpha.exp().detach()

        # ---------------------------- update critic ---------------------------- #
        # Get predicted next-state actions and Q values from target models
        with torch.no_grad():
            next_states_mask = self._env.mask_actions(next_states)
            _, action_probs, log_pis = self.actor_local.evaluate(next_states, mask=next_states_mask)
            Q_target1_next = self.critic1_target(next_states)
            Q_target2_next = self.critic2_target(next_states)

            # TODO: check if need to do something to mask logits and action probs
            # should check the value of Q_target1_next and Q_target2_next
            Q_target_next = action_probs * (
                        torch.min(Q_target1_next, Q_target2_next) - self.alpha.to(self.device) * log_pis)

            # Compute Q targets for current states (y_i)
            Q_targets = rewards + (gamma * (1 - dones) * Q_target_next.sum(dim=1).unsqueeze(-1))

            # Compute critic loss
        q1 = self.critic1(states).gather(1, actions.long())
        q2 = self.critic2(states).gather(1, actions.long())

        critic1_loss = 0.5 * F.mse_loss(q1, Q_targets)
        critic2_loss = 0.5 * F.mse_loss(q2, Q_targets)

        # Update critics
        # critic 1
        self.critic1_optimizer.zero_grad()
        critic1_loss.backward(retain_graph=True)
        clip_grad_norm_(self.critic1.parameters(), self.clip_grad_param)
        self.critic1_optimizer.step()
        # critic 2
        self.critic2_optimizer.zero_grad()
        critic2_loss.backward()
        clip_grad_norm_(self.critic2.parameters(), self.clip_grad_param)
        self.critic2_optimizer.step()

        # ----------------------- update target networks ----------------------- #
        self.soft_update(self.critic1, self.critic1_target)
        self.soft_update(self.critic2, self.critic2_target)

        return actor_loss.item(), alpha_loss.item(), critic1_loss.item(), critic2_loss.item(), current_alpha

    def soft_update(self, local_model, target_model):
        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target
        Params
        ======
            local_model: PyTorch model (weights will be copied from)
            target_model: PyTorch model (weights will be copied to)
            tau (float): interpolation parameter 
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(self._config.sac_config.tau * local_param.data + (1.0 - self._config.sac_config.tau) * target_param.data)

    def train(self):
        # np.random.seed(config.seed)
        # random.seed(config.seed)
        # torch.manual_seed(config.seed)
        #

        steps = 0
        average10 = deque(maxlen=10)

        buffer_size: int = self._config.sac_config.buffer_size
        batch_size: int = self._config.sac_config.batch_size
        n_episodes: int = self._config.sac_config.n_episodes

        wandb_config = {
            "algorithm": self.name,
            "episodes": n_episodes,
            "buffer_size": buffer_size,
            "batch_size": batch_size,
            "environment": self._env.name,
            "sales_model": self._env.sales_model_name

        }

        with wandb.init(project=self._config.wandb_config.project, entity=self._config.wandb_config.entity,
                        config=wandb_config, mode=self._config.wandb_config.mode):

            wandb.watch(self, log="gradients", log_freq=10)

            buffer = ReplayBuffer(buffer_size=buffer_size, batch_size=batch_size, device=self.device)

            collect_random(env=self._env, dataset=buffer, num_samples=self._config.sac_config.initial_random_steps,
                           state_transform_fn=self.state_transform_fn)

            for i in range(1, n_episodes + 1):
                print(f"Episode {i} starting...")
                state, _, _, _ = self._env.reset()

                episode_steps = 0
                rewards = 0
                while True:
                    # print(state)
                    action = self.get_action(state)
                    steps += 1
                    next_state, reward, done, _ = self._env.step(action)
                    buffer.add(self.state_transform_fn(state), action, list(reward.values()),
                               self.state_transform_fn(next_state), self._env.per_product_done_signal)
                    policy_loss, alpha_loss, bellmann_error1, bellmann_error2, current_alpha = self.learn(
                        buffer.sample(), gamma=self._config.sac_config.gamma)
                    state = next_state
                    rewards += np.sum(list(reward.values()))
                    # if rewards >300:
                    #     pass
                    episode_steps += 1
                    if done:
                        break
                    print(f"[{episode_steps}] next state: {next_state}, reward: {reward}, done: {done}")

                average10.append(rewards)
                print("Episode: {} | Reward: {} | Polciy Loss: {} | Steps: {}".format(i, rewards, policy_loss, steps, ))

                wandb.log({"Reward": rewards,
                           "Average10": np.mean(average10),
                           "Episodic steps": episode_steps,
                           "Policy Loss": policy_loss,
                           "Alpha Loss": alpha_loss,
                           "Bellmann error 1": bellmann_error1,
                           "Bellmann error 2": bellmann_error2,
                           "Alpha": current_alpha,
                           "Steps": steps,
                           "Episode": i,
                           "Buffer size": buffer.__len__()})
