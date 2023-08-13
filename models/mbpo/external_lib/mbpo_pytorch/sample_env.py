import torch

from env.gelateria import GelateriaState
from env_wrapper.output_gelateria_state_wrapper import OutputGelateriaStateWrapper


class EnvSampler():
    def __init__(self, env, max_path_length=1000):
        self.env = env

        self.path_length = 0
        self.current_state = None
        self.max_path_length = max_path_length
        self.path_rewards = []
        self.sum_reward = 0

        # check if the env is a OutputGelateriaStateWrapper
        self._is_OutputGelateriaStateWrapper = isinstance(env, OutputGelateriaStateWrapper)



    def sample(self, agent, eval_t=False):

        if self._is_OutputGelateriaStateWrapper:
            return self.sample_output_gelateria_state_wrapper(agent, eval_t)

        if self.current_state is None:
            self.current_state = self.env.reset()

        cur_state = self.current_state
        orig_done_signal = self.env.state.per_product_done_signal
        action_mask = self.env.mask_actions()
        action = agent.select_action(torch.from_numpy(self.current_state), eval_t, mask=action_mask)
        next_state, reward, terminal, info = self.env.step(action)
        done_signal = self.env.state.per_product_done_signal
        self.path_length += 1
        self.sum_reward += reward.sum()

        # TODO: Save the path to the env_pool
        if terminal or self.path_length >= self.max_path_length:
            self.current_state = None
            self.path_length = 0
            self.path_rewards.append(self.sum_reward)
            self.sum_reward = 0
        else:
            self.current_state = next_state

        # TODO: double check if this is still valid
        return cur_state, action, next_state, reward, done_signal, info, action_mask, orig_done_signal

    def sample_output_gelateria_state_wrapper(self, agent, eval_t=False):
        if self.current_state is None:
            self.current_state: GelateriaState = self.env.reset()

        cur_state = self.current_state
        cur_state_obs = self.env.state.get_public_observations()
        action_mask = self.env.mask_actions()
        action = agent.select_action(cur_state_obs, eval_t, mask=action_mask)
        next_state, reward, terminal, info = self.env.step(action)
        self.path_length += 1
        self.sum_reward += reward.sum()

        # TODO: Save the path to the env_pool
        if terminal or self.path_length >= self.max_path_length:
            self.current_state = None
            self.path_length = 0
            self.path_rewards.append(self.sum_reward)
            self.sum_reward = 0
        else:
            self.current_state = next_state

        return cur_state, action, next_state, reward, terminal, info

    def reset(self):
        """
        Reset the environment and return the information regarding the initial state
        """
        self.current_state, init_state_info = self.env.reset(get_info=True)
        self.path_length = 0
        self.path_rewards = []
        self.sum_reward = 0
        return init_state_info
