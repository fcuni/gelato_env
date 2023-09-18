import pickle
from collections import deque
from datetime import timedelta
from itertools import count
from pathlib import Path
from typing import Deque, Optional, Dict, Any, Union

import gym
import torch

from env_wrapper.output_gelateria_state_wrapper import OutputGelateriaStateWrapper
from env_wrapper.wrapper import DefaultGelatoEnvWrapper
from models.base_rl_agent import RLAgent
from models.mbpo.utils import *
from models.sac.sac_discrete import SACDiscrete
from utils.config import MBPOConfig
from utils.logging import EpisodeLogger
from utils.types import TensorType
from models.mbpo.external_lib.mbpo_pytorch.model import EnsembleDynamicsModel
from models.mbpo.external_lib.mbpo_pytorch.predict_env import PredictEnv
from models.mbpo.external_lib.mbpo_pytorch.sample_env import EnvSampler
from wandb.wandb_run import Run


class MBPO(RLAgent):

    def __init__(self, env: DefaultGelatoEnvWrapper,
                 agent: RLAgent,
                 config: MBPOConfig,
                 name: str = "MBPO",
                 device: Optional[torch.device] = None,
                 run_name: Optional[str] = None):

        super().__init__(env=env, name=f"{name}-{agent.name}", run_name=run_name)
        self._config: MBPOConfig = config

        assert isinstance(self._env.action_space, gym.spaces.Discrete), "only discrete action space is supported"
        if device is None:
            self._device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        # Initialise the models
        self.initialise_models()

        # Initialise the Env Buffer and Model Buffer
        self.initialise_buffers()

        # Initialise the agent
        self._agent = agent

        # initialise the sampler for the environment
        self._env_sampler = EnvSampler(OutputGelateriaStateWrapper(self._env),
                                       max_path_length=self._config.max_path_length)

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
        raise NotImplementedError

    @property
    def configs(self) -> Dict[str, Any]:
        """Return the configurations of the agent."""
        agent_configs = {}
        for k, v in self._agent.configs.items():
            if "/" in k:
                agent_configs[k] = v
        return {
            "episodes": self._config.num_epoch,
            "buffer_size": self._config.replay_size,
            "mbpo/num_networks": self._config.num_networks,
            "mbpo/num_elites": self._config.num_elites,
            "mbpo/pred_hidden_size": self._config.pred_hidden_size,
            "mbpo/use_decay": self._config.use_decay,
            "mbpo/rollout_batch_size": self._config.rollout_batch_size,
            "mbpo/model_retain_epochs": self._config.model_retain_epochs,
            "mbpo/max_path_length": self._config.max_path_length,
            "mbpo/init_exploration_steps": self._config.init_exploration_steps,
            "mbpo/epoch_length": self._config.epoch_length,
            "mbpo/min_pool_size": self._config.min_pool_size,
            "mbpo/model_train_freq": self._config.model_train_freq,
            "mbpo/real_ratio": self._config.real_ratio,
            "mbpo/predict_model_batch_size": self._config.predict_model_batch_size,
            "mbpo/predict_model_holdout_ratio": self._config.predict_model_holdout_ratio,
            "mbpo/train_every_n_steps": self._config.train_every_n_steps,
            "mbpo/max_train_repeat_per_step": self._config.max_train_repeat_per_step,
            "mbpo/num_train_repeat": self._config.num_train_repeat,
            "mbpo/policy_train_batch_size": self._config.policy_train_batch_size,
            "mbpo/rollout_min_length": self._config.rollout_min_length,
            "mbpo/rollout_max_length": self._config.rollout_max_length,
            "mbpo/rollout_max_epoch": self._config.rollout_max_epoch,
            "mbpo/rollout_min_epoch": self._config.rollout_min_epoch,
            **agent_configs
        }

    def save(self, path: Optional[Path] = None):
        """Save the trained models.

        Args:
            path (Optional[Path]): Path to save the models to. Defaults to None.
        """

        # Set default path if none is provided
        if path is None:
            path = Path.cwd() / "experiment_data" / self.run_name / "trained_models"

        # Create directory if it does not exist
        if not path.exists():
            path.mkdir(parents=True)

        # Save the agent
        self._agent.save(path / self._agent.name)


    def load(self):
        """Load the agent."""
        raise NotImplementedError

    def initialise_models(self):
        self._env_model = EnsembleDynamicsModel(self._config.num_networks, self._config.num_elites,
                                                self._state_size, self._action_size,
                                                self._reward_size, self._config.pred_hidden_size,
                                                use_decay=self._config.use_decay)
        # Predict environments
        self._predict_env = PredictEnv(self._env_model, self._env.name, 'pytorch')

    def initialise_buffers(self):
        # Initial pool for env
        self._env_pool = ReplayMemory(self._config.replay_size)
        # Initial pool for model
        rollouts_per_epoch = self._config.rollout_batch_size * self._config.epoch_length / self._config.model_train_freq
        model_steps_per_epoch = int(1 * rollouts_per_epoch)
        new_pool_size = self._config.model_retain_epochs * model_steps_per_epoch
        self._model_pool = ReplayMemory(new_pool_size)

    def rollout_model(self, rollout_length):
        state, action, reward, next_state, done, current_date = transform_sample_from_buffer(
            self._env_pool.sample_all_batch(self._config.rollout_batch_size), filter_terminal=True)
        last_actions = None
        date_step_fn = lambda d: d + timedelta(days=self._config.days_per_step)
        base_price = state[:, 2]  # get the base price (won't change)
        for i in range(rollout_length):

            action_mask = self._env.mask_actions(state=torch.from_numpy(state), current_dates=current_date)

            # Get a batch of actions
            action = self._agent.select_action(torch.from_numpy(state).float().to(self._device), mask=action_mask)
            next_states, rewards, terminals, info = self._predict_env.step(state, action[:, None])

            # GelateriaEnv-specific code
            next_states[:, 0] = np.round(action / 100, 2)
            next_states[:, 2] = base_price
            next_states[:, 3] = np.array([(d.timetuple().tm_yday - 1) / 365 for d in current_date])

            # Push a batch of samples
            self._model_pool.push_batch(
                [(state[j], action[j], rewards[j], next_states[j], terminals[j], current_date[j]) for j in
                 range(len(state))])
            nonterm_mask = ~terminals.squeeze(-1)
            if nonterm_mask.sum() == 0:
                break
            state = next_states[nonterm_mask]
            base_price = base_price[nonterm_mask]
            current_date = date_step_fn(current_date[nonterm_mask])

    def train_predict_model(self):
        # Get all samples from environment
        state_obs, action, reward, next_state_obs, done, current_date = \
            transform_sample_from_buffer(self._env_pool.sample(len(self._env_pool)), filter_terminal=True)
        delta_state_obs = next_state_obs - state_obs
        inputs = np.concatenate((state_obs, action[:, None]), axis=-1)
        labels = np.concatenate((reward[:, None], delta_state_obs), axis=-1)

        self._predict_env.model.train(inputs, labels, batch_size=self._config.predict_model_batch_size,
                                      holdout_ratio=self._config.predict_model_holdout_ratio)

    def train_policy_repeats(self, total_step, train_step, cur_step):
        if total_step % self._config.train_every_n_steps > 0:
            return 0

        if train_step > self._config.max_train_repeat_per_step * total_step:
            return 0

        for i in range(self._config.num_train_repeat):
            env_batch_size = int(self._config.policy_train_batch_size * self._config.real_ratio)
            model_batch_size = self._config.policy_train_batch_size - env_batch_size

            env_pool_samples = self._env_pool.sample(int(env_batch_size))
            env_state, env_action, env_reward, env_next_state, env_done, env_date = transform_sample_from_buffer(
                env_pool_samples)

            if model_batch_size > 0 and len(self._model_pool) > 0:
                model_pool_samples = self._model_pool.sample_all_batch(int(model_batch_size))
                model_state, model_action, model_reward, model_next_state, model_done, model_date = model_pool_samples
                batch_state = np.concatenate((env_state, model_state), axis=0)
                batch_action = np.concatenate((env_action, model_action), axis=0)
                batch_reward = np.concatenate((np.reshape(env_reward, (env_reward.shape[0], -1)), model_reward), axis=0)
                batch_next_state = np.concatenate((env_next_state, model_next_state), axis=0)
                batch_done = np.concatenate((np.reshape(env_done, (env_done.shape[0], -1)), model_done), axis=0)
                batch_date = np.concatenate((env_date, model_date), axis=0)

            else:
                batch_state, batch_action, batch_reward, batch_next_state, batch_done, batch_date = \
                    env_state, env_action, env_reward, env_next_state, env_done, env_date

            batch_reward, batch_done = np.squeeze(batch_reward), np.squeeze(batch_done)
            batch_done = (~batch_done).astype(int)

            batch_mask = self._env.mask_actions(state=batch_state, current_dates=batch_date)

            batch_experience = (
                torch.from_numpy(batch_state).float(),
                torch.from_numpy(batch_action)[:, None].float(),
                torch.from_numpy(batch_next_state).float(),
                torch.from_numpy(batch_reward)[:, None].float(),
                torch.from_numpy(batch_done)[:, None],
                batch_mask
            )

            self._agent.update_parameters(batch_experience, i)

        return self._config.num_train_repeat

    def train(self, wandb_run: Optional[Run] = None):

        # Initialise variables
        total_step = 0
        reward_sum = 0
        rollout_length = 1

        average_10_episode_reward: Deque = deque(maxlen=10)

        exploration_before_start(self._config, self._env_sampler, self._env_pool, self._agent)

        for epoch_step in range(self._config.num_epoch):
            start_step = total_step
            train_policy_steps = 0
            for i in count():
                cur_step = total_step - start_step

                if cur_step >= self._config.epoch_length and len(self._env_pool) > self._config.min_pool_size:
                    break

                if cur_step > 0 and cur_step % self._config.model_train_freq == 0 and self._config.real_ratio < 1.0:
                    self.train_predict_model()

                    new_rollout_length = set_rollout_length(self._config, epoch_step)
                    if rollout_length != new_rollout_length:
                        rollout_length = new_rollout_length
                        self._model_pool = resize_model_pool(self._config, rollout_length, self._model_pool)

                    self.rollout_model(rollout_length)

                cur_state, action, next_state, reward, done, info = self._env_sampler.sample(self._agent)
                self._env_pool.push(cur_state, action, reward, next_state, done, info['current_date'])

                if len(self._env_pool) > self._config.min_pool_size:
                    train_policy_steps += self.train_policy_repeats(total_step, train_policy_steps, cur_step)

                total_step += 1

                if total_step % self._config.epoch_length == 0:
                    logger = EpisodeLogger()
                    init_state_info = self._env_sampler.reset()
                    sum_reward = 0
                    done = False
                    test_step = 0
                    logger.log_info(init_state_info)

                    while (not done) and (test_step != self._config.max_path_length):
                        cur_state, action, next_state, reward, done, info = self._env_sampler.sample(self._agent,
                                                                                                     eval_t=True)
                        sum_reward += np.sum(reward)
                        test_step += 1
                        logger.log_info(info)

                    average_10_episode_reward.append(sum_reward)

                    print(f"Epoch {epoch_step:03d} - Step Reward: {total_step} {sum_reward}")

                    episode_summary = logger.get_episode_summary()

                    # Use wandb to record rewards per episode
                    if wandb_run is not None:
                        fig = logger.plot_episode_summary(title=f"Episode {epoch_step}")

                        wandb_log = {
                            "env_buffer_usage": len(self._env_pool),
                            "model_buffer_usage": len(self._model_pool),
                            "episode_reward": sum_reward,
                            "average_10_episode_reward": 0.0 if len(average_10_episode_reward) == 0 else np.mean(
                                average_10_episode_reward),
                            # "cumulative_reward": cumulative_reward,
                            "episode_step": test_step,
                            "epoch": epoch_step,
                            "global_step": total_step,
                            "summary_plots": fig,
                            "total_revenue": episode_summary["total_revenue"],
                            **{f"rewards/average_{k}": v for k, v in episode_summary["mean_product_reward_per_type"].items()}
                        }

                        wandb_run.log(wandb_log)

                    # save checkpoint (model and episode summary)
                    checkpoint_path = Path.cwd() / "experiment_data" / self.run_name / "checkpoints" \
                                      / f"epoch_{epoch_step:04d}"
                    # save agent
                    self._agent.save(checkpoint_path / "agent")
                    # save episode summary
                    with open(checkpoint_path / "episode_summary.pkl", "wb") as f:
                        pickle.dump(episode_summary, f)
