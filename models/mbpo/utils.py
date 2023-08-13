from typing import Tuple

import numpy as np

from env.gelateria import GelateriaState
from models.mbpo.external_lib.mbpo_pytorch.sac.replay_memory import ReplayMemory


def exploration_before_start(args, env_sampler, env_pool, agent):
    for i in range(args.init_exploration_steps):
        cur_state, action, next_state, reward, done, info = env_sampler.sample(agent)
        env_pool.push(cur_state, action, reward, next_state, done, info['current_date'])


def resize_model_pool(args, rollout_length, model_pool):
    rollouts_per_epoch = args.rollout_batch_size * args.epoch_length / args.model_train_freq
    model_steps_per_epoch = int(rollout_length * rollouts_per_epoch)
    new_pool_size = args.model_retain_epochs * model_steps_per_epoch

    sample_all = model_pool.return_all()
    new_model_pool = ReplayMemory(new_pool_size)
    new_model_pool.push_batch(sample_all)

    return new_model_pool


def set_rollout_length(args, epoch_step):
    rollout_length = (min(max(args.rollout_min_length + (epoch_step - args.rollout_min_epoch) / (
                args.rollout_max_epoch - args.rollout_min_epoch) * (args.rollout_max_length - args.rollout_min_length),
                              args.rollout_min_length), args.rollout_max_length))
    return int(rollout_length)


# def rollout_model(args, predict_env, agent, model_pool, env_pool, rollout_length, action_mask_fn):
#     state, action, reward, next_state, done, current_date = transform_sample_from_buffer(env_pool.sample_all_batch(args['rollout_batch_size']), filter_terminal=True)
#     last_actions = None
#     date_step_fn = lambda d: d + timedelta(days=args['days_per_step'])
#     base_price = state[:, 2]  # get the base price (won't change)
#     for i in range(rollout_length):
#
#         action_mask = args['action_mask_fn'](state=torch.from_numpy(state), current_dates=current_date)
#
#         # TODO: Get a batch of actions
#         action = agent.select_action(torch.from_numpy(state), mask=action_mask)
#         next_states, rewards, terminals, info = predict_env.step(state, action[:, None])
#
#         # GelateriaEnv-specific code
#         next_states[:, 0] = np.round(action/100, 2)
#         next_states[:, 2] = base_price
#         next_states[:, 3] = np.array([(d.timetuple().tm_yday-1)/365 for d in current_date])
#
#         # TODO: Push a batch of samples
#         model_pool.push_batch([(state[j], action[j], rewards[j], next_states[j], terminals[j], current_date[j]) for j in range(len(state))])
#         nonterm_mask = ~terminals.squeeze(-1)
#         if nonterm_mask.sum() == 0:
#             break
#         state = next_states[nonterm_mask]
#         base_price = base_price[nonterm_mask]
#         current_date = date_step_fn(current_date[nonterm_mask])

#
# def train_predict_model(args, env_pool, predict_env):
#     # Get all samples from environment
#     state_obs, action, reward, next_state_obs, done, current_date = \
#         transform_sample_from_buffer(env_pool.sample(len(env_pool)), filter_terminal=True)
#     delta_state_obs = next_state_obs - state_obs
#     inputs = np.concatenate((state_obs, action[:, None]), axis=-1)
#     labels = np.concatenate((reward[:, None], delta_state_obs), axis=-1)
#
#     predict_env.model.train(inputs, labels, batch_size=args['predict_model_batch_size'], holdout_ratio=args['predict_model_holdout_ratio'])


def decompose_state_from_buffer(state: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    state_obs, done_signal = [], []

    if isinstance(state[0], GelateriaState):
        for s_i, s in enumerate(state):
            state_obs.append(s.get_public_observations().numpy())
            done_signal.append(s.per_product_done_signal)

        state_obs = np.concatenate(state_obs, axis=0)
        done_signal = np.concatenate(done_signal, axis=0)

    else:
        raise NotImplementedError

    return state_obs, done_signal


def transform_sample_from_buffer(
        sample_from_buffer: Tuple[GelateriaState, np.ndarray, np.ndarray, GelateriaState, bool],
        filter_terminal: bool = False):
    state, action, reward, next_state, done, current_dates = sample_from_buffer
    state_obs, state_dones = decompose_state_from_buffer(state)
    next_state_obs, next_state_dones = decompose_state_from_buffer(next_state)
    action = np.concatenate(action, axis=0)
    reward = np.concatenate(reward, axis=0)
    if len(state) > 0:
        current_dates = np.repeat(current_dates, state[0].n_products, axis=0)
    if filter_terminal:
        return state_obs[~state_dones], action[~state_dones], reward[~state_dones], next_state_obs[~state_dones], \
            next_state_dones[~state_dones], current_dates[~state_dones]
    else:
        return state_obs, action, reward, next_state_obs, next_state_dones, current_dates
