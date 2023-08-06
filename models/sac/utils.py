from typing import Optional, Callable

import numpy as np
# import torch
from utils.misc import convert_dict_to_numpy


# def save(args, save_name, model, wandb, ep=None):
#     import os
#     save_dir = './trained_models/'
#     if not os.path.exists(save_dir):
#         os.makedirs(save_dir)
#     if ep is not None:
#         torch.save(model.state_dict(), save_dir + args.run_name + save_name + str(ep) + ".pth")
#         wandb.save(save_dir + args.run_name + save_name + str(ep) + ".pth")
#     else:
#         torch.save(model.state_dict(), save_dir + args.run_name + save_name + ".pth")
#         wandb.save(save_dir + args.run_name + save_name + ".pth")


def collect_random(env, dataset, num_samples=200, state_transform_fn: Optional[Callable] = None):
    # if no state_transform_fn is provided, use identity function
    if state_transform_fn is None:
        state_transform_fn: Callable = lambda x: x

    state, *_, _, _ = env.reset()
    idx = 0
    for _ in range(num_samples):
        while idx < 100:
            action = [env.action_space.sample() for _ in range(env.state.n_products)]
            next_state, reward, done, _ = env.step(action)
            state = next_state
            idx += 1
        action = [env.action_space.sample() for _ in range(env.state.n_products)]
        next_state, reward, done, _ = env.step(action)
        dataset.add(state=state_transform_fn(state), action=action, reward=list(reward.values()),
                    next_state=state_transform_fn(next_state), terminated=env.state.per_product_done_signal)
        state = next_state
        if done:
            idx = 0
            state, _, _, _ = env.reset()


def collect_random_v2(env, dataset, num_samples=200, state_transform_fn: Optional[Callable] = None):
    # if no state_transform_fn is provided, use identity function
    if state_transform_fn is None:
        state_transform_fn: Callable = lambda x: x

    state, _, _, _ = env.reset()
    for _ in range(num_samples):
        action_mask = env.mask_actions().astype(np.int8)
        action = np.array([env.action_space.sample(mask=action_mask[i]) for i in range(env.state.n_products)])
        next_state, reward, _, _ = env.step(action, action_dtype="int")
        dataset.add(state=state_transform_fn(state), action=action, reward=convert_dict_to_numpy(reward),
                    next_state=state_transform_fn(next_state), terminated=env.state.per_product_done_signal)
        sample_state, _ = env.sample_from_current_store()
        env.reset()
        env.set_state(sample_state)
        state = env.get_observations(env.state)
    env.reset()


def collect_random_v3(env, dataset, num_samples=200, state_transform_fn: Optional[Callable] = None):

    state = env.reset()
    dones = np.zeros(env.state.n_products, dtype=bool)
    for _ in range(num_samples):
        action_mask = env.mask_actions().astype(np.int8)
        action = np.array([env.action_space.sample(mask=action_mask[i]) for i in range(env.state.n_products)])
        next_state, reward, done, _ = env.step(action)

        dataset.add(state=state[~dones], action=action[~dones], reward=reward[~dones],
                    next_state=next_state[~dones], terminated=env.state.per_product_done_signal[~dones])
        dones = env.state.per_product_done_signal
        state = next_state
        if done:
            state = env.reset()
            dones = np.zeros(env.state.n_products, dtype=bool)

