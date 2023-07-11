# import math
# import torch

# def create_log_gaussian(mean, log_std, t):
#     quadratic = -((0.5 * (t - mean) / (log_std.exp())).pow(2))
#     l = mean.shape
#     log_z = log_std
#     z = l[-1] * math.log(2 * math.pi)
#     log_p = quadratic.sum(dim=-1) - log_z.sum(dim=-1) - 0.5 * z
#     return log_p

# def logsumexp(inputs, dim=None, keepdim=False):
#     if dim is None:
#         inputs = inputs.view(-1)
#         dim = 0
#     s, _ = torch.max(inputs, dim=dim, keepdim=True)
#     outputs = s + (inputs - s).exp().sum(dim=dim, keepdim=True).log()
#     if not keepdim:
#         outputs = outputs.squeeze(dim)
#     return outputs

# def soft_update(target, source, tau):
#     for target_param, param in zip(target.parameters(), source.parameters()):
#         target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)

# def hard_update(target, source):
#     for target_param, param in zip(target.parameters(), source.parameters()):
#         target_param.data.copy_(param.data)

from typing import Optional, Callable
import torch
from utils.misc import get_flatten_observation_from_state

def save(args, save_name, model, wandb, ep=None):
    import os
    save_dir = './trained_models/' 
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    if not ep == None:
        torch.save(model.state_dict(), save_dir + args.run_name + save_name + str(ep) + ".pth")
        wandb.save(save_dir + args.run_name + save_name + str(ep) + ".pth")
    else:
        torch.save(model.state_dict(), save_dir + args.run_name + save_name + ".pth")
        wandb.save(save_dir + args.run_name + save_name + ".pth")

def collect_random(env, dataset, num_samples=200, state_transform_fn: Optional[Callable] = None):

    # if no state_transform_fn is provided, use identity function
    if state_transform_fn is None:
        state_transform_fn = lambda x: x

    state, _, _, _ = env.reset()
    idx=0
    for _ in range(num_samples):
        while idx<100 :
            action = [env.action_space.sample() for _ in range(env.state.n_products)]
            next_state, reward, done, _ = env.step(action)
            state = next_state
            idx += 1
        action = [env.action_space.sample() for _ in range(env.state.n_products)]
        next_state, reward, done, _ = env.step(action)
        dataset.add(state_transform_fn(state), action, list(reward.values()), state_transform_fn(next_state), env.per_product_done_signal)
        state = next_state
        if done:
            idx = 0 
            state, _, _, _ = env.reset()
