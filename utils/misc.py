from typing import Any, Union
from pathlib import Path
import os
from typing import Optional
import numpy as np
import torch
from torch.distributions.categorical import Categorical
from torch import einsum
from einops import  reduce

def get_root_dir() -> Path:
    return Path(os.path.dirname(os.path.realpath(__file__))).parent


def first_not_none(*args: Any) -> Any:
    try:
        return next(item for item in args if item is not None)
    except StopIteration:
        raise ValueError(f"All items in list evaluated to None: {list(args)}")


def custom_collate_fn(data):
    public_obs, target = list(zip(*data))
    public_obs_tensor = torch.vstack(public_obs)
    target_tensor = torch.vstack(target)

    return {"public_obs": public_obs_tensor,
            "target": target_tensor}

# only output the public observation of the state 
def get_flatten_observation_from_state(state, obs_type:Optional[str]="public_obs")->Optional[np.ndarray]:
    if state is None:
        return None
    if obs_type is None:
        obs_type = "public_obs"
    if isinstance(state, tuple):
        extracted_state =  state[0][obs_type]
        if isinstance(extracted_state, torch.Tensor):
            return extracted_state.numpy()
    elif isinstance(state, dict):
        if obs_type in list(state.keys()):
            extracted_state =  state[obs_type]
            if isinstance(extracted_state, torch.Tensor):
                return extracted_state.numpy()
    else:
        return None

    

class CategoricalMasked(Categorical):
    def __init__(self, probs: Optional[torch.Tensor] = None, logits: Optional[torch.Tensor] = None, mask: Optional[torch.Tensor] = None):
        if (probs is None) == (logits is None):
            raise ValueError("Either `probs` or `logits` must be specified, but not both.")
        probs_or_logits = probs if probs is not None else logits
        self.mask = mask
        probs_or_logits = torch.atleast_2d(probs_or_logits)
        self.batch, self.nb_action = probs_or_logits.size()
        if mask is None:
            if logits is not None:
                super(CategoricalMasked, self).__init__(logits=probs_or_logits)
            else:
                super(CategoricalMasked, self).__init__(probs=probs_or_logits)
        else:
            self.mask = mask.bool()
            self.mask_value = torch.tensor(torch.finfo(probs_or_logits.dtype).min, dtype=probs_or_logits.dtype) if logits is not None else torch.tensor(0, dtype=probs_or_logits.dtype)
            probs_or_logits = torch.where(self.mask, probs_or_logits, self.mask_value)
            if logits is not None:
                super(CategoricalMasked, self).__init__(logits=probs_or_logits)
            else:
                super(CategoricalMasked, self).__init__(probs=probs_or_logits)

    def entropy(self):
        if self.mask is None:
            return super().entropy()
        # Elementwise multiplication
        p_log_p = einsum("ij,ij->ij", self.logits, self.probs)
        # Compute the entropy with possible action only
        p_log_p = torch.where(
            self.mask,
            p_log_p,
            torch.tensor(0, dtype=p_log_p.dtype, device=p_log_p.device),
        )
        return -reduce(p_log_p, "b a -> b", "sum", b=self.batch, a=self.nb_action)