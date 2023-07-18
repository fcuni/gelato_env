from collections import namedtuple
from typing import Any, Dict
from pathlib import Path
import os
from typing import Optional
import numpy as np
import torch


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


def get_flatten_observation_from_state(state_obs, obs_type: Optional[str] = "public_obs") -> Optional[np.ndarray]:
    """
    Extract the public observation (unless specified otherwise) from the state and convert it to a numpy array.

    Args:
        state_obs: state to extract the observation from.
        obs_type: type of observation to extract. Defaults to "public_obs".

    Returns:
        numpy array of the public observation of the state.
    """
    if state_obs is None:
        return None
    if obs_type is None:
        obs_type = "public_obs"
    if isinstance(state_obs, tuple):
        extracted_state = state_obs[0][obs_type]
        if isinstance(extracted_state, torch.Tensor):
            return extracted_state.numpy()
    elif isinstance(state_obs, dict):
        if obs_type in list(state_obs.keys()):
            extracted_state = state_obs[obs_type]
            if isinstance(extracted_state, torch.Tensor):
                return extracted_state.numpy()
    else:
        return None


def convert_dict_to_numpy(dictionary: Dict[str, Any]) -> np.ndarray:
    """
    Convert the values of a dictionary into a numpy array

    Args:
        dictionary: dictionary to convert from.

    Returns:
        numpy array of the values of the dictionary.
    """

    return np.array(list(dictionary.values()))


Experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])