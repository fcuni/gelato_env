from typing import Any, Dict, Union, Sequence, Optional
from pathlib import Path
import os
from typing import Optional
import numpy as np
import torch
from gym.spaces import MultiBinary
from torch.utils.data import random_split, DataLoader, TensorDataset
from utils.types import TensorType

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


def to_tensor(x: TensorType, device: Optional[torch.device] = None) -> torch.Tensor:
    """
    Helper function to convert a numpy array or torch.Tensor to a torch.Tensor.

    Args:
        x: input to convert.
        device: device to have the output tensor assigned to. Defaults to None.

    Returns:
        torch.Tensor of the input.
    """
    if isinstance(x, torch.Tensor):
        return x if device is None else x.to(device)
    if isinstance(x, np.ndarray):
        return torch.from_numpy(x) if device is None else torch.from_numpy(x).to(device)
    raise ValueError("Input must be torch.Tensor or np.ndarray.")


def split_dataset_to_train_valid_loaders(dataset: TensorDataset, train_fraction: float = 0.8, batch_size: int = 128,
                                         shuffle: bool = True, seed: Optional[int] = None):
    """
    Split a TensorDataset into train_loader and valid_loader.

    Args:
        dataset (TensorDataset): The TensorDataset to be split.
        train_fraction (float, optional): Fraction of data to be used for training. Default is 0.8.
        batch_size (int, optional): Batch size for DataLoader. Default is 128.
        shuffle (bool, optional): Whether to shuffle the data. Default is True.
        seed (int, optional): Random seed for reproducibility. Default is None.

    Returns:
        train_loader (DataLoader): DataLoader for the training set.
        valid_loader (DataLoader): DataLoader for the validation set.
    """
    # Calculate the sizes of training and validation sets
    num_samples = len(dataset)
    num_train_samples = int(train_fraction * num_samples)
    num_valid_samples = num_samples - num_train_samples

    # Set random seed for reproducibility if provided
    if seed is not None:
        torch.manual_seed(seed)

    # Split the dataset into training and validation sets
    train_dataset, valid_dataset = random_split(dataset, [num_train_samples, num_valid_samples])

    # Create DataLoader for training set
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle)

    # Create DataLoader for validation set
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=shuffle)

    return train_loader, valid_loader


class OneHotEncoding(MultiBinary):
    def __init__(self, n: Union[np.ndarray, Sequence[int], int],
                 seed: Optional[Union[int, np.random.Generator]] = None):
        assert isinstance(n, int), "n must be an int"
        super().__init__(n=n, seed=seed)

    def sample(self, mask: Optional[np.ndarray] = None) -> np.ndarray:
        one_hot_arr = np.zeros(self.n, dtype=self.dtype)
        index = self.np_random.integers(low=0, high=self.n, dtype=int)
        one_hot_arr[index] = 1
        return one_hot_arr

    def contains(self, x) -> bool:
        """Return boolean specifying if x is a valid member of this space."""
        if isinstance(x, Sequence):
            x = np.array(x)  # Promote list to array for contains check

        return bool(
            isinstance(x, np.ndarray)
            and self.shape == x.shape
            and np.all((x == 0) | (x == 1))
            and np.sum(x) == 1
        )

    def __repr__(self):
        return "OneHotEncoding(%d)" % self.n

    def __eq__(self, other):
        return self.n == other.n
