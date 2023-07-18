from pathlib import Path
from typing import Tuple, Optional, Sequence, Union, Any

import numpy as np
import random

import pandas as pd
import torch
from collections import deque

from utils.misc import Experience

import pickle

class ReplayBuffer:
    """Fixed-size buffer to store experience tuples."""

    def __init__(self, buffer_size, batch_size,
                 seed: Optional[int] = None,
                 device: Optional[torch.device] = None):
        """Initialize a ReplayBuffer object.
        Params
        ======
            buffer_size (int): maximum size of buffer
            batch_size (int): size of each training batch
            seed (int): random seed
        """
        if device is None:
            device = torch.device("cpu")
        self.device = device
        self.memory = deque(maxlen=buffer_size)
        self.batch_size = batch_size
        self.experience = Experience
        if seed is not None:
            random.seed(seed)
            self.seed = seed

    def to_csv(self, path: Path):
        """Save the buffer to a CSV file."""
        pd.DataFrame(self.memory).to_csv(path)

    def save(self, path: Path):
        """Save the buffer to a file."""
        if path.is_dir():
            # create the directory and its parent directories, if they doesn't exist
            path.mkdir(parents=True, exist_ok=True)
            # append file name to the path (since it is not given)
            path = path / "buffer.pkl"
        else:
            # create the parent directories if they don't exist
            path.parent.mkdir(parents=True, exist_ok=True)

        # save the buffer to the file
        pickle.dump(self.memory, open(path, 'wb'))

    def load(self, path: Optional[Path] = None):
        """Load the buffer from a file.

        Args:
            path: Path to the file to load. If `None`, no file will be loaded.
        """

        if path is None:
            print(f"No buffer file path given, a new buffer will be created")
            return
        if not path.exists():
            print(f"Buffer file {path} does not exist")
            return
        # TODO: check if the following file loading logic is good enough
        self.memory = pickle.load(open(path, 'rb'))

    def add(self, state: Sequence[Any], action: Sequence[Union[int, float]], reward: Sequence[float], next_state: Sequence[Any], done: Sequence[bool]):
        """Add a new experience to memory."""
        for i in range(len(state)):
            e = self.experience(state[i], action[i], reward[i], next_state[i], done[i])
            self.memory.append(e)

    def sample(self, batch_size: Optional[int] = None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Randomly sample a batch of experiences from memory."""
        if batch_size is None:
            batch_size = self.batch_size
        experiences = random.sample(self.memory, k=batch_size)

        states = torch.from_numpy(np.stack([e.state for e in experiences if e is not None])).float().to(self.device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).float().to(self.device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(self.device)
        next_states = torch.from_numpy(np.stack([e.next_state for e in experiences if e is not None])).float().to(
            self.device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(
            self.device)

        return states, actions, rewards, next_states, dones

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)
