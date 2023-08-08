from pathlib import Path
from typing import Tuple, Optional, Sequence, Union, Any

import numpy as np
import random

import pandas as pd
import torch
from collections import deque

from torch.utils.data import DataLoader, TensorDataset

from utils.misc import to_tensor
from utils.types import Transition, TensorType

import pickle


class ReplayBuffer:
    """Fixed-size buffer to store transition tuples."""

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
        self.transition = Transition
        if seed is not None:
            random.seed(seed)
            self.seed = seed

    @property
    def state_dim(self):
        """Return the dimension of the state."""
        assert len(self.memory) > 0, "The buffer is empty"
        if len(self.memory[0].state.shape) == 0:
            return 1
        else:
            return self.memory[0].state.shape[0]

    @property
    def action_dim(self):
        """Return the dimension of the action."""
        assert len(self.memory) > 0, "The buffer is empty"
        if len(self.memory[0].action.shape) == 0:
            return 1
        else:
            return self.memory[0].action.shape[0]


    def to_csv(self, path: Path):
        """Save the buffer to a CSV file."""
        pd.DataFrame(self.memory).to_csv(path)

    def to_tensor_dataset(self) -> TensorDataset:
        """
        Function to export a TensorDataset from the current `ReplayBuffer` object.

        Returns:
            dataset: `TensorDataset` object
        """
        states = []
        actions = []
        next_states = []
        rewards = []
        terminations = []
        # truncations = []

        for transition in self.memory:
            states.append(transition.state)
            actions.append(transition.action)
            next_states.append(transition.next_state)
            rewards.append(transition.reward)
            terminations.append(transition.terminated)
            # truncations.append(transition.truncated)

        states = torch.tensor(np.array(states), dtype=torch.float32)
        actions = torch.tensor(np.array(actions), dtype=torch.float32)
        next_states = torch.tensor(np.array(next_states), dtype=torch.float32)
        rewards = torch.tensor(np.array(rewards), dtype=torch.float32)
        terminations = torch.tensor(np.array(terminations), dtype=torch.float32)
        # truncations = torch.tensor(np.array(truncations), dtype=torch.float32)

        return TensorDataset(states, actions, next_states, rewards, terminations)
        # return TensorDataset(states, actions, next_states, rewards, terminations, truncations)

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

    def add(self, state: Sequence[Any], action: Sequence[Union[int, float]], next_state: Sequence[Any],
            reward: Sequence[float], terminated: Sequence[bool], truncated: Optional[Sequence[bool]] = None):
        """Add a new transition to memory."""
        for i in range(len(state)):
            if truncated is None:
                e = self.transition(state[i], action[i], next_state[i], reward[i], terminated[i], None)
            else:
                e = self.transition(state[i], action[i], next_state[i], reward[i], terminated[i], truncated[i])
            self.memory.append(e)

    def sample(self, batch_size: Optional[int] = None) -> Tuple[
               torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Randomly sample a batch of transitions from memory."""
        if batch_size is None:
            batch_size = self.batch_size
        experiences = random.sample(self.memory, k=batch_size)

        states = to_tensor(np.stack([e.state for e in experiences if e is not None]), device=self.device).float()
        actions = to_tensor(np.vstack([e.action for e in experiences if e is not None]),
                            device=self.device).float()
        next_states = to_tensor(np.stack([e.next_state for e in experiences if e is not None]),
                                device=self.device).float()
        rewards = to_tensor(np.vstack([e.reward for e in experiences if e is not None]),
                            device=self.device).float()
        terminateds = to_tensor(np.vstack([e.terminated for e in experiences if e is not None]).astype(np.uint8),
                                device=self.device).float()
        # truncateds = to_tensor(np.vstack([e.truncated for e in experiences if e is not None]).astype(np.uint8),
        #                         device=self.device).float()

        return states, actions, next_states, rewards, terminateds

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)

    def __getitem__(self, item) -> Transition:
        return self.memory[item]

class ReplayBufferForTrajectories:
    """Fixed-size buffer to store experience tuples."""

    def __init__(self, buffer_size: int, batch_size: int,
                 max_trajectory_length: int,
                 seed: Optional[int] = None,
                 device: Optional[torch.device] = None):
        """Initialize a ReplayBuffer object.
        Params
        ======
            buffer_size (int): maximum size of buffer
            batch_size (int): size of each training batch
            max_trajectory_length (int): maximum length of a single trajectory
            seed (int): random seed
        """
        if device is None:
            device = torch.device("cpu")
        self.device = device
        self.memory = deque(maxlen=buffer_size)
        self.batch_size = batch_size
        self.max_trajectory_length = max_trajectory_length
        self.current_trajectory = []
        self.transition = Transition
        if seed is not None:
            random.seed(seed)
            self.seed = seed

    def to_csv(self, path: Path):
        """Save the buffer to a CSV file."""
        pd.DataFrame(self.memory).to_csv(path)

    def save(self, path: Path):
        """Save the buffer to a file."""
        if path.is_dir():
            # create the directory and its parent directories, if they don't exist
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

    def add(self, state: Sequence[Any], action: Sequence[Union[int, float]], reward: Sequence[float],
            next_state: Sequence[Any], done: Sequence[bool]):
        """Add a new experience to memory."""
        for i in range(len(state)):
            e = self.transition(state[i], action[i], reward[i], next_state[i], done[i])
            self.current_trajectory.append(e)

            if done[i] or len(self.current_trajectory) >= self.max_trajectory_length:
                self.memory.append(self.current_trajectory)
                self.current_trajectory = []

    def sample(self, batch_size: Optional[int] = None) -> Tuple[
        torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Randomly sample a batch of experiences from memory."""
        if batch_size is None:
            batch_size = self.batch_size
        trajectories = random.sample(self.memory, k=batch_size)

        # Flatten the trajectories and sample transitions
        experiences = [exp for traj in trajectories for exp in traj]
        experiences = random.sample(experiences, k=batch_size)

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


from utils.types import Transition_new
class ReplayBuffer_v2:
    """Fixed-size buffer to store transition tuples."""

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
        self.transition = Transition_new
        if seed is not None:
            random.seed(seed)
            self.seed = seed

    @property
    def state_dim(self):
        """Return the dimension of the state."""
        assert len(self.memory) > 0, "The buffer is empty"
        if len(self.memory[0].state.shape) == 0:
            return 1
        else:
            return self.memory[0].state.shape[0]

    @property
    def action_dim(self):
        """Return the dimension of the action."""
        assert len(self.memory) > 0, "The buffer is empty"
        if len(self.memory[0].action.shape) == 0:
            return 1
        else:
            return self.memory[0].action.shape[0]


    def to_csv(self, path: Path):
        """Save the buffer to a CSV file."""
        pd.DataFrame(self.memory).to_csv(path)

    def to_tensor_dataset(self) -> TensorDataset:
        """
        Function to export a TensorDataset from the current `ReplayBuffer` object.

        Returns:
            dataset: `TensorDataset` object
        """
        states = []
        actions = []
        next_states = []
        rewards = []
        terminations = []
        # truncations = []
        action_mask = []

        for transition in self.memory:
            states.append(transition.state)
            actions.append(transition.action)
            next_states.append(transition.next_state)
            rewards.append(transition.reward)
            terminations.append(transition.terminated)
            # truncations.append(transition.truncated)
            action_mask.append(transition.action_mask)

        states = torch.tensor(np.array(states), dtype=torch.float32)
        actions = torch.tensor(np.array(actions), dtype=torch.float32)
        next_states = torch.tensor(np.array(next_states), dtype=torch.float32)
        rewards = torch.tensor(np.array(rewards), dtype=torch.float32)
        terminations = torch.tensor(np.array(terminations), dtype=torch.float32)
        # truncations = torch.tensor(np.array(truncations), dtype=torch.float32)
        action_mask = torch.tensor(np.array(action_mask), dtype=torch.bool)

        return TensorDataset(states, actions, next_states, rewards, terminations)
        # return TensorDataset(states, actions, next_states, rewards, terminations, truncations)

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

    def add(self, state: Sequence[Any], action: Sequence[Union[int, float]], next_state: Sequence[Any],
            reward: Sequence[float], terminated: Sequence[bool], truncated: Optional[Sequence[bool]] = None, action_mask: Optional = None):
        """Add a new transition to memory."""
        for i in range(len(state)):
            if truncated is None:
                e = self.transition(state[i], action[i], next_state[i], reward[i], terminated[i], None, np.atleast_2d(action_mask[i]))
            else:
                e = self.transition(state[i], action[i], next_state[i], reward[i], terminated[i], truncated[i], np.atleast_2d(action_mask[i]))
            self.memory.append(e)

    def sample(self, batch_size: Optional[int] = None) -> Tuple[
               torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Randomly sample a batch of transitions from memory."""
        if batch_size is None:
            batch_size = self.batch_size
        experiences = random.sample(self.memory, k=batch_size)

        states = to_tensor(np.stack([e.state for e in experiences if e is not None]), device=self.device).float()
        actions = to_tensor(np.vstack([e.action for e in experiences if e is not None]),
                            device=self.device).float()
        next_states = to_tensor(np.stack([e.next_state for e in experiences if e is not None]),
                                device=self.device).float()
        rewards = to_tensor(np.vstack([e.reward for e in experiences if e is not None]),
                            device=self.device).float()
        terminateds = to_tensor(np.vstack([e.terminated for e in experiences if e is not None]).astype(np.uint8),
                                device=self.device).float()
        # truncateds = to_tensor(np.vstack([e.truncated for e in experiences if e is not None]).astype(np.uint8),
        #                         device=self.device).float()
        action_masks = to_tensor(np.vstack([e.action_mask for e in experiences if e is not None]).astype(bool))

        return states, actions, next_states, rewards, terminateds, action_masks

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)

    def __getitem__(self, item) -> Transition_new:
        return self.memory[item]
