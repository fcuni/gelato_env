from collections import namedtuple
from dataclasses import dataclass
from typing import Union, Optional, Tuple, Literal
import torch
import numpy as np


# EnvType = Union[GelateriaEnv, DefaultGelatoEnvWrapper]
TensorType = Union[torch.Tensor, np.ndarray]

Transition = namedtuple("Transition",
                        field_names=["state", "action", "next_state", "reward", "terminated", "truncated"])

Transition_With_Action_Mask = namedtuple("Transition_With_Action_Mask",
                                         field_names=["state", "action", "next_state", "reward", "terminated", "truncated", "action_mask"])

Activations = Literal[tuple(Literal[act] for act in torch.nn.modules.activation.__all__)]


@dataclass
class TransitionBatch:
    """Represents a batch of transitions"""

    obs: Optional[TensorType]
    act: Optional[TensorType]
    next_obs: Optional[TensorType]
    rewards: Optional[TensorType]
    terminateds: Optional[TensorType]
    truncateds: Optional[TensorType] = None

    def __len__(self):
        return self.obs.shape[0]

    def astuple(self) -> Transition:
        return (
            self.obs,
            self.act,
            self.next_obs,
            self.rewards,
            self.terminateds,
            self.truncateds,
        )

    def __getitem__(self, item):
        if self.truncateds is None:
            return TransitionBatch(
                self.obs[item],
                self.act[item],
                self.next_obs[item],
                self.rewards[item],
                self.terminateds[item]
            )
        return TransitionBatch(
            self.obs[item],
            self.act[item],
            self.next_obs[item],
            self.rewards[item],
            self.terminateds[item],
            self.truncateds[item],
        )

    @staticmethod
    def _get_new_shape(old_shape: Tuple[int, ...], batch_size: int):
        new_shape = list((1,) + old_shape)
        new_shape[0] = batch_size
        new_shape[1] = old_shape[0] // batch_size
        return tuple(new_shape)

    def add_new_batch_dim(self, batch_size: int):
        if not len(self) % batch_size == 0:
            raise ValueError(
                "Current batch of transitions size is not a "
                "multiple of the new batch size. "
            )
        return TransitionBatch(
            self.obs.reshape(self._get_new_shape(self.obs.shape, batch_size)),
            self.act.reshape(self._get_new_shape(self.act.shape, batch_size)),
            self.next_obs.reshape(self._get_new_shape(self.obs.shape, batch_size)),
            self.rewards.reshape(self._get_new_shape(self.rewards.shape, batch_size)),
            self.terminateds.reshape(
                self._get_new_shape(self.terminateds.shape, batch_size)
            ),
            self.truncateds.reshape(
                self._get_new_shape(self.truncateds.shape, batch_size)
            ),
        )


ModelInput = Union[torch.Tensor, TransitionBatch]

