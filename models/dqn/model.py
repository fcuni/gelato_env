from typing import Optional, Sequence, Tuple
import numpy as np
import torch
import torch.nn as nn
from models.model import BaseNetwork
from utils.distributions import CategoricalMasked


class QNetwork(BaseNetwork):
    def __init__(self, num_observation: int, num_action: int, hidden_layers: Optional[Sequence[int]] = (128, 128),
                 activation: Optional[str] = "ReLU"):
        """Initialize parameters and build Q network.

        Args:
            num_observation (int): Dimension of each state observation
            num_action (int): Dimension of each action
            hidden_layers (list): Number of nodes in hidden layers of the network (default: (32, 64))
            activation (str): Activation function to use between hidden layers (default: "ReLU")
        """
        super(QNetwork, self).__init__()

        in_dim = num_observation
        layers = []
        if hidden_layers is not None:
            for dim in hidden_layers:
                layers += [nn.Linear(in_features=in_dim, out_features=dim)]
                in_dim = dim
                if activation is not None:
                    layers += [getattr(nn, activation)()]
        layers += [nn.Linear(in_features=in_dim, out_features=num_action)]
        self._model = nn.Sequential(*layers)

    def forward(self, state):
        """Given states, return the Q values of each state-action pair."""
        return self._model(state)

    def get_action(self, state_obs: torch.Tensor, mask: Optional[np.ndarray] = None) -> np.ndarray:
        with torch.no_grad():

            logits = self.forward(state_obs)

            if mask is None:
                mask = np.ones(logits.shape, dtype=bool)
            if isinstance(mask, np.ndarray):
                mask_tensor = torch.from_numpy(mask).bool()
            else:
                mask_tensor = mask.bool()

            dist = CategoricalMasked(logits=logits, mask=mask_tensor)

            action = torch.argmax(dist.probs, dim=1).detach().cpu().int().numpy()

        return action