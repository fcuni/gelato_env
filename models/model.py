import os
from pathlib import Path

import torch
from torch import nn


class BaseNetwork(nn.Module):
    def save(self, path: Path):
        os.makedirs(path, exist_ok=True)
        torch.save(self.state_dict(), path)

    def load(self, path: Path):
        if os.path.exists(path):
            self.load_state_dict(torch.load(path))
        else:
            raise FileNotFoundError(f"File {path} not found")