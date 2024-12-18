from typing import Generic, TypeVar

import numpy as np

import torch 
from torch import nn

from src.config import Config


T = TypeVar('T', bound=Config)


class BaseModule(nn.Module, Generic[T]):
    def __init__(self, config: Config, in_dim: int|np.ndarray):
        super().__init__()
        self.config: T = config
        self.activation = config.get_activation()
        self.in_dim = in_dim

    def forward(self, data_in) -> torch.Tensor:
        """Forward method. Overwrite in derived class."""
        raise NotImplementedError
