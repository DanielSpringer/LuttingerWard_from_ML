import numpy as np

import torch 
from torch import nn

from .. import config


class BaseModule(nn.Module):
    def __init__(self, config: config.Config, in_dim: int|np.ndarray):
        super(BaseModule, self).__init__()
        self.config = config
        self.activation = config.get_activation()
        self.in_dim = in_dim

    def forward(self, data_in):
        """Forward method. Overwrite in derived class."""
        raise NotImplementedError


class AutoEncoderVertex(BaseModule):
    def __init__(self, config: config.VertexConfig, in_dim: int):
        super(AutoEncoderVertex, self).__init__(config, in_dim)
        if config.positional_encoding:
            self.in_dim += 3
        
        self.embedding = nn.Sequential(
            nn.Linear(self.in_dim, config.hidden_dims[0])
        )

        self.encode = nn.Sequential(
            self.activation,
            nn.Linear(config.hidden_dims[0], config.hidden_dims[1]),
            self.activation,
            nn.Linear(config.hidden_dims[1], config.hidden_dims[2]),
            self.activation,
            nn.Linear(config.hidden_dims[2], config.hidden_dims[3])
        )

        self.decode = nn.Sequential(
            self.activation,
            nn.Linear(config.hidden_dims[3], config.hidden_dims[2]),
            self.activation,
            nn.Linear(config.hidden_dims[2], config.hidden_dims[1]),
            self.activation,
            nn.Linear(config.hidden_dims[1], config.out_dim)
        )

    def forward(self, data_in):
        # torch.Size([20, 3]) torch.Size([20, 1728])
        if self.config.positional_encoding:
            y = data_in[0] / (self.in_dim / 3)
            x = self.embedding(torch.cat([y, data_in[1]], axis=1))
            x = self.encode(x)
            x = self.decode(x)
        else:
            x = self.embedding(data_in)
            x = self.encode(x)
            x = self.decode(x)
        return x
