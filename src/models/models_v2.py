from typing import Generic, TypeVar

import numpy as np

import torch 
from torch import nn

from .. import config


T = TypeVar('T', bound=config.Config)


class BaseModule(nn.Module, Generic[T]):
    def __init__(self, config: config.Config, in_dim: int|np.ndarray):
        super(BaseModule, self).__init__()
        self.config: T = config
        self.activation = config.get_activation()
        self.in_dim = in_dim

    def forward(self, data_in) -> torch.Tensor:
        """Forward method. Overwrite in derived class."""
        raise NotImplementedError


class AutoEncoderVertex(BaseModule[config.VertexConfig]):
    def __init__(self, config: config.VertexConfig, in_dim: int):
        super(AutoEncoderVertex, self).__init__(config, in_dim)
        self.latent_space: torch.Tensor
        if config.positional_encoding:
            self.in_dim += 3
        
        self.embedding = nn.Sequential(
            nn.Linear(self.in_dim, config.hidden_dims[0])
        )

        encoder_layers = [m for i in range(len(config.hidden_dims) - 1) for m in 
                          (self.activation, nn.Linear(config.hidden_dims[i], config.hidden_dims[i + 1]))]
        self.encoder = nn.Sequential(*encoder_layers)

        decoder_dims = reversed(config.hidden_dims[1:]) + [config.out_dim]
        decoder_layers = [m for i in range(len(decoder_dims) - 1) for m in 
                          (self.activation, nn.Linear(decoder_dims[i], decoder_dims[i + 1]))]
        self.decoder = nn.Sequential(*decoder_layers)

    def forward(self, data_in) -> torch.Tensor:
        x = self.encode(data_in)
        x = self.decode(x)
        return x
    
    def encode(self, data_in) -> torch.Tensor:
        if self.config.positional_encoding:
            y = data_in[0] / (self.in_dim / 3)
            x = self.embedding(torch.cat([y, data_in[1]], axis=1))
        else:
            x = self.embedding(data_in)
        x = self.encoder(x)
        return x

    def decode(self, data_in) -> torch.Tensor:
        x = self.decoder(data_in)
        return x