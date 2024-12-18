import torch 
from torch import nn

from src.config.vertex import *
from . import BaseModule


class AutoEncoderVertex(BaseModule[VertexConfig]):
    def __init__(self, config: VertexConfig, in_dim: int):
        super().__init__(config, in_dim)
        self.matrix_dim = config.matrix_dim
        if config.positional_encoding:
            self.in_dim += self.matrix_dim
        
        self.embedding = nn.Sequential(
            nn.Linear(self.in_dim, config.hidden_dims[0])
        )

        encoder_layers = [m for i in range(len(config.hidden_dims) - 1) for m in 
                          (self.activation, nn.Linear(config.hidden_dims[i], config.hidden_dims[i + 1]))]
        self.encoder = nn.Sequential(*encoder_layers)

        decoder_dims = config.hidden_dims[1:][::-1] + [config.out_dim]
        decoder_layers = [m for i in range(len(decoder_dims) - 1) for m in 
                          (self.activation, nn.Linear(decoder_dims[i], decoder_dims[i + 1]))]
        self.decoder = nn.Sequential(*decoder_layers)

    def forward(self, data_in) -> torch.Tensor:
        x = self.encode(data_in)
        x = self.decode(x)
        return x
    
    def encode(self, data_in) -> torch.Tensor:
        if self.config.positional_encoding:
            idcs = data_in[0] // (self.in_dim / self.matrix_dim)
            x = self.embedding(torch.cat([idcs, data_in[1]], axis=1))
        else:
            x = self.embedding(data_in)
        x = self.encoder(x)
        return x

    def decode(self, data_in) -> torch.Tensor:
        x = self.decoder(data_in)
        return x
