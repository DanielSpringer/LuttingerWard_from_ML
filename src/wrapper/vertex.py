import torch

from src.config.vertex import *
from src.models.vertex import *
from . import BaseWrapper


class VertexWrapper(BaseWrapper[AutoEncoderVertex, VertexConfig]):
    ''' Wrapper for the vertex compression '''
    def __init__(self, config: VertexConfig, in_dim: int):
        super().__init__(config, in_dim)
        self.positional_encoding = config.positional_encoding
    
    def get_inputs_and_targets(self, batch: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if self.positional_encoding:
            inputs = (batch[0], batch[1])
        else:
            inputs = batch[1]
        return inputs, batch[2].float()
