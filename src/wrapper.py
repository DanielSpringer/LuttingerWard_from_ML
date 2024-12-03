import numpy as np
import lightning as L
import torch

from . import config


class BaseWrapper(L.LightningModule):
    def __init__(self, config: config.Config, in_dim: int|np.ndarray):
        super().__init__()
        self.model = config.model(config, in_dim)
        self.criterion: torch.nn = config.get_criterion()
        self.config = config
    
    def forward(self, batch: torch.Tensor):
        return self.model(batch)
    
    def get_inputs_and_targets(self, batch: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Separate batch into inputs and targets. Overwrite according to Dataloader/Dataset."""
        return batch[0], batch[1]
    
    def step(self, batch: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Model step used for training and validation. Overwrite fro custom model usage."""
        inputs, targets = self.get_inputs_and_targets(batch)
        pred = self.forward(inputs)
        loss = self.criterion(pred, targets)
        return pred, targets, loss
    
    def training_step(self, batch: torch.Tensor) -> torch.Tensor:
        """Training step. Overwrite for custom training behaviour."""
        pred, targets, loss = self.step(batch)
        self.log('train_loss', loss.item())
        return loss
    
    def validation_step(self, batch: torch.Tensor) -> torch.Tensor:
        """Validation step. Overwrite for custom validation behaviour."""
        pred, targets, loss = self.step(batch)
        self.log('val_loss', loss.item(), prog_bar=True)
        return loss
    
    def configure_optimizers(self) -> dict[str, torch.optim.Optimizer]:
        optimizer = self.config.get_optimizer(params=self.model.parameters(), lr=self.config.learning_rate, 
                                              weight_decay=self.config.weight_decay)
        # Optimizer: type[torch.optim.Optimizer] = pydoc.locate(self.config.optimizer)
        # optimizer = Optimizer(params=self.model.parameters(), lr=self.config.learning_rate, 
        #                       weight_decay=self.config.weight_decay, **self.config.optimizer_kwargs)
        # TODO: check parameter stability
        return {"optimizer": optimizer}
    
    def load_model_state(self, path: str):
        checkpoint = torch.load(path, map_location='cuda:0')
        self.model.load_state_dict(checkpoint['state_dict'])


class VertexWrapper(BaseWrapper):
    ''' Wrapper for the vertex compression '''
    def __init__(self, config: config.VertexConfig, in_dim: int):
        super().__init__(config, in_dim)
        self.positional_encoding = config.positional_encoding
    
    def get_inputs_and_targets(self, batch: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if self.positional_encoding:
            inputs = (batch[0], batch[1])
        else:
            inputs = batch[1]
        return inputs, batch[2].float()
