from typing import Generic, TypeVar

import numpy as np
import lightning as L
import torch

from src.config import Config
from src.models import BaseModule


S = TypeVar('S', bound=BaseModule)
T = TypeVar('T', bound=Config)


class BaseWrapper(L.LightningModule, Generic[S, T]):
    def __init__(self, config: Config, in_dim: int|np.ndarray):
        """
        Wrapper class that handles the model training.

        Parameters
        ----------
        config : Config
            A Config instance.
        in_dim : int | np.ndarray
            Input dimensions.
        """
        super().__init__()
        self.model: S = config.model(config, in_dim)
        self.criterion: torch.nn = config.get_criterion()
        self.config: T = config
    
    def __call__(self, *args, **kwds) -> torch.Tensor:
        return super().__call__(*args, **kwds)
    
    def forward(self, batch: torch.Tensor) -> torch.Tensor:
        return self.model(batch)
    
    def get_inputs_and_targets(self, batch: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Separate batch into inputs and targets.\n
        Overwrite according batch-implememtation in the Dataloader/Dataset.
        """
        return batch[0], batch[1]
    
    def step(self, batch: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Model step used for training and validation. Overwrite for custom training process.
        """
        inputs, targets = self.get_inputs_and_targets(batch)
        pred = self.forward(inputs)
        loss = self.criterion(pred, targets)
        return pred, targets, loss
    
    def training_step(self, batch: torch.Tensor) -> torch.Tensor:
        """
        Training step. Overwrite for custom training behaviour.\n
        See `lightning.LightningModule` for more details.
        """
        pred, targets, loss = self.step(batch)
        self.log('train_loss', loss.item())
        return loss
    
    def validation_step(self, batch: torch.Tensor) -> torch.Tensor:
        """
        Validation step. Overwrite for custom validation behaviour.\n
        See `lightning.LightningModule` for more details.
        """
        pred, targets, loss = self.step(batch)
        self.log('val_loss', loss.item(), prog_bar=True)
        return loss
    
    def configure_optimizers(self) -> dict[str, torch.optim.Optimizer]:
        """
        Instantiate and return the optimizer.\n
        See `lightning.LightningModule` for more details.
        """
        optimizer = self.config.get_optimizer(params=self.model.parameters(), lr=self.config.learning_rate, 
                                              weight_decay=self.config.weight_decay)
        return {"optimizer": optimizer}
    
    def load_model_state(self, path: str) -> None:
        """
        Load a model checkpoint from disk to the model-instance.

        Parameters
        ----------
        path : str
            File-path of model checkpoint.
        """
        checkpoint = torch.load(path, map_location='cuda:0')
        self.model.load_state_dict(checkpoint['state_dict'])
