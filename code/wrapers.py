import torch 
from torch import nn
from torch_geometric.nn import MessagePassing, global_mean_pool
import copy
# import lightning as L
import pytorch_lightning as pl

class model_wraper_ae(pl.LightningModule):
    def __init__(self, config):
        super().__init__()
        module = __import__("models")
        self.model = getattr(module, config["MODEL_NAME"])(config)
        self.criterion_mse = nn.MSELoss()
        self.config = config
        self.val_pred = []
        self.val_loss = []

    def forward(self, batch: torch.Tensor):
        return self.model(batch)

    def training_step(self, batch: torch.Tensor, batch_idx: int) -> torch.Tensor:
        pred = self.forward(batch[0])
        target = batch[1]
        loss = self.criterion_mse(pred, target)
        self.log('train_loss', loss.item())
        return loss

    def validation_step(self, batch: torch.Tensor, batch_idx: int) -> torch.Tensor:
        pred = self.forward(batch[0])
        target = batch[1]
        loss = self.criterion_mse(pred, target)
        self.val_pred.append([target, pred])
        self.val_loss.append(loss)
        self.log('val_loss', loss.item(), prog_bar=True)
        return loss

    def configure_optimizers(self) -> dict:
        optimizer = torch.optim.AdamW(params=self.model.parameters(), lr=self.config['learning_rate'], weight_decay=self.config['weight_decay'])
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.2, patience=20, min_lr=1e-5)
        return {"optimizer": optimizer, "lr_scheduler": scheduler, "monitor": "val_loss"}
    
    def load_model_state(self, PATH):
        checkpoint = torch.load(PATH, map_location='cuda:0')
        self.model.load_state_dict(checkpoint['state_dict'])



class model_wraper_gnn(pl.LightningModule):

    def __init__(self, config):
        super().__init__()
        module = __import__("models")
        self.model = getattr(module, config["MODEL_NAME"])(config)
        self.criterion_mse = nn.MSELoss()
        self.config = config
        self.val_pred = []
        self.val_loss = []

    def forward(self, batch):
        return self.model(batch)

    def training_step(self, batch, batch_idx):
#         pred = self.forward(batch[0])
        pred = self.forward(batch)
        target = batch["target"][0]
        loss = self.criterion_mse(pred, target)
        # loss = self.criterion_mse(pred, target) + 2 * self.criterion_mse(pred[0:10], target[0:10])
        # loss = self.criterion_mse(pred, target) + 30 * self.criterion_mse(pred[0:5], target[0:5])   # VERSION 2
        self.log('train_loss', loss.item())
        return loss

    def validation_step(self, batch, batch_idx):
#         pred = self.forward(batch[0])
        pred = self.forward(batch)
        target = batch["target"][0]
        loss = self.criterion_mse(pred, target)
        # loss = self.criterion_mse(pred, target) + 2 * self.criterion_mse(pred[0:10], target[0:10])
        # loss = self.criterion_mse(pred, target) + 30 * self.criterion_mse(pred[0:5], target[0:5])   # VERSION 2
        self.val_pred.append([target, pred])
        self.val_loss.append(loss)
        self.log('val_loss', loss.item())
        return loss

    # def on_validation_epoch_end(self):
    #     pass

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(params=self.model.parameters(), lr=self.config['learning_rate'], weight_decay=self.config['weight_decay'])
        return optimizer
    
    def load_model_state(self, PATH):
        checkpoint = torch.load(PATH, map_location='cuda:0')
        self.model.load_state_dict(checkpoint['state_dict'])