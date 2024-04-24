import torch
import torch.nn as nn
import pytorch_lightning as L

from utils.LossFunctions import *
from utils.misc import *

def AE_config_to_hparams(config: dict) -> dict:
    """
    This extracts all model relevant parameters from the config 
    dict (which also contains runtime related information).
    """
    hparams = {}
    hparams['batch_size'] = config['batch_size']
    hparams['lr'] = config['learning_rate']
    hparams['dropout_in'] = config['dropout_in']
    hparams['dropout'] = config['dropout']
    hparams['activation'] = config['activation']
    hparams['in_dim'] = config['in_dim']
    hparams['latent_dim'] = config['latent_dim']
    hparams['n_layers'] = config['n_layers']
    hparams['with_batchnorm'] = config['with_batchnorm']
    hparams['optimizer'] = config['optimizer']
    hparams['loss'] = config['loss']
    hparams['weight_decay'] = config['weight_decay']
    return hparams


# ================ Implementation =================
class AutoEncoder_01(L.LightningModule):
    def __init__(self, config):
        super(AutoEncoder_01, self).__init__()
        
        hparams = AE_config_to_hparams(config)
        for key in hparams.keys():
            self.hparams[key]=hparams[key]

        self.dropout_in = nn.Dropout(self.hparams['dropout_in']) if self.hparams['dropout_in'] > 0 else nn.Identity()
        self.dropout = nn.Dropout(self.hparams['dropout']) if self.hparams['dropout'] > 0  else nn.Identity()
        self.activation = activation_str_to_layer(self.hparams['activation'])
        self.loss = loss_str_to_layer(self.hparams['loss'])
        self.lr = self.hparams['lr']

        self.encoder = []
        self.decoder = []

        # tracks current out dim while building network layer-wise
        self.out_dim = self.hparams["in_dim"]

        def linear_block(Nout, last_layer=False):
            res = [
                self.dropout_in if i == 0 else self.dropout,
                nn.Linear(self.out_dim, Nout),
                nn.BatchNorm1d(Nout) if (not last_layer) and self.hparams["with_batchnorm"] else nn.Identity(),
                self.activation if (not last_layer) else nn.Identity() 
            ]
            self.out_dim = Nout
            return res
            
        ae_step_size =  (self.hparams["in_dim"] - self.hparams['latent_dim']) // self.hparams['n_layers']
        bl_encoder = []
        for i in range(self.hparams['n_layers']):
            if i == self.hparams['n_layers'] - 1:
                bl_encoder.extend(linear_block(self.hparams['latent_dim'], last_layer = True))
            else:
                bl_encoder.extend(linear_block(self.out_dim - ae_step_size))

        bl_decoder = []
        for i in range(self.hparams['n_layers']):
            if i == self.hparams['n_layers'] - 1:
                bl_decoder.extend(linear_block(self.hparams["in_dim"], last_layer = True))
            else:
                bl_decoder.extend(linear_block(self.out_dim + ae_step_size))
        self.encoder = nn.Sequential(*bl_encoder) #nn.ModuleList()
        self.decoder = nn.Sequential(*bl_decoder) #nn.ModuleList()

        for layer in self.encoder:
            if isinstance(layer, nn.Linear):
                nn.init.kaiming_normal_(layer.weight, mode="fan_in", nonlinearity="relu")
                nn.init.zeros_(layer.bias)
        for layer in self.decoder:
            if isinstance(layer, nn.Linear):
                nn.init.kaiming_normal_(layer.weight, mode="fan_in", nonlinearity="relu")
                nn.init.zeros_(layer.bias)

        self.save_hyperparameters(self.hparams)

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

    def training_step(self, batch: torch.Tensor, batch_idx: int) -> torch.Tensor:
        x, y = batch
        pred = self.forward(x)
        loss = self.loss(pred, x)
        self.log("train/loss", loss, prog_bar=False)
        return loss

    def validation_step(self, batch: torch.Tensor, batch_idx: int) -> torch.Tensor:
        x, y = batch
        pred = self.forward(x)
        loss = self.loss(pred, x)
        self.log("val/loss", loss, prog_bar=True)
        return loss


    def configure_optimizers(self):
        if self.hparams["optimizer"] == "SGD":
            optimizer = torch.optim.SGD(self.parameters(), lr=self.lr,
                                    momentum=self.hparams["SGD_momentum"],
                                    weight_decay=self.hparams["SGD_weight_decay"],
                                    dampening=self.hparams["SGD_dampening"],
                                    nesterov=self.hparams["SGD_nesterov"])
        elif self.hparams["optimizer"] == "AdamW":
            optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr)
        elif self.hparams["optimizer"] == "Adam":
            optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        else:
            raise ValueError("unkown optimzer: " + self.hparams["optimzer"])
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.1, patience=8, threshold=1e-3,
                                                               threshold_mode='rel', verbose=True)
        return {"optimizer": optimizer, 
                "lr_scheduler": scheduler, 
                "monitor": "val/loss"}
    
    def load_model_state(self, PATH):
        checkpoint = torch.load(PATH, map_location='cuda:0')
        self.load_state_dict(checkpoint['state_dict'])
    
    
    
class auto_encoder_conv(torch.nn.Module):
    def __init__(self, config):
        super(auto_encoder_conv, self).__init__()
        self.config = config
        self.embedding = nn.Sequential(
            nn.Linear(100, config["embedding_dim"])
        )

        self.encoding = nn.Sequential(
            nn.Conv1d(in_channels=2, out_channels=8, kernel_size=32),
            nn.AvgPool1d(kernel_size=(16), stride=1),
            nn.Conv1d(in_channels=8, out_channels=32, kernel_size=16),
            nn.AvgPool1d(kernel_size=(8), stride=1),
            nn.Conv1d(in_channels=32, out_channels=2, kernel_size=8),
            nn.AvgPool1d(kernel_size=(7), stride=2),
        )

        self.decoding = nn.Sequential(
            nn.Linear(24, config["hidden2_dim"]),
            nn.Linear(config["hidden2_dim"], config["hidden1_dim"]),
            nn.Linear(config["hidden1_dim"], 200)
        )

    def forward(self, data_in):
        x = self.embedding(data_in)
        x = torch.reshape(x, [self.config["batch_size"],2,-1])
        x = self.encoding(x)
        x = self.decoding(x)
        return x[:,0,:]
    
    