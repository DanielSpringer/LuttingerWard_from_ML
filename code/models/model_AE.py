import torch
import torch.nn as nn
import pytorch_lightning as L
import sys
sys.path.append('G:\\Codes\\LuttingerWard_from_ML\\code\\utils')
from LossFunctions import *
from utils import *

# ================ Implementation =================
class AutoEncoder_01(L.LightningModule):
    def __init__(self, config):
        super().__init__()
        
        self.hparams = AE_config_to_hparams(config)
        self.dropout_in = nn.Dropout(self.hparams['dropout_in']) if self.hparams['dropout_in'] > 0 else nn.Identity()
        self.dropout = nn.Dropout(self.hparams['dropout']) if self.hparams['dropout'] > 0  else nn.Identity()
        self.activation = activation_str_to_layer(self.hparams['activation'])
        self.loss = loss_str_to_layer(self.hparams['loss'])

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
    
    