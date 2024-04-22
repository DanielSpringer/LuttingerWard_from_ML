import torch
import torch.nn as nn
import pytorch_lightning as L
import numpy as np
from utils.LossFunctions import *
from utils.misc import *
import matplotlib.pyplot as plt

#TODO: abstract encoding/docing/G_to_Sigma parts away. Define this model to consist of 3 models.

def FC_config_to_hparams(config: dict) -> dict:
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
    hparams['AE_layers'] = config['AE_layers']
    hparams['latent_layers'] = config['FC_layers']
    hparams['with_batchnorm'] = config['with_batchnorm']
    hparams['optimizer'] = config['optimizer']
    hparams['loss'] = config['loss']
    hparams['weight_decay'] = config['weight_decay']
    hparams['out_dim'] = config['out_dim']
    return hparams


def linear_block(in_dim, out_dim, activation, 
                 dropout_in, dropout, with_batchnorm, 
                 first_layer = False, last_layer=False):
    res = [
        dropout_in if first_layer else dropout,
        nn.Linear(in_dim, out_dim),
        nn.BatchNorm1d(out_dim) if (not last_layer) and with_batchnorm else nn.Identity(),
        activation if (not last_layer) else nn.Identity() 
    ]
    return res

#TODO: this could be a single "coder" class. there are only two sign changes
class LinEncoder(nn.Module):
    def __init__(self, AE_layers, in_dim, out_dim, activation, dropout_in, dropout, with_batchnorm):
        super(LinEncoder, self).__init__()

        current_dim = in_dim
        ae_step_size =  (in_dim - out_dim) // AE_layers
        self.activation = activation
        self.dropout_in = nn.Dropout(dropout_in) if dropout_in > 0 else nn.Identity()
        self.dropout = nn.Dropout(dropout) if dropout > 0  else nn.Identity()
        self.with_batchnorm = with_batchnorm
        layers = []
        for i in range(AE_layers):
            next_dim = current_dim - ae_step_size if (i == AE_layers - 1) else current_dim - ae_step_size
            layers.extend(linear_block(current_dim, next_dim, activation, 
                                        self.dropout_in, self.dropout, with_batchnorm,
                                        last_layer = (i == AE_layers - 1), first_layer = (i == 0)
                                        ))
            print("enc: " + str(current_dim) + " -> " + str(next_dim))
            current_dim = next_dim

        self.encoder = nn.Sequential(*layers)
        for layer in self.encoder:
            if isinstance(layer, nn.Linear):
                nn.init.kaiming_normal_(layer.weight, mode="fan_in", nonlinearity="relu")
                nn.init.zeros_(layer.bias)  

    def forward(self, x):
        return self.encoder(x)
    
class LinDecoder(nn.Module):
    def __init__(self, AE_layers, in_dim, out_dim, activation, dropout, with_batchnorm):
        super(LinDecoder, self).__init__()

        current_dim = in_dim
        ae_step_size =  (out_dim - in_dim) // AE_layers
        self.activation = activation
        self.dropout = nn.Dropout(dropout) if dropout > 0  else nn.Identity()
        self.with_batchnorm = with_batchnorm
        layers = []
        for i in range(AE_layers):
            next_dim = out_dim if (i == AE_layers - 1) else current_dim + ae_step_size
            layers.extend(linear_block(current_dim, next_dim,  
                                        activation, nn.Identity(), self.dropout, with_batchnorm,
                                        last_layer = (i == AE_layers - 1), first_layer = (i == 0)
                                        ))
            print("dec: " + str(current_dim) + " -> " + str(next_dim))
            current_dim = next_dim
            

        self.decoder = nn.Sequential(*layers)
        for layer in self.decoder:
            if isinstance(layer, nn.Linear):
                nn.init.kaiming_normal_(layer.weight, mode="fan_in", nonlinearity="relu")
                nn.init.zeros_(layer.bias)  

    def forward(self, x):
        return self.decoder(x)



class AE_FC_01(L.LightningModule):
    """Simple fully connected model with pytorch lightning."""
    def __init__(self, config: dict) -> None:
        super().__init__()

        hparams = FC_config_to_hparams(config)
        for key in hparams.keys():
            self.hparams[key]=hparams[key]

        self.dropout_in = nn.Dropout(self.hparams['dropout_in']) if self.hparams['dropout_in'] > 0 else nn.Identity()
        self.dropout = nn.Dropout(self.hparams['dropout']) if self.hparams['dropout'] > 0  else nn.Identity()
        self.activation = activation_str_to_layer(self.hparams['activation'])
        self.reconstr_loss_f = loss_str_to_layer(self.hparams['loss'])
        self.lr = self.hparams['lr']

        self.plot_worst_examples = False
        self.worst_losses_ids = np.zeros(2,dtype=int)    # track the worst 3 example indices
        self.worst_losses_data  = [None, None]           # track the worst 3 example losses
        print("G:")
        self.G_encoder  = LinEncoder(hparams['AE_layers'], hparams['in_dim'], hparams['latent_dim'], 
                                     self.activation, hparams['dropout_in'], hparams['dropout'], 
                                     hparams['with_batchnorm'])
        #self.SE_encoder = LinEncoder(hparams['AE_layers'], hparams['in_dim'], hparams['latent_dim'], 
        #                             self.activation, hparams['dropout_in'], hparams['dropout'], 
        #                             hparams['with_batchnorm'])
        print("G:")
        self.G_decoder  = LinDecoder(hparams['AE_layers'], hparams['latent_dim'], hparams['out_dim'], 
                                     self.activation, hparams['dropout'], 
                                     hparams['with_batchnorm'])
        print("SE:")
        self.SE_decoder = LinDecoder(hparams['AE_layers'], hparams['latent_dim'], hparams['out_dim'], 
                                     self.activation, hparams['dropout'], 
                                     hparams['with_batchnorm'])


        
        bl_fc_net = []
        for i in range(self.hparams['latent_layers']):
            bl_fc_net.extend(linear_block(self.hparams['latent_dim'], self.hparams['latent_dim'],  
                                        self.activation, nn.Identity(), self.dropout, self.hparams['with_batchnorm'],
                                        last_layer = (i == self.hparams['latent_layers'] - 1), first_layer = False
                                        ))
        self.fc_net     = nn.Sequential(*bl_fc_net) if self.hparams['latent_layers'] > 0 else nn.Sequential(nn.Identity())

        for layer in self.fc_net:
            if isinstance(layer, nn.Linear):
                nn.init.kaiming_normal_(layer.weight, mode="fan_in", nonlinearity="relu")
                nn.init.zeros_(layer.bias)
        self.save_hyperparameters(self.hparams)
    
    def G_to_SE(self, G):
        SE = self.fc_net(G)
        return SE
    
    def forward(self, G):
        G_latent = self.G_encoder(G)
        SE_latent = self.G_to_SE(G_latent)
        SE = self.SE_decoder(SE_latent)
        return G_latent, SE_latent, SE



    def training_step(self, train_batch, batch_idx):
        G_in, SE_in = train_batch
        #ndens, G_in = torch.split(G_in_n, [1, G_in_n.size(1)], dim=0)
        G_latent, SE_latent, SE_hat = self(G_in)
        G_hat = self.G_decoder(G_latent)
        #TODO: think about thrid loss: train SE encoder/decoder on data as well
        G_reconstr = self.reconstr_loss_f(G_in, G_hat)
        SE_reconstr = self.reconstr_loss_f(SE_in, SE_hat)

        loss = G_reconstr/torch.clamp(-10*torch.log(SE_reconstr), min=1) + SE_reconstr
        self.log("train/loss", loss, prog_bar=False)
        self.log("train/G_reconstr", G_reconstr, prog_bar=False)
        self.log("train/SE_reconstr", SE_reconstr, prog_bar=False)
        return loss


    def validation_step(self, batch: torch.Tensor, batch_idx: int) -> torch.Tensor:
        G_in, SE_in = batch
        G_latent, SE_latent, SE_hat = self(G_in)
        loss = self.reconstr_loss_f(SE_in, SE_hat)
        #if self.plot_worst_examples:
        #    wi = np.argmin(self.worst_losses)
        #    if loss > wi:
        #        self.worst_losses_data[wi] = (G_in, SE_hat, SE_in)
        #        self.worst_losses[wi] = loss
        #else: # just save the first two
        #    if batch_idx == 0:
        #        self.worst_losses_data[0] = (G_in[0:2,:], SE_hat[0:2,:], SE_in[0:2,:])
        #        self.worst_losses[0] = loss
        #        self.worst_losses_data[1] = (G_in[2:4,:], SE_hat[2:4,:], SE_in[2:4,:])
        #        self.worst_losses[1] = loss

        self.log("val/loss", loss, prog_bar=True)
        return loss
    
    def on_validation_epoch_start(self):
        # reset worst examples
        self.worst_losses_data = [None, None]
        self.worst_losses    = np.zeros(2)

    def on_validation_epoch_end(self):
        # plot worst 2 
        if False:
            for ii,batch_i in enumerate(self.worst_losses_data):
                if batch_i is not None:
                    G_in, S_hat, S_in = batch_i
                    batch_len = S_in.size(0)
                    fig, axs = plt.subplots(batch_len,3, figsize=(24,12))
                    for i in range(batch_len):
                        axs[i,0].plot(G_in[i,:].cpu(), linewidth=2)
                        axs[i,1].plot(S_in[i,:].cpu(), label="ground truth", linewidth=2)
                        axs[i,1].plot(S_hat[i,:].cpu(), label="prediction", linewidth=2)
                        axs[i,0].set_title(f"Batch {ii}")
                        axs[i,1].legend()
                        axs[i,2].plot(np.abs(S_hat[i,:].cpu() - S_in[i,:].cpu()), label="Log Diff")
                        axs[i,0].set_xlabel("nu")
                        axs[i,0].set_ylabel("G_in")
                        axs[i,1].set_xlabel("nu")
                        axs[i,1].set_ylabel("Sigma_in")
                        axs[i,2].set_xlabel("nu")
                        axs[i,2].set_ylabel("Delta Sigma")
                        axs[i,2].set_yscale('log')
                    self.logger.experiment[f"val/worst_examples_{ii}"].append(fig)

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
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.2, patience=15, verbose=True)
        return {"optimizer": optimizer, 
                "lr_scheduler": scheduler, 
                "monitor": "val/loss"}
    
    def load_model_state(self, PATH):
        checkpoint = torch.load(PATH, map_location='cuda:0')
        self.load_state_dict(checkpoint['state_dict'])