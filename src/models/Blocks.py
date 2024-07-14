import torch
import torch.nn as nn

class LinEncoder(nn.Module):
    def __init__(self, input_dim, latent_dim, n_layers, activation):
        super(LinEncoder, self).__init__()

        ae_step_size =  (input_dim - latent_dim) // n_layers
        bl_encoder = []
        self.dropout_in = nn.Dropout(self.hparams['dropout_in']) if self.hparams['dropout_in'] > 0 else nn.Identity()
        self.dropout = nn.Dropout(self.hparams['dropout']) if self.hparams['dropout'] > 0  else nn.Identity()
        self.activation = activation_str_to_layer(self.hparams['activation'])
        self.loss = loss_str_to_layer(self.hparams['loss'])

        self.encoder = []
        self.decoder = []

        # tracks current out dim while building network layer-wise
        self.out_dim = self.hparams["in_dim"]


    def forward(self, x):
        return self.encoder(x.view(-1, 784))

class LinDecoder(nn.Module):
    def __init__(self, input_dim=784, hidden_dim=400, latent_dim=200):
        super(LinDecoder, self).__init__()
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            activation,
            nn.Linear(hidden_dim, input_dim),
            )

    def forward(self, x):
        return self.decoder(x)