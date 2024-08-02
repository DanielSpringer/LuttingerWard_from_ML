import sys
sys.path.append('/home/daniel/Projects/RevMat/LuttingerWard_from_ML/')

# from code.models.model_AE import *
# from code.models.model_FC import *
# from code.models.GNN import *

import torch 
from torch import nn
from torch_geometric.nn import MessagePassing, global_mean_pool
import copy
# import lightning as L
import pytorch_lightning as pl

### ARE THESE USED?
# class Encoder(torch.nn.Module):
#     """
#     Encodes Input data, for now with hardcoded dimensions and layers.
#     """
#     def __init__(self, config):
#         super().__init__()
#         self.encode = nn.Sequential(
#             self.activation,
#             nn.Linear(config["embedding_dim"], config["hidden1_dim"]),
#             self.activation,
#             nn.Linear(config["hidden1_dim"], config["hidden2_dim"]),
#             self.activation,
#             nn.Linear(config["hidden2_dim"], config["encoder_dim"])
#         )

#     def forward(self, x):
#         return self.encode(x)

# class Decoder(torch.nn.Module):
#     """
#     Decodes Output data, for now with hardcoded dimensions and layers.
#     """
#     def __init__(self, config):
#         super().__init__()
#         self.decode = nn.Sequential(
#             self.activation,
#             nn.Linear(config["encoder_dim"], config["hidden2_dim"]),
#             self.activation,
#             nn.Linear(config["hidden2_dim"], config["hidden1_dim"]),
#             self.activation,
#             nn.Linear(config["hidden1_dim"], config["out_dim"])
#         )
    
#     def forward(self, x):
#         return self.decode(x)
### ARE THESE USED?

class encoder(torch.nn.Module):
    def __init__(self, config):
        super(encoder, self).__init__()
        self.config = config
        self.activation = nn.ReLU() #nn.SiLU()# nn.LeakyReLU()

        self.embedding = nn.Sequential(
            nn.Linear(config["in_dim"], config["in_dim"]),
            nn.Linear(config["in_dim"], config["in_dim"]),
            nn.Linear(config["in_dim"], config["in_dim"]),
            nn.Linear(config["in_dim"], config["embedding_dim"])
        )

        self.encode = nn.Sequential(
            self.activation,
            nn.Linear(config["embedding_dim"], config["hidden1_dim"]),
            torch.nn.BatchNorm1d(config["hidden1_dim"], eps=1e-05, momentum=0.1, affine=True, track_running_stats=True, device=None, dtype=None),
            # nn.Dropout(p=0.5, inplace=False),
            self.activation,
            nn.Linear(config["hidden1_dim"], config["hidden2_dim"]),
            self.activation,
            nn.Linear(config["hidden2_dim"], config["hidden3_dim"]),
            # nn.Dropout(p=0.3, inplace=False),
            self.activation,
            nn.Linear(config["hidden3_dim"], config["hidden4_dim"]),
            torch.nn.BatchNorm1d(config["hidden4_dim"], eps=1e-05, momentum=0.1, affine=True, track_running_stats=True, device=None, dtype=None),
            self.activation,
            nn.Linear(config["hidden4_dim"], config["hidden5_dim"]),
            # nn.Dropout(p=0.1, inplace=False),
            self.activation,
            nn.Linear(config["hidden5_dim"], config["encoder_dim"])
            # self.activation
        )

    def forward(self, data_in):
        x = self.embedding(data_in.float())
        x = self.encode(x)
        return x


class auto_encoder(torch.nn.Module):
    def __init__(self, config):
        super(auto_encoder, self).__init__()
        self.config = config
        self.activation = nn.ReLU() #nn.SiLU()# nn.LeakyReLU()

        self.embedding = nn.Sequential(
            nn.Linear(config["in_dim"], config["embedding_dim"])
        )

        self.encode = nn.Sequential(
            self.activation,
            nn.Linear(config["embedding_dim"], config["hidden1_dim"]),
            self.activation,
            nn.Linear(config["hidden1_dim"], config["hidden2_dim"]),
            self.activation,
            nn.Linear(config["hidden2_dim"], config["encoder_dim"])
        )

        self.decode = nn.Sequential(
            self.activation,
            nn.Linear(config["encoder_dim"], config["hidden2_dim"]),
            self.activation,
            nn.Linear(config["hidden2_dim"], config["hidden1_dim"]),
            self.activation,
            nn.Linear(config["hidden1_dim"], config["out_dim"])
            
        )

    def forward(self, data_in):
        x = self.embedding(data_in)
        x = self.encode(x)
        x = self.decode(x)
        return x
    
    
    
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
    

class auto_encoder_injection_1(torch.nn.Module):
    def __init__(self, config):
        super(auto_encoder_injection_1, self).__init__()
        self.config = config
        self.activation = nn.ReLU() #nn.SiLU()# nn.LeakyReLU()

        self.embedding = nn.Sequential(
            nn.Linear(config["in_dim"] + config["injection_dim"], config["embedding_dim"])
        )

        self.encode1 = nn.Sequential(
            nn.Linear(config["embedding_dim"] + config["injection_dim"], config["hidden1_dim"]),
            self.activation
        )
        self.encode2 = nn.Sequential(
            nn.Linear(config["hidden1_dim"] + config["injection_dim"], config["hidden2_dim"]),
            self.activation
        )
        self.encode3 = nn.Sequential(
            nn.Linear(config["hidden2_dim"] + config["injection_dim"], config["encoder_dim"]),
            self.activation
        )

        self.decode = nn.Sequential(
            nn.Linear(config["encoder_dim"], config["hidden2_dim"]),
            self.activation,
            nn.Linear(config["hidden2_dim"], config["hidden1_dim"]),
            self.activation,
            nn.Linear(config["hidden1_dim"], config["out_dim"])
        )

    def forward(self, data_in, g0_inject):
        x = torch.cat([data_in,g0_inject], dim=1)
        x = self.embedding(x)
        x = torch.cat([x,g0_inject], dim=1)
        x = self.encode1(x)
        x = torch.cat([x,g0_inject], dim=1)
        x = self.encode2(x)
        x = torch.cat([x,g0_inject], dim=1)
        x = self.encode3(x)
        x = self.decode(x)
        return x    


class auto_encoder_injection_2(torch.nn.Module):
    def __init__(self, config):
        super(auto_encoder_injection_2, self).__init__()
        self.config = config
        self.activation = nn.ReLU() #nn.SiLU()# nn.LeakyReLU()

        self.embedding = nn.Sequential(
            nn.Linear(config["in_dim"] + config["injection_dim"], config["embedding_dim"])
        )

        self.encode1 = nn.Sequential(
            self.activation,
            nn.Linear(config["embedding_dim"] + config["injection_dim"], config["hidden1_dim"])
        )
        self.encode2 = nn.Sequential(
            self.activation,
            nn.Linear(config["hidden1_dim"] + config["injection_dim"], config["hidden2_dim"])
        )
        self.encode3 = nn.Sequential(
            self.activation,
            nn.Linear(config["hidden2_dim"] + config["injection_dim"], config["encoder_dim"])
        )

        self.decode = nn.Sequential(
            self.activation,
            nn.Linear(config["encoder_dim"] + config["injection_dim"], config["hidden2_dim"]),
            self.activation,
            nn.Linear(config["hidden2_dim"] + config["injection_dim"], config["hidden1_dim"]),
            self.activation,
            nn.Linear(config["hidden1_dim"], config["out_dim"])
        )

    def forward(self, data_in, g0_inject):
        x = self.embedding(data_in)
        x = torch.cat([x,g0_inject], dim=1)
        x = self.encode1(x)
        x = torch.cat([x,g0_inject], dim=1)
        x = self.encode2(x)
        x = torch.cat([x,g0_inject], dim=1)
        x = self.encode3(x)
        x = self.decode(x)
        return x    


################################ GRAPH NETWORKS
class Swish(nn.Module):
    def __init__(self, beta=1):
        super(Swish, self).__init__()
        self.beta = beta

    def forward(self, x):
        return x * torch.sigmoid(self.beta * x)


class GNN_Layer(MessagePassing):
    def __init__(
        self, 
        message_in_dim,
        message_hidden_dim,
        message_out_dim,
        update_in_dim,
        update_hidden_dim,
        update_out_dim,
    ):
        super(GNN_Layer, self).__init__(node_dim=-2, aggr='mean')

        self.message_net = nn.Sequential(
            nn.Linear(message_in_dim, message_hidden_dim),
            Swish(),
            nn.BatchNorm1d(int(message_hidden_dim)),
            nn.Linear(int(message_hidden_dim), int(message_hidden_dim)),
            Swish(),
            nn.BatchNorm1d(int(message_hidden_dim)),
            nn.Linear(int(message_hidden_dim), int(message_hidden_dim)),
            Swish(),
            nn.BatchNorm1d(int(message_hidden_dim)),
            nn.Linear(int(message_hidden_dim), message_out_dim),
            Swish()
        )
        self.update_net = nn.Sequential(
            nn.Linear(update_in_dim, int(update_hidden_dim)),
            Swish(),
            nn.Linear(int(update_hidden_dim), int(update_hidden_dim)),
            Swish(),
            nn.Linear(int(update_hidden_dim), int(update_hidden_dim)),
            Swish(),
            nn.Linear(int(update_hidden_dim), update_out_dim),
            Swish()
        )

    def forward(self, x, edge_index, v):
        """ Propagate messages along edges """
        propagate = self.propagate(edge_index, x=x, v=v)
        return propagate

    def message(self, x_i, x_j, x):
        """ Message creation over neighbours """
        features = torch.cat((x_i, x_j[:, :int(x_j.shape[1]/2)]), dim=-1)
        # print(x_i.shape)
        # print(x_j.shape)
        # print(features.shape)
        # print("MESSAGE")
        message = self.message_net(features)
        return message

    def update(self, agg_message, x, v):
        """ Node update """
        # print(v.shape)
        # print(x.shape)
        # print(agg_message.shape)
        # print(torch.cat((v, x, agg_message), dim=-1).shape)
        # print("UPDATE")
        # print(self.update_net)
        # print(self.update_net(torch.cat((v, x, agg_message), dim=-1)).shape)
        # k = pp
        x += self.update_net(torch.cat((v, x, agg_message), dim=-1))
        return x



class GNN_basis(torch.nn.Module):
    def __init__(self, config):
        super(GNN_basis, self).__init__()
        self.config = config

        self.out_dim = config["out_dim"]
        self.message_in_dim = config["message_in_dim"]                           # 2 Elements: neighbouring feature (v, G)
        self.message_hidden_dim = config["message_hidden_dim"]
        self.message_out_dim = config["message_out_dim"]
        self.update_in_dim = config["message_out_dim"] + int(2*config["message_in_dim"]/2) # 3 Elements: agg message, local v, local feature (v, G)
        self.update_hidden_dim = config["update_hidden_dim"]
        self.update_out_dim = config["update_out_dim"] # config["omega_steps"] + 1*config["omega_steps"]
        self.nr_coefficients = config["nr_coefficients"]
        self.hidden_layer = config["hidden_layer"]
        self.pre_pool_hidden_dim = config["pre_pool_hidden_dim"]
        self.pre_pool_out_dim = config["pre_pool_out_dim"]
        self.post_pool_hidden_dim = config["post_pool_hidden_dim"]
        self.post_pool_out_dim = config["nr_coefficients"]

        # in_features have to be of the same size as out_features for the time being
        self.green_gnn = torch.nn.ModuleList(
            modules=[GNN_Layer(
                self.message_in_dim, self.message_hidden_dim, self.message_out_dim, self.update_in_dim, self.update_hidden_dim, self.update_out_dim, 
            ) for _ in range(self.hidden_layer)]
        )

        self.head_pre_pool = nn.Sequential(
            nn.Linear(self.update_out_dim, int(self.pre_pool_hidden_dim)),
            Swish(),
            nn.Linear(int(self.pre_pool_hidden_dim), int(self.pre_pool_hidden_dim * 1)),
            Swish(),
            nn.Linear(int(self.pre_pool_hidden_dim * 1), self.pre_pool_hidden_dim),
            Swish(),
            nn.Linear(self.pre_pool_hidden_dim, self.pre_pool_out_dim))

        self.head_post_pool = nn.Sequential(
            nn.Linear(self.pre_pool_out_dim, self.post_pool_hidden_dim),
            Swish(),
            nn.Linear(self.post_pool_hidden_dim, self.nr_coefficients))

        # self.embedding_mlp = nn.Sequential(
        #     nn.Linear(self.in_features, self.embedding_features))

    def forward(self, data): #, G):
        edge_index = data["edge_index"][0]
        x = data["node_feature"][0]
        x1 = data["vectors"][0]
        
        # Not sure whether deepcopy is really needed...idea is to preserve basis vectors.
        v_shape = int(x.shape[1]/2)
        # x1 = copy.deepcopy(x)[:,:v_shape]
        x2 = copy.deepcopy(x)

        for i in range(self.hidden_layer):
            x2 = self.green_gnn[i](x2, edge_index, v=x1)

        x2 = self.head_pre_pool(x2)
        batch = torch.zeros(x2.size(0), dtype=torch.long, device=x2.device)
        x2 = global_mean_pool(x2, batch)
        coefficients = self.head_post_pool(x2)

        if self.config["weird"] == True:
            x3 = torch.zeros(self.out_dim, device=x2.device, dtype=torch.float64)
            for n in range(0, coefficients.shape[0]):
                x3 += x1[n,:] * coefficients[n,:]

        if self.config["weird"] == False:
            x3 = torch.zeros(self.out_dim, device=x2.device, dtype=torch.float64)
            # print(x1.shape, coefficients.shape)
            # k = pp
            for n in range(0, coefficients.shape[1]):
                x3 += x1[n,:] * coefficients[0,n]
        
        
        return x3
