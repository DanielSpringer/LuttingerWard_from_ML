import sys
sys.path.append('/home/fs71922/hessl3/data/ML_Luttinger/LuttingerWard_from_ML/')

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
        #self.activation = x2_Activation()
        # self.activation = xN_Activation(1.5)

              
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
    
    
class x2_Activation(nn.Module):
    def __init__(self):
        super(x2_Activation, self).__init__()
        
    def forward(self, x):
        return torch.where(x < 0, torch.tensor(0.0, device=x.device, dtype=x.dtype), x * x/2)


class x1_Activation(nn.Module):
    def __init__(self):
        super(x1_Activation, self).__init__()
        
    def forward(self, x):
        return torch.where(x < 0, torch.tensor(0.0, device=x.device, dtype=x.dtype), x)
    

class xN_Activation(nn.Module):
    def __init__(self,N):
        super(xN_Activation, self).__init__()
        self.N = N
        
    def forward(self, x):
        act = nn.ReLU()
        y=act(x)
        return y**(self.N)/self.N

class auto_encoder_AD(torch.nn.Module):
    def __init__(self, config):
        super(auto_encoder_AD, self).__init__()
        self.config = config
        #self.activation = x2_Activation()
        self.activation = xN_Activation(2)
        # self.activation = nn.GELU()
        #self.activation = nn.ReLU()
        #self.activation = nn.Sigmoid()

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
    

class auto_encoder_vertex(torch.nn.Module):
    def __init__(self, config):
        super(auto_encoder_vertex, self).__init__()
        self.config = config
        self.activation = nn.ReLU() #nn.SiLU()# nn.LeakyReLU()

        in_dim = config["in_dim"];

        if (config["positional_encoding"]):
            in_dim += 3

        self.embedding = nn.Sequential(
            nn.Linear(in_dim, config["embedding_dim"])
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
        if (self.config["positional_encoding"]):
            y = (data_in[0]) / 576.0
            x = self.embedding(torch.cat([y, data_in[1]], axis=1))
            x = self.encode(x)
            x = self.decode(x)
        else:
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


class GNN_1_Layer(MessagePassing):
    def __init__(self, config):
        
        super(GNN_1_Layer, self).__init__(node_dim=-2, aggr='mean')
        
        message_in_dim = config["message_in_dim"]
        message_hidden_dim = config["message_hidden_dim"]
        message_out_dim = config["message_out_dim"]
        update_in_dim = config["update_in_dim"]
        update_hidden_dim = config["update_hidden_dim"]
        update_out_dim = config["update_out_dim"]
        n_nodes = config["n_nodes"]

        self.message_net = nn.Sequential(
            nn.Linear(message_in_dim, message_hidden_dim),
            Swish(),
            nn.BatchNorm1d(int(n_nodes*n_nodes)),
            nn.Linear(int(message_hidden_dim), int(message_hidden_dim)),
            Swish(),
            nn.BatchNorm1d(int(n_nodes*n_nodes)),
            nn.Linear(int(message_hidden_dim), int(message_hidden_dim)),
            Swish(),
            nn.BatchNorm1d(int(n_nodes*n_nodes)),
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
        # print("PROPAGATE")
        # print("Node Features 2xiv (ImG | Ve):" , x.shape)
        # print("Vectors:" , v.shape)
        return propagate

    def message(self, x_i, x_j, x):
        """ Message creation over neighbours 
        x_i: node_i [available/incoming messages for each node i] (including own)
                DIM[x_i]: [batch, n_nodes**2, node_feature]

        x_j: node_j [messages sent by node j] (N1 identical lines if node1 is connect to N1 neighbours)
                DIM[x_j]: [batch, n_nodes**2, node_feature]

        features: Concatenation of x_i and x_j generates all possible combinations of messages 
                DIM[features]: [batch, n_nodes**2, 2*node_feature]
            First Layer:         [vector_i | ImG] | [vector_j | ImG]
            Follow up Layers:   [FeatureVector_i] | [FeatureVector_j]

        message_net(features): creates local messages for all the concatenated vectors (i.e. all combination)
        """
        features = torch.cat((x_i, x_j[:, :int(x_j.shape[1])]), dim=-1)
        message = self.message_net(features)
        # print(x_i.shape)
        # print(x_j.shape)
        # print(x_j[:, :int(x_j.shape[1])].shape)
        # print(x_i[0,:,0:4])
        # print(x_j[0,:,0:4])
        # print(features.shape)
        # print("MESSAGE")
        # print(message.shape)
        # print("  ---------------  ")
        return message

    def update(self, agg_message, x, v):
        """ Node update 
        v: Original vectors
            DIM[v]: [batch, n_nodes, omega_steps]

        agg_message: Node-wise output of messageNet
            DIM[agg_message]: [batch, n_nodes, message_out_dim]

        x: Node-wise features (node_features) before messageNet
            DIM[x]: [batch, n_nodes, node_feature]

        x += update_net(cat[v,x,message]): 
            > The concatenation of [message | node_feature | vector] is reminiscent (concat != summation) 
              of a ResNet with respect to message_net (x is the residual and v is the super residual that never changes)
            > update_net output is identical in dimension as x to be again added to the residual x (ResNet with respect to update_net)
        """
        x += self.update_net(torch.cat((v, x, agg_message), dim=-1))
        # print("v: ", v.shape)
        # print("x: ", x.shape)
        # print("Message: ", agg_message.shape)
        # print(v[0,:,0:4])
        # print(x[0,:,0:4])
        # print(agg_message[0,:,0:4])
        # print(torch.cat((v, x, agg_message), dim=-1).shape)
        # print("UPDATE")
        # print("UpdateNet Output: ", self.update_net(torch.cat((v, x, agg_message), dim=-1)).shape)
        # print("  ---------------  ")
        return x

class GNN_1(torch.nn.Module):
    def __init__(self, config):
        super(GNN_1, self).__init__()
        self.config = config

        self.out_dim = config["out_dim"]
        self.message_in_dim = config["message_in_dim"]                           # 2 Elements: neighbouring feature (v, G)
        self.message_hidden_dim = config["message_hidden_dim"]
        self.message_out_dim = config["message_out_dim"]

        if "update_in_dim" in config:
            self.update_in_dim = config["update_in_dim"]  
        else:
            self.update_in_dim = config["message_out_dim"] + int(config["message_in_dim"]) # 3 Elements: agg message, local v, local feature (v, G)
        
        self.update_hidden_dim = config["update_hidden_dim"]
        self.update_out_dim = config["update_out_dim"] # config["omega_steps"] + 1*config["omega_steps"]
        self.nr_coefficients = config["nr_coefficients"]
        self.hidden_layer = config["hidden_layer"]
        self.pre_pool_hidden_dim = config["pre_pool_hidden_dim"]
        self.pre_pool_out_dim = config["pre_pool_out_dim"]
        self.post_pool_hidden_dim = config["post_pool_hidden_dim"]
        self.post_pool_out_dim = config["nr_coefficients"]
        self.n_nodes = config["n_nodes"]

        # in_features have to be of the same size as out_features for the time being
        self.green_gnn = torch.nn.ModuleList(
            modules=[GNN_1_Layer(config) for _ in range(self.hidden_layer)]
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
            nn.Linear(self.post_pool_hidden_dim, 1))
            # nn.Linear(self.post_pool_hidden_dim, self.nr_coefficients))

    def forward(self, data): #, G):
        edge_index = data["edge_index"][0]
        x = data["node_feature"][:]
        x1 = data["vectors"][:]
        
        x2 = copy.deepcopy(x)

        for i in range(self.hidden_layer):
            x2 = self.green_gnn[i](x2, edge_index, v=x1)
        x2 = self.head_pre_pool(x2)
        # batch = torch.zeros(x2.size(1), dtype=torch.long, device=x2.device)
        # x2 = global_mean_pool(x2, batch)
        coefficients = self.head_post_pool(x2)
        
        # return coefficients

        x3 = torch.zeros((x1.shape[0],x1.shape[2]), device=x2.device, dtype=torch.float64)
        for b in range(0, coefficients.shape[0]):
            for n in range(0, coefficients.shape[1]):
                x3[b,:] += x1[b,n,:] * coefficients[b,n,0]
        return x3

class GNN_1_base(torch.nn.Module):
    def __init__(self, config):
        super(GNN_1_base, self).__init__()
        self.config = config

        self.out_dim = config["out_dim"]
        self.message_in_dim = config["message_in_dim"]                           # 2 Elements: neighbouring feature (v, G)
        self.message_hidden_dim = config["message_hidden_dim"]
        self.message_out_dim = config["message_out_dim"]

        if "update_in_dim" in config:
            self.update_in_dim = config["update_in_dim"]  
        else:
            self.update_in_dim = config["message_out_dim"] + int(config["message_in_dim"]) # 3 Elements: agg message, local v, local feature (v, G)
        
        self.update_hidden_dim = config["update_hidden_dim"]
        self.update_out_dim = config["update_out_dim"] # config["omega_steps"] + 1*config["omega_steps"]
        self.nr_coefficients = config["nr_coefficients"]
        self.hidden_layer = config["hidden_layer"]
        self.pre_pool_hidden_dim = config["pre_pool_hidden_dim"]
        self.pre_pool_out_dim = config["pre_pool_out_dim"]
        self.post_pool_hidden_dim = config["post_pool_hidden_dim"]
        self.post_pool_out_dim = config["nr_coefficients"]
        self.n_nodes = config["n_nodes"]

        # in_features have to be of the same size as out_features for the time being
        self.green_gnn = torch.nn.ModuleList(
            modules=[GNN_1_Layer(config) for _ in range(self.hidden_layer)]
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
            nn.Linear(self.post_pool_hidden_dim, 1))
            # nn.Linear(self.post_pool_hidden_dim, self.nr_coefficients))

    def forward(self, data): #, G):
        edge_index = data["edge_index"][0]
        x = data["node_feature"][:]
        x1 = data["vectors"][:]
        
        x2 = copy.deepcopy(x)

        for i in range(self.hidden_layer):
            x2 = self.green_gnn[i](x2, edge_index, v=x1)
        x2 = self.head_pre_pool(x2)
        # batch = torch.zeros(x2.size(1), dtype=torch.long, device=x2.device)
        # x2 = global_mean_pool(x2, batch)
        coefficients = self.head_post_pool(x2)

        x3 = torch.zeros((x1.shape[0],x1.shape[2]), device=x2.device, dtype=torch.float64)
        for b in range(0, coefficients.shape[0]):
            for n in range(0, coefficients.shape[1]):
                x3[b,:] += x1[b,n,:] * coefficients[b,n,0]
        return x3
        
class GNN_1_direct(torch.nn.Module):
    def __init__(self, config):
        super(GNN_1_direct, self).__init__()
        self.config = config

        self.out_dim = config["out_dim"]
        self.message_in_dim = config["message_in_dim"]                           # 2 Elements: neighbouring feature (v, G)
        self.message_hidden_dim = config["message_hidden_dim"]
        self.message_out_dim = config["message_out_dim"]

        if "update_in_dim" in config:
            self.update_in_dim = config["update_in_dim"]  
        else:
            self.update_in_dim = config["message_out_dim"] + int(config["message_in_dim"]) # 3 Elements: agg message, local v, local feature (v, G)
        
        self.update_hidden_dim = config["update_hidden_dim"]
        self.update_out_dim = config["update_out_dim"] # config["omega_steps"] + 1*config["omega_steps"]
        self.nr_coefficients = config["nr_coefficients"]
        self.hidden_layer = config["hidden_layer"]
        self.pre_pool_hidden_dim = config["pre_pool_hidden_dim"]
        self.pre_pool_out_dim = config["pre_pool_out_dim"]
        self.post_pool_hidden_dim = config["post_pool_hidden_dim"]
        self.post_pool_out_dim = config["nr_coefficients"]
        self.n_nodes = config["n_nodes"]

        # in_features have to be of the same size as out_features for the time being
        self.green_gnn = torch.nn.ModuleList(
            modules=[GNN_1_Layer(config) for _ in range(self.hidden_layer)]
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
            nn.Linear(self.post_pool_hidden_dim, 1))
            # nn.Linear(self.post_pool_hidden_dim, self.nr_coefficients))

    def forward(self, data): #, G):
        edge_index = data["edge_index"][0]
        x = data["node_feature"][:]
        x1 = data["vectors"][:]
        
        x2 = copy.deepcopy(x)

        for i in range(self.hidden_layer):
            x2 = self.green_gnn[i](x2, edge_index, v=x1)
        x2 = self.head_pre_pool(x2)
        # batch = torch.zeros(x2.size(1), dtype=torch.long, device=x2.device)
        # x2 = global_mean_pool(x2, batch)
        coefficients = self.head_post_pool(x2)

        return coefficients[:,:,0]

        # x3 = torch.zeros((x1.shape[0],x1.shape[2]), device=x2.device, dtype=torch.float64)
        # for b in range(0, coefficients.shape[0]):
        #     for n in range(0, coefficients.shape[1]):
        #         x3[b,:] += x1[b,n,:] * coefficients[b,n,0]
        # return x3


class GNN_1_trainbase(torch.nn.Module):
    def __init__(self, config):
        super(GNN_1_trainbase, self).__init__()
        self.config = config

        self.omega_steps = config["omega_steps"]
        self.vec_emb_hidden_dim = config["vec_emb_hidden_dim"]
        self.out_dim = config["out_dim"]
        self.message_in_dim = 2*config["message_in_dim"] #config["message_in_dim"]                           # 2 Elements: neighbouring feature (v, G)
        self.message_hidden_dim = config["message_hidden_dim"]
        self.message_out_dim = config["message_out_dim"]

        if "update_in_dim" in config:
            self.update_in_dim = config["update_in_dim"]  
        else:
            self.update_in_dim = config["message_out_dim"] + int(config["message_in_dim"]) # 3 Elements: agg message, local v, local feature (v, G)
        
        self.update_hidden_dim = config["update_hidden_dim"]
        self.update_out_dim = config["update_out_dim"] # config["omega_steps"] + 1*config["omega_steps"]
        self.nr_coefficients = config["nr_coefficients"]
        self.hidden_layer = config["hidden_layer"]
        self.pre_pool_hidden_dim = config["pre_pool_hidden_dim"]
        self.pre_pool_out_dim = config["pre_pool_out_dim"]
        self.post_pool_hidden_dim = config["post_pool_hidden_dim"]
        self.post_pool_out_dim = config["nr_coefficients"]
        self.n_nodes = config["n_nodes"]

        # in_features have to be of the same size as out_features for the time being
        self.green_gnn = torch.nn.ModuleList(
            modules=[GNN_1_Layer(config) for _ in range(self.hidden_layer)]
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
            nn.Linear(self.post_pool_hidden_dim, 1))
            # nn.Linear(self.post_pool_hidden_dim, self.nr_coefficients))

        self.vec_embedding_mlp = nn.Sequential(
            nn.Linear(self.omega_steps, self.vec_emb_hidden_dim),
            Swish(),
            nn.Linear(self.vec_emb_hidden_dim, self.vec_emb_hidden_dim),
            Swish(),
            nn.Linear(self.vec_emb_hidden_dim, self.omega_steps))

    
    def forward(self, data):
        edge_index = data["edge_index"][0]
        x = data["node_feature"][:]
        x1 = data["vectors"][:]
        
        x2 = copy.deepcopy(x)
        x1 = self.vec_embedding_mlp(x1)
        
        for i in range(self.hidden_layer):
            x2 = self.green_gnn[i](x2, edge_index, v=x1)
        x2 = self.head_pre_pool(x2)
        coefficients = self.head_post_pool(x2)
        
        x3 = torch.zeros((x1.shape[0],x1.shape[2]), device=x2.device, dtype=torch.float64)
        for b in range(0, coefficients.shape[0]):
            for n in range(0, coefficients.shape[1]):
                x3[b,:] += x1[b,n,:] * coefficients[b,n,0]
        return x3

class GNN_2(torch.nn.Module):
    def __init__(self, config):
        super(GNN_2, self).__init__()
        self.config = config

        self.omega_steps = config["omega_steps"]
        self.vec_emb_hidden_dim = config["vec_emb_hidden_dim"]
        self.out_dim = config["out_dim"]
        self.message_in_dim = 2*config["message_in_dim"] #config["message_in_dim"]                           # 2 Elements: neighbouring feature (v, G)
        self.message_hidden_dim = config["message_hidden_dim"]
        self.message_out_dim = config["message_out_dim"]

        if "update_in_dim" in config:
            self.update_in_dim = config["update_in_dim"]  
        else:
            self.update_in_dim = config["message_out_dim"] + int(config["message_in_dim"]) # 3 Elements: agg message, local v, local feature (v, G)
        
        self.update_hidden_dim = config["update_hidden_dim"]
        self.update_out_dim = config["update_out_dim"] # config["omega_steps"] + 1*config["omega_steps"]
        self.nr_coefficients = config["nr_coefficients"]
        self.hidden_layer = config["hidden_layer"]
        self.pre_pool_hidden_dim = config["pre_pool_hidden_dim"]
        self.pre_pool_out_dim = config["pre_pool_out_dim"]
        self.post_pool_hidden_dim = config["post_pool_hidden_dim"]
        self.post_pool_out_dim = config["nr_coefficients"]
        self.n_nodes = config["n_nodes"]

        # in_features have to be of the same size as out_features for the time being
        self.green_gnn = torch.nn.ModuleList(
            modules=[GNN_1_Layer(config) for _ in range(self.hidden_layer)]
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
            nn.Linear(self.post_pool_hidden_dim, 1))
            # nn.Linear(self.post_pool_hidden_dim, self.nr_coefficients))

        self.vec_embedding_mlp = nn.Sequential(
            nn.Linear(self.omega_steps, self.vec_emb_hidden_dim),
            Swish(),
            nn.Linear(self.vec_emb_hidden_dim, self.vec_emb_hidden_dim),
            Swish(),
            nn.Linear(self.vec_emb_hidden_dim, self.omega_steps))

    
    def forward(self, data):
        edge_index = data["edge_index"][0]
        x = data["node_feature"][:]
        x1 = data["vectors"][:]
        
        x2 = copy.deepcopy(x)
        x1 = self.vec_embedding_mlp(x1)
        
        for i in range(self.hidden_layer):
            x2 = self.green_gnn[i](x2, edge_index, v=x1)
        x2 = self.head_pre_pool(x2)
        coefficients = self.head_post_pool(x2)
        
        x3 = torch.zeros((x1.shape[0],x1.shape[2]), device=x2.device, dtype=torch.float64)
        for b in range(0, coefficients.shape[0]):
            for n in range(0, coefficients.shape[1]):
                x3[b,:] += x1[b,n,:] * coefficients[b,n,0]
        return x3


############# OLD GNN VERSIONS BEFORE 2025 #############
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
        features = torch.cat((x_i, x_j[:, :int(x_j.shape[1])]), dim=-1)
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

        if "update_in_dim" in config:
            self.update_in_dim = config["update_in_dim"]  
        else:
            self.update_in_dim = config["message_out_dim"] + int(config["message_in_dim"]) # 3 Elements: agg message, local v, local feature (v, G)
            
        
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
        # v_shape = int(x.shape[1]/2)
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
            ### HARDCODED FOR BATCH 1 TO PREDICT IMAG ONLY!!!
            # x3 = torch.zeros(self.out_dim, device=x2.device, dtype=torch.float64)
            x3 = torch.zeros(self.config["out_dim"], device=x2.device, dtype=torch.float64)
            for n in range(0, coefficients.shape[1]):
                x3 += x1[n,:] * coefficients[0,n]
        return x3
    
    
class GNN_Layer_2(MessagePassing):
    def __init__(
        self, 
        message_in_dim,
        message_hidden_dim,
        message_out_dim,
        update_in_dim,
        update_hidden_dim,
        update_out_dim,
        n_nodes,
    ):
        super(GNN_Layer_2, self).__init__(node_dim=-2, aggr='mean')

        self.message_net = nn.Sequential(
            nn.Linear(message_in_dim, message_hidden_dim),
            Swish(),
            # nn.BatchNorm1d(int(message_hidden_dim*n_nodes)),
            nn.BatchNorm1d(int(n_nodes*n_nodes)),
            nn.Linear(int(message_hidden_dim), int(message_hidden_dim)),
            Swish(),
            # nn.BatchNorm1d(int(message_hidden_dim*n_nodes)),
            nn.BatchNorm1d(int(n_nodes*n_nodes)),
            nn.Linear(int(message_hidden_dim), int(message_hidden_dim)),
            Swish(),
            # nn.BatchNorm1d(int(message_hidden_dim*n_nodes)),
            nn.BatchNorm1d(int(n_nodes*n_nodes)),
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
        features = torch.cat((x_i, x_j[:, :int(x_j.shape[1])]), dim=-1)
        # print(x_i.shape)
        # print(x_i[0,:5])
        # print(x_j.shape)
        # print(x_j[0,:5])
        # print(x_i-x_j)
        # print(features.shape)
        # print("MESSAGE")
        # p = kk
        message = self.message_net(features)
        return message

    def update(self, agg_message, x, v):
        """ Node update """
        # print("update start")
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



class GNN_basis_2(torch.nn.Module):
    def __init__(self, config):
        super(GNN_basis_2, self).__init__()
        self.config = config

        self.omega_steps = config["omega_steps"]

        self.in_dim = config["in_dim"]
        self.embedding_features = config["embedding_features"]
        self.vec_emb_hidden_dim = config["vec_emb_hidden_dim"]
        self.out_dim = config["out_dim"]
        self.message_in_dim = 2*config["embedding_features"] #config["message_in_dim"]                           # 2 Elements: neighbouring feature (v, G)
        self.message_hidden_dim = config["message_hidden_dim"]
        self.message_out_dim = config["message_out_dim"]

        if "update_in_dim" in config:
            self.update_in_dim = config["update_in_dim"]  
        else:
            self.update_in_dim = config["message_out_dim"] + int(config["message_in_dim"]) # 3 Elements: agg message, local v, local feature (v, G)
            
        
        self.update_hidden_dim = config["update_hidden_dim"]
        self.update_out_dim = config["update_out_dim"] # config["omega_steps"] + 1*config["omega_steps"]
        self.nr_coefficients = config["nr_coefficients"]
        self.hidden_layer = config["hidden_layer"]
        self.pre_pool_hidden_dim = config["pre_pool_hidden_dim"]
        self.pre_pool_out_dim = config["pre_pool_out_dim"]
        self.post_pool_hidden_dim = config["post_pool_hidden_dim"]
        self.post_pool_out_dim = config["nr_coefficients"]
        self.n_nodes = config["n_nodes"]

        # in_features have to be of the same size as out_features for the time being
        self.green_gnn = torch.nn.ModuleList(
            modules=[GNN_Layer_2(
                self.message_in_dim, self.message_hidden_dim, self.message_out_dim, self.update_in_dim, self.update_hidden_dim, self.update_out_dim, self.n_nodes
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

        self.G_embedding_mlp = nn.Sequential(
            nn.Linear(self.in_dim, self.embedding_features))

        self.vec_embedding_mlp = nn.Sequential(
            nn.Linear(self.omega_steps, self.vec_emb_hidden_dim),
            Swish(),
            nn.Linear(self.vec_emb_hidden_dim, self.vec_emb_hidden_dim),
            Swish(),
            nn.Linear(self.vec_emb_hidden_dim, self.omega_steps))

    def forward(self, data): #, G):
        ### DOES THIS WORK FOR BATCHES???
        edge_index = data["edge_index"][0]
        x = data["node_feature"][:]
        x1 = data["vectors"][:]
        
        # Not sure whether deepcopy is really needed...idea is to preserve basis vectors.
        # v_shape = int(x.shape[1]/2)
        # x1 = copy.deepcopy(x)[:,:v_shape]
        x2 = copy.deepcopy(x)

        x2 = self.G_embedding_mlp(x2)
        x1 = self.vec_embedding_mlp(x1)
        for i in range(self.hidden_layer):
            x2 = self.green_gnn[i](x2, edge_index, v=x1)

        x2 = self.head_pre_pool(x2)
        batch = torch.zeros(x2.size(1), dtype=torch.long, device=x2.device)
        x2 = global_mean_pool(x2, batch)
        coefficients = self.head_post_pool(x2)

        if self.config["weird"] == True:
            x3 = torch.zeros(self.out_dim, device=x2.device, dtype=torch.float64)
            for n in range(0, coefficients.shape[0]):
                x3 += x1[n,:] * coefficients[n,:]

        if self.config["weird"] == False:
            ### HARDCODED FOR BATCH 1 TO PREDICT IMAG ONLY!!!
            # x3 = torch.zeros(self.out_dim, device=x2.device, dtype=torch.float64)
            x3 = torch.zeros((x1.shape[0],x1.shape[2]), device=x2.device, dtype=torch.float64)
            # print(x3.shape, coefficients.shape, x1.shape)
            for b in range(0, coefficients.shape[0]):
                for n in range(0, coefficients.shape[2]):
                    # print((x1[b,n,:] * coefficients[b,0,n]).shape)
                    x3[b,:] += x1[b,n,:] * coefficients[b,0,n]
        
        
        return x3




class GNN_Layer_batched(MessagePassing):
    def __init__(
        self, 
        message_in_dim,
        message_hidden_dim,
        message_out_dim,
        update_in_dim,
        update_hidden_dim,
        update_out_dim,
        n_nodes
    ):
        super(GNN_Layer_batched, self).__init__(node_dim=-2, aggr='mean')

        self.message_net = nn.Sequential(
            nn.Linear(message_in_dim, message_hidden_dim),
            Swish(),
            nn.BatchNorm1d(int(n_nodes*n_nodes)),
            nn.Linear(int(message_hidden_dim), int(message_hidden_dim)),
            Swish(),
            nn.BatchNorm1d(int(n_nodes*n_nodes)),
            nn.Linear(int(message_hidden_dim), int(message_hidden_dim)),
            Swish(),
            nn.BatchNorm1d(int(n_nodes*n_nodes)),
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
        features = torch.cat((x_i, x_j[:, :int(x_j.shape[1])]), dim=-1)
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



class GNN_batched(torch.nn.Module):
    def __init__(self, config):
        super(GNN_batched, self).__init__()
        self.config = config

        self.out_dim = config["out_dim"]
        self.message_in_dim = config["message_in_dim"]                           # 2 Elements: neighbouring feature (v, G)
        self.message_hidden_dim = config["message_hidden_dim"]
        self.message_out_dim = config["message_out_dim"]

        if "update_in_dim" in config:
            self.update_in_dim = config["update_in_dim"]  
        else:
            self.update_in_dim = config["message_out_dim"] + int(config["message_in_dim"]) # 3 Elements: agg message, local v, local feature (v, G)
            
        
        self.update_hidden_dim = config["update_hidden_dim"]
        self.update_out_dim = config["update_out_dim"] # config["omega_steps"] + 1*config["omega_steps"]
        self.nr_coefficients = config["nr_coefficients"]
        self.hidden_layer = config["hidden_layer"]
        self.pre_pool_hidden_dim = config["pre_pool_hidden_dim"]
        self.pre_pool_out_dim = config["pre_pool_out_dim"]
        self.post_pool_hidden_dim = config["post_pool_hidden_dim"]
        self.post_pool_out_dim = config["nr_coefficients"]
        self.n_nodes = config["n_nodes"]

        # in_features have to be of the same size as out_features for the time being
        self.green_gnn = torch.nn.ModuleList(
            modules=[GNN_Layer_batched(
                self.message_in_dim, self.message_hidden_dim, self.message_out_dim, self.update_in_dim, self.update_hidden_dim, self.update_out_dim, self.n_nodes
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
        # batch = data.batch
        ### DOES THIS WORK FOR BATCHES???
        edge_index = data["edge_index"][0]
        x = data["node_feature"][:]
        x1 = data["vectors"][:]
        
        # Not sure whether deepcopy is really needed...idea is to preserve basis vectors.
        # v_shape = int(x.shape[1]/2)
        # x1 = copy.deepcopy(x)[:,:v_shape]
        x2 = copy.deepcopy(x)

        for i in range(self.hidden_layer):
            x2 = self.green_gnn[i](x2, edge_index, v=x1)

        ### 21.12.2024 not sure whether this makes any sense...
        x2 = self.head_pre_pool(x2)
        batch = torch.zeros(x2.size(1), dtype=torch.long, device=x2.device)
        x2 = global_mean_pool(x2, batch)
        coefficients = self.head_post_pool(x2)

        if self.config["weird"] == True:
            x3 = torch.zeros(self.out_dim, device=x2.device, dtype=torch.float64)
            for n in range(0, coefficients.shape[0]):
                x3 += x1[n,:] * coefficients[n,:]

        if self.config["weird"] == False:
            ### HARDCODED FOR BATCH 1 TO PREDICT IMAG ONLY!!!
            # x3 = torch.zeros(self.out_dim, device=x2.device, dtype=torch.float64)
            x3 = torch.zeros((x1.shape[0],x1.shape[2]), device=x2.device, dtype=torch.float64)
            # print(x3.shape, coefficients.shape, x1.shape)
            for b in range(0, coefficients.shape[0]):
                for n in range(0, coefficients.shape[2]):
                    # print((x1[b,n,:] * coefficients[b,0,n]).shape)
                    x3[b,:] += x1[b,n,:] * coefficients[b,0,n]
        
        return x3
    
