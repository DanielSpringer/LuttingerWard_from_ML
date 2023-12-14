import torch 
from torch import nn
from torch_geometric.nn import MessagePassing, global_mean_pool
import copy
import pytorch_lightning as pl


class model_wraper(pl.LightningModule):

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
        return loss

    def validation_step(self, batch, batch_idx):
#         pred = self.forward(batch[0])
        pred = self.forward(batch)
        target = batch["target"][0]
        loss = self.criterion_mse(pred, target)
        self.val_pred.append([target, pred])
        self.val_loss.append(loss)
        return loss

    # def on_validation_epoch_end(self):
    #     pass

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(params=self.model.parameters(), lr=self.config['learning_rate'], weight_decay=self.config['weight_decay'])
        return optimizer
    
    def load_model_state(self, PATH):
        checkpoint = torch.load(PATH, map_location='cuda:0')
        self.model.load_state_dict(checkpoint['state_dict'])


class auto_encoder(torch.nn.Module):
    def __init__(self, config):
        super(auto_encoder, self).__init__()
        self.config = config

        self.embedding = nn.Sequential(
            nn.Linear(config["in_dim"], config["embedding_dim"])
        )

        self.encoding = nn.Sequential(
            nn.Linear(config["embedding_dim"], config["hidden1_dim"]),
            nn.Linear(config["hidden1_dim"], config["hidden2_dim"]),
            nn.Linear(config["hidden2_dim"], config["encoder_dim"])
        )

        self.decoding = nn.Sequential(
            nn.Linear(config["encoder_dim"], config["hidden2_dim"]),
            nn.Linear(config["hidden2_dim"], config["hidden1_dim"]),
            nn.Linear(config["hidden1_dim"], config["in_dim"])
        )

    def forward(self, data_in):
        x = self.embedding(data_in)
        x = self.encoding(x)
        x = self.decoding(x)
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
    
    

################################ GRAPH 
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
        features = torch.cat((x_i, x_j[:, :int(x_j.shape[1]/3)]), dim=-1)
        message = self.message_net(features)
        return message

    def update(self, agg_message, x, v):
        """ Node update """
        x += self.update_net(torch.cat((v, x, agg_message), dim=-1))
        return x



class GreenGNN(torch.nn.Module):
    def __init__(self, config):
        super(GreenGNN, self).__init__()

        omega_steps = 100
        tau_steps = 100
        message_in_dim = 400
        message_hidden_dim = 100
        message_out_dim = 100
        update_hidden_dim = 100
        update_out_dim = 300
        pre_pool_hidden_dim = 100
        pre_pool_out_dim = 100
        post_pool_hidden_dim = 100
        nr_coefficients = 100
        hidden_layer = 2

        self.out_dim = 100
        self.message_in_dim = message_in_dim                           # 2 Elements: neighbouring feature (v, G)
        self.message_hidden_dim = message_hidden_dim
        self.message_out_dim = message_out_dim
        self.update_in_dim = message_out_dim + int(2*message_in_dim/2) # 3 Elements: agg message, local v, local feature (v, G)
        self.update_hidden_dim = update_hidden_dim
        self.update_out_dim = omega_steps + 2*tau_steps
        self.nr_coefficients = nr_coefficients
        self.hidden_layer = hidden_layer
        self.pre_pool_hidden_dim = pre_pool_hidden_dim
        self.pre_pool_out_dim = pre_pool_out_dim
        self.post_pool_hidden_dim = post_pool_hidden_dim
        self.post_pool_out_dim = nr_coefficients

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
            nn.Linear(post_pool_hidden_dim, self.nr_coefficients))

        # self.embedding_mlp = nn.Sequential(
        #     nn.Linear(self.in_features, self.embedding_features))

    def forward(self, data): #, G):
        edge_index = data["edge_index"][0]
        x = data["node_feature"][0]
        x1 = data["vectors"][0]
#         print(" **********************2 ***************************  ")
#         print(" *************************************************  ")
#         print(" *************************************************  ")
#         print(" *************************************************  ")
#         print(" *************************************************  ")
#         print(" *************************************************  ")
        
#         print(data["ei"])
#         edge_index = data[1][0]
#         x = data[0][0]
#         x1 = data[3][0]
        
        # Not sure whether deepcopy is really needed...idea is to preserve basis vectors.
        v_shape = int(x.shape[1]/3)
        x1 = copy.deepcopy(x)[:,:v_shape]
        x2 = copy.deepcopy(x)

        for i in range(self.hidden_layer):
            x2 = self.green_gnn[i](x2, edge_index, v=x1)
#             x2 = self.green_gnn[i](x2, edge_index, batch, v=x1)

        x2 = self.head_pre_pool(x2)
        batch = torch.zeros(x2.size(0), dtype=torch.long, device=x2.device)
        x2 = global_mean_pool(x2, batch)
        coefficients = self.head_post_pool(x2)
        x3 = torch.zeros(self.out_dim, device=x2.device, dtype=torch.float64)
#         print(batch.dtype, x3.dtype, x1.dtype, coefficients.dtype)

#         print(out.shape, x1.shape)
        for n in range(0, coefficients.shape[0]):
#             print(x3.dtype, x1.dtype, coefficients.dtype)
            x3 += x1[n,:] * coefficients[n,:]
            
        return x3
    
    
    