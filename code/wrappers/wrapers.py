import torch 
from torch import nn
from torch_geometric.nn import MessagePassing, global_mean_pool
import json
# import lightning as L
import pytorch_lightning as L

from wrapper_AE import *


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


### Model wrapers for 3-step convergence concept:
class model_wraper_encgiv(pl.LightningModule):
    ''' First part of convergence model. Wraper to train actual auto-encoding, i.e. the input and target are identical. '''
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
        target = batch[0]
        loss = self.criterion_mse(pred, target)
        self.log('train_loss', loss.item())
        return loss

    def validation_step(self, batch: torch.Tensor, batch_idx: int) -> torch.Tensor:
        pred = self.forward(batch[0])
        target = batch[0]
        loss = self.criterion_mse(pred, target)
        self.val_pred.append([target, pred])
        self.val_loss.append(loss)
        self.log('val_loss', loss.item(), prog_bar=True)
        return loss

    def configure_optimizers(self) -> dict:
        optimizer = torch.optim.AdamW(params=self.model.parameters(), lr=self.config['learning_rate'], weight_decay=self.config['weight_decay'])
        return {"optimizer": optimizer}
    
    def load_model_state(self, PATH):
        checkpoint = torch.load(PATH, map_location='cuda:0')
        self.model.load_state_dict(checkpoint['state_dict'])


class model_wraper_G0injection(pl.LightningModule):
    ''' Second part of convergence model. In this wraper an injection vector (encoding for g0) is passed to the model to be used (injected) in the model.
        This wraper does not update the g0 encoding network. 
    '''
    def __init__(self, config):
        super().__init__()

        self.config = config
        config_encoder = json.load(open(config["ENCODER_MODEL_SAVEPATH"]+'config.json'))

        # LW prediction model
        module = __import__("models")
        self.model = getattr(module, config["MODEL_NAME"])(config)

        # Loading g0 encoder -> Different config for encoder. 
        encoder_wraper = __import__("wrapers")
        self.encoder_model = getattr(encoder_wraper, config["ENCODER_MODEL_WRAPER"])(config_encoder)
        filename = config["ENCODER_MODEL_SAVEPATH"] + config["ENCODER_MODEL_SAVEFILE"]
        if torch.cuda.is_available() == False:
            checkpoint = torch.load(filename, map_location=torch.device('cpu'))
        if torch.cuda.is_available() == True:
            checkpoint = torch.load(filename)
        self.encoder_model.load_state_dict(checkpoint['state_dict'])
        # No update of encoder
        self.encoder_model.model.requires_grad_(False)

        self.criterion_mse = nn.MSELoss()
        self.val_pred = []
        self.val_loss = []

    def forward(self, batch: torch.Tensor, g0):
        return self.model(batch, g0)

    def training_step(self, batch: torch.Tensor, batch_idx: int) -> torch.Tensor:
        # Encode the injection vector g0 = batch[2]
        x = self.encoder_model.model.embedding(batch[2])
        x = self.encoder_model.model.encode(x)
        # Predicting based on input G and injection vector
        pred = self.forward(batch[0], x)
        target = batch[1]
        loss = self.criterion_mse(pred, target)
        self.log('train_loss', loss.item())
        return loss

    def validation_step(self, batch: torch.Tensor, batch_idx: int) -> torch.Tensor:
        # Encode the injection vector g0 = batch[2]
        x = self.encoder_model.model.embedding(batch[2])
        x = self.encoder_model.model.encode(x)
        # Predicting based on input G and injection vector
        pred = self.forward(batch[0], x)
        target = batch[1]
        loss = self.criterion_mse(pred, target)
        self.val_pred.append([target, pred])
        self.val_loss.append(loss)
        self.log('val_loss', loss.item(), prog_bar=True)
        return loss

    def configure_optimizers(self) -> dict:
        optimizer = torch.optim.AdamW(params=self.model.parameters(), lr=self.config['learning_rate'], weight_decay=self.config['weight_decay'])
        return {"optimizer": optimizer}
    
    def load_model_state(self, PATH):
        checkpoint = torch.load(PATH, map_location='cuda:0')
        self.model.load_state_dict(checkpoint['state_dict'])


class model_wraper_convergence(pl.LightningModule):
    ''' Third part of convergence model. In this wraper the . '''
    def __init__(self, config):
        super().__init__()

        self.config = config
        config_stativ_LW = json.load(open(config["INJECTION_MODEL_SAVEPATH"]+'config.json'))
        config_encoder = json.load(open(config["ENCODER_MODEL_SAVEPATH"]+'config.json'))

        # module = __import__("models")
        wraper = __import__("wrapers")

        # Loading Encoder  
        self.encoder_model = getattr(wraper, config["ENCODER_MODEL_WRAPER"])(config_encoder)
        filename = config["ENCODER_MODEL_SAVEPATH"] + config["ENCODER_MODEL_SAVEFILE"]
        if torch.cuda.is_available() == False:
            checkpoint = torch.load(filename, map_location=torch.device('cpu'))
        if torch.cuda.is_available() == True:
            checkpoint = torch.load(filename)
        self.encoder_model.load_state_dict(checkpoint['state_dict'])
        # Only encoder is updated
        self.encoder_model.model.requires_grad_(True)

        # Loading LW model  
        self.static_LW_model = getattr(wraper, config["INJECTION_MODEL_WRAPER"])(config_stativ_LW)
        filename = config["INJECTION_MODEL_SAVEPATH"] + config["INJECTION_MODEL_SAVEFILE"]
        if torch.cuda.is_available() == False:
            checkpoint = torch.load(filename, map_location=torch.device('cpu'))
        if torch.cuda.is_available() == True:
            checkpoint = torch.load(filename)
        self.static_LW_model.load_state_dict(checkpoint['state_dict'])
        # No update of LW model
        self.static_LW_model.model.requires_grad_(False)

        self.criterion_mse = nn.MSELoss()
        self.val_pred = []
        self.val_loss = []
    
    def smoothing(self, data):
        ''' This function smoothes the output by applying a Gaussian filter. However, if this is applied in the convergence model but not in the 
            injection model, the quality of the pre-training is artificially reduced.
        '''
        k_sz = 7
        s = int((k_sz-1)/2)
        n = torch.linspace(-s,s,k_sz)
        sigma = 1.6
        w = 1/sigma/torch.sqrt(torch.tensor(2*torch.pi)) * torch.exp(-n**2/2/sigma**2)

        conv = torch.nn.Conv1d(
            in_channels=1, out_channels=1, kernel_size=k_sz, padding=s, bias=False
        )
        conv.weight.data = w[None,None]
        if torch.cuda.is_available():
            dev = data.get_device()
            conv.to(device=dev)
        real = conv(data[:,:,:200])
        imag = conv(data[:,:,200:400])
        out = torch.cat([real,imag], dim=2)
        return out[0,:,:]

    def forward(self, batch: torch.Tensor, g0):
        return self.static_LW_model(batch, g0)

    def training_step(self, batch: torch.Tensor, batch_idx: int) -> torch.Tensor:
        # Target is the input Green's function g
        target = batch[0]
        # Encode the g0 injection vector
        g0 = batch[2]
        injg0 = self.encoder_model.model.embedding(g0)
        injg0 = self.encoder_model.model.encode(injg0)
        # Predicting based on input g and injection vector injg0
        out = self.forward(batch[0], injg0)
        # Smoothing the output - probably not a good idea if only in convergence run!
        if self.config["smoothing"]==True:
            out = self.smoothing(out[:,None,:])
            print(">>> Using Gaussian Smoothing")
        else:
            out = out

        # Prediction of input g via Dyson equation
        sigma_sz = int(out.shape[1]/2)
        sigma = out[:,:sigma_sz] + 1j*out[:,sigma_sz:]
        g0 = g0[:,:sigma_sz] + 1j*g0[:,sigma_sz:]
        pred = 1/( 1/g0 - sigma )
        pred = torch.cat([pred.real, pred.imag],axis=1)
        
        loss = self.criterion_mse(pred[:,200:399], target[:,200:399])# + 200 * self.criterion_mse(pred[0,200:205], target[0,200:205]) + 500 * self.criterion_mse(pred[0,200:202], target[0,200:202])
        self.log('train_loss', loss.item())
        return loss

    def configure_optimizers(self) -> dict:
        # optimizer = torch.optim.AdamW(params=list(self.encoder_model.model.parameters()) + list(self.static_LW_model.model.parameters()), lr=self.config['learning_rate'], weight_decay=self.config['weight_decay'])
        # optimizer = torch.optim.AdamW(params=self.encoder_model.model.parameters(), lr=self.config['learning_rate'], weight_decay=self.config['weight_decay'])
        optimizer = torch.optim.RAdam(params=list(self.encoder_model.model.embedding.parameters()) + list(self.encoder_model.model.encode.parameters()), lr=self.config['learning_rate'], weight_decay=self.config['weight_decay'])
        return {"optimizer": optimizer}
    
    def load_model_state(self, PATH):
        checkpoint = torch.load(PATH, map_location='cuda:0')
        self.static_LW_model.load_state_dict(checkpoint['state_dict'])
