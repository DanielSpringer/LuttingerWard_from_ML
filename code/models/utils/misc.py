import torch.optim
from torch import nn

from .LossFunctions import *

# ==================== Helpers ====================
def activation_str_to_layer(activation_str_in: str) -> nn.Module:
        act = activation_str_in.lower()
        if act == "leakyrelu":
            return nn.LeakyReLU() 
        elif act == "silu":
            return nn.SiLU()
        elif act == "relu":
            return nn.ReLU()
        else:
            raise ValueError("unkown activation: " + act)
        
def loss_str_to_layer(loss_str_in: str, ylen=None) -> nn.Module:
        loss_str = loss_str_in.lower()
        if loss_str == "MSE".lower():
            return nn.MSELoss()
        elif loss_str == "WeightedMSE".lower():
            return WeightedLoss(ylen)
        elif loss_str == "WeightedMSE2".lower():
            return WeightedLoss2(ylen)
        elif loss_str == "ScaledMSE".lower():
            return ScaledLoss(ylen)
        elif loss_str == "WeightedScaledLoss".lower():
            return WeightedScaledLoss(ylen)
        else:
            raise ValueError("unkown activation: " + loss_str)
        
def optimizer_str_to_obj(model):
    opt_str = model.hparams['optimizer'].lower()
    if opt_str == "SGD":
        return torch.optim.SGD(model.parameters(), lr=model.lr,
                                momentum=model.hparams["SGD_momentum"],
                                weight_decay=model.hparams["SGD_weight_decay"],
                                dampening=model.hparams["SGD_dampening"],
                                nesterov=model.hparams["SGD_nesterov"])
    elif opt_str == "RMSprop":
        return torch.optim.RMSprop(model.parameters(), lr=model.lr,
                    momentum=model.hparams["SGD_momentum"],
                    weight_decay=model.hparams["SGD_weight_decay"],
                    alpha=model.hparams["RMSprop_alpha"])
    
    elif opt_str == "AdamW":
        return torch.optim.AdamW(model.parameters(), lr=model.lr)
    elif opt_str == "Adam":
        return torch.optim.Adam(model.parameters(), lr=model.lr)
    else:
        raise ValueError("unkown optimzer: " + model.hparams["optimizer"])
    
def dtype_str_to_type(dtype_str: str):
    if dtype_str.lower() == "float32":
        return torch.float32
    elif dtype_str.lower() == "float64":
        return torch.float64
    else:
        raise ValueError("unkown dtype: " + dtype_str)