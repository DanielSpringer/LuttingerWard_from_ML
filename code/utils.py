import torch.optim

# ==================== Helpers ====================
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
    return hparams

def activation_str_to_layer(activation_str_in: str) -> nn.Module:
        act = activation_str_in.lower()
        if act == "LeakyReLU":
            return nn.LeakyReLU() 
        elif act == "SiLU":
            return nn.SiLU()
        elif act == "ReLU":
            return nn.ReLU()
        else:
            raise ValueError("unkown activation: " + act)
        
def loss_str_to_layer(loss_str_in: str) -> nn.Module:
        loss_str = loss_str_in.lower()
        if loss_str == "MSE":
            return nn.MSELoss()
        elif loss_str == "WeightedMSE":
            return WeightedLoss(ylen)
        elif loss_str == "WeightedMSE2":
            return WeightedLoss2(ylen)
        elif loss_str == "ScaledMSE":
            return ScaledLoss(ylen)
        elif loss_str == "WeightedScaledLoss":
            return WeightedScaledLoss(ylen)
        else:
            raise ValueError("unkown activation: " + loss_str)
        
def optimizer_str_to_obj(model):
    opt_str = opt_str_in.hparams['optimizer'].lower()
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