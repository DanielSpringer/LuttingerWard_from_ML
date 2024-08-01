#%%
import sys, os
#sys.path.append('/gpfs/data/fs72150/springerd/Projects/LuttingerWard_from_ML/code/')
#sys.path.append(r'C:\Users\Daniel\OneDrive - TU Wien\Uni\6. Semester\Bachelorarbeit\autoencoder\LuttingerWard_from_ML\code')
sys.path.append(os.getcwd())
import torch
# from models import models as models
# from wrappers import wrapers
from torch.utils.data import DataLoader
import datetime
from pytorch_lightning.loggers import TensorBoardLogger
import pytorch_lightning as pl
from pytorch_lightning.plugins.environments import LightningEnvironment
import json
import numpy as np
from train_scripts.trainer_mode_enum import TrainerModes
import importlib
import src.load_data
from src.wrappers import wrapers
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping

trainer_mode = TrainerModes.JUPYTERGPU


def train():

    torch.set_float32_matmul_precision('high')
    ### JSON File contains full information about entire run (model, data, hyperparameters)
    ### TODO 
    MODEL_NAME = "AUTO_ENCODER_VERTEX"
    config = json.load(open(os.path.join('configs', 'confmod_auto_encoder.json')))[MODEL_NAME]
    # MODEL_NAME = "AUTO_ENCODER_1"
    # config = json.load(open('confmod_auto_encoder.json'))[MODEL_NAME]

    ''' Dataloading '''
    data_set = getattr(src.load_data, config["DATA_LOADER"])(config)
    train_set, validation_set = torch.utils.data.random_split(data_set, [int(len(data_set)*0.8), int(len(data_set)*0.2)], generator=torch.Generator().manual_seed(42))
    train_dataloader = DataLoader(train_set, batch_size=config["batch_size"], shuffle=True, num_workers=8, persistent_workers=True, pin_memory=True)
    validation_dataloader = DataLoader(validation_set, batch_size=config["batch_size"], num_workers=8, persistent_workers=True, pin_memory=True)


    ''' Model setup '''
    model = getattr(wrapers, config["MODEL_WRAPER"])(config)
    
    ''' Model loading from save file '''
    if config["continue"] == True:
        SAVEPATH = config["SAVEPATH"]
        checkpoint = torch.load(SAVEPATH)
        model.load_state_dict(checkpoint['state_dict'])
        print(" >>> Loaded checkpoint")


    ''' Logging and saving '''
    DATA_NAME = 'vertex'

    PATH = ''
    CONFIGURATION = os.path.join(os.getcwd(), "saves", DATA_NAME, f"save_{config['MODEL_NAME']}_BS{config['batch_size']}_{datetime.datetime.now().date()}")
    # CONFIGURATION = f"../saves/save_{config['MODEL_NAME']}_Nodes{config['n_nodes']}_BS{config['batch_size']}_{datetime.datetime.now().date()}"
    logger = TensorBoardLogger(PATH, name=CONFIGURATION)

    checkpoint_callback = ModelCheckpoint(
        save_top_k=10,  # Save top 10 models
        monitor='val_loss',  # Monitor validation loss
        mode='min',  # 'min' for minimizing the validation loss
        verbose=True
    )
    
    early_stopping_callback = EarlyStopping(
        monitor='val_loss',  # Monitor validation loss
        mode='min',  # 'min' for minimizing the validation loss
        patience=10,  # Number of epochs with no improvement after which training will be stopped
        verbose=True
    )



    # '''Define (pytorch_lightning) Trainer '''
    # ### > SLURM Training
    if trainer_mode == TrainerModes.SLURM:
        trainer = pl.Trainer(max_epochs=config["epochs"], accelerator=config["device_type"], devices=config["devices"], num_nodes=config["num_nodes"], strategy='ddp', logger=logger, callbacks=[checkpoint_callback, early_stopping_callback])
    
    # ### > Jupyter Notebook Training
    elif trainer_mode == TrainerModes.JUPYTERGPU:
        trainer = pl.Trainer(max_epochs=config["epochs"], accelerator='gpu', devices=1, strategy='auto', logger=logger, plugins=[LightningEnvironment()], callbacks=[checkpoint_callback, early_stopping_callback])

    # ### > Jupyter Notebook CPU Training
    elif trainer_mode == TrainerModes.JUPYTERCPU:
        trainer = pl.Trainer(max_epochs=1, accelerator='cpu', devices=1, strategy='auto', logger=logger, plugins=[LightningEnvironment()], callbacks=[checkpoint_callback, early_stopping_callback])

    else:
        raise NotImplementedError("trainer mode not implemented")
    
    ''' Train '''
    trainer.fit(model, train_dataloader, validation_dataloader)
    
    ''' Saving configuration file into log folder ''' 
    LOGDIR = trainer.log_dir
    json_object = json.dumps(config, indent=4)
    with open(LOGDIR+"/config.json", "w") as outfile:
        outfile.write(json_object)

# %%
