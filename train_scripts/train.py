#%%
import sys
sys.path.append('/gpfs/data/fs72150/springerd/Projects/LuttingerWard_from_ML/code/')
import torch
# from models import models as models
# from wrappers import wrapers
from torch.utils.data import DataLoader
import load_data
import datetime
from pytorch_lightning.loggers import TensorBoardLogger
import pytorch_lightning as pl
from pytorch_lightning.plugins.environments import LightningEnvironment
import json
import os


def train():
    ### JSON File contains full information about entire run (model, data, hyperparameters)
    ### TODO 
    MODEL_NAME = "CONVERGENCE_AUTO_ENCODER_1"
    config = json.load(open('../configs/confmod_auto_encoder.json'))[MODEL_NAME]
    # MODEL_NAME = "AUTO_ENCODER_1"
    # config = json.load(open('confmod_auto_encoder.json'))[MODEL_NAME]

    ''' Dataloading '''
    ### > Separate training and validation HDF5 files 
    ld = __import__("load_data", fromlist=['object'])
    train_set = getattr(ld, config["DATA_LOADER"])(config, data_type = "train", target_sample = None)
    validation_set = getattr(ld, config["DATA_LOADER"])(config, data_type = "valid")
    train_dataloader = DataLoader(train_set, batch_size=config["batch_size"], shuffle=True, num_workers=8, persistent_workers=True, pin_memory=True)
    validation_dataloader = DataLoader(validation_set, batch_size=config["batch_size"], shuffle=False, num_workers=8, persistent_workers=True, pin_memory=True)
    ### > Single HDF5 file containing training and validation data 
    # data_set = load_data.Dataset_ae(config)
    # # train_set, validation_set, unused_set = torch.utils.data.random_split(data_set, [int(data_set.__len__()*0.3), int(data_set.__len__()*0.05), int(data_set.__len__()*0.65)], generator=torch.Generator().manual_seed(42))
    # train_set, validation_set = torch.utils.data.random_split(data_set, [int(data_set.__len__()*0.8), int(data_set.__len__()*0.2)], generator=torch.Generator().manual_seed(42))
    # train_dataloader = DataLoader(train_set, batch_size=config["batch_size"], shuffle=True)
    # validation_dataloader = DataLoader(validation_set, batch_size=config["batch_size"], shuffle=True)


    ''' Model setup '''
    wrapers = __import__("wrappers.wrapers", fromlist=['object'])#.wrapers
    # import sys, inspect
    # for name, clazz in inspect.getmembers(wrapers):
    #     print(name)
    #     if inspect.isclass(clazz):
    #         print(clazz)
    model = getattr(wrapers, config["MODEL_WRAPER"])(config)
    
    ''' Model loading from save file '''
    if config["continue"] == True:
        SAVEPATH = config["SAVEPATH"]
        checkpoint = torch.load(SAVEPATH)
        model.load_state_dict(checkpoint['state_dict'])
        print(" >>> Loaded checkpoint")


    ''' Logging and saving '''
    DATA_NAME = os.path.splitext(os.path.basename(config["PATH_TRAIN"]))[0]

    PATH = ""
    CONFIGURATION = f"../saves/{DATA_NAME}/save_{config['MODEL_NAME']}_BS{config['batch_size']}_{datetime.datetime.now().date()}"
    # CONFIGURATION = f"../saves/save_{config['MODEL_NAME']}_Nodes{config['n_nodes']}_BS{config['batch_size']}_{datetime.datetime.now().date()}"
    logger = TensorBoardLogger(PATH, name=CONFIGURATION)


    # '''Define (pytorch_lightning) Trainer '''
    # ### > SLURM Training
    # trainer = pl.Trainer(max_epochs=config["epochs"], accelerator=config["device_type"], devices=config["devices"], num_nodes=config["num_nodes"], strategy='ddp', logger=logger)
    # ### > Jupyter Notebook Training
    # trainer = pl.Trainer(max_epochs=20, accelerator='gpu', devices=1, strategy='auto', logger=logger, plugins=[LightningEnvironment()])
    # ### > Jupyter Notebook CPU Training
    trainer = pl.Trainer(max_epochs=1, accelerator='cpu', devices=1, strategy='auto', logger=logger, plugins=[LightningEnvironment()])
    
    ''' Train '''
    trainer.fit(model, train_dataloader, validation_dataloader)

    ''' Saving configuration file into log folder ''' 
    LOGDIR = trainer.log_dir
    json_object = json.dumps(config, indent=4)
    with open(LOGDIR+"/config.json", "w") as outfile:
        outfile.write(json_object)




def main():
    train()

if __name__ == '__main__':
    main()
# %%
