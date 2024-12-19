#%%
import sys
sys.path.append('/home/fs71922/hessl3/data/ML_Luttinger/LuttingerWard_from_ML/src/')
import torch
# from models import models as models
# from wrappers import wrapers
from torch.utils.data import DataLoader
#sys.path.insert(1, '../src/');
import load_data
import datetime
from pytorch_lightning.loggers import TensorBoardLogger
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.plugins.environments import LightningEnvironment
import json
import os
import numpy as np
import h5py



def create_datasets(config):
    
    PATH = config["PATH_TRAIN"]
    f = h5py.File(PATH, 'r')
    #data = np.array(f["train"]['data'])
    #rint("************************************")
    #print("************************************")
    #print("************************************")
    #rint("************************************")
    #print("Size of dataset: ", data.shape)
    #print("************************************")
    #print("************************************")
    #print("************************************")
    #print("************************************")
    #train, validation = torch.utils.data.random_split(data, [int(data.__len__()*0.8), int(data.__len__())-int(data.__len__()*0.8)], generator=torch.Generator().manual_seed(42))

    if config["TRAINDATA"]==config["VALIDATIONDATA"]:
        data = np.array(f[config["TRAINDATA"]])
        train, validation = torch.utils.data.random_split(data, [int(data.__len__()*config["SPLIT"]), int(data.__len__())-int(data.__len__()*config["SPLIT"])], generator=torch.Generator().manual_seed(42))
    else:
        train = np.array(f[config["TRAINDATA"]])
        validation = np.array(f[config["VALIDATIONDATA"]])

    return train, validation


def train():
    ### JSON File contains full information about entire run (model, data, hyperparameters)
    ### TODO 
    MODEL_NAME = "AUTO_ENCODER_LW_AD_trans"
    config = json.load(open('../configs/confmod_auto_encoder.json'))[MODEL_NAME]

    ''' Dataloading '''
    train_data, validation_data = create_datasets(config)
    train_data = np.array(train_data)
    validation_data = np.array(validation_data)

    ### > Single HDF5 file containing training and validation data 
    ld = __import__("load_data", fromlist=['object'])
    # data_set = load_data.Dataset_ae(config)
    train_set = getattr(ld, config["DATA_LOADER"])(config, train_data)
    validation_set = getattr(ld, config["DATA_LOADER"])(config, validation_data)

    train_dataloader = DataLoader(train_set, batch_size=config["batch_size"], shuffle=True)
    validation_dataloader = DataLoader(validation_set, batch_size=config["batch_size"], shuffle=True)


    ''' Model setup '''
    wrapers = __import__("wrappers.wrapers", fromlist=['object'])#.wrapers
    model = getattr(wrapers, config["MODEL_WRAPER"])(config)

    ''' Model loading from save file '''
    if config["continue"] == True:
        SAVEPATH = config["SAVEPATH"]
        checkpoint = torch.load(SAVEPATH)
        model.load_state_dict(checkpoint['state_dict'])
        print(" >>> Loaded checkpoint")



    ''' Logging and saving '''
    DATA_NAME = os.path.splitext(os.path.basename(config["PATH_TRAIN"]))[0]
    print(" TRAIN DATA (slurm relevance) ")
    print(config["PATH_TRAIN"])
    print(DATA_NAME)
    print(MODEL_NAME)
    
    PATH = "/home/fs71922/hessl3/data/ML_Luttinger/saves/"
    CONFIGURATION = f"{PATH}/SigmaTrans/AD_GELU/{DATA_NAME}/save_{config['MODEL_NAME']}_BS{config['batch_size']}_{datetime.datetime.now().date()}"
    logger = TensorBoardLogger(PATH, name=CONFIGURATION)
    
    early_stop_callback = EarlyStopping(monitor="val_loss", mode="min", min_delta=0.00, patience=20, verbose=False)
    checkpoint_callback = ModelCheckpoint(save_top_k=-1)

    
    # ### '''Define (pytorch_lightning) Trainer '''
    # ### > SLURM Training
    trainer = pl.Trainer(max_epochs=config["epochs"], 
                        accelerator=config["device_type"], 
                        devices=config["devices"], 
                        num_nodes=config["num_nodes"], 
                        #strategy='ddp', 
                        strategy='ddp_find_unused_parameters_true',
                        logger=logger
                        # callbacks=[checkpoint_callback]
                        )
    # ### > Jupyter Notebook Training
    # trainer = pl.Trainer(max_epochs=config["epochs"], 
    #                      accelerator='gpu', 
    #                      devices=1, 
    #                      strategy='auto', 
    #                      logger=logger, 
    #                      # log_every_n_steps=1, 
    #                      plugins=[LightningEnvironment()], 
    #                      callbacks=[checkpoint_callback]
    #                     )

    # ### > Jupyter Notebook CPU Training
    # trainer = pl.Trainer(max_epochs=20, accelerator='cpu', devices=1, strategy='auto', logger=logger, plugins=[LightningEnvironment()])
    
    # ''' Train '''
    trainer.fit(model, train_dataloader, validation_dataloader)
    # trainer.fit(model, train_dataloader)
    
    
    # ### ''' Saving configuration file into log folder ''' 
    LOGDIR = trainer.log_dir
    json_object = json.dumps(config, indent=4)
    with open(LOGDIR+"/config.json", "w") as outfile:
        outfile.write(json_object)




def main():
    train()

if __name__ == '__main__':
    main()
# %%
