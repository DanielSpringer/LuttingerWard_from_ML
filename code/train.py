import torch
import models
from torch.utils.data import DataLoader
import load_data
import datetime
from pytorch_lightning.loggers import TensorBoardLogger
import pytorch_lightning as pl
from pytorch_lightning.plugins.environments import LightningEnvironment


def train():
    config = {}
    config["MODEL_NAME"] = "GreenGNN"
    config["n_nodes"] = 100
    config["weird"] = False
    config["PATH_TRAIN"] = "../data/batch1.hdf5"
    config["batch_size"] = 1
    config["learning_rate"] = 1e-4
    config["weight_decay"] = 1e-5

    data_set = load_data.Dataloader_graph(config)
    # train_set, validation_set, unused_set = torch.utils.data.random_split(data_set, [int(data_set.__len__()*0.3), int(data_set.__len__()*0.05), int(data_set.__len__()*0.65)], generator=torch.Generator().manual_seed(42))
    train_set, validation_set = torch.utils.data.random_split(data_set, [int(data_set.__len__()*0.8), int(data_set.__len__()*0.2)], generator=torch.Generator().manual_seed(42))
    train_dataloader = DataLoader(train_set, batch_size=config["batch_size"], shuffle=True)
    validation_dataloader = DataLoader(validation_set, batch_size=config["batch_size"], shuffle=True)

    model = models.model_wraper_gnn(config)
    # SAVEPATH = "../saves/save_weightedLoss_GreenGNN_Nodes100_2024-02-21/version_2/checkpoints/epoch=14-step=600000.ckpt"
    # checkpoint = torch.load(SAVEPATH)#, map_location=torch.device('cpu'))
    # model.load_state_dict(checkpoint['state_dict'])
    # print("...apparently loaded model.")

    PATH = ""
    # CONFIGURATION = f"../saves/UGrid/save_{config['MODEL_NAME']}_{datetime.datetime.now().date()}"
    CONFIGURATION = f"../saves/save_weightedLoss2_{config['MODEL_NAME']}_Nodes{config['n_nodes']}_{datetime.datetime.now().date()}"
    logger = TensorBoardLogger(PATH, name=CONFIGURATION)


    # ### SLURM Training
    trainer = pl.Trainer(max_epochs=20, accelerator='gpu', devices=2, num_nodes=1, strategy='ddp', logger=logger)
    # ### Jupyter Notebook Training
    # trainer = pl.Trainer(max_epochs=20, accelerator='gpu', devices=1, strategy='auto', logger=logger, plugins=[LightningEnvironment()])
    # ### Jupyter Notebook CPU Training
    # trainer = pl.Trainer(max_epochs=1, accelerator='cpu', devices=1, strategy='auto', logger=logger, plugins=[LightningEnvironment()])

    trainer.fit(model, train_dataloader, validation_dataloader)


def main():
    train()

if __name__ == '__main__':
    main()