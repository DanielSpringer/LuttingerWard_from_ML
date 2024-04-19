import torch
import code.models.models as models
from torch.utils.data import DataLoader
import load_data
import datetime
from pytorch_lightning.loggers import TensorBoardLogger
import pytorch_lightning as pl
from pytorch_lightning.plugins.environments import LightningEnvironment


def train():
    config = {}
    config["MODEL_NAME"] = "GreenGNN"
    config["PATH_TRAIN"] = "../data/batch1.hdf5"
    config["batch_size"] = 1
    config["learning_rate"] = 1e-4
    config["weight_decay"] = 1e-5

    data_set = load_data.Dataloader_graph(config)
    train1_set, validation_set = torch.utils.data.random_split(data_set, [int(data_set.__len__()*0.8), int(data_set.__len__()*0.2)], generator=torch.Generator().manual_seed(42))
    train_dataloader = DataLoader(train1_set, batch_size=config["batch_size"], shuffle=True)
    validation_dataloader = DataLoader(validation_set, batch_size=config["batch_size"], shuffle=True)

    model = models.model_wraper_gnn(config)
    # SAVEPATH = "/gpfs/data/fs72150/springerd/Projects/LuttingerWard_from_ML/saves/save_GreenGNN_2023-12-16/version_2/checkpoints/"
    # checkpoint = torch.load(PATH)#, map_location=torch.device('cpu'))
    # model.load_state_dict(checkpoint['state_dict'])
    # print("...apparently loaded model.")

    PATH = ""
    CONFIGURATION = f"../saves/save_{config['MODEL_NAME']}_{datetime.datetime.now().date()}"
    logger = TensorBoardLogger(PATH, name=CONFIGURATION)

    # ### SLURM Training
    # trainer = pl.Trainer(max_epochs=20, accelerator='gpu', devices=2, num_nodes=8, strategy='ddp', logger=logger)
    trainer = pl.Trainer(max_epochs=20, devices=128, num_nodes=1, strategy='ddp', logger=logger)

    # ### Jupyter Notebook Training
    # # trainer = pl.Trainer(max_epochs=20, accelerator='gpu', devices=1, strategy='auto', logger=logger, plugins=[LightningEnvironment()])

    trainer.fit(model, train_dataloader, validation_dataloader)


def main():
    train()

if __name__ == '__main__':
    main()