import lightning as L
from lightning.pytorch.tuner import Tuner
from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor, EarlyStopping, StochasticWeightAveraging
import torch
torch.set_float32_matmul_precision("medium")
import models
import load_data as ld
from torch.utils.data import DataLoader
    
if __name__ == "__main__":

    config = {}
    config["PATH_TRAIN"] = "D:/data_test1.hdf5"
    data_set = ld.Dataset_baseline(config)
    config["MODEL_NAME"] = "auto_encoder"

    config["in_dim"] = data_set.data_in.shape[1]
    config["out_dim"] = data_set.data_target.shape[1]
    config["batch_size"] = 64
    config["learning_rate"] = 1e-3
    config["weight_decay"] = 1e-5
    config["embedding_dim"] = 128 #int(config["in_dim"]/2)
    config["hidden1_dim"] = int(config["embedding_dim"]/2)
    config["hidden2_dim"] = int(config["embedding_dim"]/4)
    config["encoder_dim"] = int(config["embedding_dim"]/8)

    train1_set, validation_set = torch.utils.data.random_split(data_set, [int(data_set.__len__()*0.8), int(data_set.__len__()*0.2)], generator=torch.Generator().manual_seed(42))
    train_dataloader = DataLoader(train1_set, batch_size=config["batch_size"], shuffle=False, num_workers=1,persistent_workers=True)
    validation_dataloader = DataLoader(validation_set, batch_size=config["batch_size"], shuffle=False, num_workers=1,persistent_workers=True)
    model = models.model_wraper_AE(config)

    lr_monitor = LearningRateMonitor(logging_interval='step')
    early_stopping = EarlyStopping(monitor="val_loss",patience=50)
    val_ckeckpoint = ModelCheckpoint(filename="{epoch}-{step}-{val_loss:.8f}",monitor="val_loss", mode="min",save_top_k=5)
    swa = StochasticWeightAveraging(swa_lrs=0.0001,swa_epoch_start=50,)
    callbacks = [val_ckeckpoint, lr_monitor, early_stopping, swa]
    trainer = L.Trainer(enable_checkpointing=True, max_epochs=500, accelerator="gpu", callbacks=callbacks) #precision="16-mixed", 
    trainer.fit(model, train_dataloader, validation_dataloader)
    trainer = pl.Trainer(max_epochs=200, devices=1, strategy='auto', logger=logger, plugins=[LightningEnvironment()])