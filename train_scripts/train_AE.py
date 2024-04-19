
import sys
#TODO I hate the python import system. someone else fix this please.
sys.path.append('G:\\Codes\\LuttingerWard_from_ML\\code\\models')

from model_AE import AutoEncoder_01
sys.path.append('G:\\Codes\\LuttingerWard_from_ML\\code\\IO')
import DataMod_AE

import lightning as L
from lightning.pytorch.tuner import Tuner
from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor, EarlyStopping, StochasticWeightAveraging, GradientAccumulationScheduler, RichModelSummary, DeviceStatsMonitor
from lightning.pytorch.profilers import PyTorchProfiler
import torch
from pytorch_lightning.loggers import TensorBoardLogger
from argparse import ArgumentParser

import json
from os.path import dirname, abspath, join

torch.set_float32_matmul_precision("highest")
torch.set_default_dtype(torch.float64)


def main(args):
    config = json.load(open(join(dirname(abspath(__file__)),'../configs/confmod_auto_encoder.json')))
    torch.manual_seed(config['seed'])
    model = AutoEncoder_01(config) 
    dataMod = DataMod_AE(config)

    lr_monitor = LearningRateMonitor(logging_interval='step')
    logger = TensorBoardLogger("lightning_logs", name="VAE_Linear")
    val_ckeckpoint = ModelCheckpoint( # saved in `trainer.default_root_dir`/`logger.version`/`checkpoint_callback.dirpath`
            filename="{epoch}-{step}-{val_loss:.8f}",
            monitor="val_loss",
            mode="min",
            save_top_k=2,
            save_last =True
            )
    early_stopping = EarlyStopping(monitor="val_loss",patience=40)
    swa = StochasticWeightAveraging(swa_lrs=1e-8,annealing_epochs=40, swa_epoch_start=220,)
    accumulator = GradientAccumulationScheduler(scheduling={0: 128, 12: 64, 16: 32, 24: 16, 32: 8, 40: 4, 48: 1})
    callbacks = [lr_monitor, early_stopping, val_ckeckpoint, swa, accumulator]
    trainer = L.Trainer(enable_checkpointing=True, max_epochs=config["epochs"],
                      callbacks=callbacks, logger=logger, gradient_clip_val=0.5) #precision="16-mixed", 

    trainer.fit(model, datamodule=dataMod)