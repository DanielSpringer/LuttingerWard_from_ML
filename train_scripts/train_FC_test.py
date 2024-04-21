
import sys
from os.path import dirname, abspath, join

#TODO I hate the python import system. someone else fix this please.
sys.path.append(join(dirname(__file__),'../code/models'))
from model_AE_FC import AE_FC_01
sys.path.append(join(dirname(__file__),'../code/models/IO'))
from DataMod_FC import *

import pytorch_lightning as L
from pytorch_lightning.tuner import Tuner
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor, EarlyStopping, StochasticWeightAveraging, GradientAccumulationScheduler,  DeviceStatsMonitor
from pytorch_lightning.profilers import PyTorchProfiler
import torch
from pytorch_lightning.loggers import TensorBoardLogger, NeptuneLogger
from argparse import ArgumentParser

import json

torch.set_float32_matmul_precision("medium")


if __name__ == '__main__':
    config = json.load(open(join(dirname(__file__),'../configs/confmod_AE_FC_test.json')))
    torch.manual_seed(config['seed'])
    model = AE_FC_01(config) 
    dataMod = DataMod_FC(config)
    neptune_logger = NeptuneLogger(
                    project="stobbe.julian/LW-AEpFC",
                    name="TEST_"+config['MODEL_NAME'],
                    description="",
                    tags=["code_test"],
                    )


    lr_monitor = LearningRateMonitor(logging_interval='step')
    accumulator = GradientAccumulationScheduler(scheduling={0: 128, 12: 64, 16: 32, 24: 16, 32: 8, 40: 4, 48: 1})
    callbacks = [lr_monitor, accumulator]
    trainer = L.Trainer(enable_checkpointing=False, max_epochs=config["epochs"],
                    callbacks=callbacks, logger=neptune_logger) #[TB_logger, neptune_logger]
    trainer.fit(model, datamodule=dataMod)