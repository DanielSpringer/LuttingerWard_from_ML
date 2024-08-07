
import sys
from os.path import dirname, abspath, join

#TODO I hate the python import system. someone else fix this please.
sys.path.append(join(dirname(__file__),'../code/models'))
from model_AE import AutoEncoder_01
sys.path.append(join(dirname(__file__),'../code/models/IO'))
from DataMod_AE import *

import pytorch_lightning as L
from pytorch_lightning.tuner import Tuner
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor, EarlyStopping, StochasticWeightAveraging, GradientAccumulationScheduler, RichModelSummary, DeviceStatsMonitor
from pytorch_lightning.profilers import PyTorchProfiler
import torch
from pytorch_lightning.loggers import TensorBoardLogger
from argparse import ArgumentParser


import json

torch.set_float32_matmul_precision("highest")
torch.set_default_dtype(torch.float64)


def main(args):
    config = json.load(open(join(dirname(__file__),'../configs/confmod_auto_encoder.json')))
    torch.manual_seed(config['seed'])
    pr = neptune.init_project("LW-AEpFC")
    models_table = pr.fetch_models_table().to_pandas()
    model_key = "AETEST"+mode.upper()
    if not any(models_table["sys/id"].str.contains("LWAEP-"+model_key)):
        modelN = neptune.init_model(key=model_key,name=config['MODEL_NAME'],project="stobbe.julian/LW-AEpFC")
    model_version = neptune.init_model_version(model=f"LWAEP-AETEST01",name=f"EXAMPLE",project="stobbe.julian/LW-AEpFC")
    model = AutoEncoder_01(config) 
    dataMod = DataMod_AE(config)

    lr_monitor = LearningRateMonitor(logging_interval='step')
    logger = TensorBoardLogger("lightning_logs", name=config['MODEL_NAME'])
    neptune_logger = NeptuneLogger(    
                                    project="stobbe.julian/LW-AEpFC",
                                    name=config['MODEL_NAME'],
                                    description="Simple Autoencoder. Example run.",
                                    tags=["AE", "Test"],
                                    capture_hardware_metrics=False,
                                    capture_stdout=False,
                                    )
    val_ckeckpoint = ModelCheckpoint( # saved in `trainer.default_root_dir`/`logger.version`/`checkpoint_callback.dirpath`
            filename="{epoch}-{step}-{val_loss:.8f}",
            monitor="val/loss",
            mode="min",
            save_top_k=2,
            save_last =True
            )
    early_stopping = EarlyStopping(monitor="val/loss",patience=40)
    swa = StochasticWeightAveraging(swa_lrs=1e-8,annealing_epochs=40, swa_epoch_start=220,)
    accumulator = GradientAccumulationScheduler(scheduling={0: 128, 12: 64, 16: 32, 24: 16, 32: 8, 40: 4, 48: 1})
    callbacks = [lr_monitor, early_stopping, val_ckeckpoint, swa, accumulator]
    trainer = L.Trainer(enable_checkpointing=True, max_epochs=config["epochs"],
                      callbacks=callbacks, logger=[neptune_logger,logger], gradient_clip_val=0.5) #precision="16-mixed", 

    trainer.fit(model, datamodule=dataMod)                
    model_version["run/id"] = neptune_logger._run_instance["sys/id"].fetch()
    neptune_logger.log_model_summary(model=model, max_depth=-1)
    neptune_logger._run_instance.stop()

if __name__ == '__main__':
    parser = ArgumentParser()
    args = parser.parse_args()
    main(args)