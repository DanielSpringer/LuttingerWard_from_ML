
import sys
from os.path import dirname, abspath, join

#TODO I hate the python import system. someone else fix this please.
model_path = join(dirname(__file__),'../code/models')
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
import neptune

import json

torch.set_float32_matmul_precision("high")
torch.set_default_dtype(torch.float64)


def main(args):
    i = 0
    config_path = join(dirname(__file__),'../configs/confmod_AE_FC.json')
    config = json.load(open(config_path))
    
    torch.manual_seed(config['seed'])
    for AE_layers in [1,2,3]:
        for FC_layers in [0,1,2,3,4,5,6,7,8]:
            if True:
                print(i)
                config['AE_layers'] = AE_layers
                config['FC_layers'] = FC_layers
                model = AE_FC_01(config) 
                dataMod = DataMod_FC(config)

                model_version = neptune.init_model(key=f"AE{AE_layers}FC{FC_layers}",name=config['MODEL_NAME'],project="stobbe.julian/LW-AEpFC")
                model_version["model/signature"].upload(config_path)
                model_script = model.to_torchscript()
                torch.jit.save(model_script, "tmp_model.pt")
                model_version["model/definition"].upload("tmp_model.pt")

                lr_monitor = LearningRateMonitor(logging_interval='step')
                TB_logger = TensorBoardLogger(
                    save_dir="TB_logs", 
                    name=config['MODEL_NAME'], 
                    default_hp_metric=False,
                    version=i)
                
                neptune_logger = NeptuneLogger(    
                    project="stobbe.julian/LW-AEpFC",
                    name=config['MODEL_NAME'],
                    description="Simple Autoencoder structure with FC layers. Reconstruction Loss of input is taken into account.",
                    tags=["test"],
                    )
                val_ckeckpoint = ModelCheckpoint( # saved in `trainer.default_root_dir`/`logger.version`/`checkpoint_callback.dirpath`
                        filename="{epoch}-{step}-{val_loss:.8f}",
                        monitor="val/loss",
                        mode="min",
                        save_top_k=2,
                        save_last =True
                        )
                early_stopping = EarlyStopping(monitor="val/loss",patience=40)
                swa = StochasticWeightAveraging(swa_lrs=1e-8,annealing_epochs=40, swa_epoch_start=220)
                accumulator = GradientAccumulationScheduler(scheduling={0: 1024, 8: 128, 16: 64, 24: 32, 32: 16, 40: 8, 48: 4, 56: 1})
                callbacks = [lr_monitor, early_stopping, val_ckeckpoint, swa, accumulator]
                trainer = L.Trainer(enable_checkpointing=True, max_epochs=config["epochs"], #
                                callbacks=callbacks, logger=neptune_logger, gradient_clip_val=0.5)

                trainer.fit(model, datamodule=dataMod)
                neptune_logger.log_model_summary(model=model, max_depth=-1)

                model_script = model.to_torchscript()
                torch.jit.save(model_script, "tmp_model.pt")
                model_version["model/fitted"].upload("tmp_model.pt")

                model_version["validation/acc"] = trainer.callback_metrics["val/acc"]
            i += 1

if __name__ == '__main__':
    parser = ArgumentParser()
    args = parser.parse_args()
    main(args)