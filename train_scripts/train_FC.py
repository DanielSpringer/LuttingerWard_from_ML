
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

def load_checkpoint(run: neptune.Run, epoch: int):
    checkpoint_name = f"epoch_{epoch}"
    ext = run["checkpoints"][checkpoint_name].fetch_extension()
    run["checkpoints"][checkpoint_name].download()  # Download the checkpoint
    run.wait()
    checkpoint = torch_load(f"{checkpoint_name}.{ext}")  # Load the checkpoint
    return checkpoint

def main(args):
    i = 0
    config_path = join(dirname(__file__),'../configs/confmod_AE_FC.json')
    config = json.load(open(config_path))
    
    torch.manual_seed(config['seed'])
    for AE_layers in [1,2,3]:
        for FC_layers in [0,1,2,3,4,5,6,7,8]:
            if i > 4:
                print(i)
                config['AE_layers'] = AE_layers
                config['FC_layers'] = FC_layers
                model = AE_FC_01(config) 
                dataMod = DataMod_FC(config)
                if not (i == 5): 
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
                if i == 5:
                    neptune_logger = NeptuneLogger(    
                        project="stobbe.julian/LW-AEpFC",
                        name=config['MODEL_NAME'],
                        description="Simple Autoencoder structure with FC layers. Reconstruction Loss of input is taken into account.",
                        tags=["test"],
                        capture_hardware_metrics=False,
                        capture_stdout=False,
                        with_id="LWAEP-86"
                        )
                else:
                    neptune_logger = NeptuneLogger(    
                        project="stobbe.julian/LW-AEpFC",
                        name=config['MODEL_NAME'],
                        description="Simple Autoencoder structure with FC layers. Reconstruction Loss of input is taken into account.",
                        tags=["test"],
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
                swa = StochasticWeightAveraging(swa_lrs=1e-8,annealing_epochs=40, swa_epoch_start=220)
                accumulator = GradientAccumulationScheduler(scheduling={0: 512, 8: 128, 16: 64, 24: 32, 32: 16, 40: 8, 48: 4, 56: 1})
                callbacks = [lr_monitor, early_stopping, val_ckeckpoint, swa, accumulator]
                trainer = L.Trainer(enable_checkpointing=True, max_epochs=config["epochs"], #
                                callbacks=callbacks, logger=neptune_logger, gradient_clip_val=0.5)
                if  i == 5:
                    tmp_path = "G:/Codes/LuttingerWard_from_ML/.neptune/AE_FC_scale_01/LWAEP-86/checkpoints/epoch=104-step=1796678-val_loss=0.00000000.ckpt"
                    trainer.fit(model, datamodule=dataMod, ckpt_path=tmp_path)
                else:
                    trainer.fit(model, datamodule=dataMod)
                neptune_logger.log_model_summary(model=model, max_depth=-1)
                if not (i == 5): 
                    model_script = model.to_torchscript()
                    torch.jit.save(model_script, "tmp_model.pt")
                    model_version["model/fitted"].upload("tmp_model.pt")
                neptune_logger.log_model_summary(model=model, max_depth=-1)
                neptune_logger._run_instance.stop()
                #model_version["validation/acc"] = trainer.callback_metrics["val/acc"]
            i += 1

if __name__ == '__main__':
    parser = ArgumentParser()
    args = parser.parse_args()
    main(args)