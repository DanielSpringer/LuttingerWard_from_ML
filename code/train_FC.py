from model_FC import SimpleFC
import lightning as L
from lightning.pytorch.tuner import Tuner
from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor, EarlyStopping, StochasticWeightAveraging, RichModelSummary, DeviceStatsMonitor
from lightning.pytorch.profilers import PyTorchProfiler
import torch
from pytorch_lightning.loggers import TensorBoardLogger
torch.set_float32_matmul_precision("high")

if __name__ == "__main__":

    hparams = {
            "model_name": "FullCNN",
            "fc_dims": [201, 200, 200, 200, 200, 200, 200, 200],
            "dropout_in": 0.0,
            "dropout": 0.4,
            "with_batchnorm": True,
            "lr": 0.01,
            "batch_size": 64,
            "train_path": "../data/batch1.hdf5",
            "optimizer": "Adam",
            "SGD_weight_decay": 0.0,
            "SGD_momentum": 0.9,
            "SGD_dampening": 0.0,
            "SGD_nesterov": False,
               }
    model = SimpleFC(hparams)
    lr_monitor = LearningRateMonitor(logging_interval='step')
    early_stopping = EarlyStopping(
            monitor="val_loss",
            patience=50)
    logger = TensorBoardLogger("lightning_logs", name=hparams["model_name"])
    val_ckeckpoint = ModelCheckpoint( # saved in `trainer.default_root_dir`/`logger.version`/`checkpoint_callback.dirpath`
            filename="{epoch}-{step}-{val_loss:.8f}",
            monitor="val_loss",
            mode="min",
            save_top_k=10,
            save_last =True
            )
    swa = StochasticWeightAveraging(swa_lrs=0.0001,
                                    swa_epoch_start=100,
                                    )
    callbacks = [val_ckeckpoint, lr_monitor, early_stopping, swa, RichModelSummary()] #, DeviceStatsMonitor()
    profiler = PyTorchProfiler()
    trainer = L.Trainer(enable_checkpointing=True, max_epochs=600, accelerator="gpu", callbacks=callbacks, logger=logger)
    trainer.fit(model)