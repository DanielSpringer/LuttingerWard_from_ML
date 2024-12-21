import datetime
import glob
import importlib
import json
import os

from enum import Enum
from pathlib import Path
from typing import Any, Generic, TypeVar

import numpy as np
from lightning.pytorch import Trainer
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.plugins.environments import LightningEnvironment

import torch
from torch.utils.data import DataLoader, random_split

from src.config import Config
from src.load_data import FilebasedDataset
from src.wrapper import BaseWrapper


R = TypeVar('R', bound=BaseWrapper)
S = TypeVar('S', bound=FilebasedDataset)
T = TypeVar('T', bound=Config)


class TrainerModes(Enum):
    SLURM = 1
    JUPYTERGPU = 2
    JUPYTERCPU = 3
    LOCAL = 4


class BaseTrainer(Generic[T, S, R]):
    config_cls: type[T] = T.__bound__

    def __init__(self, project_name: str, config_name: str, subconfig_name: str|None = None, 
                 config_dir: str = 'configs', config_kwargs: dict[str, Any] = {}):
        """
        Main class for training and using a model.

        Parameters
        ----------
        project_name : str
            Some project name under which the results are saved.
        config_name : str
            Name of the config-file to load.
        subconfig_name : str | None, optional
            Name of a subsection in the config-file.\n
            Required if the file contains multiple config-sections. (defaults to None)
        config_dir : str, optional
            Path to the directory of the config-files. (defaults to `'configs'`)
        config_kwargs : dict[str, Any], optional
            Additional kwargs that are applied to loading the config-file.\n
            Allows to overwrite attributes from the config-file. (defaults to {})
        """
        conf_classname = repr(self.__orig_bases__[0]).split('[')[1].split(',')[0].split('.')[-1]
        conf_module = importlib.import_module('src.config')
        self.config_cls = getattr(conf_module, conf_classname, Config)

        self.project_name = project_name
        self.subconfig_name = subconfig_name

        self.config: T = self.config_cls.from_json(config_name, subconfig_name, config_dir, **config_kwargs)
        self.dataset: S = self.config.dataset(self.config)
        self.data_loader: type[DataLoader] = self.config.data_loader

        self.wrapper: R = None
    
    @property
    def input_size(self) -> int|np.ndarray:
        """
        Get the input size of the model. 
        Overwrite when having a different dataset item structure.
        """
        return self.dataset[0][0].shape[0]

    def train(self, train_mode: TrainerModes) -> None:
        """
        Main method to train the model.

        Parameters
        ----------
        train_mode : TrainerModes
            TrainerModes configure the lightning-trainer for different environments 
            (e.g. cluster, local, jupyter).
        """
        self.pre_train()
        self._train(train_mode)
        self.post_train()
    
    def predict(self, new_data_path: str, save_path: str|None = None,
                load_model: bool = False, **kwargs) -> np.ndarray:
        """
        Performs a prediction using new data. If already trained, uses the trained model, 
        otherwise loads a model from disk.

        Parameters
        ----------
        new_data_path : str
            Path to file containing the new data to predict on.
        save_path : str | None, optional
            Either path to a model checkpoint in the saves-folder 
            (like `/<project_name>/<version>/checkpoints/<chjeckpoint>.ckpt`) 
            or to a saves-folder containing the run config (ending as `/<project_name>/<version>/`).\n
            In the latter case, the newest checkpoint in the directory is loaded.\n
            Path can either be absolute or relative to `/<project_name>/`.\n
            If `None` `self.config.save_path` is used. (defaults to None)
        load_model : bool, optional
            Force load a model (loads a model even if a model is already loaded). (defaults to False)

        Returns
        -------
        np.ndarray
            Prediction as numpy array.
        """
        new_data = self.dataset.load_from_file(new_data_path)
        if self.wrapper is None or load_model or save_path:
            self.load_model(save_path)
        self.wrapper.model.eval()
        return self._predict(new_data, new_data_path, **kwargs)

    def pre_train(self) -> None:
        """
        Overwrite to perform operations before starting the training.
        """
        pass

    def post_train(self) -> None:
        """
        Overwrite to perform operations after finishing training.
        """
        pass

    def create_data_loader(self) -> tuple[DataLoader, DataLoader]:
        """
        Create DataLoader.\n
        Overwrite for custom data loading.
        """
        train_set, validation_set = random_split(self.dataset, [1 - self.config.test_ratio, self.config.test_ratio], 
                                                 generator=torch.Generator().manual_seed(42))
        train_dataloader = self.data_loader(train_set, batch_size=self.config.batch_size, shuffle=True, 
                                            num_workers=8, persistent_workers=True, pin_memory=True)
        validation_dataloader = self.data_loader(validation_set, batch_size=self.config.batch_size, 
                                                 num_workers=8, persistent_workers=True, pin_memory=True)
        return train_dataloader, validation_dataloader

    def set_logging(self) -> TensorBoardLogger:
        """
        Set TensorBoardLogger and ModelCheckpoint.\n
        Overwrite for custom logging.
        """
        save_path = self.save_prefix + str(datetime.datetime.now().date())
        logger = TensorBoardLogger(self.get_full_save_path(''), name=save_path)
        self.config.save_path = logger.log_dir
        return logger

    def _predict(self, new_data: np.ndarray, new_data_path: str, **kwargs) -> np.ndarray:
        """
        Perform a prediction using new data.\n
        Overwrite for customized prediction.

        Parameters
        ----------
        new_data : np.ndarray
            New data to predict on.
        new_data_path : str
            Path to file containing the new data to predict on.
        **kwargs :
            Additional keyword arguments received from the main `predict`-method, 
            that can be used when overwriting this method.

        Returns
        -------
        np.ndarray
            Prediction as numpy array.
        """
        input = torch.tensor(new_data, dtype=torch.float32).to('cpu')
        pred = self.wrapper(input).detach().numpy()
        p = self.get_full_save_path() / 'predictions'
        p.mkdir(exist_ok=True)
        fp = p / f'{Path(new_data_path).stem}.npy'
        np.save(fp, pred)
        return pred
    
    def get_full_save_path(self, save_path: str|None = None) -> Path:
        """
        Return the absolute save-path from a path relative to `<saves_dir>/<project_name>/`.

        Parameters
        ----------
        save_path : str | None, optional
            Path relative to `<saves_dir>/<project_name>/`.
            If `None` `self.config.save_path` is used. (defaults to None)

        Returns
        -------
        Path
            The absolute save-path.
        """
        if save_path is None:
            save_path = self.config.save_path
        return self.config.base_dir / self.config.save_dir / self.project_name / save_path
    
    @property
    def save_prefix(self) -> str:
        """
        Get the name of the saves-subfolder for this training-run.\n
        Overwrite for custom naming.
        """
        return f"save_{self.subconfig_name}_BS{self.config.batch_size}_"

    def load_config(self, save_path: str) -> None:
        """
        Sets the used Config from a config-JSON file.

        Parameters
        ----------
        save_path : str
            File-path of the config-JSON.
        """
        self.config = self.config_cls.from_json('config.json', directory=save_path, save_path=save_path)

    def load_model(self, save_path: str|None = None) -> None:
        """
        Loads a model and corresponding config into the trainer. 
        Loads the model either from a given checkpoint filepath 
        or from the newest checkpoint in a given directory.

        Parameters
        ----------
        save_path : str | None, optional
            Either path to a model checkpoint in the saves-folder 
            (like `/<project_name>/<version>/checkpoints/<chjeckpoint>.ckpt`) 
            or to a saves-folder containing the run config (ending as `/<project_name>/<version>/`).\n
            In the latter case, the newest checkpoint in the directory is loaded.\n
            Path can either be absolute or relative to `/<project_name>/`.\n
            If `None` `self.config.save_path` is used. (defaults to None)
        """
        assert save_path or self.config.save_path, ("No save_path found in the config-file, "
                                                    "please provide a save_path.")
        save_path: Path = Path(save_path)
        if not save_path.is_absolute():
            save_path = self.get_full_save_path(save_path or self.config.save_path)
        if not save_path.suffix == '.ckpt':
            ckpt_files = glob.glob((save_path / '**/*.ckpt').as_posix(), recursive=True)
            save_path = max(ckpt_files, key=os.path.getctime)
        print(f" >>> Load checkpoint from '{save_path}'")
        self.load_config(Path(save_path).parent.parent.as_posix())
        device = self.get_device_from_accelerator(self.config.device_type)
        checkpoint = torch.load(save_path, map_location=device, weights_only=True)
        self.wrapper = self.config.model_wrapper(self.config, self.input_size)
        self.wrapper.load_state_dict(checkpoint['state_dict'])

    def _train(self, train_mode: TrainerModes) -> None:
        """
        Core method for running the training.\n
        Overwrite for custom training process.

        Parameters
        ----------
        train_mode : TrainerModes
            TrainerModes configure the lightning-trainer for different environments 
            (e.g. cluster, local, jupyter).
        """
        ''' Dataloading '''
        train_dataloader, validation_dataloader = self.create_data_loader()

        ''' Model setup '''
        in_dim = self.input_size
        self.wrapper = self.config.model_wrapper(self.config, in_dim)
        
        ''' Model loading from save file '''
        if self.config.resume == True:
            checkpoint = torch.load(self.get_full_save_path())
            self.wrapper.load_state_dict(checkpoint['state_dict'])
            print(" >>> Loaded checkpoint")

        ''' Logging and checkpoint saving '''
        logger = self.set_logging()

        ''' Set pytorch_lightning Trainer '''
        callbacks = [self.config.get_model_checkpoint(), *self.config.get_callbacks()]
        if train_mode == TrainerModes.SLURM:
            trainer = Trainer(max_epochs=self.config.epochs, accelerator=self.config.device_type, devices=self.config.devices, 
                                 num_nodes=self.config.num_nodes, strategy='ddp', logger=logger, callbacks=callbacks)
        elif train_mode == TrainerModes.JUPYTERGPU:
            trainer = Trainer(max_epochs=self.config.epochs, accelerator='gpu', devices=1, strategy='auto', 
                                 logger=logger, plugins=[LightningEnvironment()], callbacks=callbacks)
        elif train_mode == TrainerModes.JUPYTERCPU:
            trainer = Trainer(max_epochs=1, accelerator='cpu', devices=1, strategy='auto', logger=logger, 
                                 plugins=[LightningEnvironment()], callbacks=callbacks)
        elif train_mode == TrainerModes.LOCAL:
            trainer = Trainer(max_epochs=self.config.epochs, accelerator=self.config.device_type, devices=1, strategy='auto', 
                                 logger=logger, plugins=[LightningEnvironment()], callbacks=callbacks)
        
        ''' Train '''
        trainer.fit(self.wrapper, train_dataloader, validation_dataloader)
        
        ''' Saving config-file ''' 
        json_object = json.dumps(self.config.as_dict(), indent=4)
        with open(self.get_full_save_path() / 'config.json', 'w') as outfile:
            outfile.write(json_object)
    
    def _load_npy(self, npy_type: str, save_path: str|None = None, 
                  file_name: str|None = None) -> np.ndarray|None:
        """
        Load result arrays stored in the numpy format (`.npy`).

        Parameters
        ----------
        npy_type : str
            Name of subfolder in the model-save-folder where the numpy-files are stored.
        save_path : str | None, optional
            Path to the model-save-folder (like `/<project_name>/<version>/`).\n
            Required if no model loaded.\n
            If `None` `self.config.save_path` is used. (defaults to None)
        file_name : str | None, optional
            Name of the `.npy`-file.\n
            If `None` the newest file in the folder is loaded. (defaults to None)

        Returns
        -------
        np.ndarray|None
            The loaded numpy array or `None` if the file does not exist.
        """
        if save_path:
            self.load_config(self.get_full_save_path(save_path))
        p = self.get_full_save_path() / npy_type
        if not file_name:
            npy_path = max(p.glob(f'*.npy'), key=os.path.getctime)
        else:
            npy_path = p / file_name
        if os.path.exists(npy_path):
            return np.load(npy_path)
    
    def load_prediction(self, save_path: str|None = None, file_name: str|None = None) -> np.ndarray|None:
        """
        Load a previous prediction stored in the numpy format (`.npy`).

        Parameters
        ----------
        save_path : str | None, optional
            Path to the model-save-folder (like `/<project_name>/<version>/`).\n
            Required if no model loaded.\n
            If `None` `self.config.save_path` is used. (defaults to None)
        file_name : str | None, optional
            Name of the `.npy`-file.\n
            If `None` the newest file in the folder is loaded. (defaults to None)

        Returns
        -------
        np.ndarray|None
            The loaded numpy array or `None` if the file does not exist.
        """
        return self._load_npy('predictions', save_path, file_name)
    
    @staticmethod
    def get_device_from_accelerator(accelerator: str) -> str:
        """
        Maps lightning accelerators to pytorch device-types.

        Parameters
        ----------
        accelerator : str
            Lightning accelerator type name.

        Returns
        -------
        str
            Pytorch device type name.

        Raises
        ------
        ValueError
            If the accelerator is not known.
        """
        mapping = {
            "cpu": "cpu",
            "gpu": "cuda",
            "mps": "mps",
            "xla": "tpu",
            "hpu": "hpu"
        }
        if accelerator in mapping:
            return mapping[accelerator]
        else:
            raise ValueError(f"Unable to map accelerator: {accelerator}")
