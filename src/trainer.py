import datetime
import glob
import json
import os
import random

from enum import Enum
from pathlib import Path
from typing import Any, Generic, TypeVar

import numpy as np
from lightning.pytorch import Trainer
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.plugins.environments import LightningEnvironment

import torch
from torch.utils.data import DataLoader, random_split

from . import config, load_data, wrapper


R = TypeVar('R', bound=wrapper.BaseWrapper)
S = TypeVar('S', bound=load_data.FilebasedDataset)
T = TypeVar('T', bound=config.Config)


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

        :param project_name: Some project name under which results are saved.
        :type project_name: str
        
        :param config_name: Name of the config-file to load.
        :type config_name: str

        :param subconfig_name: Name of a sub-config in the config-file.  
                               Required if the file contains multiple cofigurations. (defaults to None)
        :type subconfig_name: str, optional
        
        :param test_ratio: Test split ratio between 0 and 1. (defaults to 0.2)
        :type test_ratio: float, optional
        
        :param config_dir: Path to the directory of the config-files. (defaults to 'configs')
        :type config_dir: str, optional
        
        :param config_kwargs: Additional kwargs that are applied when loading the config-file.  
                              Allows to overwrite attributes from the config-file. (defaults to {})
        :type config_kwargs: dict[str, Any], optional
        """
        self.project_name = project_name
        self.subconfig_name = subconfig_name

        self.config: T = self.config_cls.from_json(config_name, subconfig_name, config_dir, **config_kwargs)
        self.dataset: S = self.config.dataset(self.config)
        self.data_loader: type[DataLoader] = self.config.data_loader

        self.wrapper: R = None
    
    @property
    def input_size(self) -> int|np.ndarray:
        """Get the input size of the model. Overwrite for custom input size."""
        return self.dataset[0][0].shape[0]

    def train(self, train_mode: TrainerModes) -> None:
        self.pre_train()
        self._train(train_mode)
        self.post_train()
    
    def predict(self, new_data_path: str, save_path: str|None = None, model_path: str|None = None, 
                load_model: bool = False, **kwargs) -> np.ndarray:
        """
        If already trained, uses the trained model, otherwise loads a model and performs a prediction on new data.

        :param new_data_path: Path to new data to predict on.
        :type new_data_path: str
        
        :param save_path: Save directory for the model run (`<model_name>/<version>/`). 
                          The newest checkpoint in the directory is loaded. 
                          If `None` `self.config.save_path` is used. (defaults to None)
        :type save_path: str | None, optional
        
        :param model_path: Load a model from an absolute path or a path relative to project-subdirectory 
                           (named after `self.project_name`). If `None` **save_path** is used. (defaults to None)
        :type model_path: str | None, optional
        
        :param load_model: Force load a model. (defaults to False)
        :type load_model: bool, optional

        :param kwargs: Additional keyword arguments handed to the `_predict` method.
        :type kwargs: dict[str, Any]
        
        :return: Prediction as numpy array.
        :rtype: np.ndarray
        """
        new_data = self.dataset.load_from_file(new_data_path)
        if self.wrapper is None or load_model or save_path or model_path:
            self.load_model(save_path, model_path)
        self.wrapper.model.eval()
        return self._predict(new_data, new_data_path, **kwargs)

    def pre_train(self) -> None:
        """Overwrite to perform operations before the main training"""
        pass

    def post_train(self) -> None:
        """Overwrite to perform operations after the main training"""
        pass

    def create_data_loader(self) -> tuple[DataLoader, DataLoader]:
        """Create DataLoader. Overwrite for custom data loading."""
        train_set, validation_set = random_split(self.dataset, [1 - self.config.test_ratio, self.config.test_ratio], 
                                                 generator=torch.Generator().manual_seed(42))
        train_dataloader = self.data_loader(train_set, batch_size=self.config.batch_size, shuffle=True, 
                                            num_workers=8, persistent_workers=True, pin_memory=True)
        validation_dataloader = self.data_loader(validation_set, batch_size=self.config.batch_size, 
                                                 num_workers=8, persistent_workers=True, pin_memory=True)
        return train_dataloader, validation_dataloader

    def set_logging(self) -> TensorBoardLogger:
        """Set TensorBoardLogger and ModelCheckpoint. Overwrite for custom logging."""
        save_path = self.save_prefix + str(datetime.datetime.now().date())
        logger = TensorBoardLogger(self.get_full_save_path(''), name=save_path)
        self.config.save_path = logger.log_dir
        return logger

    def _predict(self, new_data: np.ndarray, new_data_path: str, **kwargs) -> np.ndarray:
        """Overwrite for customized prediction."""
        input = torch.tensor(new_data, dtype=torch.float32).to('cpu')
        pred = self.wrapper(input).detach().numpy()
        fp = self.get_full_save_path() / f'{Path(new_data_path).stem}_prediction.npy'
        np.save(fp, pred)
        return pred
    
    def get_full_save_path(self, save_path: str|None = None) -> Path:
        if save_path is None:
            save_path = self.config.save_path
        return self.config.base_dir / self.config.save_dir / self.project_name / save_path
    
    @property
    def save_prefix(self) -> str:
        return f"save_{self.subconfig_name}_BS{self.config.batch_size}_"

    def load_config(self, save_path: str) -> None:
        self.config = self.config_cls.from_json('config.json', directory=save_path, save_path=save_path)

    def load_model(self, save_path: str|None = None, model_path: str|None = None) -> None:
        """
        Loads a model and corresponding config into the trainer. 
        Loads the model either from a given checkpoint filepath or from the newest checkpoint in a given directory.

        :param save_path: Save directory for the model run (`<model_name>/<version>/`). 
                          The newest checkpoint in the directory is loaded. 
                          If `None` `self.config.save_path` is used. (defaults to None)
        :type save_path: str | None, optional
        
        :param model_path: Load a model from an absolute path or a path relative to project-subdirectory 
                           (named after `self.project_name`). If `None` **save_path** is used. (defaults to None)
        :type model_path: str | None, optional
        """
        assert model_path or save_path or self.config.save_path, "No save_path found in the config-file, please provide a save_path or model_path."
        if model_path and Path(model_path).is_absolute():
            ckpt_path = model_path
        elif model_path:
            ckpt_path = self.get_full_save_path(model_path)
        else:
            save_path = save_path or self.config.save_path
            ckpt_path = self.get_full_save_path(save_path)
            ckpt_files = glob.glob((ckpt_path / '**/*.ckpt').as_posix(), recursive=True)
            ckpt_path = max(ckpt_files, key=os.path.getctime)
        print(f" >>> Load checkpoint from '{ckpt_path}'")
        self.load_config(Path(ckpt_path).parent.parent.as_posix())
        device = self.get_device_from_accelerator(self.config.device_type)
        checkpoint = torch.load(ckpt_path, map_location=device, weights_only=True)
        self.wrapper = self.config.model_wrapper(self.config, self.input_size)
        self.wrapper.load_state_dict(checkpoint['state_dict'])

    def _train(self, train_mode: TrainerModes) -> None:
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
    
    def _load_npy(self, npy_type: str, save_path: str|None = None, npy_path: str|None = None) -> np.ndarray|None:
        if save_path:
            self.load_config(save_path)
        if not npy_path:
            npy_path = max(self.get_full_save_path().glob(f'*_{npy_type}.npy'), key=os.path.getctime)
        if os.path.exists(npy_path):
            return np.load(npy_path)
    
    def load_prediction(self, save_path: str|None = None, npy_path: str|None = None) -> np.ndarray|None:
        return self._load_npy('prediction', save_path, npy_path)
    
    @staticmethod
    def get_device_from_accelerator(accelerator: str) -> str:
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


class VertexTrainer(BaseTrainer[config.VertexConfig, load_data.AutoEncoderVertexV2, wrapper.VertexWrapper]):
    def __init__(self, project_name: str, config_name: str, subconfig_name: str|None = None, 
                 config_dir: str = 'configs', config_kwargs: dict[str, Any] = {}):
        super().__init__(project_name, config_name, subconfig_name, config_dir, config_kwargs)
    
    @property
    def input_size(self) -> int:
        return self.dataset[0][1].shape[0]
    
    def pre_train(self) -> None:
        torch.set_float32_matmul_precision('high')
    
    def predict(self, vertex_path: str, save_path = None, model_path = None, load_model = False, 
                encode_only: bool = False):
        return super().predict(vertex_path, save_path, model_path, load_model, 
                               encode_only=encode_only)
    
    def _predict(self, vertex: torch.Tensor, vertex_path: str, axis: int = 3, 
                 encode_only: bool = False) -> np.ndarray:
        if encode_only:
            result = np.empty((vertex.shape[0], vertex.shape[1], self.config.hidden_dims[-1]))
        else:
            result = np.zeros_like(vertex)
        for i_cnt in range(vertex.shape[0]):
            for j_cnt in range(vertex.shape[1]):
                r_cnt = random.randint(0, vertex.shape[2] - 1)
                if axis == 1:
                    i, j, r = r_cnt, i_cnt, j_cnt
                elif axis == 2:
                    i, j, r = i_cnt, r_cnt, j_cnt
                elif axis == 3:
                    i, j, r = i_cnt, j_cnt, r_cnt
                else:
                    raise NotImplementedError("Axis not implemented")

                dim1 = vertex[i, j, :]
                dim2 = vertex[i, :, r]
                dim3 = vertex[:, j, r]
                full_input = torch.tensor([*dim1, *dim2, *dim3], dtype=torch.float32).to('cpu')
                if self.config.positional_encoding:
                    pos = torch.tensor([i, j, r], dtype=torch.float32).to('cpu')
                    full_input = (pos.unsqueeze(0), full_input.unsqueeze(0))
                if encode_only:
                    pred = self.wrapper.model.encode(full_input).detach().numpy()
                else:
                    pred = self.wrapper(full_input).detach().numpy()
                if axis == 1:
                    result[:, j, r] = pred
                elif axis == 2:
                    result[i, :, r] = pred
                elif axis == 3:
                    result[i, j, :] = pred
                del dim1, dim2, dim3, full_input
        
        # save results to disk
        vertex_name = Path(vertex_path).stem
        if encode_only:
            fp = self.get_full_save_path() / f'{vertex_name}_latentspace.npy'
        else:
            fp = self.get_full_save_path() / f'{vertex_name}_prediction.npy'
        np.save(fp, result)
        return result
    
    def load_latentspace(self, save_path: str|None = None, 
                         npy_path: str|None = None) -> np.ndarray|None:
        return self._load_npy('prediction', save_path, npy_path)
