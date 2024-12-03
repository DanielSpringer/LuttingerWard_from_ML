import importlib
import json
import os
import pydoc

from dataclasses import dataclass, field
from typing import Any

from lightning.pytorch.callbacks.callback import Callback
from torch.nn import Module
from torch.optim.optimizer import Optimizer
from torch.utils.data import DataLoader


@dataclass
class Config:
    model_name: str = 'BaseModule'
    _model_wrapper: str = 'BaseWrapper'
    resume: bool = False
    save_dir: str = 'saves'
    save_path: str = ''
    path_train: str = ''
    _dataset: str = 'FilebasedDataset'
    _data_loader: str = 'torch.utils.data.DataLoader'
    
    # model architecture
    hidden_dims: list[int]|None = None
    out_dim: int = 128

    # training
    batch_size: int = 20
    learning_rate: float = 0.0001
    weight_decay: float = 1e-05
    epochs: int = 1000
    device_type: str = 'gpu'
    devices: int = 1
    num_nodes: int = 1

    # torch modules
    _criterion: str = 'torch.nn.MSELoss'
    criterion_kwargs: dict[str, Any]|None = None
    _optimizer: str = 'torch.optim.AdamW'
    optimizer_kwargs: dict[str, Any]|None = None
    _activation = 'torch.nn.ReLU'
    activation_kwargs: dict[str, Any]|None = None

    # lightning callbacks
    _model_checkpoint: str = 'ModelCheckpoint'
    model_checkpoint_kwargs: dict[str, Any] = field(default_factory=lambda: {
        'save_top_k': 10,       # Save top 10 models
        'monitor': 'val_loss',  # Monitor validation loss
        'mode': 'min',          # 'min' for minimizing the validation loss
        'verbose': True
    })
    _callbacks: list[str]|None = None
    callbacks_kwargs: dict[str, dict[str, Any]] = field(default_factory=lambda: {
        'EarlyStopping': {
            'monitor': 'val_loss',  # Monitor validation loss
            'mode': 'min',          # 'min' for minimizing the validation loss
            'patience': 10,         # Number of epochs with no improvement after which training will be stopped
            'verbose': True
        },
    })
    
    @classmethod
    def from_json(cls, config_name: str, subconfig_name: str|None = None, directory: str = 'configs', **kwargs):
        """
        Creates a class containing all training parameters from a config-JSON. 
        Add kwargs to add attributes or overwrite attributes from the config-JSON.

        :param config_name: File name of the config-JSON.
        :type config_name: str

        :param subconfig_name: Name of a sub-config in the config-file.  
                               Required if the file contains multiple cofigurations. (defaults to None)
        :type subconfig_name: str | None, optional
        
        :param directory: Relative path to the directory of the config-file. (defaults to 'configs')
        :type directory: str, optional
        """
        config: dict[str, Any] = json.load(open(os.path.join(directory, config_name)))
        if subconfig_name is not None:
            config = config[subconfig_name]
        config.update(kwargs)
        conf = cls()
        for key, value in config.items():
            setattr(conf, key.lower(), value)
        return conf
    
    @property
    def model(self) -> type[Module]:
        return self.resolve_objectname(self.model_name, 'src.models')

    @property
    def model_wrapper(self) -> type:
        return self.resolve_objectname(self._model_wrapper, 'src.wrapper')
    
    @model_wrapper.setter
    def model_wrapper(self, value: str):
        self._model_wrapper = value
    
    @property
    def dataset(self) -> type:
        return self.resolve_objectname(self._dataset, 'src.load_data')
    
    @dataset.setter
    def dataset(self, value: str):
        self._dataset = value
    
    @property
    def data_loader(self) -> type[DataLoader]:
        if '.' in self._data_loader:
            return self.resolve_objectpath(self._data_loader)
        else:
            return self.resolve_objectname(self._data_loader, 'src.load_data')
    
    @data_loader.setter
    def data_loader(self, value: str):
        self._data_loader = value
    
    @property
    def criterion(self) -> type[Module]:
        return self.resolve_objectpath(self._criterion)
    
    @criterion.setter
    def criterion(self, value: str):
        self._criterion = value
    
    def get_criterion(self, **kwargs) -> Module:
        """Get an instantiation of the criterion class-object using `self.criterion_kwargs` as arguments. 
        Use `**kwargs` to overwrite arguments set in `self.criterion_kwargs`."""
        return self._instantiate_type('criterion', **kwargs)
    
    @property
    def optimizer(self) -> type[Optimizer]:
        return self.resolve_objectpath(self._optimizer)
    
    @optimizer.setter
    def optimizer(self, value: str):
        self._optimizer = value
    
    def get_optimizer(self, **kwargs) -> Optimizer:
        """Get an instantiation of the optimizer class-object using `self.optimizer_kwargs` as arguments.
        Use `**kwargs` to overwrite arguments set in `self.optimizer_kwargs`."""
        return self._instantiate_type('optimizer', **kwargs)
    
    @property
    def activation(self) -> type[Module]:
        return self.resolve_objectpath(self._activation)
    
    @activation.setter
    def activation(self, value: str):
        self._activation = value
    
    def get_activation(self, **kwargs) -> Module:
        """Get an instantiation of the activation class-object using `self.activation_kwargs` as arguments.
        Use `**kwargs` to overwrite arguments set in `self.activation_kwargs`."""
        return self._instantiate_type('activation', **kwargs)
    
    @property
    def model_checkpoint(self) -> type[Callback]:
        return self.resolve_objectname(self._model_checkpoint, 'lightning.pytorch.callbacks')
    
    @model_checkpoint.setter
    def model_checkpoint(self, value: str):
        self._model_checkpoint = value
    
    def get_model_checkpoint(self, **kwargs) -> Callback:
        """Get an instantiation of the model_checkpoint class-object using `self.model_checkpoint_kwargs` as arguments.
        Use `**kwargs` to overwrite arguments set in `self.model_checkpoint_kwargs`."""
        return self._instantiate_type('model_checkpoint', **kwargs)
    
    @property
    def callbacks(self) -> list[type[Callback]]:
        if not self._callbacks:
            return []
        return [self.resolve_objectname(cb, 'lightning.pytorch.callbacks') for cb in self._callbacks]
    
    @callbacks.setter
    def callbacks(self, value: list[str]):
        self._callbacks = value
    
    def get_callbacks(self, kwargs: dict[str, dict[str, Any]] = {}) -> list[Callback]:
        """Get a list of instantiations of the callback class-objects using `self.callbacks_kwargs` as arguments.
        Use `kwargs` to overwrite arguments set in `self.callbacks_kwargs`. 
        Attention: The `kwargs`-argument is a dictionary of kwarg-dictionaries using the callback-class-names as keys."""
        return [cb(**dict(self.callbacks_kwargs[cb_name], **kwargs.get(cb_name, {}))) for cb_name, cb in zip(self._callbacks, self.callbacks)]
    
    @staticmethod
    def resolve_objectpath(obj_fqn: str) -> type:
        """
        Import a class-object defined by its fully qualified name.

        :param obj_fqn: The fully qualified name of the class.
        :type obj_fqn: str
        
        :return: The class.
        :rtype: type
        """
        return pydoc.locate(obj_fqn)
    
    @staticmethod
    def resolve_objectname(obj_name: str, module_name: str) -> type:
        """
        Import a class object defined by the class-name and its fully qualified module-name

        :param obj_name: The name of the class.
        :type obj_name: str
        
        :param module_name: The fully qualified module-name where the class is contained.
        :type module_name: str
        
        :return: The class.
        :rtype: type
        """        
        module = importlib.import_module(module_name)
        return getattr(module, obj_name)
    
    def _instantiate_type(self, name: str, **kwargs) -> Any:
        """Instantiate a class-object using the kwargs from the corresponding class-attribute. 
        Kwargs from the class-attribute can be overwritten by providing kwrags in the method-call."""
        attr_kwargs = getattr(self, name + '_kwargs') or {}
        attr_kwargs = dict(attr_kwargs, **kwargs)
        return getattr(self, name)(**attr_kwargs)
    
    def __repr__(self):
        return self.__class__.__qualname__ + '(\n   ' + ',\n   '.join([f"{k}={v!r}" for k, v in self.__dict__.items()]) + '\n)'

    def __str__(self):
        return self.__class__.__qualname__ + ':\n{\n   ' + ',\n   '.join([f"{k}: {v!r}" for k, v in self.__dict__.items()]) + '\n}'
    
    def __iter__(self):
        return iter(self.__dict__.items())

    def __len__(self):
        return len(self.__dict__)

    def as_dict(self):
        return self.__dict__
    
    def save(self, path: str) -> None:
        """Save config as JSON file."""
        save_dict = {self.model_name.upper(): self.as_dict()}
        with open(path, 'w') as f:
            json.dump(save_dict, f, indent=4)


class VertexConfig(Config):
    construction_axis: int = 3
    sample_count_per_vertex: int = 2000
    positional_encoding: bool = True
