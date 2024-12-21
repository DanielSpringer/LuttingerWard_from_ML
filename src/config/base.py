import importlib
import json
import pydoc

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Generic, Literal, Type, TypeVar, TYPE_CHECKING

from lightning.pytorch.callbacks.callback import Callback
from torch.nn import Module
from torch.optim.optimizer import Optimizer
from torch.utils.data import DataLoader


if TYPE_CHECKING:
    from src import wrapper, models, load_data
R = TypeVar('R', bound='models.BaseModule')
S = TypeVar('S', bound='wrapper.BaseWrapper')
T = TypeVar('T', bound='load_data.FilebasedDataset')


@dataclass
class Config(Generic[R, S, T]):
    _base_dir: str = Path(__file__).parent.parent.parent.as_posix()
    model_name: str = 'BaseModule'
    _model_wrapper: str = 'BaseWrapper'
    resume: bool = False
    save_dir: str = 'saves'
    save_path: str = ''
    path_train: str = ''
    _dataset: str = 'FilebasedDataset'
    _data_loader: str = 'torch.utils.data.DataLoader'
    test_ratio: float = 0.2
    subset: int|float|None = None
    subset_shuffle: bool = True
    subset_seed: int = None
    
    # model architecture
    hidden_dims: list[int] = field(default_factory=lambda: [])
    out_dim: int = 128

    # training
    batch_size: int = 20
    learning_rate: float = 0.0001
    weight_decay: float = 1e-05
    epochs: int = 1000
    device_type: Literal['cpu', 'gpu', 'mps', 'xla', 'hpu'] = 'gpu'
    devices: int = 1          # >= <# devices (CPUs or GPUs) on partition> * `num_nodes`
    num_nodes: int = 1

    # torch modules
    _criterion: str = 'torch.nn.MSELoss'
    criterion_kwargs: dict[str, Any] = field(default_factory=lambda: {})
    _optimizer: str = 'torch.optim.AdamW'
    optimizer_kwargs: dict[str, Any] = field(default_factory=lambda: {})
    _activation = 'torch.nn.ReLU'
    activation_kwargs: dict[str, Any] = field(default_factory=lambda: {})

    # lightning callbacks
    _model_checkpoint: str = 'ModelCheckpoint'
    model_checkpoint_kwargs: dict[str, Any] = field(default_factory=lambda: {
        'save_top_k': 10,       # Save top 10 models
        'monitor': 'val_loss',  # Monitor validation loss
        'mode': 'min',          # 'min' for minimizing the validation loss
        'verbose': True
    })
    _callbacks: list[str] = field(default_factory=lambda: [])
    callbacks_kwargs: dict[str, dict[str, Any]] = field(default_factory=lambda: {
        'EarlyStopping': {
            'monitor': 'val_loss',  # Monitor validation loss
            'mode': 'min',          # 'min' for minimizing the validation loss
            'patience': 10,         # Number of epochs with no improvement after which training will be stopped
            'verbose': True
        },
    })
    
    @classmethod
    def from_json(cls, config_name: str, subconfig_name: str|None = None, 
                  directory: str = 'configs', **kwargs) -> 'Config':
        """
        Creates a class containing all training parameters from a config-JSON. 
        Add kwargs to add attributes or overwrite attributes from the config-JSON.

        Parameters
        ----------
        config_name : str
            File name of the config-JSON.
        subconfig_name : str | None, optional
            Name of a sub-config in the config-file.\n
            Required if the file contains multiple cofigurations. (defaults to None)
        directory : str, optional
            Path to the directory of the config-file. (defaults to 'configs')
        **kwargs :
            Adds or overwrites attributes taken from the config-JSON.

        Returns
        -------
        Config
            A Config-instance with parameters set according to the JSON-file.
        """
        directory = Path(cls._base_dir, directory)
        with open(directory / config_name) as f:
            config: dict[str, Any] = json.load(f)
        if subconfig_name is not None:
            config = config[subconfig_name]
        config.update(kwargs)
        conf = cls()
        for key, value in config.items():
            setattr(conf, key.lower(), value)
        return conf

    @property
    def base_dir(self) -> Path:
        return Path(self._base_dir)
    
    @property
    def model(self) -> Type[R]:
        return self.resolve_objectname(self.model_name, 'src.models')

    @property
    def model_wrapper(self) -> Type[S]:
        return self.resolve_objectname(self._model_wrapper, 'src.wrapper')
    
    @model_wrapper.setter
    def model_wrapper(self, value: str):
        self._model_wrapper = value
    
    @property
    def dataset(self) -> Type[T]:
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
        return [cb(**dict(self.callbacks_kwargs[cb.__name__], **kwargs.get(cb.__name__, {}))) for cb in self.callbacks]
    
    @staticmethod
    def resolve_objectpath(obj_fqn: str) -> type:
        """
        Import a class-object defined by its fully qualified name.

        Parameters
        ----------
        obj_fqn : str
            The fully qualified name of the class.

        Returns
        -------
        type
            The class.
        """
        return pydoc.locate(obj_fqn)
    
    @staticmethod
    def resolve_objectname(obj_name: str, module_name: str) -> type:
        """
        Import a class object defined by the class-name and its fully qualified module-name.

        Parameters
        ----------
        obj_name : str
            The name of the class.
        module_name : str
            The name of the class.

        Returns
        -------
        type
            The class.
        """
        module = importlib.import_module(module_name)
        return getattr(module, obj_name)
    
    def _instantiate_type(self, name: str, **kwargs) -> Any:
        """Instantiate a class-object using the kwargs from the corresponding class-attribute. 
        Kwargs from the class-attribute can be overwritten by providing kwrags in the method-call."""
        attr_kwargs = dict(getattr(self, name + '_kwargs'), **kwargs)
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
