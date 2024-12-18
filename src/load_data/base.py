from abc import ABC, abstractmethod

import torch
from torch.utils.data import Dataset

from src.config import Config


class FilebasedDataset(Dataset, ABC):
    @abstractmethod
    def __init__(self, config: Config):
        """
        :param config: A Config instance.
        :type config: Config
        
        :param subset: Number of data items to load.
                       Either as integer to specify the absolute count or as float to specifiy the percentage of the existing data. 
                       Minimum items loaded is 1. (defaults to 1.0)
        :type subset: int | float, optional
        
        :param shuffle: If loading a subset, determines if items are selected from the directory in alphabetical order or randomly. 
                        (defaults to True)
        :type shuffle: bool, optional
        """
    
    @staticmethod
    @abstractmethod
    def load_from_file(path: str) -> torch.Tensor:
        pass
