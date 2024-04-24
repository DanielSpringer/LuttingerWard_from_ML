import pytorch_lightning as L
import torch
import h5py
import copy
import numpy as np

from torch.utils.data import Dataset, DataLoader, random_split

#TODO relative import does not work...
def dtype_str_to_type(dtype_str: str):
    if dtype_str.lower() == "float32":
        return torch.float32
    elif dtype_str.lower() == "float64":
        return torch.float64
    else:
        raise ValueError("unkown dtype: " + dtype_str)

class AE_Dataset(Dataset):
    """
    Placeholder for now. 
    We may need this for large datasets or custom transformations/loss functions.
    """
    def __init__(self, x: torch.Tensor, y: torch.Tensor, dtype_default) -> None:
        self.x = x.clone().detach().to(dtype=dtype_default)
        self.y = y.clone().detach().to(dtype=dtype_default)
        self.ylen = y.shape[1] // 2

    def __len__(self) -> int:
        return len(self.x)

    def normalize_x(self, x: torch.Tensor) -> torch.Tensor:
        return x

    def unnormalize_x(self, x: torch.Tensor) -> torch.Tensor:
        return x

    def normalize_y(self, x: torch.Tensor) -> torch.Tensor:
        return x

    def unnormalize_y(self, x: torch.Tensor) -> torch.Tensor:
        return x

    def __getitem__(self, idx: int) -> tuple:
        x_norm = self.normalize_x(self.x[idx,:])
        y_norm = self.normalize_y(self.y[idx,:])
        return x_norm, y_norm
    

class DataMod_AE(L.LightningDataModule):
    def __init__(self, config):
        super().__init__()

        self.prepare_data_per_node = True
        self.train_batch_size = config['batch_size']
        self.val_batch_size = config['batch_size']
        self.test_batch_size = config['batch_size']
        self.data = config['PATH_TRAIN']
        self.dtype = dtype_str_to_type(config['dtype'])
        self.mode = config['mode']

    def setup(self, stage: str):
        """
        Download and transform datasets. 
        """
        with h5py.File(self.data, "r") as hf:
            if self.mode == 'gf':
                x = hf["Set1/GImp"][:]
            elif self.mode == 'se':
                x = hf["Set1/SImp"][:]
            else:
                raise RuntimeError("mode " + self.mode + "not found")
            #y = hf["Set1/GImp"][:]
        x = np.concatenate((x.real, x.imag), axis=1)
        y = copy.deepcopy(x)
        p = np.random.RandomState(seed=0).permutation(x.shape[0])
        x = x[p,:]
        y = y[p,:]
        x = torch.tensor(x, dtype=self.dtype)
        y = torch.tensor(y, dtype=self.dtype)

        self.train_dataset = AE_Dataset(x, y, self.dtype)
        self.train_set_size = int(len(self.train_dataset) * 0.8)
        self.val_set_size = len(self.train_dataset) - self.train_set_size

        self.train_dataset, self.val_dataset = random_split(self.train_dataset, [self.train_set_size, self.val_set_size])
        

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.train_batch_size, num_workers=8, persistent_workers=True, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.train_batch_size, num_workers=8, persistent_workers=True, shuffle=False)
    
    def test_dataloader(self):
        raise NotImplementedError("Define standard for data generation from jED.jl and create test data there!")
