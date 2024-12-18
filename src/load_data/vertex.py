import glob
import random

from copy import deepcopy

import h5py
import numpy as np

import torch

from src.config.vertex import *
from . import FilebasedDataset


class AutoEncoderVertexV2(FilebasedDataset):
    # matrix parameters
    n_freq = 24
    space_dim = 2
    k_dim = 3
    dim = k_dim
    length = n_freq**space_dim
    target_length = length
    
    def __init__(self, config: VertexConfig):
        config.matrix_dim = self.dim
        self.data_in_indices: torch.Tensor = torch.tensor([])
        self.data_in_slices: torch.Tensor = torch.tensor([])

        # Subsample files
        file_paths = glob.glob(f"{config.path_train}/*.h5")
        subset = config.subset
        if subset is not None:
            n_files = len(file_paths)
            if type(subset) == float:
                subset = int(n_files * subset)
            if subset < n_files and config.subset_shuffle:
                random.seed(config.subset_seed)
                file_paths = random.sample(file_paths, max(subset, 1))
        
        # Iterate through all files in given directory
        for file_path in file_paths:
            # Get vertex and create slices in each of the 3 dimensions
            vertex = self.load_from_file(file_path)

            # sample random indices of a 576^3 matrix and merge all rows through the sampled indices
            merged_slices, indices = self._sample(vertex, config)
        
            # Append result to data_in
            self.data_in_slices = torch.cat([self.data_in_slices, 
                                        torch.tensor(merged_slices, dtype=torch.float32)], axis=0)
            self.data_in_indices = torch.cat([self.data_in_indices, 
                                        torch.tensor(indices, dtype=torch.float32)], axis=0)
            assert self.data_in_indices.shape[0] == self.data_in_slices.shape[0]
        
        # Construct target data
        axis = config.construction_axis
        assert axis <= self.dim, "Axis invalid"
        idx_range = slice(self.target_length * (self.dim - axis), self.target_length * (self.dim - axis + 1))
        self.data_target = deepcopy(self.data_in_slices[:, idx_range])
        assert list(self.data_target[0]) == list(self.data_in_slices[0][idx_range])
    
    def _sample(self, vertex: np.ndarray, config: VertexConfig) -> tuple[np.ndarray, np.ndarray]:
        indices = random.sample(range(self.length**self.k_dim), config.sample_count_per_vertex)
        indices = np.array([[(x // self.length**i) % self.length for i in range(self.k_dim)] for x in indices])

        # Create and merge all row combinations
        merged_slices = list(self.get_vector_from_vertex(vertex, x, y, z) for x, y, z in indices)
        return merged_slices, indices
    
    @staticmethod
    def get_vector_from_vertex(vertex: np.ndarray, x: int, y: int, z: int) -> list[np.ndarray]:
        return [
            *vertex[x, y, :], 
            *vertex[x, :, z], 
            *vertex[:, y, z],
        ]

    def __len__(self):
        return self.data_in_slices.shape[0]

    def __getitem__(self, idx):
      if torch.is_tensor(idx):
          idx = idx.tolist()
      return self.data_in_indices[idx], self.data_in_slices[idx], self.data_target[idx]

    @staticmethod
    def load_from_file(path: str) -> np.ndarray:
        with h5py.File(path, 'r') as f:
            for name, data in f["V"].items():
                if name.startswith("step"):
                    return data[()]


class AutoEncoderVertex24x6(AutoEncoderVertexV2):
    dim = AutoEncoderVertexV2.space_dim * AutoEncoderVertexV2.k_dim
    target_length = AutoEncoderVertexV2.n_freq # * AutoEncoderVertexV2.space_dim

    def _sample(self, vertex: np.ndarray, config: VertexConfig) -> tuple[np.ndarray, np.ndarray]:
        # sample `sample_count_per_vertex` random indices of a 24^6 matrix
        indices = random.sample(range(self.n_freq**self.dim), config.sample_count_per_vertex)
        indices = np.array([[(x // self.n_freq**i) % self.n_freq for i in range(self.dim)] for x in indices])

        # Retrive and merge all row combinations by the sampled indices
        l_idcs = np.arange(self.length)
        x_range, y_range = l_idcs % self.n_freq, l_idcs // self.n_freq
        merged_slices = np.array([[
            *vertex[k1x + self.n_freq * k1y, k2x + self.n_freq * k2y, y_range == k3y],   # k3y
            *vertex[k1x + self.n_freq * k1y, k2x + self.n_freq * k2y, x_range == k3x],   # k3x
            *vertex[k1x + self.n_freq * k1y, y_range == k2y, k3x + self.n_freq * k3y],   # k2y
            *vertex[k1x + self.n_freq * k1y, x_range == k2x, k3x + self.n_freq * k3y],   # k2x
            *vertex[y_range == k1y, k2x + self.n_freq * k2y, k3x + self.n_freq * k3y],   # k1y
            *vertex[x_range == k1x, k2x + self.n_freq * k2y, k3x + self.n_freq * k3y],   # k1x
        ] for k1x, k1y, k2x, k2y, k3x, k3y in indices])
        return merged_slices, indices
    
    @staticmethod
    def get_vector_from_vertex(vertex: np.ndarray, k1x: int, k1y: int, k2x: int, k2y: int, k3x: int, k3y: int, 
                               x_range: np.ndarray, y_range: np.ndarray) -> list[np.ndarray]:
        # ???
        return [
            *vertex[k1y, k2y, y_range == k3y],   # k3y
            *vertex[k1x, k2x, x_range == k3x],   # k3x
            *vertex[k1y, y_range == k2y, k3y],   # k2y
            *vertex[k1x, x_range == k2x, k3x],   # k2x
            *vertex[y_range == k1y, k2y, k3y],   # k1y
            *vertex[x_range == k1x, k2x, k3x],   # k1x
        ]
