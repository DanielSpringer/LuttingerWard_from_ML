import random

from pathlib import Path
from typing import Any

import numpy as np
import torch

from tqdm import tqdm

from src.config.vertex import *
from src.load_data.vertex import *
from src.wrapper.vertex import *
from . import BaseTrainer


class VertexTrainer(BaseTrainer[VertexConfig, AutoEncoderVertexDataset, VertexWrapper]):
    @property
    def input_size(self) -> int:
        return self.dataset[0][1].shape[0]
    
    def pre_train(self) -> None:
        torch.set_float32_matmul_precision('high')
    
    def predict(self, vertex_path: str, save_path = None, load_model = False, 
                axis: int = 3, encode_only: bool = False):
        return super().predict(vertex_path, save_path, load_model, 
                               axis=axis, encode_only=encode_only)
    
    def _prepare_prediction_input(self, vertex: torch.Tensor, i: int, j: int, 
                                  r: int) -> tuple[torch.Tensor, dict[str, Any]]:
        full_input = torch.tensor(self.dataset.get_vector_from_vertex(vertex, i, j, r),
                                  dtype=torch.float32).to('cpu')
        if self.config.positional_encoding:
            pos = torch.tensor([i, j, r], dtype=torch.float32).to('cpu')
            full_input = (pos.unsqueeze(0), full_input.unsqueeze(0))
        return full_input, {}
    
    def _copy_prediction_to_matrix(self, pred: np.ndarray, result: np.ndarray, i: int, j: int, r: int, 
                                   axis: int) -> None:
        if axis == 1:
            result[:, j, r] = pred
        elif axis == 2:
            result[i, :, r] = pred
        elif axis == 3:
            result[i, j, :] = pred
    
    def _predict(self, vertex: torch.Tensor, vertex_path: str, axis: int = 3, 
                 encode_only: bool = False) -> np.ndarray:
        if encode_only:
            shape = list(vertex.shape)
            shape[axis - 1] = self.config.hidden_dims[-1]
            result = np.empty(tuple(shape))
        else:
            result = np.zeros_like(vertex)
        
        with tqdm(total=self.dataset.length**(self.dataset.dim - 1)) as prog:
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

                    full_input = torch.tensor(self.dataset.get_vector_from_vertex(vertex, i, j, r),
                                            dtype=torch.float32).to('cpu')
                    if self.config.positional_encoding:
                        pos = torch.tensor([i, j, r], dtype=torch.float32).to('cpu')
                        full_input = (pos.unsqueeze(0), full_input.unsqueeze(0))
                    
                    # get prediction of length 576 for a k axis
                    if encode_only:
                        pred = self.wrapper.model.encode(full_input).detach().numpy()
                    else:
                        pred = self.wrapper(full_input).detach().numpy()
                    
                    # feed back predictions into 576^3-matrix
                    if axis == 1:
                        result[:, j, r] = pred
                    elif axis == 2:
                        result[i, :, r] = pred
                    elif axis == 3:
                        result[i, j, :] = pred
                    del full_input
                    prog.update()
        
        # save results to disk
        vertex_name = Path(vertex_path).stem
        if encode_only:
            p = self.get_full_save_path() / 'latentspaces'
        else:
            p = self.get_full_save_path() / 'predictions'
        p.mkdir(exist_ok=True)
        fp = p / f'{vertex_name}.npy'
        np.save(fp, result)
        return result
    
    def load_latentspace(self, save_path: str|None = None, 
                         file_name: str|None = None) -> np.ndarray|None:
        return self._load_npy('latentspaces', save_path, file_name)


class VertexTrainer24x6(VertexTrainer):
    def __init__(self, project_name, config_name, subconfig_name = None, config_dir = 'configs', config_kwargs = ...):
        super().__init__(project_name, config_name, subconfig_name, config_dir, config_kwargs)
        self.dataset: AutoEncoderVertex24x6Dataset = self.dataset
    
    def predict(self, vertex_path: str, save_path = None, load_model = False, 
                axis: int = 6, encode_only: bool = False):
        return super().predict(vertex_path, save_path, load_model, axis, encode_only)

    def _predict(self, vertex: torch.Tensor, vertex_path: str, axis: int = 6, 
                 encode_only: bool = False) -> np.ndarray:
        assert axis >= 1 and axis <= 6, "Axis must be in range [1,6]"

        if encode_only:
            shape = list(vertex.shape)
            shape[axis - 1] = self.config.hidden_dims[-1]
            result = np.empty(tuple(shape))
        else:
            result = np.zeros_like(vertex)
        
        with tqdm(total=self.dataset.n_freq**(self.dataset.dim - 1)) as prog:
            for a_idx in range(vertex.shape[0]):
                for b_idx in range(vertex.shape[1]):
                    for c_idx in range(vertex.shape[2]):
                        for d_idx in range(vertex.shape[3]):
                            for e_idx in range(vertex.shape[4]):
                                random_idx = random.randint(0, vertex.shape[5] - 1)
                                coord = [a_idx, b_idx, c_idx, d_idx, e_idx]
                                coord.insert(axis - 1, random_idx)  # on the current axis the random index is selected

                                full_input = torch.tensor(self.dataset.get_vector_from_vertex(vertex, *coord),
                                                        dtype=torch.float32).to('cpu')
                                if self.config.positional_encoding:
                                    pos = torch.tensor(coord, dtype=torch.float32).to('cpu')
                                    full_input = (pos.unsqueeze(0), full_input.unsqueeze(0))
                                
                                # get prediction of length 24 for an axis
                                if encode_only:
                                    pred = self.wrapper.model.encode(full_input).detach().numpy()
                                else:
                                    pred = self.wrapper(full_input).detach().numpy()
                                
                                # feed back predictions into 24^6-matrix
                                coord[axis -1] = slice(None)  # set the index for the prediction axis to the full axis
                                result[*coord] = pred      # assign the prediction to this axis
                                del full_input
                                prog.update()
        
        # save results to disk
        vertex_name = Path(vertex_path).stem
        if encode_only:
            p = self.get_full_save_path() / 'latentspaces'
        else:
            p = self.get_full_save_path() / 'predictions'
        p.mkdir(exist_ok=True)
        fp = p / f'{vertex_name}.npy'
        np.save(fp, result)
        return result
