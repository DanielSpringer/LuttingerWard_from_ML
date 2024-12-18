import random

from pathlib import Path
from typing import Any

import numpy as np

import torch

from src.config.vertex import *
from src.load_data.vertex import *
from src.wrapper.vertex import *
from . import BaseTrainer


class VertexTrainer(BaseTrainer[VertexConfig, AutoEncoderVertexV2, VertexWrapper]):
    @property
    def input_size(self) -> int:
        return self.dataset[0][1].shape[0]
    
    def pre_train(self) -> None:
        torch.set_float32_matmul_precision('high')
    
    def predict(self, vertex_path: str, save_path = None, model_path = None, load_model = False, 
                encode_only: bool = False):
        return super().predict(vertex_path, save_path, model_path, load_model, 
                               encode_only=encode_only)
    
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
                 encode_only: bool = False, prepare_kwargs: dict[str, Any] = {}) -> np.ndarray:
        if encode_only:
            shape = list(vertex.shape)
            shape[axis - 1] = self.config.hidden_dims[-1]
            result = np.empty(tuple(shape))
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

                full_input, copy_kwargs = self._prepare_prediction_input(vertex, i, j, r, **prepare_kwargs)
                
                # get prediction of length 576 for a k axis
                if encode_only:
                    pred = self.wrapper.model.encode(full_input).detach().numpy()
                else:
                    pred = self.wrapper(full_input).detach().numpy()
                
                # feed back predictions into 576^3-matrix
                self._copy_prediction_to_matrix(pred, result, i, j, r, axis, encode_only, **copy_kwargs)
                del full_input
        
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


class VertexTrainer24x6(VertexTrainer):
    def _prepare_prediction_input(self, vertex: torch.Tensor, i: int, j: int, r: int,x_range: np.ndarray, 
                                  y_range: np.ndarray) -> tuple[torch.Tensor, dict[str, Any]]:
        k1x, k1y = i % self.dataset.n_freq, i // self.dataset.n_freq
        k2x, k2y = j % self.dataset.n_freq, j // self.dataset.n_freq
        k3x, k3y = r % self.dataset.n_freq, r // self.dataset.n_freq
        full_input = torch.tensor([
            *vertex[x_range == k1x, k2x, k3x],   # k1x
            *vertex[y_range == k1y, k2y, k3y],   # k1y
            *vertex[k1x, x_range == k2x, k3x],   # k2x
            *vertex[k1y, y_range == k2y, k3y],   # k2y
            *vertex[k1x, k2x, x_range == k3x],   # k3x
            *vertex[k1y, k2y, y_range == k3y],   # k3y
        ], dtype=torch.float32).to('cpu')
        if self.config.positional_encoding:
            pos = torch.tensor([k1x, k1y, k2x, k2y, k3x, k3y], dtype=torch.float32).to('cpu')
            full_input = (pos.unsqueeze(0), full_input.unsqueeze(0))
        copy_kwargs = {'k1x': k1x, 'k1y': k1y, 'k2x': k2x, 'k2y': k2y, 'k3x': k3x, 'k3y': k3y, 
                       'x_range': x_range, 'y_range': y_range}
        return full_input, copy_kwargs
    
    def _copy_prediction_to_matrix(self, pred: np.ndarray, result: np.ndarray, i: int, j: int, r: int, 
                                   axis: int, encode_only: bool, **kwargs) -> None:
        if encode_only:
            super()._copy_prediction_to_matrix(pred, result, i, j, r, axis)
        else:
            k1x, k1y, k2x, k2y, k3x, k3y, x_range, y_range = (kwargs[k] for k in ('k1x', 'k1y', 'k2x', 'k2y', 'k3x', 
                                                                                  'k3y', 'x_range', 'y_range'))
            kx, ky = pred[:, :self.dataset.n_freq], pred[:, self.dataset.n_freq:]  # lengths: 24, 24
            # TODO: only 48 of the 576 elements for each vector are set
            # either we need 576 predicted items or we need 12 predictions to assign to one 576-vector
            # -> predict k_i_y[24] (out_dim: 24), repeat for each k_i_x[24]
            if axis == 1:
                result[x_range == k1x, j, r] = kx
                result[y_range == k1y, j, r] = ky
            elif axis == 2:
                result[i, x_range == k2x, r] = kx
                result[i, y_range == k2y, r] = ky
            elif axis == 3:
                result[i, j, x_range == k3x] = kx
                result[i, j, y_range == k3y] = ky

    def _predict(self, vertex: torch.Tensor, vertex_path: str, axis: int = 3, 
                 encode_only: bool = False) -> np.ndarray:
        l_idcs = np.arange(self.dataset.length)
        x_range, y_range = l_idcs % self.dataset.n_freq, l_idcs // self.dataset.n_freq
        prepare_kwargs = {'x_range': x_range, 'y_range': y_range}
        #return super(VertexTrainer)._predict(vertex, vertex_path, axis, encode_only, prepare_kwargs)

        if encode_only:
            shape = list(vertex.shape)
            shape[axis - 1] = self.config.hidden_dims[-1]
            result = np.empty(tuple(shape))
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

                full_input, copy_kwargs = self._prepare_prediction_input(vertex, i, j, r, **prepare_kwargs)
                
                # get prediction of length 48 for a k_x and k_y axis
                if encode_only:
                    pred = self.wrapper.model.encode(full_input).detach().numpy()
                else:
                    pred = self.wrapper(full_input).detach().numpy()
                
                # feed back predictions into 576^3-matrix
                self._copy_prediction_to_matrix(pred, result, i, j, r, axis, encode_only, **copy_kwargs)
                del full_input
        
        # save results to disk
        vertex_name = Path(vertex_path).stem
        if encode_only:
            fp = self.get_full_save_path() / f'{vertex_name}_latentspace.npy'
        else:
            fp = self.get_full_save_path() / f'{vertex_name}_prediction.npy'
        np.save(fp, result)
        return result
