import glob
import random
from abc import ABC, abstractmethod
from copy import deepcopy

import h5py
import numpy as np
from scipy.special import eval_legendre

import torch
from torch.utils.data import Dataset

from . import config


class FilebasedDataset(Dataset, ABC):
    @abstractmethod
    def __init__(self, config: config.Config, subset: int|float = 1.0, shuffle: bool = True):
        """
        :param config: A Config instance.
        :type config: config.Config
        
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


class Dataset_baseline(Dataset):
    """Dataset with minimal number of features.
    includes only impurity Green's function, electron density and self-energy (labels) without basis sets.
    """
    def __init__(self, config):
        PATH = config["PATH_TRAIN"]
        f = h5py.File(PATH, 'r')
        x = np.array(f["Set1"]["GImp"])
        ndens_in = np.array(f["Set1"]["dens"])
        y = np.array(f["Set1"]["SImp"])
        x = np.concatenate((x.real, x.imag), axis=1)
        y = np.concatenate((y.real, y.imag), axis=1)
        x = np.c_[ndens_in, x]
        p = np.random.RandomState(seed=0).permutation(x.shape[0])
        x = x[p,:]
        y = y[p,:]
        self.data_in = torch.tensor(x, dtype=torch.float32)
        self.data_target = torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        return self.data_in.shape[0]

    def __getitem__(self, idx):
        #if torch.is_tensor(idx):
        #    idx = idx.tolist()
        return self.data_in[idx], self.data_target[idx]


class Dataset_baseline_conv(Dataset):
    def __init__(self, config):
        PATH = config["PATH_TRAIN"]
        f = h5py.File(PATH, 'r')
        data_in = np.array(f["Set1"]["GImp"])
        data_target = np.array(f["Set1"]["SImp"])
        self.data_in = torch.stack([torch.tensor(data_in.real, dtype=torch.float32), torch.tensor(data_in.imag, dtype=torch.float32)],dim=1)
        self.data_target = torch.cat([torch.tensor(data_target.real, dtype=torch.float32), torch.tensor(data_target.imag, dtype=torch.float32)], axis=1)
#         self.data_target = torch.stack([torch.tensor(data_target.real, dtype=torch.float32), torch.tensor(data_target.imag, dtype=torch.float32)],dim=1)
#         print(self.data_in.shape)

    def __len__(self):
        return self.data_in.shape[0]

    def __getitem__(self, idx):
      if torch.is_tensor(idx):
          idx = idx.tolist()
      return self.data_in[idx], self.data_target[idx]


class Dataset_ae(Dataset):
    def __init__(self, config):
        PATH = config["PATH_TRAIN"]
        f = h5py.File(PATH, 'r')
        data_in = np.array(f["Set1"]["GImp"])
        data_target = np.array(f["Set1"]["SImp"])
        self.data_in = torch.cat([torch.tensor(data_in.real, dtype=torch.float32), torch.tensor(data_in.imag, dtype=torch.float32)], axis=1)
        self.data_target = torch.cat([torch.tensor(data_target.real, dtype=torch.float32), torch.tensor(data_target.imag, dtype=torch.float32)], axis=1)

    def __len__(self):
        return self.data_in.shape[0]

    def __getitem__(self, idx):
      if torch.is_tensor(idx):
          idx = idx.tolist()
      return self.data_in[idx], self.data_target[idx]


class Dataset_ae_vertex(Dataset):
    def __init__(self, config):
        self.data_in_indices: torch.Tensor = torch.tensor([])
        self.data_in_slices: torch.Tensor = torch.tensor([])

        SAMPLE_COUNT = config["sample_count_per_vertex"]

        # Iterate through all files in given directory
        for file_path in glob.glob(f"{config['PATH_TRAIN']}/*.h5"):

            # Get vertex and create slices in each of the 3 dimensions
            full_vertex = self.get_vertex_from_filepath(file_path)

            length = 576
            indices = random.sample(range(length ** 3), SAMPLE_COUNT)
            indices = [(x % length, (x // length) % length, (x // (length**2)) % length) for x in indices]

            # Create and merge all row combinations
            merged_slices = list([*full_vertex[x, y, :], *full_vertex[x, :, z], *full_vertex[:, y, z]] for x, y, z in indices)
        
            # Append result to data_in
            self.data_in_slices = torch.cat([self.data_in_slices, 
                                        torch.tensor(merged_slices, dtype=torch.float32)], axis=0)
            
            self.data_in_indices = torch.cat([self.data_in_indices, 
                                        torch.tensor(indices, dtype=torch.float32)], axis=0)
            
            assert self.data_in_indices.shape[0] == self.data_in_slices.shape[0]
                
        axis = config.get("construction_axis", 3)

        # Construct target data
        match axis:
            case 1: 
                self.data_target = deepcopy(self.data_in_slices[:, 2*length:])
                assert list(self.data_target[0]) == list(self.data_in_slices[0][2*length:])
            case 2:
                self.data_target = deepcopy(self.data_in_slices[:,length:2*length])
                assert list(self.data_target[0]) == list(self.data_in_slices[0][length:2*length])
            case 3:
                self.data_target = deepcopy(self.data_in_slices[:, :length])
                assert list(self.data_target[0]) == list(self.data_in_slices[0][:length])
            case _:
                raise NotImplementedError("Axis invalid")
        

    def __len__(self):
        return self.data_in_slices.shape[0]

    def __getitem__(self, idx):
      if torch.is_tensor(idx):
          idx = idx.tolist()
      return self.data_in_indices[idx], self.data_in_slices[idx], self.data_target[idx]

    def get_vertex_from_filepath(self, path: str):
        with h5py.File(path, 'r') as f:
            for name, data in f["V"].items():
                if name.startswith("step"):
                    return data[()]


class Dataset_ae_vertex_analysis(Dataset):
    def __init__(self, config):
        input_data = []
        
        # Iterate through all files in given directory
        for file_path in glob.glob(f"{config['PATH_TRAIN']}/*.h5"):

            # Get vertex and create slices in each of the 3 dimensions
            full_vertex = self.get_vertex_from_filepath(file_path)
        
            # Append result to data_in
            input_data.append(torch.tensor(full_vertex, dtype=torch.float32))
                
        # create deepcopy of result into data_target
        self.data_in = torch.stack(input_data)
        self.data_target = deepcopy(self.data_in)

    def __len__(self):
        return self.data_in.shape[0]

    def __getitem__(self, idx):
      if torch.is_tensor(idx):
          idx = idx.tolist()
      return self.data_in[idx], self.data_target[idx]

    def get_vertex_from_filepath(self, path: str):
        with h5py.File(path, 'r') as f:
            for name, data in f["V"].items():
                if name.startswith("step"):
                    return data[()]


class AutoEncoderVertexV2(FilebasedDataset):
    def __init__(self, config: config.Config, subset: int|float = 1.0, shuffle: bool = True):
        self.data_in_indices: torch.Tensor = torch.tensor([])
        self.data_in_slices: torch.Tensor = torch.tensor([])
        length = -1

        # Iterate through all files in given directory
        file_paths = glob.glob(f"{config.path_train}/*.h5")
        n_files = len(file_paths)
        if type(subset) == float:
            subset = int(n_files * subset)
        if subset < n_files and shuffle:
            file_paths = random.sample(file_paths, min(subset, 1))
        for file_path in glob.glob(f"{config.path_train}/*.h5"):
            # Get vertex and create slices in each of the 3 dimensions
            full_vertex = self.load_from_file(file_path)
            length = full_vertex.shape[0]
            indices = random.sample(range(length ** 3), config.sample_count_per_vertex)
            indices = [(x % length, (x // length) % length, (x // (length**2)) % length) for x in indices]

            # Create and merge all row combinations
            merged_slices = list([*full_vertex[x, y, :], *full_vertex[x, :, z], *full_vertex[:, y, z]] for x, y, z in indices)
        
            # Append result to data_in
            self.data_in_slices = torch.cat([self.data_in_slices, 
                                        torch.tensor(merged_slices, dtype=torch.float32)], axis=0)
            self.data_in_indices = torch.cat([self.data_in_indices, 
                                        torch.tensor(indices, dtype=torch.float32)], axis=0)
            assert self.data_in_indices.shape[0] == self.data_in_slices.shape[0]
        axis = config.construction_axis

        # Construct target data
        match axis:
            case 1: 
                self.data_target = deepcopy(self.data_in_slices[:, 2*length:])
                assert list(self.data_target[0]) == list(self.data_in_slices[0][2*length:])
            case 2:
                self.data_target = deepcopy(self.data_in_slices[:,length:2*length])
                assert list(self.data_target[0]) == list(self.data_in_slices[0][length:2*length])
            case 3:
                self.data_target = deepcopy(self.data_in_slices[:, :length])
                assert list(self.data_target[0]) == list(self.data_in_slices[0][:length])
            case _:
                raise NotImplementedError("Axis invalid")

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


class Dataset_ae_split(Dataset):
    ''' Compared to Dataset_ae, this dataset expects input data where the split between training and
        validation has already been performed, potentially by physically meaningful aspects. 
        >> VARIABLES:
        PATH_TRAIN: full path to data hdf5
        data_type: "train" or "valid"
    '''
    def __init__(self, config, **kwargs):
        PATH = config["PATH_TRAIN"]
        f = h5py.File(PATH, 'r')
        
        data_in = np.array(f[kwargs["data_type"]]["data"]["30.0"][:,0,:])
        data_target = np.array(f[kwargs["data_type"]]["data"]["30.0"][:,1,:])
        self.data_in = torch.cat([torch.tensor(data_in.real, dtype=torch.float32), torch.tensor(data_in.imag, dtype=torch.float32)], axis=1)
        self.data_target = torch.cat([torch.tensor(data_target.real, dtype=torch.float32), torch.tensor(data_target.imag, dtype=torch.float32)], axis=1)

    def __len__(self):
        return self.data_in.shape[0]

    def __getitem__(self, idx):
      if torch.is_tensor(idx):
          idx = idx.tolist()
      return self.data_in[idx], self.data_target[idx]


class Dataset_encgiv_split(Dataset):
    ''' Compared to Dataset_ae_split, this dataset returns the non-interacting G0 as input and target by solving the Dyson equation.
    '''
    def __init__(self, config, **kwargs):
        PATH = config["PATH_TRAIN"]
        f = h5py.File(PATH, 'r')
        g0 = np.array(f[kwargs["data_type"]]["data"][:,2,:])
        self.data_in = torch.cat([torch.tensor(g0.real, dtype=torch.float32), torch.tensor(g0.imag, dtype=torch.float32)], axis=1)
        self.data_target = self.data_in

    def __len__(self):
        return self.data_in.shape[0]

    def __getitem__(self, idx):
      if torch.is_tensor(idx):
          idx = idx.tolist()
      return self.data_in[idx], self.data_target[idx]


class Dataset_injection_split(Dataset):
    ''' Compared to Dataset_ae_split, this dataset returns the non-interacting G0 as input and target by solving the Dyson equation.
    '''
    def __init__(self, config, **kwargs):
        PATH = config["PATH_TRAIN"]
        f = h5py.File(PATH, 'r')
        data_in = np.array(f[kwargs["data_type"]]["data"][:,0,:])
        data_target = np.array(f[kwargs["data_type"]]["data"][:,1,:])
        self.data_in = torch.cat([torch.tensor(data_in.real, dtype=torch.float32), torch.tensor(data_in.imag, dtype=torch.float32)], axis=1)
        self.data_target = torch.cat([torch.tensor(data_target.real, dtype=torch.float32), torch.tensor(data_target.imag, dtype=torch.float32)], axis=1)

        g0 = np.array(f[kwargs["data_type"]]["data"][:,2,:])
        self.g0 = torch.cat([torch.tensor(g0.real, dtype=torch.float32), torch.tensor(g0.imag, dtype=torch.float32)], axis=1)

    def __len__(self):
        return self.data_in.shape[0]

    def __getitem__(self, idx):
      if torch.is_tensor(idx):
          idx = idx.tolist()
      return self.data_in[idx], self.data_target[idx], self.g0[idx]


class Dataset_convergence_split(Dataset):
    ''' Compared to Dataset_ae_split, this dataset returns the non-interacting G0 as input and target by solving the Dyson equation.
    '''
    def __init__(self, config, **kwargs):
        sample_idx = kwargs["target_sample"]
        amp = config["convergence_noise"]
        
        ### Using noise to produce a gradient
        # PATH = config["PATH_TRAIN"]
        # f = h5py.File(PATH, 'r')
        # data_in = np.array(f[kwargs["data_type"]]["data"][sample_idx,0,:])[None]
        # # data_target = np.array(f[kwargs["data_type"]]["data"][sample_idx,1,:])[None]
        # self.data_target = torch.cat([torch.tensor(data_in.real, dtype=torch.float32), torch.tensor(data_in.imag, dtype=torch.float32)], axis=1)
        # self.noise = amp * (np.random.rand(*data_in.shape)-0.5)*2 # in[-1,1]
        # data_in = data_in + self.noise
        # self.data_in = torch.cat([torch.tensor(data_in.real, dtype=torch.float32), torch.tensor(data_in.imag, dtype=torch.float32)], axis=1)
        # g0 = np.array(f[kwargs["data_type"]]["data"][sample_idx,2,:])[None]
        # self.g0 = torch.cat([torch.tensor(g0.real, dtype=torch.float32), torch.tensor(g0.imag, dtype=torch.float32)], axis=1)

        ### Using more validation samples with half of them the actual target to produce a gradient
        PATH = config["PATH_TRAIN"]
        f = h5py.File(PATH, 'r')
        data_in = []
        g0 = []
        for n in range(0,sample_idx.shape[0]):
            data_in.append(f[kwargs["data_type"]]["data"][n,0,:])
            g0.append(f[kwargs["data_type"]]["data"][n,2,:])
        data_in = torch.tensor(data_in)#[None] # BATCH DIMENSION
        g0 = torch.tensor(g0)#[None] # BATCH DIMENSION
        # data_target = np.array(f[kwargs["data_type"]]["data"][sample_idx,1,:])[None]
        self.data_target = torch.cat([torch.tensor(data_in.real, dtype=torch.float32), torch.tensor(data_in.imag, dtype=torch.float32)], axis=1)
        # self.noise = amp * (np.random.rand(*data_in.shape)-0.5)*2 # in[-1,1]
        # data_in = data_in + self.noise
        self.data_in = torch.cat([torch.tensor(data_in.real, dtype=torch.float32), torch.tensor(data_in.imag, dtype=torch.float32)], axis=1)
        self.g0 = torch.cat([torch.tensor(g0.real, dtype=torch.float32), torch.tensor(g0.imag, dtype=torch.float32)], axis=1)
        
        
    def __len__(self):
        return self.data_in.shape[0]

    def __getitem__(self, idx):
      if torch.is_tensor(idx):
          idx = idx.tolist()
      return self.data_in[idx], self.data_target[idx], self.g0[idx]


class Dataset_graph_split(Dataset):
    def __init__(self, config, **kwargs):
        self.config = config
        PATH = config["PATH_TRAIN"]
        f = h5py.File(PATH, 'r')

        self.data_in = np.array(f[kwargs["data_type"]]["data"][:,0,:])
        self.data_target = np.array(f[kwargs["data_type"]]["data"][:,1,:])
        
        self.n_nodes = config["n_nodes"]
        n_freq = self.data_in.shape[1]
        leg_pol = np.linspace(0, n_freq-1, self.n_nodes)
        beta = 30 ### Later this needs to be dynamics
        iv = np.linspace(0, (2*n_freq+1)*np.pi/beta, n_freq)
        iv2 = np.linspace(0, 1, n_freq)

        self.vectors = torch.zeros((n_freq, n_freq))
        for p in leg_pol:
            self.vectors[int(p),:] = torch.tensor(eval_legendre(int(p), iv2), dtype=torch.torch.float64)
        self.n_vectors = self.vectors.shape[0]
        
        edge_index = torch.zeros((2, self.n_nodes**2))
        k = 0
        for i in range(self.n_nodes):
            for j in range(self.n_nodes):
                edge_index[0, k] = i
                edge_index[1, k] = j
                k += 1
        self.edge_index = edge_index.long()

    def __len__(self):
        return self.data_in.shape[0]

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        # : Node Features
        node_features = torch.zeros((self.n_nodes, 3*self.n_vectors)) 
        for w in range(self.n_nodes):
            node_features[w,:] = torch.cat([self.vectors[w], torch.tensor(self.data_in[idx].real, dtype=torch.torch.float64), torch.tensor(self.data_in[idx].imag, dtype=torch.torch.float64)])
#         graph = Data(x = node_features, edge_index = self.edge_index, y = self.data_target[idx])
        sample = {}
        sample["node_feature"] = node_features
        sample["edge_index"] = self.edge_index
        sample["target"] = torch.tensor(self.data_target[idx].imag, dtype=torch.torch.float64)
        sample["vectors"] = self.vectors
        return sample #[node_features, self.edge_index, self.data_target[idx], self.vectors] #, graph


class Dataset_graph_direct_split(Dataset):
    def __init__(self, config, **kwargs):
        self.config = config
        PATH = config["PATH_TRAIN"]
        f = h5py.File(PATH, 'r')

        self.data_in = np.array(f[kwargs["data_type"]]["data"][:,0,:])
        self.data_target = np.array(f[kwargs["data_type"]]["data"][:,1,:])
        
        self.n_nodes = config["n_nodes"]
        self.n_freq = self.data_in.shape[1]
        leg_pol = np.linspace(0, self.n_freq-1, self.n_nodes)
        beta = 30 ### Later this needs to be dynamics
        iv = np.linspace(0, (2*self.n_freq+1)*np.pi/beta, self.n_freq)
        iv2 = np.linspace(0, 1, self.n_freq)

        edge_index = torch.zeros((2, self.n_nodes**2))
        k = 0
        for i in range(self.n_nodes):
            for j in range(self.n_nodes):
                edge_index[0, k] = i
                edge_index[1, k] = j
                k += 1
        self.edge_index = edge_index.long()

    def __len__(self):
        return self.data_in.shape[0]

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        # : Node Features
        node_features = torch.zeros((self.n_nodes, self.n_freq)) 
        for w in range(self.n_nodes):
            node_features[w,:] = torch.cat([torch.tensor(self.data_in[idx].real, dtype=torch.torch.float64), torch.tensor(self.data_in[idx].imag, dtype=torch.torch.float64)])
#         graph = Data(x = node_features, edge_index = self.edge_index, y = self.data_target[idx])
        sample = {}
        sample["node_feature"] = node_features
        sample["edge_index"] = self.edge_index
        sample["target"] = torch.tensor(self.data_target[idx].imag, dtype=torch.torch.float64)
        return sample 


class Dataset_RevMat_ae(Dataset):

    def __init__(self, config, **kwargs):
        self.data_in = []
        self.data_target = []
        PATH = config["PATH_TRAIN"]
        print(config["PATH_TRAIN"])
        f = h5py.File(PATH, 'r')
        for k1 in f.keys():
            if k1 != '.tempaxis':
                for k2 in f[k1].keys():
                    for k3 in f[k1][k2].keys():
                        ### VERSION LIN
                        # x = np.array([float(k1),float(k2),1e6*float(k3)])
                        ### VERSION LOGLOG
                        # x = np.array([np.log(float(k1)),np.log(float(k2)),np.log(float(k3))])
                        ### VERSION SEMILOG
                        # x = np.array([(float(k1)),(float(k2)),np.log(float(k3))])
                        ### VERSION SEMILOG 2 VALUES
                        x = np.array([(float(k2)),np.log(float(k3))])
                        self.data_target.append(x)
                        y = f[k1][k2][k3]['kubo']['L11']
                        ymin = np.min(y)
                        # y -= ymin # For log
                        ymax = np.max(y)
                        y = y/ymax
                        # y = y + 2
                        y_fin = np.zeros(y.shape[0]+2)
                        y_fin[0] = ymin
                        y_fin[1] = ymax
                        y_fin[2:] = np.log(y)
                        # print("----- ")
                        # print(y[0:10])
                        # print(y_fin[0:10])
                        # print("----- ")
                        self.data_in.append(y_fin)
        self.data_in = torch.tensor(self.data_in, dtype=torch.torch.float64)
        self.data_target = torch.tensor(self.data_target, dtype=torch.torch.float64)
        # self.data_in.astype('double')
        # self.data_target.astype('double')

    def __len__(self):
        return self.data_in.shape[0]

    def __getitem__(self, idx):
      if torch.is_tensor(idx):
          idx = idx.tolist()
      return self.data_in[idx], self.data_target[idx]

class Dataset_generic(Dataset):
    def __init__(self, config, dataset):
        
        ### MODIFY THIS PART ACCORDING TO YOUR DATA        
        data_in = dataset[:,0]
        data_target = dataset[:,1]
        # self.data_target = torch.cat([torch.tensor(data_target.real, dtype=torch.float32), torch.tensor(data_target.imag, dtype=torch.float32)], axis=1)
        # self.data_in = torch.cat([torch.tensor(data_in.real, dtype=torch.float32), torch.tensor(data_in.imag, dtype=torch.float32)], axis=1)
        self.data_target = torch.tensor(data_target.imag, dtype=torch.float32)
        self.data_in = torch.tensor(data_in.imag, dtype=torch.float32)
        
    def __len__(self):
        return self.data_in.shape[0]

    def __getitem__(self, idx):
      if torch.is_tensor(idx):
          idx = idx.tolist()
      return self.data_in[idx], self.data_target[idx]


class Dataset_graph_generic(Dataset):
    def __init__(self, config, dataset):

        ### MODIFY THIS PART ACCORDING TO YOUR DATA        
        data_in = dataset[:,0]
        data_target = dataset[:,1]
        # self.data_target = torch.cat([torch.tensor(data_target.real, dtype=torch.float32), torch.tensor(data_target.imag, dtype=torch.float32)], axis=1)
        # self.data_in = torch.cat([torch.tensor(data_in.real, dtype=torch.float32), torch.tensor(data_in.imag, dtype=torch.float32)], axis=1)
        self.data_target = torch.tensor(data_target.imag, dtype=torch.torch.float64)
        self.data_in = torch.tensor(data_in.imag, dtype=torch.torch.float64)
        # torch.tensor(self.data_target[idx].imag, dtype=torch.torch.float64)
        # print(self.data_in.shape)
        # print("okok")
        # p = kk
        
        self.n_nodes = config["n_nodes"]
        n_freq = self.data_in.shape[1]
        # leg_pol = np.linspace(0, n_freq-1, self.n_nodes)
        leg_pol = np.linspace(0, config["nr_coefficients"]-1, config["nr_coefficients"])
        beta = 30 ### Later this needs to be dynamics
        iv = np.linspace(0, (2*n_freq+1)*np.pi/beta, n_freq)
        iv2 = np.linspace(0, 1, n_freq)

        self.vectors = torch.zeros((config["nr_coefficients"], n_freq))
        for p in leg_pol:
            self.vectors[int(p),:] = torch.tensor(eval_legendre(int(p), iv2), dtype=torch.torch.float64)
        self.n_vectors = self.vectors.shape[0]
        
        edge_index = torch.zeros((2, self.n_nodes**2))
        k = 0
        for i in range(self.n_nodes):
            for j in range(self.n_nodes):
                edge_index[0, k] = i
                edge_index[1, k] = j
                k += 1
        self.edge_index = edge_index.long()

    def __len__(self):
        return self.data_in.shape[0]

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        # : Node Features
        node_features = torch.zeros((self.n_nodes, 2*self.vectors.shape[1])) 
        for w in range(self.n_nodes):
            node_features[w,:] = torch.cat([self.vectors[w], self.data_in[idx]])
#         graph = Data(x = node_features, edge_index = self.edge_index, y = self.data_target[idx])

        sample = {}
        sample["node_feature"] = node_features
        sample["edge_index"] = self.edge_index
        sample["target"] = self.data_target[idx]
        sample["vectors"] = self.vectors
        return sample #[node_features, self.edge_index, self.data_target[idx], self.vectors] #, graph
