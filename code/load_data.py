from torch.utils.data import Dataset
import torch
import numpy as np
import h5py
from scipy.special import eval_legendre

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


class Dataloader_graph(Dataset):
    def __init__(self, config):
        self.config = config
        PATH = config["PATH_TRAIN"]
        f = h5py.File(PATH, 'r')
        self.data_in = np.array(f["Set1"]["GImp"])
        self.data_target = np.array(f["Set1"]["SImp"])
        
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
    
