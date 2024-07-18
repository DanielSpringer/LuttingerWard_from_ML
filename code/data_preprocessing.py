#%%
import numpy as np
import h5py
import matplotlib.pyplot as plt

### DATA STRUCTURE
### Rewriting all data into an array [beta, ek, vk, [G, S]]
PATH = "../data/U2c0_b10b50.jld2"
f = h5py.File(PATH, 'r')

### Unique parameters
unique_ek = []
unique_vk = [] 
unique_beta = [] 
for n,para in enumerate(f["Set1"]["Parameters"]):
    if para[0] not in unique_ek:
        unique_ek.append(para[0])
    if para[2] not in unique_vk:
        unique_vk.append(para[2])
    if para[6] not in unique_beta:
        unique_beta.append(para[6])
unique_beta = np.array(unique_beta)
unique_ek = np.array(unique_ek)
unique_vk = np.array(unique_vk)

### Book-keeping between indices of 'data' and actual parameter [beta, ek, vk]
idx_to_para = np.zeros((unique_beta.shape[0], unique_ek.shape[0], unique_vk.shape[0], 3))
n = 0
for b, beta in enumerate(unique_beta):    
    for e, ek in enumerate(unique_ek):    
        for v, vk in enumerate(unique_vk):    
            idx_to_para[b,e,v] = np.array([beta, ek, vk])

### Array
data = np.zeros((unique_beta.shape[0], unique_ek.shape[0], unique_vk.shape[0], 2, 200), dtype=complex)
for n,para in enumerate(f["Set1"]["Parameters"]):
    # jld2 Format fksp complex numbers and stores them into tuples.
    g = []
    s = []
    for gg in f["Set1"]["GImp"][n,:]:
        g.append(gg[0] + 1j*gg[1])
    g = np.array(g)
    for ss in f["Set1"]["SImp"][n,:]:
        s.append(ss[0] + 1j*ss[1])
    s = np.array(s)
    idx = np.argwhere((idx_to_para == np.array([para[6],  para[0], para[2]])).all(-1))[0]
    data[idx[0],idx[1],idx[2],0] = g
    data[idx[0],idx[1],idx[2],1] = s



### STRATEGY 1
### Detecting g_max and s_max as a function of ek (vk fixed) and vk (ek fixed)
nb = unique_beta.shape[0]
ne = unique_ek.shape[0]
nv = unique_vk.shape[0]
g_max_e = np.zeros((nb,ne,2))
g_max_v = np.zeros((nb,nv,2))
s_max_e = np.zeros((nb,ne,2))
s_max_v = np.zeros((nb,nv,2))
for b in range(0,nb):
    for e in range(0,ne):
        for v in range(0,nv):
            if data[b,e,v,0,0].imag < g_max_e[b,v,1]:
                g_max_e[b,v,0] = e
                g_max_e[b,v,1] = data[b,e,v,0,0].imag
            if data[b,e,v,0,0].imag < g_max_v[b,e,1]:
                g_max_v[b,e,0] = v
                g_max_v[b,e,1] = data[b,e,v,0,0].imag

            if data[b,e,v,1,0].imag < s_max_e[b,v,1]:
                s_max_e[b,v,0] = e
                s_max_e[b,v,1] = data[b,e,v,1,0].imag
            if data[b,e,v,1,0].imag < s_max_v[b,e,1]:
                s_max_v[b,e,0] = v
                s_max_v[b,e,1] = data[b,e,v,1,0].imag



###
data_train_1 = []
data_valid_1 = []
para_train_1 = []
para_valid_1 = []
for b in range(0,nb):
    for e in range(0,ne):
        for v in range(0,nv):
            if v < g_max_v[b,e,0] - 10 or v > g_max_v[b,e,0] + 10:
                data_train_1.append(data[b,e,v])
                para_train_1.append(idx_to_para[b,e,v])
            else:
                data_valid_1.append(data[b,e,v])
                para_valid_1.append(idx_to_para[b,e,v])

data_train_1 = np.array(data_train_1)
data_valid_1 = np.array(data_valid_1)
para_train_1 = np.array(para_train_1)
para_valid_1 = np.array(para_valid_1)

#%%
f = h5py.File("../data/U2c0_b10b50_gmax_10x10.hdf5", "w")
f.create_dataset('train/data', data = data_train_1)
f.create_dataset('train/parameters', data = para_train_1)
f.create_dataset('valid/data', data = data_valid_1)
f.create_dataset('valid/parameters', data = para_valid_1)
f.close()
#%%
import h5py
f = h5py.File("/gpfs/data/fs72150/springerd/Projects/LuttingerWard_from_ML/data/U2c0_b10b50_gmax_10x10.hdf5", "r")


print(f["train"].keys())
print(f["train"]["parameters"].shape)
print(f["train"]["data"].shape)
print(f["train"]["parameters"][35000])

f.close()
