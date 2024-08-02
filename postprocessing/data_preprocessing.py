#%%
import numpy as np
import h5py
import matplotlib.pyplot as plt
import os
os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"
    
    
### DATA STRUCTURE
### Rewriting all data into an array [beta, ek, vk, [G, S]]
PATH = "../data/U2c0_b30.jld2"
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


#%%
# ### STRATEGY 1
# ### Detecting g_max and s_max as a function of ek (vk fixed) and vk (ek fixed)
# nb = unique_beta.shape[0]
# ne = unique_ek.shape[0]
# nv = unique_vk.shape[0]
# g_max_e = np.zeros((nb,ne,2))
# g_max_v = np.zeros((nb,nv,2))
# s_max_e = np.zeros((nb,ne,2))
# s_max_v = np.zeros((nb,nv,2))
# for b in range(0,nb):
#     for e in range(0,ne):
#         for v in range(0,nv):
#             if data[b,e,v,0,0].imag < g_max_e[b,v,1]:
#                 g_max_e[b,v,0] = e
#                 g_max_e[b,v,1] = data[b,e,v,0,0].imag
#             if data[b,e,v,0,0].imag < g_max_v[b,e,1]:
#                 g_max_v[b,e,0] = v
#                 g_max_v[b,e,1] = data[b,e,v,0,0].imag

#             if data[b,e,v,1,0].imag < s_max_e[b,v,1]:
#                 s_max_e[b,v,0] = e
#                 s_max_e[b,v,1] = data[b,e,v,1,0].imag
#             if data[b,e,v,1,0].imag < s_max_v[b,e,1]:
#                 s_max_v[b,e,0] = v
#                 s_max_v[b,e,1] = data[b,e,v,1,0].imag

# ###
# data_train_1 = []
# data_valid_1 = []
# para_train_1 = []
# para_valid_1 = []
# for b in range(0,nb):
#     for e in range(0,ne):
#         for v in range(0,nv):
#             if v < g_max_v[b,e,0] - 10 or v > g_max_v[b,e,0] + 10:
#                 data_train_1.append(data[b,e,v])
#                 para_train_1.append(idx_to_para[b,e,v])
#             else:
#                 data_valid_1.append(data[b,e,v])
#                 para_valid_1.append(idx_to_para[b,e,v])

# data_train_1 = np.array(data_train_1)
# data_valid_1 = np.array(data_valid_1)
# para_train_1 = np.array(para_train_1)
# para_valid_1 = np.array(para_valid_1)

# f = h5py.File("../data/U2c0_b10b50_gmax_10x10.hdf5", "w")
# f.create_dataset('train/data', data = data_train_1)
# f.create_dataset('train/parameters', data = para_train_1)
# f.create_dataset('valid/data', data = data_valid_1)
# f.create_dataset('valid/parameters', data = para_valid_1)
# f.close()

#%%
### STRATEGY 2
### If gradient of sigma is positive at origin -> Insulator
data_split = {}
data_split["met"] = {}
data_split["iso"] = {}
phase = np.zeros((100,100))
grad = np.zeros((100,100))

for b in range(0,1): # Currently only a single beta in each file (Julia script does that)
    met = []
    iso = []
    for n1 in range(0,100):
        for n2 in range(0,100):
            d = data[b,n1,n2,1,1].imag - data[b,n1,n2,1,0].imag
            grad[n1,n2] = d
            if d < 0:
                phase[n1,n2] = -1
                met.append([n1,n2])
            if d > 0:
                phase[n1,n2] = 1
                iso.append([n1,n2])
    met = np.array(met)
    iso = np.array(iso)
    data_split["iso"]["parameters"] = {}
    data_split["met"]["parameters"] = {}
    data_split["iso"]["data"] = {}
    data_split["met"]["data"] = {}
    data_split["iso"]["parameters"][str(unique_beta[b])] = idx_to_para[b,iso[:,0],iso[:,1]]
    data_split["met"]["parameters"][str(unique_beta[b])] = idx_to_para[b,met[:,0],met[:,1]]
    data_split["iso"]["data"][str(unique_beta[b])] = data[b,iso[:,0],iso[:,1]]
    data_split["met"]["data"][str(unique_beta[b])] = data[b,met[:,0],met[:,1]]

print(data_split["met"]["data"]["30.0"].shape)


f = h5py.File(f"../data/Strategy_2/U2c0_b{unique_beta[0]}_metiso_1.hdf5", "w")
for b in unique_beta:
    print(b)
    f.create_dataset(f'metallic/data/{str(b)}', data = data_split["met"]["data"][str(b)])
    f.create_dataset(f'metallic/parameters/{str(b)}', data = data_split["met"]["parameters"][str(b)])
    f.create_dataset(f'insulating/data/{str(b)}', data = data_split["iso"]["data"][str(b)])
    f.create_dataset(f'insulating/parameters/{str(b)}', data = data_split["iso"]["parameters"][str(b)])
f.close()


#%%

fig, ax = plt.subplots(ncols=2, nrows=1, figsize=(15,5))
p = ax[0].imshow(phase[:,:].real, aspect="auto", cmap="jet")
ax[0].set_xlabel("ek")
ax[0].set_ylabel("vk")
ax[0].set_title("Metall/Insulating")
# fig.colorbar(p, ax=ax[0])

p = ax[1].imshow(grad[:,:].real, aspect="auto", cmap="jet")
ax[1].set_xlabel("ek")
ax[1].set_ylabel("vk")
ax[1].set_title("Sigma Gradient")
fig.colorbar(p, ax=ax[1])


















############################################################################################################
############################################################################################################
############################################################################################################
############################################################################################################
############################################################################################################







#%%
print(data.shape)
print(unique_beta)


#%%
x = np.random.rand(5,5,9)
n = [1,2]
print(x[n,0].shape)

#%%
import matplotlib.pyplot as plt




# %%
import h5py
f = h5py.File("/gpfs/data/fs72150/springerd/Projects/LuttingerWard_from_ML/data/Strategy_2/U2c0_b30.0_metiso_1.hdf5")
print(f["metallic"]["data"].keys())
print(f["metallic"]["data"]["30.0"].shape)
# print(f["insulating"]["data"]["30.0"].shape)
# data = f

fig, ax = plt.subplots(ncols=2, nrows=1, figsize=(20,5))
fign = 0
ax[fign].plot(f["metallic"]["data"]["30.0"][0,0,:].imag)
ax[fign].plot(f["metallic"]["data"]["30.0"][110,0,:].imag)
fign = 1
ax[fign].plot(f["insulating"]["data"]["30.0"][0,0,:].imag)
ax[fign].plot(f["insulating"]["data"]["30.0"][110,0,:].imag)

fig, ax = plt.subplots(ncols=2, nrows=1, figsize=(20,5))
fign = 0
ax[fign].plot(f["metallic"]["data"]["30.0"][0,1,:].imag)
ax[fign].plot(f["metallic"]["data"]["30.0"][110,1,:].imag)
fign = 1
ax[fign].plot(f["insulating"]["data"]["30.0"][0,1,:].imag)
ax[fign].plot(f["insulating"]["data"]["30.0"][110,1,:].imag)


#%%

print([met[0]])
print(data.shape)

fig, ax = plt.subplots(ncols=2, nrows=1, figsize=(20,5))
fign = 0
# ax[fign].plot(data_split["iso"]['10.0'][2020,1,:20].imag)
# ax[fign].plot(data_split["iso"]['50.0'][2020,1,:20].imag)
ax[fign].plot(data[0,met[20,0],met[20][1],1,:20].imag)
ax[fign].plot(data[0,met[1220,0],met[1220][1],1,:20].imag)
fign = 1
ax[fign].plot(data[0,iso[0,0],iso[0][1],1,:20].imag)
ax[fign].plot(data[0,iso[0,0],iso[0][1],1,:20].imag)
# ax[fign].plot(data[0,iso[20,0],iso[20][1],1,:20].imag)
# ax[fign].plot(data[0,iso[1220,0],iso[1220][1],1,:20].imag)


# fig, ax = plt.subplots(ncols=2, nrows=1, figsize=(20,5))
# fign = 0
# ax[fign].plot(data[0,met[0,0],met[0,1],0,:20].imag)
# ax[fign].plot(data[0,met[20,0],met[20,1],0,:20].imag)
# ax[fign].plot(data[0,met[1220,0],met[1220,1],0,:20].imag)
# fign = 1
# ax[fign].plot(data[0,iso[0,0],iso[0,1],0,:20].imag)
# ax[fign].plot(data[0,iso[20,0],iso[20,1],0,:20].imag)
# ax[fign].plot(data[0,iso[1220,0],iso[1220,1],0,:20].imag)

#%%
n1 = np.linspace(0,99,10, dtype=int)
n2 = np.linspace(0,99,10, dtype=int)
print(n1)
fig, ax = plt.subplots(ncols=3, nrows=1, figsize=(20,5))
fign = 0
for n in n1[2:8]:
    ax[fign].plot(data[0,n,40,0,:20].imag)
fign = 1
for n in n1[2:8]:
    ax[fign].plot(data[0,n,40,1,:20].imag)
fign = 2
for n in n2:
    ax[fign].plot(data[0,10,n,0,:20].imag)

#%%
import h5py
import matplotlib.pyplot as plt


f = h5py.File("/gpfs/data/fs72150/springerd/Projects/LuttingerWard_from_ML/data/Strategy_1/U2c0_b10_gmax_10x10.hdf5")
print(f.keys())
print(f['train'].keys())
print(f['train']['data'].shape)
print(f['valid']['data'].shape)

fig, ax = plt.subplots(ncols=3, nrows=1, figsize=(20,5))
fign = 0
ax[fign].plot(f['train']['data'][5000,0,:].imag)
fign = 1
ax[fign].plot(f['train']['data'][5000,1,:])


print(f['train']['data'][5000,0,:3])
print(f['train']['data'][5000,1,:3])
