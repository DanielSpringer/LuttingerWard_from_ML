#%%
# !pip3 install git+https://github.com/ml-jku/hopfield-layers

#%%
from hflayers import Hopfield
import numpy as np
import torch 


hopfield = Hopfield(
    num_heads=10,
    input_size=200,
    hidden_size=64,
    pattern_projection_size=200)

data = torch.rand(1,10,200)
basis = torch.rand(1,200,200)
print(data.shape)
out = hopfield((basis, data, basis))
print(out.shape)


# %%
print(hopfield)
