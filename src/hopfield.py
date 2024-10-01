#%%
# !pip3 install git+https://github.com/ml-jku/hopfield-layers

#%%
from hflayers import Hopfield
# import numpy as np

hopfield = Hopfield(
    input_size=200)

print(hopfield)