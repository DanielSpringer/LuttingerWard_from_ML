# LuttingerWard_from_ML

## Usage
### Setup environment
```shell
conda env create -f conda/conda_ml24_1.yml
```

### Run training
* create config in `/configs/`
* local execution:
  * TODO
* execution on VSC cluster:
  * create slurm in `/slurm/`
  * deploy slurm:
    ```shell
    sbatch [slurm_filename].slrm
    ```
* Jupyter-hub on VSC:  
  https://jupyterhub.vsc.ac.at/hub/
  * install dependencies (run in Jupyter-cell):
    ```jupyter
    !pip install torch-geometric
    !pip install pytorch-lightning
    !pip install h5py
    !pip install matplotlib
    !pip install tensorboard
    
    from IPython.display import display, HTML
    display(HTML("<style>.jp-Cell { width: 100% !important;margin-left:00px}</style>"))
    ```
* start tensorboard:
  ```shell
  tensorboard --logdir=saves
  ```


## ML-framework
Create new ML-project by deriving classes from the provided base classes and overwrite methods were necessary for customization.
1. Create a config-class if additional attributes are required:
   - Add a new class to `/src/config.py`, inheriting from `src.configs.Config`.
   - Add additional attributes.
   - For attributes of type `type`, the fully qualified class-name has to be set.
     - For convenience of accessing/importing types from strings, add each attribute as `_<attribute-name>`, add `<attribute-name>_kwargs` to store arguments for later instantiation and add property-getter, -setter and a `get_...`-method which returns an instantiation.
2. Create a Dataset-class:
   - Add a new class to `/src/load_data.py`, inheriting from `src.load_data.FilebasedDataset`.
3. Create a model class:
   - Add a new class to `/src/models/models_v2.py`, inheriting from `src.models.BaseModule`.
4. Create a wrapper-class if custom behaviour is required:
   - Add a new class to `/src/wrapper.py`, inheriting from `src.wrapper.BaseWrapper`.
   - Overwrite methods where custom behaviour is required.
5. Create a trainer-class if custom behaviour is required:
   - Add a new class to `/src/trainer.py`, inheriting from `src.trainer.BaseTrainer`.
   - Assign the corresponding Config-class to the `config_cls` class attribute.
   - Overwrite methods where custom behaviour is required.

see `ML_framework_demo.ipynb` for example usage.


---


## OLD README
### Setup

Install [pytorch](https://pytorch.org/get-started/locally/) and pytorch lightning
Dependencies (TODO complete list)
```
conda create -name LW_ML
conda install -c conda-forge neptune-client
```

For deployment use
```
python -m build
python -m pip install .
```

For development use:
```
python -m build
python -m pip install -e .
```


### TODO:
content:
    - define s

code: 
    - define standardized input from jED, split off test data as separate file
    - finish refactor
        - I think the wrappers are a needless complication for now
    - set up git LFS for saves (also, move to neptune AI?)
    - documentation
    - complete requirements.txt/pyproject.toml
