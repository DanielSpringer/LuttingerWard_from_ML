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
Create a new ML-project by deriving classes from the provided base classes and overwrite methods were necessary for customization. It's recommended to put your custom classes in a module-file named after your ML-project in the respective component folders (e.g. `models`, `trainer`, `load_data`, etc.)
1. Create a config-JSON in containing required parameters. Add a sub-section with SLURM-settings if you want to run the model via SLURM.
2. Create a config-class if additional attributes are required. Tip: You can also create the config-class with default parameter-values first and then save it as a config-JSON by calling the `save`-method.
   - Add a new class to `/src/config.py`, inheriting from `src.configs.base.Config`.
   - Add additional attributes.
   - For attributes of type `type`, the fully qualified class-name has to be set.
     - For convenience of accessing/importing types from strings, add each attribute as `_<attribute-name>`, add `<attribute-name>_kwargs` to store arguments for later instantiation and add property-getter, -setter and a `get_<attribute-name>`-method which already returns an imported instance of from the provided class-name.
3. Create a Dataset-class:
   - Add a new class to `/src/load_data.py`, inheriting from `src.load_data.base.FilebasedDataset`, which is based on `torch.utils.data.Dataset`.
   - As usual overwrite the `torch.utils.data.Dataset`-methods where required.
4. Create a model class:
   - Add a new class to `/src/models/models_v2.py`, inheriting from `src.models.base.BaseModule`.
   - Overwrite at least the `forward`-method.
5. Create a wrapper-class if custom behaviour is required:
   - Add a new class to `/src/wrapper.py`, inheriting from `src.wrapper.base.BaseWrapper`.
   - Overwrite methods where custom behaviour is required.
6. Create a trainer-class if custom behaviour is required:
   - Add a new class to `/src/trainer.py`, inheriting from `src.trainer.base.BaseTrainer`.
   - Overwrite methods where custom behaviour is required.

### Usage
There are different options to run the trainer and generate scripts to run it on the cluster:
* run training directly from a Jupyter-notebook by instantiating a Trainer-class in `src.trainer` and running the `train`-method with the appropriate `train_mode`. By using the `config_kwargs`-argument, you can overwrite parameters in the config for this Trainer-instance.
* call `src.utils.slurm_generate.create` to generate a SLURM-script and a Python train-script to run the training via SLURM. You can also overwrite parameters from the config here by using the `trainer_kwargs`-argument.
* run `src/util/slurm_generate.py` directly from the command line to generate a SLURM-script and a Python train-script to run the training via SLURM. You can specify the name of the subsection containing the SLURM-settings as an optional argument.  
    ```shell
    python src/util/slurm_generate.py path/to/config/json <OPTIONAL_NAME_OF_SLURM_SETTINGS_SECTION>
    ```


**Tipps & Tricks:**
- don't copy classes: derive them and overwrite methods where necessary
- don't duplicate code: if a method needs partial customization, try to split it into multiple methods and overwrite only the necessary ones
- variables from one class can be easily shared to other classes by writing it into the config-instance

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
