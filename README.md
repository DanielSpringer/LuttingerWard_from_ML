# LuttingerWard_from_ML

# Setup

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

# TODO:

content:
    - define s

code: 
    - define standardized input from jED, split off test data as separate file
    - finish refactor
        - I think the wrappers are a needless complication for now
    - set up git LFS for saves (also, move to neptune AI?)
    - documentation
    - complete requirements.txt/pyproject.toml