import importlib
import json
import sys

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from src.config import Config


@dataclass
class SlurmOptions:
    n: int = 1
    mail_type: str = 'BEGIN'
    mail_user: str = '<email@address.at>'
    partition: str = 'zen3_0512_a100x2'    # see available resources on VSC via `sqos`-command
    qos: str = 'zen3_0512_a100x2'
    ntasks_per_node: int = 128             # max. <#devices of partition> (CPUs or GPUs)
    nodes: int = 1
    time: str = '01:00:00'                 # must be <= '00:10:00' for '_devel' nodes
    device_type: Literal['cpu', 'gpu', 'mps', 'xla', 'hpu'] = 'gpu'   # must be set according to the `partition`

    def __post_init__(self) -> None:
        self.partition = self.qos.removesuffix('_devel')
    
    def set_from_config(self, config: Config) -> None:
        self.nodes = config.num_nodes
        self.ntasks_per_node = config.devices // self.nodes


def create_train_script(project_name: str, base_dir: str|Path, device_type: Literal['cpu', 'gpu'],
                        trainer: str|None = None, trainer_kwargs: dict[str, Any]|None = None) -> None:
    def create_args_string(dic: dict[str, Any]) -> str:
        return ', '.join([f'{k}={repr(v)}' for k, v in dic.items()])
    
    etxra_kwargs = trainer_kwargs.pop('config_kwargs', {})
    trainer_kwargs['device_type'] = device_type
    args_string = create_args_string(trainer_kwargs)
    args_string += ', ' + create_args_string(etxra_kwargs)

    s = f"""import sys, os
sys.path.append(os.getcwd())

from src.trainer import TrainerModes, {trainer}


def train():
    trainer = {trainer}('{project_name}', {args_string})
    trainer.train(train_mode=TrainerModes.SLURM)


if __name__ == '__main__':
    train()
"""
    fdir = Path(base_dir, 'train_scripts', project_name)
    fdir.mkdir(parents=True, exist_ok=True)
    (fdir / f'train_{project_name}.py').write_text(s)


def create(project_name: str, script_name: str, pyenv_dir: str, pyenv_name: str, 
           path_to_repo: str, slurm_options: SlurmOptions = SlurmOptions(), 
           train_script_name: str|None = None, trainer: str|None = None, 
           trainer_kwargs: dict[str, Any]|None = None) -> None:
    """
    Create a slurm script and optionally create a python train-script.

    Parameters
    ----------
    project_name : str
        A name for the project.
    script_name : str
        A name for the generated slurm-file.
    pyenv_dir : str
        Path to the directory of the python environment.\n
        Given directory should contain the `/bin/` folder.
    pyenv_name : str
        Name of the python environment.
    path_to_repo : str
        Path to the LuttingerWard_from_ML repository.\n
        Given directory should contain the `/PhysML/` folder.
    slurm_options : SlurmOptions, optional
        `SlurmOptions`-instance containing the SBATCH options.
    train_script_name : str | None, optional
        Name of the train-script in the `train_scripts` folder to use.\n
        If None, generate a train-script. (defaults to None)
    trainer : str | None, optional
        Name of trainer-class to use in the train-script.\n
        Only required if `train_script_name` is not given. (defaults to None)
    trainer_kwargs : dict[str, Any] | None, optional
        Kwargs for instantiating the trainer-class in the train-script.\n
        Only required if `train_script_name` is not given. (defaults to None)
    """
    config_cls: type[Config] = getattr(importlib.import_module('src.trainer'), trainer).config_cls
    config = config_cls.from_json(trainer_kwargs['config_name'], trainer_kwargs['subconfig_name'], 
                                  trainer_kwargs['config_dir'], **trainer_kwargs['config_kwargs'])
    slurm_options.set_from_config(config)
    venv_files = Path(pyenv_dir, '*').as_posix()
    source_path = Path(pyenv_dir, 'bin/activate').as_posix()
    current_base_dir = Path(__file__).parent.parent.parent
    
    if not train_script_name:
        create_train_script(project_name, current_base_dir, slurm_options.device_type, trainer, trainer_kwargs)
        train_script_name = f'train_{project_name}.py'
    train_script_path = Path(path_to_repo, 'LuttingerWard_from_ML', 'train_scripts', project_name, 
                             train_script_name).as_posix()
    
    s = f"""#!/bin/bash
#
#SBATCH -J {project_name}
#SBATCH -N {slurm_options.n}
#SBATCH --mail-type={slurm_options.mail_type}    # first have to state the type of event to occur 
#SBATCH --mail-user={slurm_options.mail_user}    # and then your email address

#SBATCH --partition={slurm_options.partition}
#SBATCH --qos {slurm_options.qos}
#SBATCH --ntasks-per-node={slurm_options.ntasks_per_node}
#SBATCH --nodes={slurm_options.nodes}
#SBATCH --time={slurm_options.time}

FILES=({venv_files})
source {source_path}
conda init bash
conda activate {pyenv_name}

srun python {train_script_path}
"""
    fdir = current_base_dir / 'slurm' / project_name
    fdir.mkdir(parents=True, exist_ok=True)
    (fdir / f'{script_name}.slrm').write_text(s)


def create_from_config(config_file: str|Path, slurm_config_name: str = 'SLURM_CONFIG') -> None:
    """
    Reads the section with the key-name given by `slurm_config_name` from a config-JSON 
    and creates a SLURM-script and train-script from the settings contained.

    Parameters
    ----------
    config_file : str | Path
        Path to config-JSON.
    slurm_config_name : str, optional
        Name of the section in the config-JSON to read the settings from.\n
        (defaults to `'SLURM_CONFIG'`)
    """
    with open(config_file) as f:
        config: dict[str, Any] = json.load(f)
    config = config[slurm_config_name]
    if 'slurm_options' in config:
        config['slurm_options'] = SlurmOptions(**config['slurm_options'])
    create(**config)


if __name__ == '__main__':
    create_from_config(sys.argv[1])
