import json
import sys

from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass
class SlurmOptions:
    n: int = 1
    mail_type: str = 'BEGIN'
    mail_user: str = '<email@address.at>'
    partition: str = 'zen3_0512_a100x2'
    qos: str = 'zen3_0512_a100x2'
    ntasks_per_node: int = 128
    nodes: int = 1
    time: str = '01:00:00'


def create_train_script(project_name: str, base_dir: str|Path, trainer: str|None = None, 
                        trainer_kwargs: dict[str, Any]|None = None) -> None:
    s = f"""import sys, os
sys.path.append(os.getcwd())

from src.trainer import TrainerModes, {trainer}


def train():
    trainer = {trainer}('{project_name}', '{trainer_kwargs['config_name']}', '{trainer_kwargs['subconfig_name']}')
    trainer.train(train_mode=TrainerModes.SLURM)
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

    :param project_name: Name of the project.
    :type project_name: str
    
    :param script_name: Name of the generated slurm-file.
    :type script_name: str
    
    :param venv_dir: Path to the directory of the python environment. 
                     Given directory should contain the `/bin/` folder.
    :type venv_dir: str
    
    :param venv_name: Name of the python environment.
    :type venv_name: str
    
    :param path_to_repo: Path to the LuttingerWard_from_ML repository. 
                         Given directory should contain the `/LuttingerWard_from_ML/` folder.
    :type path_to_repo: str

    :param slurm_options: SlurmOptions instance containing the SBATCH options. (defaults to None)
    :type slurm_options: SlurmOptions | None, optional
    
    :param train_script_name: Name of the train-script in the `train_scripts` folder to use.
                              If None, generate a train-script. (defaults to None)
    :type train_script_name: str | None, optional
    
    :param trainer: If generating a train-script, name of trainer-class to use in the train-script.
                    Must be provided if no train_script_name is given. (defaults to None)
    :type trainer: str | None, optional
    
    :param trainer_kwargs: If generating a train-script, kwargs for instantiating the trainer-class in the train-script. 
                           Must be provided if no train_script_name is given. (defaults to None)
    :type trainer_kwargs: dict[str, Any] | None, optional
    """
    venv_files = Path(pyenv_dir, '*').as_posix()
    source_path = Path(pyenv_dir, 'bin/activate').as_posix()
    current_base_dir = Path(__file__).parent.parent.parent
    
    if not train_script_name:
        create_train_script(project_name, current_base_dir, trainer, trainer_kwargs)
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
    Reads the section with the key `slurm_config_name` from a config-JSON and runs `create` from the settings contained.

    :param config_file: Path to config-JSON
    :type config_file: str | Path

    :param slurm_config_name: Name of the section in the config-JSON to read the settings from. (defaults to 'SLURM_CONFIG')
    :type slurm_config_name: str, optional
    """
    with open(config_file) as f:
        config: dict[str, Any] = json.load(f)
    config = config[slurm_config_name]
    create(**config)


if __name__ == '__main__':
    create_from_config(sys.argv[1])
