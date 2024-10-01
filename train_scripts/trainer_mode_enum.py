from enum import Enum
class TrainerModes(Enum):
    SLURM = 1
    JUPYTERGPU = 2
    JUPYTERCPU = 3