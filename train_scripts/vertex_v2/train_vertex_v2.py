import sys, os
sys.path.append(os.getcwd())

from src.trainer import TrainerModes, VertexTrainer


def train():
    trainer = VertexTrainer('vertex_v2', 'confmod_auto_encoder_v2.json', 'AUTO_ENCODER_VERTEX_V2')
    trainer.train(train_mode=TrainerModes.SLURM)


if __name__ == '__main__':
    train()
