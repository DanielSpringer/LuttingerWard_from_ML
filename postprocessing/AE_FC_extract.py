import torch
import math
import pandas as pd
from os.path import dirname, abspath, join
from os import scandir


parent_path = dirname(abspath(__file__))




# ==================== Latent Dimension Scaling ====================
model_dir = join(parent_path, "../lightning_logs/VAE_Linear")
versions = [f.path for f in scandir(abspath(model_dir)) if f.is_dir()]

df = pd.DataFrame({'int': [], 'int': [], 'float': [], 'int': [], 'int': []}, columns = ['VersionID', 'best_epoch', 'val_loss', 'batch_size', 'latent_dim', 'n_layers'])
for i,version_path in enumerate(versions):
    path_i = join(version_path, "checkpoints/last.ckpt")

    try:
        versionID = int(version_path[str(version_path).rfind('_')+1:])
        #print(f"version: {versionID}")
        checkpoint = torch.load(path_i,map_location=torch.device('cpu'))
        best_epoch = int(checkpoint["epoch"])
        validation_loss = math.inf
        for callback in checkpoint.get('callbacks', []):
            if isinstance(callback, str) and callback.startswith("ModelCheckpoint"):
                validation_loss = checkpoint["callbacks"][callback]["best_model_score"].item()
                break

        batch_size = checkpoint["hyper_parameters"]["batch_size"]
        n_layers = checkpoint["hyper_parameters"]["n_layers"]
        latent_dim = checkpoint["hyper_parameters"]["latent_dim"]
        print(versionID, best_epoch, validation_loss, batch_size, n_layers, latent_dim)
        df.loc[i] = {'VersionID': versionID, 'best_epoch': best_epoch, 'val_loss': validation_loss, 'batch_size': batch_size, 'latent_dim': latent_dim, 'n_layers': n_layers}

    except:
        print("skipping "+version_path)


    #row = pd.DataFrame({'int': [versionID], 'int': [best_epoch], 'float': [validation_loss], 'int': [batch_size], 'int': [n_layers]}, columns = ['VersionID', 'best_epoch', 'val_loss', 'batch_size', 'fc_dims'])
    #print(row)


df.to_csv('scan_nPrune_02_AE_FC_01.csv', index=False)