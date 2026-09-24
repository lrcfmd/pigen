import sys
sys.path.insert(0, '.')  # ensure pigen importable for the .pt load

import torch
import pandas as pd
import pickle
import numpy as np
from sklearn.preprocessing import StandardScaler
from pathlib import Path

# ---- load the cached dataset (needs pigen on path, but only here) ----
from pigen.assets.simple_dataset import SimpleCrystDataset
from dataclasses import asdict

train_df = pd.read_csv('data/Pearson/train.csv')
train_dataset = SimpleCrystDataset(
    df=train_df,   # any df — .pt exists so preprocess is skipped
    save_path='data/Pearson/train_ori_bravais_idx_natoms.pt',
    prop=['bravais_idx', 'natoms'],
    target_energy=False,
    gpus=1,
)

# ---- helper to strip to plain sklearn StandardScaler ----
def to_portable_scaler(s):
    out = StandardScaler()
    out.mean_             = np.array(s.mean_)
    out.scale_            = np.array(s.scale_)
    out.var_              = np.array(s.var_)
    out.n_features_in_    = int(s.n_features_in_)
    out.n_samples_seen_   = int(s.n_samples_seen_)
    return out

# ---- lattice_scaler: single scaler ----
portable_lattice = to_portable_scaler(train_dataset.lattice_scaler)

# ---- prop scaler: list of scalers, one per prop ----
portable_prop = [to_portable_scaler(s) for s in train_dataset.scaler]

# ---- save ----
save_dir = Path('data/Pearson')

with open(save_dir / 'lattice_scaler.pkl', 'wb') as f:
    pickle.dump(portable_lattice, f)

with open(save_dir / 'scaler_bravais_idx_natoms.pkl', 'wb') as f:
    pickle.dump(portable_prop, f)

print("lattice mean:", portable_lattice.means)
print("prop scalers:")
for prop, s in zip(['bravais_idx', 'natoms'], portable_prop):
    print(f"  {prop}: mean={s.means}") #, scale={s.scale_}")
