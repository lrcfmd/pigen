import pandas as pd
import numpy as np
import torch
from tqdm import tqdm
import sys

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using {device}')


# ── parse AMD vectors ─────────────────────────────────────────────────────────
def parse_amd(series):
    return np.vstack([
        np.array([float(i) for i in x[1:-1].split(',')])
        for x in series
    ])

df = pd.read_csv('/home/u6fo/andrij.u6fo/pigen/data/PDD/alex_mp20_pdd.csv')
df_amds = parse_amd(df['amd'])   # (N_ref,   k)
print('ALEX MP20 AMD parsed:', df_amds.shape)
# Precompute cumsums on GPU
print('Precompute cumsums on GPU for Alex...')
df_t = torch.tensor(df_amds, dtype=torch.float32, device=device)
print(df_t.shape)

bfs = ['/lus/lfs1aip2/projects/u6fo/pigen/generated/PDDMIN/gen_PDDMIN3/denovo_[3.0]_2.0_compact5141_spp.csv_AMD.pickle',
       '/lus/lfs1aip2/projects/u6fo/pigen/generated/PDDMIN/gen_PDDMIN4/denovo_[4.0]_2.0_compact4519_spp.csv_AMD.pickle',
       '/lus/lfs1aip2/projects/u6fo/pigen/generated/PDDMIN/gen_PDDMIN5/denovo_[5.0]_2.0_compact3815_spp.csv_AMD.pickle',
       '/projects/u6fo/pigen/generated/PDDMIN/PIGEN_MLED9_Compactness07/denovo_9_07_g2_pdd.pickle']

#bfs = ['/home/u6fo/andrij.u6fo/pigen/generated/PDDMIN/gen_PDDMIN3_det/denovo_[3.0]_2.0_compact5293_spp.csv_AMD.pickle',
#       '/home/u6fo/andrij.u6fo/pigen/generated/PDDMIN/gen_PDDMIN4_det/denovo_[4.0]_2.0_compact4524_spp.csv_AMD.pickle',
#       '/home/u6fo/andrij.u6fo/pigen/generated/PDDMIN/gen_PDDMIN5_det/denovo_[5.0]_2.0_compact3657_spp.csv_AMD.pickle']

#bfs = ['/home/u6fo/andrij.u6fo/pigen/generated/PDDMIN/gen_PDDMIN3/denovo_[3.0]_2.0.csv_AMD.pickle',
#       '/home/u6fo/andrij.u6fo/pigen/generated/PDDMIN/gen_PDDMIN4/denovo_[4.0]_2.0.csv_AMD.pickle',
#       '/home/u6fo/andrij.u6fo/pigen/generated/PDDMIN/gen_PDDMIN5/denovo_[5.0]_2.0.csv_AMD.pickle']
#
#bfs = ['/home/u6fo/andrij.u6fo/pigen/generated/PCD/PCD_hP24_generated_pearson_compact6989_spp.csv_AMD.pickle']

# ── GPU EMD via cumsum L1 ─────────────────────────────────────────────────────
# For monotone 1D AMD vectors: EMD = L1(cumsum(u), cumsum(v))
# Vectorised over all pairs in a block: (block, k) vs (N_ref, k)

for bfile in bfs:
    bf1 = pd.read_pickle(bfile)
    bf_amds = np.vstack(bf1['amd'].to_numpy()).astype(np.float32)
    print('Read:', bfile)
    print('AMD in bfile:', bf_amds.shape)
    bf_t = torch.tensor(bf_amds, dtype=torch.float32, device=device)
    N_probe = bf_t.shape[0]
    N_ref   = df_t.shape[0]
    PDDMIN     = np.full(N_probe, np.inf)
    PDDMIN_idx = np.zeros(N_probe, dtype=int)
    BLOCK_REF  = 5000
    for j in tqdm(range(0, N_ref, BLOCK_REF)):
        ref_chunk = df_t[j:j+BLOCK_REF]
        D_block = (bf_t.unsqueeze(1) - ref_chunk.unsqueeze(0)).abs().mean(-1)
        D_np = D_block.cpu().numpy()
        block_min_vals = D_np.min(axis=1)
        block_min_idxs = D_np.argmin(axis=1) + j
        update = block_min_vals < PDDMIN
        PDDMIN[update]     = block_min_vals[update]
        PDDMIN_idx[update] = block_min_idxs[update]
    bf1['PDDMIN']     = PDDMIN
    bf1['PDDMIN_ref'] = df.iloc[PDDMIN_idx].index.values
    bf1.to_csv(f'{bfile}_PDDMIN_L1.csv', index=False)
