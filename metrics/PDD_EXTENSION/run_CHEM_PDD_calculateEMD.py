import numpy as np
import sys
import pdd
import time
from pdd_scaled import PDD, PDD_scale, PDD_to_AMD, PDD_scale_split
import pandas as pd
from pymatgen.io.cif import CifWriter

from pdd_chemistry_same_cross import (split_to_amds_soft, 
                                     pdd_split,
                                     _extract_motif_and_cell,
                                     _nearest_neighbours
                                      )

def s2cif(s):
    try:
        s = str(CifWriter(s))
    except:
        s = None
    return s

def compute_periodic_sets(df, col='structure'):
    periodic_sets = []
    could_not = 0
    for structure in df[col]:
        try:
            if structure is None:
                periodic_sets.append(None)
                continue
            psets = pdd.read_cif_pmg(structure)
            if psets is None:
                periodic_sets.append(None)
            else:
                periodic_sets.append(psets)
        except Exception:
            could_not += 1
            periodic_sets.append(None)
    print('Could not build periodic sets:', could_not, '/', df.shape[0])
    return periodic_sets


def compute_split_and_amd(ps, k):
    if ps is None:
        return None
    motif, cell, asym, mult = pdd._extract_motif_and_cell(ps)
    scale = (motif.shape[0] / abs(np.linalg.det(cell))) ** (1/3)

    dists_raw, cloud, inds = _nearest_neighbours(motif, cell, k, asym)
    dists = dists_raw * scale

    occ = ps.tags.get('occupancy_vectors', None)   # None for ordered structures
    # simple version first
    #occ = None
    pdd_same, pdd_cross, soft, weights = pdd_split(
        ps, k=k,
        occupancy_vectors=occ,
    )
    pdd_same[:, 1:]  *= scale
    pdd_cross[:, 1:] *= scale

    if occ is not None:
        # disordered: soft weighted AMD
        amd_same, amd_cross = split_to_amds_soft(dists, soft, weights)
    else:
        # ordered: standard weighted mean on compacted scaled PDD matrices
        amd_same  = np.average(pdd_same[:, 1:],  weights=pdd_same[:, 0],  axis=0)
        amd_cross = np.average(pdd_cross[:, 1:], weights=pdd_cross[:, 0], axis=0)

    return amd_same, amd_cross

if __name__ == '__main__':
    import sys, os
    from tqdm import tqdm
    k = 100
#   df = pd.read_pickle(
#       '../Contrastive_learning_MPDS_structure_space/'
#       'split_data_test_val_train/StrClas_data/test.pickle'
#   )
#
#   periodic_sets = compute_periodic_sets(df)
#   valid = [ps for ps in periodic_sets if ps is not None]
#   print(f'{len(valid)} / {len(periodic_sets)} structures built successfully.')
#
#   # ── 1) original: scaled PDD → AMD → all-to-all EMD ───────────────────────
#   print(f'\n[1] Computing scaled PDD (k={k}) ...')
#   pdds_scaled = [
#       PDD_scale(ps, k) if ps is not None else None
#       for ps in periodic_sets
#   ]
#
#   df['amd_scaled'] = [
#       PDD_to_AMD(p) if p is not None else None
#       for p in pdds_scaled
#   ]
#
#   print('[1] Computing all-to-all EMD on scaled PDD ...')
#   t0 = time.time()
#   emd_matrix_orig = compute_emd_matrix(pdds_scaled, pdd.emd)
#   print(f'    done in {time.time() - t0:.1f}s')

# ── 2) new: split scaled AMD → all-to-all AMD-split EMD ──────────────────
    # From CIF -- scratch
    df = pd.read_pickle('../PCD_DATA/PCD_201K_AMD_AMDcross.pickle')

    files = [f for  _, _, ff in os.walk('../PCD_DATA/201841_complete_structure') for f in ff]
    periodic_sets = [None] * len(files)
    for i, f in tqdm(enumerate(files), total=len(files), desc="Computing Periodic sets"):
        try:
            periodic_sets[i] = pdd.read_cif(f'../PCD_DATA/201841_complete_structure/{f}')
        except Exception:
            pass

    print(f'\n[2] Computing scaled split AMD (k={k}) ...')

    #amd_pairs = [compute_split_and_amd(ps, k) for ps in periodic_sets[:10]]
    amd_pairs = [compute_split_and_amd(ps, k) for ps in periodic_sets]
    df['amd_split_occ'] = amd_pairs
    df.to_pickle('../PCD_DATA/PCD_201K_AMD_AMDcross.pickle')
    sys.exit(0)

    df['amd_same_scaled']  = [p[0] if p is not None else None for p in amd_pairs]
    df['amd_cross_scaled'] = [p[1] if p is not None else None for p in amd_pairs]

    def amd_split_distance(pair_i, pair_j, alpha=0.5, beta=0.5):
        amd_same_i,  amd_cross_i  = pair_i
        amd_same_j,  amd_cross_j  = pair_j
        d_same  = np.mean(np.abs(amd_same_i  - amd_same_j))
        d_cross = np.mean(np.abs(amd_cross_i - amd_cross_j))
        return alpha * d_same + beta * d_cross

    print('[2] Computing all-to-all AMD-split distance matrix ...')
    t0 = time.time()
    emd_matrix_amd_split = compute_emd_matrix(
        amd_pairs,
        lambda a, b: amd_split_distance(a, b, alpha=0.5, beta=0.5)
    )
    print(f'    done in {time.time() - t0:.1f}s')

    # ── save ─────────────────────────────────────────────────────────────────
    df.to_pickle('ConstrastiveLearning_StrSpace/test_with_split_amd.pickle')
    np.save('ConstrastiveLearning_StrSpace/emd_matrix_orig.npy',      emd_matrix_orig)
    np.save('ConstrastiveLearning_StrSpace/emd_matrix_amd_split.npy', emd_matrix_amd_split)

    print('\nSaved:')
    print('  test_with_split_amd.pickle     (df with amd_scaled, amd_same_scaled, amd_cross_scaled)')
    print('  emd_matrix_orig.npy            (all-to-all, original scaled PDD-EMD)')
    print('  emd_matrix_amd_split.npy       (all-to-all, AMD-split L1, alpha=beta=0.5)')
