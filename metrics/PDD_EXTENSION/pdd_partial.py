"""
PDD extension — Option 2c: partial PDDs (pair-type decomposition)

For a structure with species {A, B, ...}, computes one PDD per ordered
species pair (A->A, A->B, B->A, B->B, ...), directly analogous to
partial radial distribution functions (pRDF) in MD.

Relationship to 2f (same/cross split):
    pdd_same  = union of all (X->X) partial PDDs
    pdd_cross = union of all (X->Y, X!=Y) partial PDDs

New public API:
    pdd_partial(periodic_set, k)
        -> dict {(src_type, tgt_type): (weights, amd_vector)}

    amd_partial(periodic_set, k)
        -> dict {(src_type, tgt_type): amd_vector}  (k,) each

    distance_partial(partial_1, partial_2, pair_weights=None)
        -> scalar L1 distance, optionally weighted per pair

All existing functions (pdd, emd, pdd_split, emd_split) are unchanged.
"""

import numpy as np
import collections
from scipy.spatial.distance import pdist, squareform
from scipy.stats import wasserstein_distance
import pdd
from pdd import PeriodicSet

from pdd_chemistry_same_cross import (
    _extract_motif_and_cell,
    _nearest_neighbours,
    _tile_types,
)


# ── core: partial PDD computation ────────────────────────────────────────────

def pdd_partial(periodic_set, k=100):
    """
    Compute partial PDDs for all ordered species pairs present in the structure.

    For each ordered pair (src_species, tgt_species):
      - query atoms  : all sites of src_species
      - neighbour filter : only distances to tgt_species sites are kept
      - output row i : sorted distances to the k closest tgt_species neighbours
                       zero-padded (Option B: compacted + tail-padded) if fewer
                       than k tgt_species neighbours exist in the search cloud

    Parameters
    ----------
    periodic_set : PeriodicSet with `.types` tag (integer species labels)
    k            : number of nearest neighbours per pair component

    Returns
    -------
    partials : dict  {(src_type, tgt_type): np.ndarray shape (n_src_sites, k+1)}
               column 0 is the normalised row weight (site multiplicity / total)
               columns 1..k are sorted distances

    pair_types : list of (src_type, tgt_type) tuples — ordered species pairs present
    """
    if not isinstance(periodic_set, PeriodicSet) or 'types' not in periodic_set.tags:
        raise ValueError(
            "pdd_partial requires periodic_set.types — integer species labels "
            "stored as a PeriodicSet tag."
        )

    motif, cell, asym, mult = _extract_motif_and_cell(periodic_set)
    types = periodic_set.types                          # (n_atoms,)

    if asym is not None:
        query_indices = asym
    else:
        query_indices = np.arange(len(motif))

    # ── single k-NN call for all pairs ───────────────────────────────────────
    # We need the k closest neighbours of *any* type per query atom.
    # For pair-filtered distances we need enough cloud points so that each
    # (src, tgt) pair can find k tgt-type neighbours — use k * n_species as
    # a safe overestimate for the search width.
    n_species = len(np.unique(types))
    k_search  = min(k * n_species, k * 10)             # cap to avoid explosion

    dists_raw, cloud, inds = _nearest_neighbours(motif, cell, k_search, asym)
    # dists_raw : (n_query, k_search)
    # inds      : (n_query, k_search) — indices into cloud

    tiled_types     = _tile_types(types, len(motif), len(cloud))
    neighbour_types = tiled_types[inds]                # (n_query, k_search)
    query_types     = types[query_indices]             # (n_query,)

    # ── number density scale ─────────────────────────────────────────────────
    scale = (motif.shape[0] / abs(np.linalg.det(cell))) ** (1 / 3)
    dists = dists_raw * scale

    # ── site weights ─────────────────────────────────────────────────────────
    if mult is None:
        site_weights = np.full(len(query_indices), 1 / len(motif))
    else:
        site_weights = mult / np.sum(mult)             # (n_query,)

    # ── build one partial PDD per ordered pair ────────────────────────────────
    unique_types = np.unique(types)
    partials     = {}

    for src_type in unique_types:
        src_mask = query_types == src_type             # (n_query,) bool
        if not src_mask.any():
            continue
        src_indices   = np.where(src_mask)[0]          # rows in dists/inds
        src_weights   = site_weights[src_mask]
        src_weights  /= src_weights.sum()              # normalise within src group

        for tgt_type in unique_types:
            # for each src atom, collect distances to tgt_type neighbours only
            rows = []
            for qi in src_indices:
                tgt_mask_k = neighbour_types[qi] == tgt_type   # (k_search,) bool
                tgt_dists  = np.sort(dists[qi, tgt_mask_k])    # compacted, sorted
                # Option B: tail-pad to length k
                row = np.zeros(k)
                n   = min(len(tgt_dists), k)
                row[:n] = tgt_dists[:n]
                rows.append(row)

            rows = np.array(rows)                      # (n_src, k)

            # attach weights as first column
            partial_matrix = np.hstack((src_weights[:, None], rows))
            partials[(src_type, tgt_type)] = partial_matrix

    pair_types = list(partials.keys())
    return partials, pair_types


# ── AMD from partial PDDs ─────────────────────────────────────────────────────

def amd_partial(periodic_set, k=100):
    """
    Compute AMD vectors for all ordered species pairs.

    Returns
    -------
    amds : dict {(src_type, tgt_type): np.ndarray shape (k,)}
    """
    partials, pair_types = pdd_partial(periodic_set, k=k)
    amds = {}
    for pair, matrix in partials.items():
        weights = matrix[:, 0]
        dists   = matrix[:, 1:]
        # zero rows (tail padding) should not contribute to mean
        # mask rows where all distances are zero (no tgt neighbours found)
        nonzero = dists.any(axis=1)
        if nonzero.any():
            amds[pair] = np.average(dists[nonzero],
                                    weights=weights[nonzero],
                                    axis=0)
        else:
            amds[pair] = np.zeros(k)
    return amds


# ── distance between two partial AMD dicts ────────────────────────────────────

def distance_partial(amds_1, amds_2, pair_weights=None):
    """
    Wasserstein (Earth Mover's) distance between two partial AMD representations.

    Each AMD vector is treated as a 1D distribution over neighbour shells
    (uniform weights across k positions). scipy.stats.wasserstein_distance
    gives the exact 1D EMD — strictly more correct than L1 norm, and free
    since 1D Wasserstein reduces to sorting with no transport solve needed.

    Pairs present in one structure but absent in the other contribute
    the full Wasserstein distance against a zero vector — penalising
    compositional mismatch naturally.

    Parameters
    ----------
    amds_1, amds_2  : dicts {(src, tgt): amd_vector} as from amd_partial()
    pair_weights    : dict {(src, tgt): float} or None.
                      If None, all pairs weighted equally.
                      Useful to upweight homoatomic (X->X) or
                      cross-species (X->Y) components independently.

    Returns
    -------
    float : weighted mean Wasserstein distance across all pairs
    """
    all_pairs = set(amds_1.keys()) | set(amds_2.keys())
    k         = next(iter(amds_1.values())).shape[0]

    total_dist   = 0.0
    total_weight = 0.0

    for pair in all_pairs:
        v1 = amds_1.get(pair, np.zeros(k))
        v2 = amds_2.get(pair, np.zeros(k))
        w  = pair_weights.get(pair, 1.0) if pair_weights is not None else 1.0
        # 1D Wasserstein: exact, no binning, no hyperparameters
        total_dist   += w * wasserstein_distance(v1, v2)
        total_weight += w

    return total_dist / total_weight if total_weight > 0 else 0.0


# ── consistency check: recover full AMD from partial sum ─────────────────────

def partial_to_full_amd(amds):
    """
    Sum all partial AMD vectors to recover the full (geometry-only) AMD.

    Analogous to GRID's property that summing all groups recovers the RDF.
    Use this to verify that pdd_partial is consistent with the original pdd():

        amd_orig  = PDD_to_AMD(pdd(periodic_set, k=k))
        amd_check = partial_to_full_amd(amd_partial(periodic_set, k=k))
        assert np.allclose(amd_orig, amd_check, atol=1e-6)

    Parameters
    ----------
    amds : dict {(src_type, tgt_type): amd_vector (k,)} from amd_partial()

    Returns
    -------
    amd_full : (k,) float — should match original AMD up to floating point
    """
    if not amds:
        raise ValueError("Empty partial AMD dict.")
    k        = next(iter(amds.values())).shape[0]
    amd_full = np.zeros(k)
    for vec in amds.values():
        amd_full += vec
    # normalise by number of pairs — each distance appears once per
    # ordered pair (A->B and B->A are separate), consistent with AMD
    # which averages over all sites without pair distinction
    return amd_full / len(amds)


# ── pair weight constructors (convenience) ────────────────────────────────────

def uniform_pair_weights(pair_types):
    """All pairs weighted equally — baseline."""
    return {p: 1.0 for p in pair_types}


def homoatomic_upweight(pair_types, homo_weight=2.0):
    """Upweight X->X pairs relative to X->Y pairs."""
    return {p: homo_weight if p[0] == p[1] else 1.0 for p in pair_types}


def heteroatomic_upweight(pair_types, hetero_weight=2.0):
    """Upweight X->Y (X!=Y) pairs — emphasises coordination chemistry."""
    return {p: hetero_weight if p[0] != p[1] else 1.0 for p in pair_types}


# ── all-to-all distance matrix ────────────────────────────────────────────────

def compute_partial_amd_matrix(periodic_sets, k=100, pair_weights=None, compute_matrix=False):
    """
    All-to-all L1 distance matrix using partial AMD representations.

    Parameters
    ----------
    periodic_sets : list of PeriodicSet or None
    k             : number of nearest neighbours
    pair_weights  : dict or callable(pair_types) -> dict, or None
                    if callable, called with the union of all pair_types

    Returns
    -------
    mat   : (n, n) float array, NaN where either structure is None
    amds  : list of partial AMD dicts (or None), for storage/reuse
    """
    n    = len(periodic_sets)
    amds = []

    print(f'  Computing partial AMDs for {n} structures ...')
    all_pair_types = set()
    for ps in periodic_sets:
        if ps is None:
            amds.append(None)
        else:
            try:
                a = amd_partial(ps, k=k)
                amds.append(a)
                all_pair_types.update(a.keys())
            except Exception as e:
                print(f'    Warning: {e}')
                amds.append(None)

    #print(f'  Pair types found across dataset: {sorted(all_pair_types)}')

    mat   = np.full((n, n), np.nan)

    if compute_matrix:
        # resolve pair_weights
        if callable(pair_weights):
            pair_weights = pair_weights(list(all_pair_types))
        # None -> uniform inside distance_partial

        print(f'  Building {n}x{n} distance matrix ...')
        pairs = [(i, j) for i in range(n) for j in range(i + 1, n)
                 if amds[i] is not None and amds[j] is not None]
        for i, j in pairs:
            d          = distance_partial(amds[i], amds[j], pair_weights=pair_weights)
            mat[i, j]  = d
            mat[j, i]  = d
        np.fill_diagonal(mat, 0.0)

    return mat, amds

# ── relationship to 2f: sanity check ─────────────────────────────────────────

def amd_partial_to_split(amds):
    """
    Collapse partial AMDs back to the 2f same/cross representation.
    Useful for verifying consistency between Option 2c and 2f.

    Returns amd_same, amd_cross as (k,) vectors.
    """
    homo_pairs  = {p: v for p, v in amds.items() if p[0] == p[1]}
    cross_pairs = {p: v for p, v in amds.items() if p[0] != p[1]}

    def _mean_across_pairs(pair_dict):
        if not pair_dict:
            k = next(iter(amds.values())).shape[0]
            return np.zeros(k)
        vecs = np.array(list(pair_dict.values()))   # (n_pairs, k)
        return vecs.mean(axis=0)                    # unweighted mean across pairs

    return _mean_across_pairs(homo_pairs), _mean_across_pairs(cross_pairs)

if __name__ == '__main__':
    import sys, os
    from tqdm import tqdm
    from pathlib import Path
    import pandas as pd
    import pickle
    k = 100
    DATA_DIR = Path('/Users/andrij/Programmes/Contrastive_learning_MPDS_structure_space/PCD_DATA/')
    STRUCT_DIR = DATA_DIR / '201841_complete_structure'
    
    df = pd.read_pickle(DATA_DIR / 'PCD_201K_AMD_AMDcross.pickle')
    print(df.shape)
    
#   cif_files = list(STRUCT_DIR.rglob('*.cif'))
#
#   periodic_sets = [None] * len(cif_files)
#   print('LEN cif files:', len(cif_files))
#
#   for i, f in tqdm(enumerate(cif_files), total=len(cif_files), desc="Computing periodic sets"):
#        try:
#           periodic_sets[i] = pdd.read_cif(f)
#        except Exception:
#            pass

#    files = [f for  _, _, ff in os.walk('../PCD_DATA/201841_complete_structure') for f in ff]
#    for i, f in tqdm(enumerate(files), total=len(files), desc="Computing Periodic sets"):

    CHUNK_SIZE = 5000


#   for chunk_idx, chunk_start in enumerate(range(0, len(files), CHUNK_SIZE)):
#       periodic_sets = []
#       entry_ids = []
#       chunk = files[chunk_start : chunk_start + CHUNK_SIZE]
#       for f in tqdm(chunk, desc=f"Chunk {chunk_idx}"):
#           entry_ids.append(f.split('.')[0])
#           try:
#               periodic_sets.append(pdd.read_cif(f'../PCD_DATA/201841_complete_structure/{f}'))
#           except Exception:
#               periodic_sets.append(None)
#   
#       payload = {"periodic_sets": periodic_sets, "entries": entry_ids}
#   
#       with open(DATA_DIR / f'periodic_sets_chunk_{chunk_idx}.pickle', 'wb') as fout:
#           pickle.dump(payload, fout)
#   
#       del periodic_sets, entry_ids, payload
#
#   sys.exit(0)

    print(f'\n[2] Computing scaled patial AMD (k={k}) ...')
    #_, amds = compute_partial_amd_matrix(periodic_sets, k=k, compute_matrix=False)
    #df['amd_partial'] = amds

    amds_all = []
    chunk_files = sorted(DATA_DIR.glob('periodic_sets_chunk_*.pickle'))
    
    for chunk_path in tqdm(chunk_files, desc="Processing chunks"):
        with open(chunk_path, 'rb') as f:
            periodic_sets = pickle.load(f)
            print(chunk_path, len(periodic_sets['periodic_sets']), periodic_sets.keys())
        
        _, amds = compute_partial_amd_matrix(periodic_sets['periodic_sets'], k=k, compute_matrix=False)
        amds_all.extend(amds)
        del periodic_sets

    assert len(amds_all) == len(df), f"Length mismatch: {len(amds_all)} amds vs {len(df)} rows"
    df['amd_partial'] = amds_all
    df.to_pickle(DATA_DIR / 'PCD_201K_AMD_AMDcross_AMDpartial.pickle')
