"""
PDD with chemistry extension: same/cross-species split.

Extends the original PDD implementation with two parallel distance
distribution components:
  - pdd_same:  k-NN distances to atoms of the *same* species
  - pdd_cross: k-NN distances to atoms of *different* species

For disordered structures, same/cross flags are replaced by the
continuous probability P(same|i,j) = sum_s p_s(i) * p_s(j),
where p_s(i) is the occupancy of species s at site i.

New public API (drop-in alongside originals):
  pdd_split(periodic_set, k, ...)  -> (pdd_same, pdd_cross)
  emd_split(ps1, ps2, alpha, beta) -> scalar distance

All original functions are unchanged.
"""

import collections
from itertools import product, combinations
import pdd as opdd
from pdd import PeriodicSet

import numpy as np
from scipy.spatial import KDTree
from scipy.spatial.distance import pdist, cdist, squareform

# ── helpers carried over verbatim from the original ──────────────────────────

def _extract_motif_and_cell(periodic_set):
    asymmetric_unit, multiplicities = None, None
    if isinstance(periodic_set, PeriodicSet):
        motif, cell = periodic_set.motif, periodic_set.cell
        if ('asymmetric_unit' in periodic_set.tags and
                'wyckoff_multiplicities' in periodic_set.tags):
            asymmetric_unit = periodic_set.asymmetric_unit
            multiplicities  = periodic_set.wyckoff_multiplicities
    elif isinstance(periodic_set, np.ndarray):
        motif, cell = periodic_set, None
    else:
        motif, cell = periodic_set[0], periodic_set[1]
    return motif, cell, asymmetric_unit, multiplicities


def _collapse_into_groups(overlapping):
    overlapping = squareform(overlapping)
    group_nums  = {}
    group       = 0
    for i, row in enumerate(overlapping):
        if i not in group_nums:
            group_nums[i] = group
            group += 1
            for j in np.argwhere(row).T[0]:
                if j not in group_nums:
                    group_nums[j] = group_nums[i]
    groups = collections.defaultdict(list)
    for row_ind, group_num in sorted(group_nums.items()):
        groups[group_num].append(row_ind)
    return list(groups.values())


def _dist(xy, z):
    s = z ** 2
    for val in xy:
        s += val ** 2
    return s


def _distkey(pt):
    s = 0
    for val in pt:
        s += val ** 2
    return s


def _generate_integer_lattice(dims):
    ymax = collections.defaultdict(int)
    d = 0
    if dims == 1:
        yield np.array([[0]])
        while True:
            d += 1
            yield np.array([[-d], [d]])
    while True:
        positive_int_lattice = []
        while True:
            batch = []
            for xy in product(range(d + 1), repeat=dims - 1):
                if _dist(xy, ymax[xy]) <= d ** 2:
                    batch.append((*xy, ymax[xy]))
                    ymax[xy] += 1
            if not batch:
                break
            positive_int_lattice += batch
        positive_int_lattice.sort(key=_distkey)
        int_lattice = []
        for p in positive_int_lattice:
            int_lattice.append(p)
            for n_reflections in range(1, dims + 1):
                for indexes in combinations(range(dims), n_reflections):
                    if all((p[i] for i in indexes)):
                        p_ = list(p)
                        for i in indexes:
                            p_[i] *= -1
                        int_lattice.append(p_)
        yield np.array(int_lattice)
        d += 1


def _generate_concentric_cloud(motif, cell):
    int_lattice_generator = _generate_integer_lattice(cell.shape[0])
    while True:
        int_lattice = next(int_lattice_generator) @ cell
        yield np.concatenate([motif + translation for translation in int_lattice])


def _nearest_neighbours(motif, cell, k, asymmetric_unit=None):
    if asymmetric_unit is not None:
        asym_unit = motif[asymmetric_unit]
    else:
        asym_unit = motif
    cloud_generator = _generate_concentric_cloud(motif, cell)
    n_points = 0
    cloud    = []
    while n_points <= k:
        l = next(cloud_generator)
        n_points += l.shape[0]
        cloud.append(l)
    cloud.append(next(cloud_generator))
    cloud = np.concatenate(cloud)
    tree         = KDTree(cloud, compact_nodes=False, balanced_tree=False)
    pdd_, inds   = tree.query(asym_unit, k=k + 1, workers=-1)
    pdd          = np.zeros_like(pdd_)
    while not np.allclose(pdd, pdd_, atol=1e-12, rtol=0):
        pdd   = pdd_
        cloud = np.vstack((cloud,
                           next(cloud_generator),
                           next(cloud_generator)))
        tree       = KDTree(cloud, compact_nodes=False, balanced_tree=False)
        pdd_, inds = tree.query(asym_unit, k=k + 1, workers=-1)
    return pdd_[:, 1:], cloud, inds[:, 1:]


def _network_simplex(source_demands, sink_demands, network_costs):
    # (unchanged from original — omitted here for brevity but must be present)
    from scipy.sparse.csgraph import minimum_spanning_tree  # placeholder import
    raise NotImplementedError("Paste the original _network_simplex body here.")


# ── PeriodicSet (unchanged) ───────────────────────────────────────────────────

#lass PeriodicSet:
#   def __init__(self, motif, cell, name=None, **kwargs):
#       self.motif = motif
#       self.cell  = cell
#       self.name  = name
#       self.tags  = kwargs
#
#   def __getattr__(self, attr):
#       if 'tags' not in self.__dict__:
#           self.tags = {}
#       if attr in self.tags:
#           return self.tags[attr]
#       raise AttributeError(
#           f"{self.__class__.__name__} object has no attribute or tag {attr}"
#       )


# ── original pdd / emd (unchanged) ───────────────────────────────────────────

def pdd(periodic_set, k=100, lexsort=True, collapse=True, collapse_tol=1e-4):
    motif, cell, asymmetric_unit, multiplicities = opdd_extract_motif_and_cell(periodic_set)
    dists, _, _ = _nearest_neighbours(motif, cell, k, asymmetric_unit=asymmetric_unit)
    groups = [[i] for i in range(len(dists))]
    if multiplicities is None:
        weights = np.full((motif.shape[0],), 1 / motif.shape[0])
    else:
        weights = multiplicities / np.sum(multiplicities)
    if collapse:
        overlapping = pdist(dists, metric='chebyshev')
        overlapping = overlapping < collapse_tol
        if overlapping.any():
            groups  = _collapse_into_groups(overlapping)
            weights = np.array([sum(weights[group]) for group in groups])
            ordering = [group[0] for group in groups]
            dists    = dists[ordering]
    result = np.hstack((weights[:, None], dists))
    if lexsort:
        lex_ordering = np.lexsort(np.rot90(dists))
        result       = result[lex_ordering]
    return result


def emd(pdd1, pdd2, metric='chebyshev', **kwargs):
    dm           = cdist(pdd1[:, 1:], pdd2[:, 1:], metric=metric, **kwargs)
    emd_dist, _  = _network_simplex(pdd1[:, 0], pdd2[:, 0], dm)
    return emd_dist


# ── NEW: chemistry helpers ────────────────────────────────────────────────────

def _tile_types(types, motif_size, cloud_size):
    """
    Reconstruct per-point species labels for the concentric cloud.

    The cloud is built by stacking full copies of the motif for each
    lattice translation, so cloud point i belongs to atom (i % motif_size).
    We only need the first `cloud_size` points.
    """
    n_full_tiles = cloud_size // motif_size
    remainder    = cloud_size  % motif_size
    tiled = np.tile(types, n_full_tiles)
    if remainder:
        tiled = np.concatenate([tiled, types[:remainder]])
    return tiled                                          # shape (cloud_size,)


def _same_species_flags(query_types, inds, tiled_types):
    """
    Boolean array (n_query, k): True where neighbour j is same species as
    query atom i.

    Parameters
    ----------
    query_types  : (n_query,) int  — species of each query atom
    inds         : (n_query, k)    — cloud indices of the k nearest neighbours
    tiled_types  : (cloud_size,)   — species labels tiled over the cloud
    """
    neighbour_types = tiled_types[inds]                   # (n_query, k)
    return neighbour_types == query_types[:, None]        # broadcast


def _same_species_flags_soft(occupancy_query, occupancy_neighbours):
    """
    Soft (probabilistic) same-species flag for disordered structures.

    P(same | i, j) = sum_s  p_s(i) * p_s(j)
                   = dot(occ_i, occ_j)

    Parameters
    ----------
    occupancy_query      : (n_query, n_species)  float  — occupancy vectors
    occupancy_neighbours : (n_query, k, n_species) float

    Returns
    -------
    flags : (n_query, k) float in [0, 1]
    """
    # einsum: for each (i, j), dot product over species axis
    return np.einsum('is,iks->ik', occupancy_query, occupancy_neighbours)


# ── NEW: pdd_split ────────────────────────────────────────────────────────────

def _build_split_pdds(dists, flags, weights,
                      collapse=True, collapse_tol=1e-4, lexsort=True,
                      fill_empty=0.0):
    """
    Internal: given distance matrix, boolean flag matrix, and row weights,
    return (pdd_same, pdd_cross).

    For rows where a particular flag never fires (e.g. a pure-element structure
    has no cross-species distances), the corresponding distances are filled with
    `fill_empty` (default 0.0 — a safe sentinel that sorts first).

    Each output has shape (n_rows_after_collapse, 1 + k) where column 0 is the
    row weight.  Rows with zero weight (empty component) are retained so that
    both matrices have the same number of rows — this keeps them aligned for
    emd_split.
    """
    n_rows, k = dists.shape

    # ── build component distance matrices ────────────────────────────────────
    # For each row i, take distances where flag==True / False, sort them,
    # and zero-pad to length k so all rows have the same width.
    def a_extract_component(flag_mask):
        out = np.full((n_rows, k), fill_empty, dtype=float)
        for i in range(n_rows):
            vals = np.sort(dists[i, flag_mask[i]])
            out[i, :len(vals)] = vals
        return out


    def _extract_component(flag_mask):
        out = np.full((n_rows, k), 0.0, dtype=float)
        for i in range(n_rows):
            vals = np.sort(dists[i, flag_mask[i]])   # only flagged distances
            out[i, :len(vals)] = vals                 # left-pack, tail stays zero
        return out

    dists_same  = _extract_component(flags)
    dists_cross = _extract_component(~flags)

    def _finalise(d):
        w      = weights.copy()
        groups = [[i] for i in range(n_rows)]
        if collapse:
            overlapping = pdist(d, metric='chebyshev') < collapse_tol
            if overlapping.any():
                groups   = _collapse_into_groups(overlapping)
                w        = np.array([sum(w[g]) for g in groups])
                ordering = [g[0] for g in groups]
                d        = d[ordering]
        result = np.hstack((w[:, None], d))
        if lexsort:
            result = result[np.lexsort(np.rot90(d))]
        return result

    return _finalise(dists_same), _finalise(dists_cross)


def pdd_split(periodic_set, k=100, lexsort=True,
              collapse=True, collapse_tol=1e-4,
              occupancy_vectors=None):
    """
    Compute the same-species / cross-species split PDD (experiment 2f).

    Parameters
    ----------
    periodic_set     : PeriodicSet with a `.types` tag (integer species labels)
    k                : number of nearest neighbours
    lexsort          : lexicographic row ordering (mirrors original pdd())
    collapse         : merge identical rows (mirrors original pdd())
    collapse_tol     : Chebyshev tolerance for collapse
    occupancy_vectors: (n_atoms, n_species) float, optional.
                       If provided, uses soft P(same) flags for disordered
                       structures instead of the hard boolean flag.
                       Rows should sum to 1 (normalised occupancies).

    Returns
    -------
    pdd_same  : np.ndarray, shape (m, k+1)  — col 0 is weight
    pdd_cross : np.ndarray, shape (m, k+1)
    """
    motif, cell, asymmetric_unit, multiplicities = opdd._extract_motif_and_cell(
        periodic_set)

    if not isinstance(periodic_set, PeriodicSet) or 'types' not in periodic_set.tags:
        raise ValueError(
            "pdd_split requires periodic_set.types — integer species labels "
            "per atom site, stored as a PeriodicSet tag."
        )
    types = periodic_set.types                            # (n_atoms,)

    # query atoms: either full motif or asymmetric unit
    if asymmetric_unit is not None:
        query_indices = asymmetric_unit
    else:
        query_indices = np.arange(len(motif))
    query_types = types[query_indices]                    # (n_query,)

    # k-NN distances AND cloud indices
    dists, cloud, inds = _nearest_neighbours(
        motif, cell, k, asymmetric_unit=asymmetric_unit)
    # inds: (n_query, k) — indices into `cloud`

    # ── species flags ─────────────────────────────────────────────────────────
    if occupancy_vectors is None:
        # ordered / hard flag
        tiled_types = _tile_types(types, len(motif), len(cloud))
        flags = _same_species_flags(query_types, inds, tiled_types)
        soft  = flags.astype(float)          # hard 0/1 — soft is trivially defined
        # flags: (n_query, k) bool
    else:
        # disordered / soft flag — returns float in [0,1]
        # occupancy_vectors shape (n_atoms, n_species)
        occ_query      = occupancy_vectors[query_indices]           # (nq, ns)
        tiled_occ      = np.tile(occupancy_vectors,
                                 (len(cloud) // len(motif) + 1, 1))[:len(cloud)]
        occ_neighbours = tiled_occ[inds]                            # (nq, k, ns)
        soft           = _same_species_flags_soft(occ_query, occ_neighbours)
        # binarise at 0.5 for the split — or keep soft for downstream uses
        flags          = soft >= 0.5

    # ── weights ───────────────────────────────────────────────────────────────
    if multiplicities is None:
        weights = np.full((len(query_indices),), 1 / len(motif))
    else:
        weights = multiplicities / np.sum(multiplicities)

    pdd_same, pdd_cross = _build_split_pdds(dists, flags, weights,
                             collapse=collapse,
                             collapse_tol=collapse_tol,
                             lexsort=lexsort)
    return pdd_same, pdd_cross, soft, weights 


# ── NEW: emd_split ────────────────────────────────────────────────────────────

def compute_split_and_amd(ps, k, occupancy_vectors=None):
    if ps is None:
        return None

    # ── scaling factor ────────────────────────────────────────────────────
    motif, cell, asym, mult = opdd._extract_motif_and_cell(ps)
    scale = (motif.shape[0] / abs(np.linalg.det(cell))) ** (1/3)

    # ── single k-NN call ──────────────────────────────────────────────────
    dists_raw, cloud, inds = _nearest_neighbours(motif, cell, k, asym)
    dists = dists_raw * scale                  # ← scale once, reuse everywhere

    occ = ps.tags.get('occupancy_vectors', None)
    pdd_same, pdd_cross, soft, weights = pdd_split(
        ps, k=k, occupancy_vectors=occ,
        precomputed=(dists, cloud, inds)       # ← pass in to avoid recomputing
    )
    # pdd_same/pdd_cross distances are already scaled if pdd_split uses
    # the precomputed dists — otherwise scale their distance columns too:
    pdd_same[:, 1:]  *= scale
    pdd_cross[:, 1:] *= scale

    if occ is not None:
        amd_same, amd_cross = split_to_amds_soft(dists, soft, weights)
    else:
        amd_same  = np.average(pdd_same[:, 1:],  weights=pdd_same[:, 0],  axis=0)
        amd_cross = np.average(pdd_cross[:, 1:], weights=pdd_cross[:, 0], axis=0)

    # amd_same, amd_cross are derived from scaled dists — no further scaling needed

    return pdd_same, pdd_cross, amd_same, amd_cross

def split_to_amds_soft(dists, soft, weights):
    """
    dists   : (m, k)   — PDD distance rows
    soft    : (m, k)   — P(same|i,j) = occ_i · occ_j
    weights : (m,)     — site weights wᵢ
    """
    W_same  = weights[:, None] * soft          # (m, k)
    W_cross = weights[:, None] * (1 - soft)    # (m, k)

    # normalise column-wise to get proper weighted averages
    amd_same  = np.sum(W_same  * dists, axis=0) / (np.sum(W_same,  axis=0) + 1e-12)
    amd_cross = np.sum(W_cross * dists, axis=0) / (np.sum(W_cross, axis=0) + 1e-12)

    return amd_same, amd_cross   # each shape (k,)


def emd_split(split1, split2,
              alpha=0.5, beta=0.5,
              metric='chebyshev', **kwargs):
    """
    EMD distance between two split-PDD pairs.

    distance = alpha * EMD(pdd_same_1, pdd_same_2)
             + beta  * EMD(pdd_cross_1, pdd_cross_2)

    alpha + beta need not equal 1; set alpha=1, beta=0 to use only the
    same-species component, or alpha=beta=0.5 for equal weighting.

    Parameters
    ----------
    split1, split2 : tuples (pdd_same, pdd_cross) as returned by pdd_split()
    alpha, beta    : component weights
    metric         : passed to cdist for the ground cost matrix

    Returns
    -------
    float : combined EMD distance
    """
    pdd_same_1,  pdd_cross_1  = split1
    pdd_same_2,  pdd_cross_2  = split2

    dist = 0.0
    if alpha != 0.0:
        dm_same   = cdist(pdd_same_1[:, 1:],  pdd_same_2[:, 1:],
                          metric=metric, **kwargs)
        d_same, _ = _network_simplex(pdd_same_1[:, 0], pdd_same_2[:, 0], dm_same)
        dist     += alpha * d_same

    if beta != 0.0:
        dm_cross   = cdist(pdd_cross_1[:, 1:], pdd_cross_2[:, 1:],
                           metric=metric, **kwargs)
        d_cross, _ = _network_simplex(pdd_cross_1[:, 0], pdd_cross_2[:, 0],
                                      dm_cross)
        dist      += beta * d_cross

    return dist


# ── convenience: combined distance (geometry + chemistry) ────────────────────

def emd_combined(periodic_set_1, periodic_set_2,
                 k=100,
                 gamma=0.5,
                 alpha=0.5, beta=0.5,
                 metric='chebyshev',
                 **pdd_kwargs):
    """
    Single-call combined distance:

        (1 - gamma) * emd(pdd_1, pdd_2)          # geometry only
      +      gamma  * emd_split(split_1, split_2) # chemistry split

    gamma=0 recovers the original geometry-only EMD.
    gamma=1 uses only the chemistry-split EMD.
    """
    pdd1    = pdd(periodic_set_1, k=k, **pdd_kwargs)
    pdd2    = pdd(periodic_set_2, k=k, **pdd_kwargs)
    split1  = pdd_split(periodic_set_1, k=k, **pdd_kwargs)
    split2  = pdd_split(periodic_set_2, k=k, **pdd_kwargs)

    d_geom  = emd(pdd1, pdd2, metric=metric)
    d_chem  = emd_split(split1, split2, alpha=alpha, beta=beta, metric=metric)

    return (1 - gamma) * d_geom + gamma * d_chem
