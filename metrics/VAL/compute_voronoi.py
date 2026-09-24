"""
Voronoi-based local packing metric at scale (pymatgen + ASE + SciPy)

Implements:
0) Print radii comparison: pymatgen Element.atomic_radius vs ASE covalent_radii
2) Precompute V_occ[element] = 4/3*pi*r_cov(element)^3 using ASE covalent radii
3) Input df with CIF strings in column 'cif'
4) Parse Structure.from_str(..., fmt='cif')
5a) Build 3x3x3 supercell
6a) SciPy Voronoi on supercell points; compute Voronoi cell volume for central-image sites
5b/6b) Pymatgen VoronoiNN per site (fallback / comparison route)
7) Per-structure per-species stats of phi_i = V_occ(species)/V_voro(site)
8) Time route a vs b
9) Store per-structure stats dict into new df column; save df to pickle
10) Aggregate across all structures per species
10a) Aggregate “metals” across structures that DO NOT contain any anions in list
10b) Aggregate “halides” across structures that contain any of {Cl, Br, I}
11) Plot KDEs of selected stat (default median) for a given element/group

Notes:
- This assumes 3D periodic crystals. Slabs / huge vacuum / molecular crystals may behave poorly.
- SciPy Voronoi is not periodic; we approximate periodicity by 3x3x3 replication and
  only trust volumes for central-image sites (usually bounded). We catch failures.
- For disordered sites / partial occupancies, this script uses site.specie when possible,
  otherwise uses the majority species in a Composition.

Dependencies: pandas, numpy, pymatgen, ase, scipy, matplotlib
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from ase.data import covalent_radii as ase_covalent_radii
from pymatgen.core import Element, Structure
from pymatgen.analysis.local_env import VoronoiNN

from scipy.spatial import Voronoi, ConvexHull, QhullError
from scipy.stats import gaussian_kde
import matplotlib.pyplot as plt
import seaborn as sns


ANIONS = {"O", "S", "Cl", "Br", "I", "Se", "Te", "N", "F"}
HALIDES = {"Cl", "Br", "I"}


# -------------------------
# 0) Radii comparison
# -------------------------
def print_radii_comparison(max_rows: int = 118) -> None:
    """
    Print a per-element comparison:
      pymatgen Element.atomic_radius (Angstrom) vs ASE covalent_radii[Z] (Angstrom)

    Note: Pymatgen atomic_radius may be None for some elements.
    """
    rows = []
    for Z in range(1, max_rows + 1):
        try:
            el = Element.from_Z(Z)
        except Exception:
            continue

        pmg_r = el.atomic_radius  # may be None
        ase_r = float(ase_covalent_radii[Z]) if Z < len(ase_covalent_radii) else float("nan")
        rows.append((Z, el.symbol, pmg_r if pmg_r is not None else np.nan, ase_r))

    df = pd.DataFrame(rows, columns=["Z", "element", "pmg_atomic_radius_A", "ase_covalent_radius_A"])
    #df = df[["pmg_atomic_radius_A", "ase_covalent_radius_A"]]
    #print(df.to_string(index=False))
    ax = sns.scatterplot(df, x="pmg_atomic_radius_A", y="ase_covalent_radius_A")
    plt.show()


# -------------------------
# 2) Precompute V_occ
# -------------------------
def precompute_vocc_from_ase_covalent() -> Dict[str, float]:
    """
    V_occ[element] = 4/3*pi*r_cov^3, with r from ASE covalent radii table.
    Returns Å^3 volumes keyed by element symbol.
    """
    vocc: Dict[str, float] = {}
    for Z in range(1, len(ase_covalent_radii)):
        r = float(ase_covalent_radii[Z])
        if not np.isfinite(r) or r <= 0:
            continue
        try:
            sym = Element.from_Z(Z).symbol
        except Exception:
            continue
        vocc[sym] = (4.0 * math.pi * (r ** 3)) / 3.0
    return vocc


# -------------------------
# Helpers
# -------------------------
def site_symbol(site) -> str:
    """
    Robustly extract a single element symbol for a site.
    - If ordered: site.specie.symbol
    - If disordered: choose the species with highest occupancy
    """
    try:
        return site.specie.symbol  # ordered
    except Exception:
        # disordered: site.species is a Composition-like mapping {Specie: occ}
        sp_map = site.species
        # pick the max-occupancy species
        sp = max(sp_map.items(), key=lambda kv: kv[1])[0]
        return sp.symbol


def safe_percentile(x: np.ndarray, p: float) -> float:
    if x.size == 0:
        return float("nan")
    return float(np.percentile(x, p))


def stats_from_phi(phi_by_el: Dict[str, List[float]]) -> Dict[str, Dict[str, float]]:
    out: Dict[str, Dict[str, float]] = {}
    for el, vals in phi_by_el.items():
        a = np.asarray(vals, dtype=float)
        out[el] = {
            "mean": float(np.mean(a)) if a.size else float("nan"),
            "median": float(np.median(a)) if a.size else float("nan"),
            "p95": safe_percentile(a, 95.0),
            "max": float(np.max(a)) if a.size else float("nan"),
            "n": int(a.size),
        }
    return out


def structure_elements(struct: Structure) -> set[str]:
    return {site_symbol(s) for s in struct.sites}


# -------------------------
# 6a) SciPy Voronoi route
# -------------------------
def _region_volume(vertices: np.ndarray) -> float:
    """
    Volume of a convex polyhedron given by its vertices (3D) via ConvexHull.
    """
    if vertices.shape[0] < 4:
        return float("nan")
    hull = ConvexHull(vertices)
    return float(hull.volume)


def voronoi_volumes_scipy_supercell(
    struct: Structure,
    supercell: Tuple[int, int, int] = (3, 3, 3),
) -> Tuple[np.ndarray, Optional[str]]:
    """
    Approximate periodic Voronoi volumes by:
    - building a 3x3x3 supercell
    - running scipy.spatial.Voronoi on the cartesian coordinates
    - extracting finite region volumes for central-image sites only

    Returns:
      volumes (Å^3) of length N_sites (original structure ordering),
      error string (None if success)

    Failure modes:
    - Voronoi degeneracies / coplanar sets -> QhullError
    - Infinite regions for central points (rare but possible) -> NaNs
    """
    n0 = len(struct)
    if n0 == 0:
        return np.array([], dtype=float), "empty structure"

    # Build supercell
    sc = struct.copy()
    sc.make_supercell(supercell)

    # Map supercell sites back to original indices + image translation
    # For a (3,3,3) supercell, images are 0,1,2 along each axis.
    # We want the "central" image (1,1,1) for each original site.
    # Pymatgen stores properties but not explicit image indices; easiest is to
    # reconstruct by iterating in the same order as make_supercell uses:
    # It replicates by lattice translations and appends sites.
    # We'll build translations explicitly and match the order.
    a, b, c = supercell
    translations = [(i, j, k) for i in range(a) for j in range(b) for k in range(c)]
    # Order in pymatgen supercell is typically translation-major then site-major.
    # We'll assume that: for each translation, all original sites are appended.
    # This matches common pymatgen behavior. If it differs, central extraction may be off;
    # we include a sanity check on length.
    if len(sc) != n0 * (a * b * c):
        return np.full(n0, np.nan), "unexpected supercell size/order"

    cart = np.asarray(sc.cart_coords, dtype=float)

    try:
        vor = Voronoi(cart)
    except QhullError as e:
        return np.full(n0, np.nan), f"scipy Voronoi QhullError: {e}"
    except Exception as e:
        return np.full(n0, np.nan), f"scipy Voronoi failed: {type(e).__name__}: {e}"

    vols = np.full(n0, np.nan, dtype=float)
    central_t = (a // 2, b // 2, c // 2)  # (1,1,1) for (3,3,3)
    t_index = translations.index(central_t)

    # For each original site i, its central copy is at index:
    # idx = t_index*n0 + i
    for i in range(n0):
        idx = t_index * n0 + i
        region_id = vor.point_region[idx]
        region = vor.regions[region_id]
        if region is None or len(region) == 0 or (-1 in region):
            # Infinite region -> cannot compute volume in this approximation
            continue
        verts = vor.vertices[np.array(region, dtype=int)]
        try:
            vols[i] = _region_volume(verts)
        except QhullError:
            continue
        except Exception:
            continue

    # If too many NaNs, consider failure (heuristic)
    nan_frac = np.mean(~np.isfinite(vols))
    if nan_frac > 0.25:
        return vols, f"too many infinite/invalid regions (nan_frac={nan_frac:.2f})"
    return vols, None


# -------------------------
# 5b/6b) Pymatgen VoronoiNN route
# -------------------------
def voronoi_volumes_pmg_voronoinn(struct: Structure) -> Tuple[np.ndarray, Optional[str]]:
    """
    Voronoi volumes per site using pymatgen.analysis.local_env.VoronoiNN
    by summing facet pyramid volumes.

    Returns:
      volumes (Å^3) length N, error string.
    """
    n0 = len(struct)
    if n0 == 0:
        return np.array([], dtype=float), "empty structure"

    vnn = VoronoiNN()
    vols = np.full(n0, np.nan, dtype=float)
    try:
        for i in range(n0):
            poly = vnn.get_voronoi_polyhedra(struct, i)
            vols[i] = sum(poly[j]["volume"] for j in poly)
    except Exception as e:
        return vols, f"VoronoiNN failed: {type(e).__name__}: {e}"

    nan_frac = np.mean(~np.isfinite(vols))
    if nan_frac > 0.05:
        return vols, f"invalid VoronoiNN volumes (nan_frac={nan_frac:.2f})"
    return vols, None


# -------------------------
# Phi computation per route
# -------------------------
def phi_stats_for_structure(
    struct: Structure,
    vocc: Dict[str, float],
    route: str,
    supercell: Tuple[int, int, int] = (3, 3, 3),
) -> Dict[str, Any]:
    """
    Compute per-species phi stats for a structure:
      phi_i = V_occ(species(site_i)) / V_voro(site_i)

    Returns dict with:
      - "route": "scipy" or "pmg"
      - "time_s": float
      - "errors": str|None
      - "per_element": { "Mo": {mean,median,p95,max,n}, ... }
      - "per_site_phi": optional list[float] (kept for downstream aggregates/plots)
    """
    t0 = time.perf_counter()

    if route == "scipy":
        vv, err = voronoi_volumes_scipy_supercell(struct, supercell=supercell)
    elif route == "pmg":
        vv, err = voronoi_volumes_pmg_voronoinn(struct)
    else:
        raise ValueError("route must be 'scipy' or 'pmg'")

    per_el: Dict[str, List[float]] = {}
    per_site_phi: List[float] = []

    # Compute phi per site where Voronoi volume is valid and V_occ is known
    for i, site in enumerate(struct.sites):
        el = site_symbol(site)
        if el not in vocc:
            continue
        v_occ = vocc[el]
        v_voro = vv[i] if i < len(vv) else np.nan
        if not np.isfinite(v_voro) or v_voro <= 0:
            continue
        phi = v_occ / float(v_voro)
        per_el.setdefault(el, []).append(phi)
        per_site_phi.append(phi)

    t1 = time.perf_counter()

    return {
        "route": route,
        "time_s": float(t1 - t0),
        "error": err,
        "per_element": stats_from_phi(per_el),
        "per_site_phi": per_site_phi,  # keep list for flexible global aggregation/KDE
        "n_sites": int(len(struct)),
        "elements": sorted(list(structure_elements(struct))),
    }


# -------------------------
# Main dataframe pipeline
# -------------------------
@dataclass
class RowResult:
    ok: bool
    error: Optional[str]
    stats: Dict[str, Any]


def process_row_cif(
    cif_str: str,
    vocc: Dict[str, float],
    supercell: Tuple[int, int, int] = (3, 3, 3),
    route: str='pmg'
) -> RowResult:
    try:
        struct = Structure.from_str(cif_str, fmt="cif")
    except Exception as e:
        return RowResult(ok=False, error=f"parse failed: {type(e).__name__}: {e}", stats={})

    if route=='scipy':
        stats_scipy = phi_stats_for_structure(struct, vocc, route="scipy", supercell=supercell)
        stats_pmg = []
    else:
        stats_pmg = phi_stats_for_structure(struct, vocc, route="pmg", supercell=supercell)
        stats_scipy = []

    out = {
        "scipy": stats_scipy,
        "pmg": stats_pmg,
    }
    return RowResult(ok=True, error=None, stats=out)


def add_phi_stats_column(
    df: pd.DataFrame,
    vocc: Dict[str, float],
    cif_col: str = "cif",
    out_pickle: str = "df_with_phi_stats.pkl",
    supercell: Tuple[int, int, int] = (3, 3, 3),
    route: str = "pmg"
) -> pd.DataFrame:
    if cif_col not in df.columns:
        raise KeyError(f"Expected CIF column '{cif_col}'")

    results: List[Dict[str, Any]] = []
    for s in df[cif_col].tolist():
        rr = process_row_cif(s, vocc=vocc, supercell=supercell, route=route)
        if rr.ok:
            results.append(rr.stats)
        else:
            results.append({"error": rr.error})

    df_out = df.copy()
    df_out["phi_stats"] = results
    df_out.to_pickle(out_pickle)
    return df_out


# -------------------------
# Aggregations across dataset
# -------------------------
def aggregate_per_element(
    df: pd.DataFrame,
    route: str = "scipy",
    stat: str = "median",
) -> Dict[str, List[float]]:
    """
    Collect a chosen stat per element across all structures.
    For each structure, takes per-element stats[el][stat] and appends to list.
    """
    agg: Dict[str, List[float]] = {}
    for d in df["phi_stats"]:
        if not isinstance(d, dict) or "error" in d:
            continue
        if route not in d:
            continue
        per_el = d[route].get("per_element", {})
        for el, stats in per_el.items():
            v = stats.get(stat, np.nan)
            if np.isfinite(v):
                agg.setdefault(el, []).append(float(v))
    return agg


def aggregate_per_site_phi_per_element(
    df: pd.DataFrame,
    route: str = "pmg",
) -> Dict[str, List[float]]:
    """
    Aggregate per-site φ values per element across all structures,
    assuming per_site_phi ordering matches site iteration ordering,
    and that sites were grouped per element during computation.

    Returns:
        {element: [phi_site_values...]}
    """

    agg: Dict[str, List[float]] = {}

    for d in df["phi_stats"]:
        if not isinstance(d, dict) or "error" in d:
            continue
        if route not in d:
            continue

        data = d[route]
        per_el = data.get("per_element", {})
        site_phis = data.get("per_site_phi", [])

        offset = 0
        for el, stats in per_el.items():
            n = stats.get("n", 0)

            # slice φ values belonging to this element
            vals = site_phis[offset:offset + n]
            offset += n

            for v in vals:
                if np.isfinite(v):
                    agg.setdefault(el, []).append(float(v))

    return agg


def aggregate_group_species_agnostic(
    df: pd.DataFrame,
    group: str,
    route: str = "scipy",
    stat: str = "median",
) -> List[float]:
    """
    Species-agnostic aggregation for:
      - group == "metals": only structures that contain NONE of ANIONS
      - group == "halides": structures that contain any of HALIDES

    Returns list of per-site phi values pooled across eligible structures,
    then summarized by chosen stat per structure? You asked species-agnostic across cifs;
    this function pools the chosen per-structure stat across all *sites* indirectly.

    Practical choice here:
      - pool per-structure site-level phis, then compute stat across pooled list for plotting.
    """
    pooled: List[float] = []
    for d in df["phi_stats"]:
        if not isinstance(d, dict) or "error" in d or route not in d:
            continue

        elems = set(d[route].get("elements", []))
        if group == "metals":
            if len(elems.intersection(ANIONS)) != 0:
                continue
        elif group == "halides":
            if len(elems.intersection(HALIDES)) == 0:
                continue
        else:
            raise ValueError("group must be 'metals' or 'halides'")

        # pool site-level phis (already computed with element-specific V_occ and site V_voro)
        phis = d[route].get("per_site_phi", [])
        for x in phis:
            if np.isfinite(x):
                pooled.append(float(x))
    return pooled


# -------------------------
# 11) KDE plotting (no seaborn)
# -------------------------
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde


def plot_kde(
    df: pd.DataFrame,
    target: str = "S",
    route: str = "scipy",
    stat: str = "median",
    agg_type: str = "per_structure",
    min_points: int = 50,
) -> None:
    """
    Plot KDE distribution for:
      - target = element symbol (e.g. 'S')
      - target = 'metals' (species-agnostic, structures without ANIONS)
      - target = 'halides' (species-agnostic, structures with Cl/Br/I)

    Falls back to histogram if insufficient points for KDE.
    """

    sns.set_style("white")
    sns.set_palette("muted")

    if target in ("metals", "halides"):
        data = np.asarray(
            aggregate_group_species_agnostic(df, target, route=route, stat=stat),
            dtype=float,
        )
        label = f"{target} (pooled site φ)"
        xlabel = "φ (V_occ / V_voro)"
    else:
        if agg_type == 'per_structure':
            per_el = aggregate_per_element(df, route=route, stat=stat)
        elif agg_type == 'per_site':
            per_el = aggregate_per_site_phi_per_element(df, route=route)
        data = np.asarray(per_el.get(target, []), dtype=float)
        label = f"{target} ({stat} per structure)"
        xlabel = f"φ_{target} ({stat})"

    data = data[np.isfinite(data)]

    plt.figure()

    if data.size >= min_points:
        # KDE route
        kde = gaussian_kde(data)
        xs = np.linspace(np.percentile(data, 1), np.percentile(data, 99), 400)
        ys = kde(xs)

        sns.lineplot(x=xs, y=ys)
        plt.ylabel("density")

    else:
        # Histogram fallback
        print(
            f"Not enough points for KDE (n={data.size}), "
            "falling back to histogram."
        )
        sns.histplot(data, kde=False, stat="density", bins="auto")
        plt.ylabel("density")

    #plt.title(f"Distribution of φ using route={route}")
    plt.xlabel(xlabel)
    plt.legend([label])

    sns.despine()          # remove top/right axes
    plt.tight_layout()
    plt.show()

import math
from typing import Dict, List, Optional

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


def plot_element_phi_grid(
    agg: Dict[str, List[float]],
    elements: Optional[List[str]] = None,
    mode: str = "kde",          # "kde" or "hist"
    min_points: int = 50,
    cols: int = 8,
    xlim: Optional[tuple[float, float]] = None,
    sharex: bool = True,
    sharey: bool = False,
    title: Optional[str] = None,
) -> None:
    """
    Plot a grid of tiny subplots (one per element) showing phi distribution.

    Parameters
    ----------
    agg : dict
        {element: [phi values]}.
    elements : list or None
        Which elements to plot (default: all keys sorted).
    mode : str
        "kde" or "hist".
    min_points : int
        Minimum points required for KDE; otherwise uses histogram.
    cols : int
        Number of subplot columns.
    xlim : (xmin, xmax) or None
        Optional shared x-limits.
    sharex, sharey : bool
        Share axes across subplots.
    title : str or None
        Figure title.
    """
    sns.set_style("white")
    sns.set_palette("muted")

    if elements is None:
        elements = sorted(agg.keys())

    # Filter elements that actually have data
    elements = [e for e in elements if e in agg and len(agg[e]) > 0]
    if not elements:
        print("No elements with data to plot.")
        return

    n = len(elements)
    rows = math.ceil(n / cols)

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2.2, rows * 1.8), sharex=sharex, sharey=sharey)
    axes = np.array(axes).reshape(rows, cols)

    for idx, el in enumerate(elements):
        r = idx // cols
        c = idx % cols
        ax = axes[r, c]

        data = np.asarray(agg[el], dtype=float)
        data = data[np.isfinite(data)]

        if data.size == 0:
            ax.set_axis_off()
            continue

        use_kde = (mode == "kde") and (data.size >= min_points)

        if use_kde:
            sns.kdeplot(x=data, bw_adjust=0.6, ax=ax, fill=False, linewidth=1.2)
        else:
            # histogram fallback
            sns.histplot(x=data, ax=ax, bins="auto", stat="density", kde=False)

        ax.set_title(f"{el} (n={data.size})", fontsize=9)

        # Clean look
        ax.tick_params(axis="both", labelsize=8)
        sns.despine(ax=ax)

        # Limits
        if xlim is not None:
            ax.set_xlim(*xlim)

        # Reduce clutter: only label left column and bottom row
        if r != rows - 1:
            ax.set_xlabel("")
        if c != 0:
            ax.set_ylabel("")

    # Turn off unused axes
    for j in range(n, rows * cols):
        rr = j // cols
        cc = j % cols
        axes[rr, cc].set_axis_off()

    if title:
        fig.suptitle(title, fontsize=14)

    plt.tight_layout(rect=(0, 0, 1, 0.97) if title else None)
    out_path = "phi_element_grid.png"
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    #plt.show()

# -------------------------
# Example usage (edit to your pipeline)
# -------------------------
if __name__ == "__main__":
    # 0) Radii comparison printout
    #print_radii_comparison()

    # 2) Precompute V_occ from ASE covalent radii
    V_OCC = precompute_vocc_from_ase_covalent()

    # 3) Load dataframe
    #df = pd.read_pickle("full_icsd_voronoi.pickle")
    df = pd.read_csv("LiNbN_trajectory.csv")
    #df2 = pd.read_pickle("df_with_phi_stats.pkl")
    #df2 = pd.read_pickle("full_icsd_voronoi.pickle")
    #df2 = pd.read_pickle("full_icsd_voronoi.pickle")
    # Expect df has column 'cif'

    # ---- Uncomment below when you have df ----
    df2 = add_phi_stats_column(
        df,
        vocc=V_OCC,
        route='scipy',
        cif_col="cif",
        #out_pickle="full_icsd_voronoi_scipy.pkl",
        out_pickle="LiNbN_with_phi_scipy_stats.pkl",
    )
    #
    # 8) Timing summary
    #print_timing_summary(df2)
    #
    # 10) Aggregations
#    agg_el = aggregate_per_element(df2, route="pmg", stat="p95") # "median"
#   plot_element_phi_grid(
#      agg_el,
#      mode="kde",
#      cols=8,
#      xlim=(0.0, 1.5),
#      title="Per-structure median φ distributions (ICSD)"
#   )
    #agg_el_per_site = aggregate_per_site_phi_per_element(df2,) 
    #print(f"Elements aggregated (pmg, median): {sorted(agg_el.keys())}")
    #
    # 11) Plot KDE for element / groups
    #plot_kde(df2, target="O", route="pmg", stat="median", agg_type='per_structure')
    #plot_kde(df2, target="O", route="pmg", stat="median", agg_type='per_site')
    #plot_kde(df2, target="metals",  route="pmg", stat="median")
    #plot_kde(df2, target="halides", route="pmg", stat="median")
