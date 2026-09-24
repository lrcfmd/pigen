#!/usr/bin/env python3
"""
compute_val.py

Given a dataframe with CIF strings (column 'cif'), compute:
- pmg Voronoi-based phi stats per structure
- VAL score using pre-fitted ICSD element GMMs saved as JSON

Outputs:
- df with new columns: phi_stats, phi_mean_per_element, val_score
- saved to pickle (or parquet if you want)

Usage:
  python compute_val.py input.pkl icsd_phi_gmms.json output.pkl
"""

from __future__ import annotations

import json
import math
import sys
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from ase.data import covalent_radii as ase_covalent_radii
from pymatgen.core import Structure, Composition
from pymatgen.analysis.local_env import VoronoiNN
from sklearn.mixture import GaussianMixture

# -------------------------
# Radii -> V_occ lookup (Å^3)
# -------------------------
def precompute_vocc_from_ase_covalent() -> Dict[str, float]:
    vocc: Dict[str, float] = {}
    for z in range(1, len(ase_covalent_radii)):
        r = float(ase_covalent_radii[z])
        if not np.isfinite(r) or r <= 0:
            continue
        # Element symbol via pymatgen atomic number mapping
        # (avoid Element import for speed; Composition can parse symbols anyway)
        # We'll just use pymatgen Element if you prefer; keeping simple:
        from pymatgen.core import Element
        sym = Element.from_Z(z).symbol
        vocc[sym] = (4.0 * math.pi * (r ** 3)) / 3.0
    return vocc


def site_symbol(site) -> str:
    try:
        return site.specie.symbol
    except Exception:
        sp_map = site.species
        sp = max(sp_map.items(), key=lambda kv: kv[1])[0]
        return sp.symbol


# -------------------------
# Voronoi volumes via pmg VoronoiNN
# -------------------------
def voronoi_volumes_pmg(struct: Structure) -> Tuple[np.ndarray, Optional[str]]:
    vnn = VoronoiNN()
    n = len(struct)
    vols = np.full(n, np.nan, dtype=float)
    try:
        for i in range(n):
            poly = vnn.get_voronoi_polyhedra(struct, i)
            vols[i] = sum(poly[j]["volume"] for j in poly)
    except Exception as e:
        return vols, f"VoronoiNN failed: {type(e).__name__}: {e}"

    if np.mean(~np.isfinite(vols)) > 0.05:
        return vols, "too many invalid Voronoi volumes"
    return vols, None


def phi_stats_pmg(struct: Structure, vocc: Dict[str, float]) -> Dict[str, Any]:
    vv, err = voronoi_volumes_pmg(struct)

    per_el_vals: Dict[str, List[float]] = {}
    per_site_phi: List[float] = []

    for i, site in enumerate(struct.sites):
        el = site_symbol(site)
        v_occ = vocc.get(el, None)
        v_voro = vv[i] if i < len(vv) else np.nan
        if v_occ is None or (not np.isfinite(v_voro)) or v_voro <= 0:
            continue
        phi = float(v_occ) / float(v_voro)
        per_el_vals.setdefault(el, []).append(phi)
        per_site_phi.append(phi)

    per_element = {}
    for el, vals in per_el_vals.items():
        a = np.asarray(vals, dtype=float)
        per_element[el] = {
            "mean": float(np.mean(a)),
            "median": float(np.median(a)),
            "p95": float(np.percentile(a, 95)),
            "max": float(np.max(a)),
            "n": int(a.size),
        }

    return {
        "route": "pmg",
        "error": err,
        "per_element": per_element,
        "per_site_phi": per_site_phi,
        "n_sites": int(len(struct)),
        "elements": sorted({site_symbol(s) for s in struct.sites}),
    }


def extract_mean_phi_per_element(phi_stats_entry: Any) -> Dict[str, float]:
    if not isinstance(phi_stats_entry, dict):
        return {}
    per_el = phi_stats_entry.get("per_element", {}) or {}
    out = {}
    for el, stats in per_el.items():
        v = stats.get("mean", np.nan)
        if np.isfinite(v):
            out[el] = float(v)
    return out


def parse_composition_to_atomic_fractions(comp_val: Any) -> Dict[str, float]:
    if isinstance(comp_val, Composition):
        comp = comp_val
    elif isinstance(comp_val, str):
        comp = Composition(comp_val)
    elif isinstance(comp_val, dict):
        comp = Composition(comp_val)
    else:
        comp = Composition(str(comp_val))

    frac = comp.fractional_composition
    return {el.symbol: float(amount) for el, amount in frac.items()}


# -------------------------
# Load GMMs from JSON (same format as save_gmms_to_json)
# -------------------------
def load_gmms_from_json(path: str, random_state: int = 0) -> Dict[str, GaussianMixture]:
    with open(path, "r") as f:
        payload = json.load(f)

    gmms: Dict[str, GaussianMixture] = {}
    for el, d in payload.items():
        k = int(d["n_components"])
        gmm = GaussianMixture(
            n_components=k,
            covariance_type="full",
            reg_covar=float(d.get("reg_covar", 0.0)),
            random_state=random_state,
        )
        gmm.weights_ = np.asarray(d["weights"], dtype=float)
        gmm.means_ = np.asarray(d["means"], dtype=float).reshape(k, 1)
        cov = np.asarray(d["covariances"], dtype=float).reshape(k, 1, 1)
        gmm.covariances_ = cov
        gmm.precisions_cholesky_ = 1.0 / np.sqrt(cov)
        gmms[el] = gmm
    return gmms


def val_score(
    phi_stats_entry: Dict[str, Any],
    composition_val: Any,
    gmms: Dict[str, GaussianMixture],
    eps_unseen: float = -50.0,
) -> float:
    """
    VAL = sum_e x_e * log p_e(phi_e_mean)
    """
    per_el_mean = extract_mean_phi_per_element(phi_stats_entry)
    if not per_el_mean:
        return float("nan")

    xfrac = parse_composition_to_atomic_fractions(composition_val)

    score = 0.0
    for el, x in xfrac.items():
        if x <= 0:
            continue

        phi_el = per_el_mean.get(el, None)
        gmm = gmms.get(el, None)

        if (phi_el is None) or (gmm is None) or (not np.isfinite(phi_el)):
            score += float(x) * float(eps_unseen)
        else:
            lp = float(gmm.score_samples(np.array([[float(phi_el)]], dtype=float))[0])
            score += float(x) * lp

    return float(score)

def _get_pmg_block(phi_stats_entry: Any) -> Optional[dict]:
    """
    Accept either:
      A) {"pmg": {...}, "scipy": ...}
      B) {"route": "pmg", ...}  (already the pmg dict)
    Return the pmg dict, else None.
    """
    if not isinstance(phi_stats_entry, dict):
        return None

#   if "pmg" in phi_stats_entry and isinstance(phi_stats_entry["pmg"], dict):
#       return phi_stats_entry["pmg"]

    if "scipy" in phi_stats_entry and isinstance(phi_stats_entry["scipy"], dict):
        return phi_stats_entry["scipy"]

    #if phi_stats_entry.get("route") == "pmg":
    if phi_stats_entry.get("route") == "scipy":
        return phi_stats_entry

    return None


def main() -> None:
    if len(sys.argv) != 4:
        raise SystemExit("Usage: python compute_val.py <input.pkl> <gmms.json> <output.pkl>")

    in_path, gmm_path, out_path = sys.argv[1], sys.argv[2], sys.argv[3]

    df = pd.read_pickle(in_path)
    if "composition" not in df.columns:
        raise KeyError("Input df must have a 'composition' column")

    # Only required if we need to compute phi_stats
    need_compute_phi = "phi_stats" not in df.columns
    if need_compute_phi:
        print('Need to compute stats - Voronoi')
    if need_compute_phi and "cif" not in df.columns:
        raise KeyError("Input df must have a 'cif' column if 'phi_stats' is not present")

    vocc = precompute_vocc_from_ase_covalent() if need_compute_phi else None
    gmms = load_gmms_from_json(gmm_path)

    phi_stats_list = []
    phi_mean_list = []
    val_list = []
    err_list = []

    # If phi_stats exists, iterate over it; otherwise compute from CIF
    if "phi_stats" in df.columns:
        print('Extracting phi_stats -- precomputed')
        iterable = zip(df["phi_stats"].tolist(), df["composition"].tolist())
        for phi_entry, comp in iterable:
            pmg_entry = _get_pmg_block(phi_entry)

            # If phi_stats is present but doesn't contain pmg, fallback to recompute if CIF available
            if pmg_entry is None:
                print('pmg_entry is None - recomputing stats!')
                if "cif" in df.columns:
                    try:
                        struct = Structure.from_str(df.loc[len(phi_stats_list), "cif"], fmt="cif")
                        pmg_entry = phi_stats_pmg(struct, vocc=precompute_vocc_from_ase_covalent())
                    except Exception as e:
                        pmg_entry = {
                            "route": "pmg",
                            "error": f"parse/phi failed: {type(e).__name__}: {e}",
                            "per_element": {},
                        }
                else:
                    pmg_entry = {"route": "pmg", "error": "no pmg phi_stats and no cif to recompute", "per_element": {}}

            phi_stats_list.append(pmg_entry)
            phi_mean = extract_mean_phi_per_element(pmg_entry)
            phi_mean_list.append(phi_mean)

            v = val_score(pmg_entry, comp, gmms)
            val_list.append(v)
            err_list.append(pmg_entry.get("error", None))

    else:
        # compute from CIF
        assert vocc is not None
        for cif_str, comp in zip(df["cif"].tolist(), df["composition"].tolist()):
            try:
                struct = Structure.from_str(cif_str, fmt="cif")
                pmg_entry = phi_stats_pmg(struct, vocc=vocc)
            except Exception as e:
                pmg_entry = {"route": "pmg", "error": f"parse/phi failed: {type(e).__name__}: {e}", "per_element": {}}

            phi_stats_list.append(pmg_entry)
            phi_mean = extract_mean_phi_per_element(pmg_entry)
            phi_mean_list.append(phi_mean)

            v = val_score(pmg_entry, comp, gmms)
            val_list.append(v)
            err_list.append(pmg_entry.get("error", None))

    df_out = df.copy()
    df_out["phi_stats"] = phi_stats_list
    df_out["phi_mean_per_element"] = phi_mean_list
    df_out["val_score"] = val_list
    df_out["phi_error"] = err_list

    df_out.to_pickle(out_path)
    print(f"Wrote: {out_path}")

if __name__ == "__main__":
    main()
