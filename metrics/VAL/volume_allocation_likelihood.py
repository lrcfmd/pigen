"""
ICSD-based composition-weighted adequacy score from phi_stats (pmg route)

Assumptions:
- df has columns:
    - 'phi_stats': dict like {"pmg": {"per_element": {el: {"mean":..,"n":..}, ...}, ...}}
      (or directly the pmg dict; this script handles both)
    - 'composition': a string formula (e.g. "MoS2", "Cu3Au", "LiFePO4") OR a dict-like mapping

Pipeline:
1) Aggregate per-element mean(phi) per structure across dataset -> training samples for ICSD reference
2) Fit per-element Gaussian Mixture Model p_el(phi) (mixture; choose #components via BIC; cap at 2 by default)
3) For each structure: compute composition-weighted score = sum_i x_i * log p_el(phi_el)
   where x_i is atomic fraction from df['composition'].

Outputs:
- df['phi_mean_per_element'] : {el: mean_phi_el_in_structure}
- df['adequacy_score']       : float (higher is "more typical" under ICSD model)
- df['adequacy_terms']       : optional per-element contributions

Dependencies: pandas, numpy, pymatgen, scikit-learn
"""

from __future__ import annotations

import math
from typing import Dict, List, Tuple, Any, Optional
import json
import numpy as np
import pandas as pd
from pymatgen.core import Composition
from sklearn.mixture import GaussianMixture


# -------------------------
# Helpers to access df row data
# -------------------------
def _get_pmg_block(phi_stats_entry: Any, route: str = "pmg") -> Optional[dict]:
    """
    Accepts either:
      - {"pmg": {...}, "scipy": ...}
      - {"route": "pmg", ...}  (already the pmg block)
    Returns the pmg dict or None.
    """
    if not isinstance(phi_stats_entry, dict):
        return None
    if "error" in phi_stats_entry:
        return None
    if route in phi_stats_entry and isinstance(phi_stats_entry[route], dict):
        return phi_stats_entry[route]
    # already a route dict?
    if phi_stats_entry.get("route") == route:
        return phi_stats_entry
    return None


def extract_mean_phi_per_element_from_row(phi_stats_entry: Any, route: str = "pmg", stat: str='mean') -> Dict[str, float]:
    """
    From row phi_stats, return {el: mean_phi} for that structure.
    """
    pmg = _get_pmg_block(phi_stats_entry, route=route)
    if pmg is None:
        return {}

    per_el = pmg.get("per_element", {}) or {}
    out: Dict[str, float] = {}
    for el, stats in per_el.items():
        v = stats.get(stat, np.nan)
        if np.isfinite(v):
            out[str(el)] = float(v)
    return out


def parse_composition_to_atomic_fractions(comp_val: Any) -> Dict[str, float]:
    """
    df['composition'] can be:
      - formula string (recommended): "MoS2"
      - dict-like: {"Mo": 1, "S": 2}
      - pymatgen Composition
    Returns atomic fractions {el: x_el} summing to 1.
    """
    if isinstance(comp_val, Composition):
        comp = comp_val
    elif isinstance(comp_val, str):
        comp = Composition(comp_val)
    elif isinstance(comp_val, dict):
        comp = Composition(comp_val)
    else:
        # try string conversion as last resort
        comp = Composition(str(comp_val))

    frac = comp.fractional_composition
    return {el.symbol: float(amount) for el, amount in frac.items()}


# -------------------------
# 1) Aggregate mean per structure per element across dataset
# -------------------------
def aggregate_element_samples(
    df: pd.DataFrame,
    route: str = "pmg",
    stat: str="mean",
    min_samples: int = 30,
) -> Dict[str, np.ndarray]:
    """
    Build training samples for each element:
      samples[el] = np.array([mean_phi_el(structure_1), mean_phi_el(structure_2), ...])

    Filters out elements with < min_samples.
    """
    agg: Dict[str, List[float]] = {}
    for entry in df["phi_stats"]:
        per_el_mean = extract_mean_phi_per_element_from_row(entry, route=route, stat=stat)
        for el, v in per_el_mean.items():
            if np.isfinite(v):
                agg.setdefault(el, []).append(float(v))

    out: Dict[str, np.ndarray] = {}
    for el, vals in agg.items():
        if len(vals) >= min_samples:
            out[el] = np.asarray(vals, dtype=float)
    return out


# -------------------------
# 2) Fit p_el(phi) as Gaussian mixture (GMM)
# -------------------------
def fit_element_gmms(
    element_samples: Dict[str, np.ndarray],
    max_components: int = 2,
    reg_covar: float = 1e-5,
    random_state: int = 0,
) -> Dict[str, GaussianMixture]:
    """
    For each element, fit a GMM to samples. Choose number of components by BIC in [1..max_components].
    """
    gmms: Dict[str, GaussianMixture] = {}
    for el, x in element_samples.items():
        X = x.reshape(-1, 1)

        best_gmm = None
        best_bic = float("inf")

        for k in range(1, max_components + 1):
            gmm = GaussianMixture(
                n_components=k,
                covariance_type="full",
                reg_covar=reg_covar,
                random_state=random_state,
            )
            gmm.fit(X)
            bic = gmm.bic(X)
            if bic < best_bic:
                best_bic = bic
                best_gmm = gmm

        if best_gmm is not None:
            gmms[el] = best_gmm

    return gmms


def log_p_el(gmm: GaussianMixture, phi: float) -> float:
    """
    Log density under the fitted GMM (1D).
    """
    return float(gmm.score_samples(np.array([[phi]], dtype=float))[0])


# -------------------------
# 3) Compute composition-weighted adequacy score per structure
# -------------------------
def compute_adequacy_score_for_row(
    phi_stats_entry: Any,
    composition_val: Any,
    gmms: Dict[str, GaussianMixture],
    route: str = "pmg",
    stat: str="mean",
    eps_unseen: float = -50.0,
    return_terms: bool = False,
) -> Tuple[float, Optional[Dict[str, float]]]:
    """
    Score = sum_el x_el * log p_el(phi_el)
      - x_el from composition atomic fractions
      - phi_el from per_element mean in this structure
    If element missing a GMM or missing phi_el -> add x_el * eps_unseen

    eps_unseen is a log-density fallback penalty (negative).
    """
    per_el_mean = extract_mean_phi_per_element_from_row(phi_stats_entry, route=route, stat=stat)
    if not per_el_mean:
        return (float("nan"), None)

    xfrac = parse_composition_to_atomic_fractions(composition_val)

    score = 0.0
    terms: Dict[str, float] = {}

    for el, x in xfrac.items():
        if x <= 0:
            continue

        phi_el = per_el_mean.get(el, None)
        gmm = gmms.get(el, None)

        if (phi_el is None) or (gmm is None) or (not np.isfinite(phi_el)):
            contrib = float(x) * float(eps_unseen)
        else:
            contrib = float(x) * log_p_el(gmm, float(phi_el))

        score += contrib
        if return_terms:
            terms[el] = contrib

    return (float(score), terms if return_terms else None)


def save_gmms_to_json(gmms: Dict[str, GaussianMixture], path: str) -> None:
    """
    Save per-element 1D GMMs to JSON in a portable parameter form.
    """
    payload = {}
    for el, gmm in gmms.items():
        payload[el] = {
            "n_components": int(gmm.n_components),
            "covariance_type": str(gmm.covariance_type),
            "weights": gmm.weights_.astype(float).tolist(),
            "means": gmm.means_.reshape(-1).astype(float).tolist(),  # 1D
            # For 1D full covariance, covariances_ is (K, 1, 1); store as (K,)
            "covariances": np.array(gmm.covariances_).reshape(-1).astype(float).tolist(),
            "reg_covar": float(getattr(gmm, "reg_covar", 0.0)),
        }

    with open(path, "w") as f:
        json.dump(payload, f, indent=2)


def load_gmms_from_json(path: str, random_state: int = 0) -> Dict[str, GaussianMixture]:
    """
    Load per-element 1D GMMs from JSON and reconstruct sklearn GaussianMixture objects.
    """
    with open(path, "r") as f:
        payload = json.load(f)

    gmms: Dict[str, GaussianMixture] = {}
    for el, d in payload.items():
        k = int(d["n_components"])
        cov_type = d.get("covariance_type", "full")
        if cov_type != "full":
            raise ValueError(f"Only covariance_type='full' supported here, got {cov_type}")

        gmm = GaussianMixture(
            n_components=k,
            covariance_type="full",
            reg_covar=float(d.get("reg_covar", 0.0)),
            random_state=random_state,
        )

        # Mark as "fitted" by setting learned attributes
        gmm.weights_ = np.asarray(d["weights"], dtype=float)
        gmm.means_ = np.asarray(d["means"], dtype=float).reshape(k, 1)

        cov = np.asarray(d["covariances"], dtype=float).reshape(k, 1, 1)
        gmm.covariances_ = cov

        # Required derived attributes for scoring
        gmm.precisions_cholesky_ = 1.0 / np.sqrt(cov)  # for 1D full cov

        gmms[el] = gmm

    return gmms

def add_adequacy_scores(
    df: pd.DataFrame,
    route: str = "pmg",
    stat: str="mean",
    composition_col: str = "composition",
    max_components: int = 4,
    min_samples_per_element: int = 30,
    eps_unseen: float = -50.0,
    store_terms: bool = False,
) -> Tuple[pd.DataFrame, Dict[str, GaussianMixture]]:
    """
    End-to-end:
      1) Build element sample distributions from df (ICSD reference)
      2) Fit GMMs per element
      3) Score each row

    Returns updated df and fitted gmms.
    """
    if "phi_stats" not in df.columns:
        raise KeyError("df must contain 'phi_stats' column")
    if composition_col not in df.columns:
        raise KeyError(f"df must contain '{composition_col}' column")

    # 1) Samples
    samples = aggregate_element_samples(df, route=route, stat=stat, min_samples=min_samples_per_element)

    # 2) Fit GMMs
    gmms = fit_element_gmms(samples, max_components=max_components)

    save_gmms_to_json(gmms, "icsd_phi_gmms_5comp_95p.json")

    # 3) Score rows
    df_out = df.copy()

    df_out["phi_mean_per_element"] = df_out["phi_stats"].apply(
        lambda d: extract_mean_phi_per_element_from_row(d, route=route, stat=stat)
    )

    scores = []
    terms_list = [] if store_terms else None

    for phi_entry, comp in zip(df_out["phi_stats"].tolist(), df_out[composition_col].tolist()):
        s, t = compute_adequacy_score_for_row(
            phi_entry,
            comp,
            gmms,
            route=route,
            stat=stat,
            eps_unseen=eps_unseen,
            return_terms=store_terms,
        )
        scores.append(s)
        if store_terms and terms_list is not None:
            terms_list.append(t if t is not None else {})

    df_out["adequacy_score"] = scores
    if store_terms and terms_list is not None:
        df_out["adequacy_terms"] = terms_list

    return df_out, gmms

import seaborn as sns
import matplotlib.pyplot as plt


def plot_adequacy_score(
    df: pd.DataFrame,
    score_col: str = "adequacy_score",
    mode: str = "kde",      # "kde" or "hist"
    min_points: int = 50,
    bins: str | int = "auto",
    title: str | None = None,
) -> None:
    """
    Plot adequacy score distribution.

    Parameters
    ----------
    df : DataFrame
        Dataframe containing adequacy_score column.
    score_col : str
        Column storing scores.
    mode : str
        "kde" or "hist".
    min_points : int
        KDE fallback threshold.
    bins : histogram bins
        Passed to seaborn histplot.
    title : optional custom title
    """

    if score_col not in df.columns:
        raise KeyError(f"{score_col} not found in dataframe")

    sns.set_style("white")
    sns.set_palette("muted")

    data = np.asarray(df[score_col], dtype=float)
    data = data[np.isfinite(data)]

    if data.size == 0:
        print("No valid adequacy scores found.")
        return

    plt.figure()

    # KDE requested but insufficient data
    if mode == "kde" and data.size < min_points:
        print("Too few samples for KDE, falling back to histogram.")
        mode = "hist"

    if mode == "kde":
        sns.kdeplot(data, fill=False)
        plt.ylabel("density")

    elif mode == "hist":
        sns.histplot(data, bins=bins, stat="density", kde=False)
        plt.ylabel("density")

    else:
        raise ValueError("mode must be 'kde' or 'hist'")

    plt.xlabel("VAL score")
    plt.title(title or "VAL score distribution")

    sns.despine()
    plt.tight_layout()
    plt.show()


# -------------------------
# Example usage
# -------------------------
if __name__ == "__main__":
    df = pd.read_pickle("full_icsd_voronoi.pickle")
    #df = pd.read_pickle("full_icsd_voronoi_scipy.pkl")
    df_scored, gmms = add_adequacy_scores(
        df,
        route="pmg",
        stat="p95",
        composition_col="composition",
        max_components=5,              # allow bimodality (e.g. S)
        min_samples_per_element=30,    # avoid fitting junk
        eps_unseen=-50.0,              # penalty if element missing model/value
        store_terms=False,             # True if you want per-element contributions
    )
    #df_scored.to_pickle("icsd_VAL_scipy_5comp_p95.pkl")
    plot_adequacy_score(df_scored, mode="hist", bins=100)
