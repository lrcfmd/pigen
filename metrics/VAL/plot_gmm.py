import json
from collections import Counter
from typing import Dict
from pymatgen.core import Element
import numpy as np
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture


def load_gmms_from_json(path: str, random_state: int = 0) -> Dict[str, GaussianMixture]:
    """
    Load per-element 1D GMMs from JSON (weights/means/covariances) and reconstruct sklearn objects.
    JSON format expected:
      { "S": {"n_components":..,"weights":[..],"means":[..],"covariances":[..],"reg_covar":..}, ... }
    """
    with open(path, "r") as f:
        payload = json.load(f)

    gmms: Dict[str, GaussianMixture] = {}
    for el, d in payload.items():
        k = int(d["n_components"])
        cov_type = d.get("covariance_type", "full")
        if cov_type != "full":
            raise ValueError(f"Only covariance_type='full' supported, got {cov_type}")

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
        gmm.precisions_cholesky_ = 1.0 / np.sqrt(cov)  # 1D full cov

        gmms[el] = gmm

    return gmms


def gmm_component_counts(gmms: Dict[str, GaussianMixture], cap=None) -> None:
    """
    Print counts of selected n_components and list elements hitting cap (if provided).
    """
    counts = Counter(g.n_components for g in gmms.values())
    print("Component count distribution:", dict(sorted(counts.items())))

    if cap is not None:
        hit = sorted([el for el, g in gmms.items() if g.n_components == cap])
        print(f"Elements hitting cap={cap} (n={len(hit)}): {hit}")


def plot_gmm_density(
    gmms: Dict[str, GaussianMixture],
    element: str,
    xs = None, #np. array
    xlim = None # tuple[float, float] | None = None,
) -> None:
    """
    Plot the GMM PDF for a chosen element.

    Notes:
    - This plots the model density p_el(phi).
    - If you want to overlay the empirical histogram, pass your sample data separately.
    """
    if element not in gmms:
        raise KeyError(f"Element '{element}' not found in loaded GMMs")

    gmm = gmms[element]

    if xs is None:
        # Pick a reasonable default range around component means
        mus = gmm.means_.reshape(-1)
        sig = np.sqrt(gmm.covariances_.reshape(-1))
        lo = float(np.min(mus - 4 * sig))
        hi = float(np.max(mus + 4 * sig))
        # Keep it in a sensible phi domain (optional)
        lo = max(lo, -0.1)
        hi = min(hi, 3.0)
        xs = np.linspace(lo, hi, 600)

    if xlim is not None:
        xs = np.linspace(xlim[0], xlim[1], 600)

    logp = gmm.score_samples(xs.reshape(-1, 1))
    p = np.exp(logp)

    plt.figure()
    plt.plot(xs, p, label=f"{element} GMM (k={gmm.n_components})")

    # Optional: plot components for intuition
    w = gmm.weights_.reshape(-1)
    mu = gmm.means_.reshape(-1)
    var = gmm.covariances_.reshape(-1)
    for j in range(gmm.n_components):
        pj = w[j] * (1.0 / np.sqrt(2 * np.pi * var[j])) * np.exp(-(xs - mu[j]) ** 2 / (2 * var[j]))
        plt.plot(xs, pj, linestyle="--", linewidth=1.0)

    plt.xlabel("phi")
    plt.ylabel("density p(phi)")
    plt.title(f"GMM density for {element}")
    plt.legend()
    plt.tight_layout()
    plt.show()


import math
from typing import Dict, List, Optional, Tuple

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.mixture import GaussianMixture


def plot_gmm_grid(
    gmms: Dict[str, GaussianMixture],
    elements: Optional[List[str]] = None,
    cols: int = 8,
    xlim: Tuple[float, float] = (0.0, 1.5),
    n_points: int = 400,
    plot_components: bool = False,
    title: Optional[str] = None,
    save_path: Optional[str] = None,
) -> None:
    """
    Plot a grid of tiny subplots (one per element) showing the fitted GMM density p(phi).
    Optionally overlay individual Gaussian components (dashed).

    Parameters
    ----------
    gmms : dict
        {element: sklearn GaussianMixture} loaded from json.
    elements : list or None
        Elements to plot (default: all keys sorted).
    cols : int
        Number of subplot columns.
    xlim : (xmin, xmax)
        Shared x-axis range for all plots.
    n_points : int
        Number of x samples for evaluating the density curve.
    plot_components : bool
        If True, overlay each Gaussian component as dashed curve.
    title : str or None
        Figure title.
    save_path : str or None
        If provided, save to file instead of showing.
    """
    sns.set_style("white")
    sns.set_palette("muted")

    if elements is None:
        elements = sorted(gmms.keys(), key=lambda el: Element(el).Z)
    else:
        elements = [e for e in elements if e in gmms]

    if not elements:
        print("No elements found to plot.")
        return

    n = len(elements)
    rows = math.ceil(n / cols)

    xs = np.linspace(xlim[0], xlim[1], n_points)
    X = xs.reshape(-1, 1)

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2.2, rows * 1.8), sharex=True, sharey=False)
    axes = np.array(axes).reshape(rows, cols)

    for idx, el in enumerate(elements):
        r = idx // cols
        c = idx % cols
        ax = axes[r, c]

        gmm = gmms[el]
        logp = gmm.score_samples(X)
        p = np.exp(logp)

        ax.plot(xs, p, linewidth=1.3)
        #ax.set_title(f"{el} (k={gmm.n_components})", fontsize=9)
        ax.set_title(f"{el}", fontsize=9)

        if plot_components:
            w = gmm.weights_.reshape(-1)
            mu = gmm.means_.reshape(-1)
            var = gmm.covariances_.reshape(-1)
            for j in range(gmm.n_components):
                pj = w[j] * (1.0 / np.sqrt(2 * np.pi * var[j])) * np.exp(-(xs - mu[j]) ** 2 / (2 * var[j]))
                ax.plot(xs, pj, linestyle="--", linewidth=0.9)

        ax.tick_params(axis="both", labelsize=8)
        sns.despine(ax=ax)

        # Reduce clutter: only label bottom row / left column
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

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Saved figure to: {save_path}")
        plt.close()
    else:
        plt.show()


# ---- Example usage ----
#gmms = load_gmms_from_json("icsd_phi_gmms_5comp.json")
gmms = load_gmms_from_json("icsd_phi_gmms_5comp_95p.json")
# plot_gmm_density(gmms, "S", xlim=(0.0, 1.5))
# plot_gmm_density(gmms, "O", xlim=(0.0, 1.5))
plot_gmm_grid(gmms, xlim=(0.0, 1.5), cols=8, plot_components=False, title="ICSD GMM densities", save_path="gmm_grid.png")
