import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


def plot_adequacy_score(
    df: pd.DataFrame,
    score_col: str = "adequacy_score",
    cutoff: int = -2,
    mode: str = "kde",      # "kde" or "hist"
    min_points: int = 50,
    bins: int = 1000,
    title = None,
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

    df = df[df[score_col]>cutoff]

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

    plt.xlabel("adequacy score")
    plt.title(title or "Adequacy score distribution")

    sns.despine()
    plt.tight_layout()
    plt.show()

# -------------------------
# Example usage
# -------------------------
if __name__ == "__main__":
    #df_scored = pd.read_pickle("icsd_with_adequacy_score.pkl")
    df_scored = pd.read_pickle("icsd_VAL_scipy_5comp_p95.pkl")
    print(df_scored.columns)
    plot_adequacy_score(df_scored, cutoff=-70, mode="hist")
    plot_adequacy_score(df_scored, cutoff=-70,mode="kde")
