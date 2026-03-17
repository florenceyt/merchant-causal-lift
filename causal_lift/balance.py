"""
balance.py
----------
Covariate balance diagnostics pre- and post-PSM matching.
Generates standardized mean difference (SMD) love plots.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


BALANCE_COLS = [
    "months_in_business",
    "avg_monthly_txn_pre",
    "num_locations",
    "has_pos_system",
    "onboarding_score",
]

COL_LABELS = {
    "months_in_business": "Months in Business",
    "avg_monthly_txn_pre": "Avg Monthly Transactions (Pre)",
    "num_locations": "Number of Locations",
    "has_pos_system": "Has POS System",
    "onboarding_score": "Onboarding Score",
}


def standardized_mean_diff(df: pd.DataFrame, col: str) -> float:
    """Compute SMD for a single covariate."""
    treated = df.loc[df["treatment"] == 1, col]
    control = df.loc[df["treatment"] == 0, col]
    pooled_sd = np.sqrt((treated.var() + control.var()) / 2)
    if pooled_sd == 0:
        return 0.0
    return (treated.mean() - control.mean()) / pooled_sd


def compute_balance_table(
    pre_df: pd.DataFrame, post_df: pd.DataFrame, cols: list[str] = BALANCE_COLS
) -> pd.DataFrame:
    """Return a balance table comparing SMD before and after matching."""
    rows = []
    for col in cols:
        rows.append({
            "covariate": COL_LABELS.get(col, col),
            "smd_pre": standardized_mean_diff(pre_df, col),
            "smd_post": standardized_mean_diff(post_df, col),
        })
    return pd.DataFrame(rows)


def plot_love_plot(
    balance_table: pd.DataFrame,
    threshold: float = 0.1,
    figsize: tuple = (8, 5),
    save_path: str | None = None,
) -> plt.Figure:
    """
    Love plot: SMD before vs after PSM matching.
    Threshold line at |SMD| = 0.10 (conventional balance criterion).
    """
    fig, ax = plt.subplots(figsize=figsize)
    bt = balance_table.sort_values("smd_pre", ascending=True)
    y = np.arange(len(bt))

    ax.scatter(bt["smd_pre"].abs(), y, label="Before Matching", color="#E07B54", s=70, zorder=3)
    ax.scatter(bt["smd_post"].abs(), y, label="After PSM", color="#4C8BB5", s=70, zorder=3)

    for i, row in enumerate(bt.itertuples()):
        ax.plot(
            [abs(row.smd_pre), abs(row.smd_post)],
            [i, i],
            color="gray",
            linewidth=0.8,
            alpha=0.6,
        )

    ax.axvline(threshold, color="red", linestyle="--", linewidth=1.2, label=f"|SMD| = {threshold}")
    ax.set_yticks(y)
    ax.set_yticklabels(bt["covariate"].tolist(), fontsize=10)
    ax.set_xlabel("Absolute Standardized Mean Difference", fontsize=11)
    ax.set_title("Covariate Balance: Before vs. After PSM Matching", fontsize=13, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(axis="x", alpha=0.3)
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig
