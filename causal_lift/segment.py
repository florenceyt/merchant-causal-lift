"""
segment.py
----------
Post-estimation segment analysis: groups merchants by CATE quartile and
by categorical features to surface highest-ROI targeting segments.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import seaborn as sns


def add_cate_segments(df: pd.DataFrame, cate: np.ndarray, n_quantiles: int = 4) -> pd.DataFrame:
    """
    Append CATE estimates and quartile labels to the DataFrame.
    Quartile 4 = highest estimated lift.
    """
    df = df.copy()
    df["cate"] = cate
    df["cate_quartile"] = pd.qcut(cate, q=n_quantiles, labels=[f"Q{i}" for i in range(1, n_quantiles + 1)])
    return df


def segment_summary(df: pd.DataFrame, group_col: str = "industry") -> pd.DataFrame:
    """
    Return mean CATE and merchant count by a categorical segment.
    """
    return (
        df.groupby(group_col)
        .agg(
            n_merchants=("cate", "count"),
            mean_cate=("cate", "mean"),
            median_cate=("cate", "median"),
        )
        .sort_values("mean_cate", ascending=False)
        .reset_index()
    )


def plot_cate_distribution(
    df: pd.DataFrame,
    figsize: tuple = (8, 4),
    save_path: str | None = None,
) -> plt.Figure:
    """Histogram of CATE estimates with ATE reference line."""
    fig, ax = plt.subplots(figsize=figsize)
    ate = df["cate"].mean()

    ax.hist(df["cate"], bins=50, color="#4C8BB5", edgecolor="white", alpha=0.85)
    ax.axvline(ate, color="#E07B54", linewidth=2, linestyle="--", label=f"ATE = ${ate:,.0f}")
    ax.axvline(0, color="black", linewidth=1, linestyle="-", alpha=0.4, label="Zero effect")

    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"${x:,.0f}"))
    ax.set_xlabel("Estimated CATE (90-day volume lift, $)", fontsize=11)
    ax.set_ylabel("Merchant Count", fontsize=11)
    ax.set_title("Distribution of Individual Treatment Effect Estimates", fontsize=13, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


def plot_cate_by_segment(
    df: pd.DataFrame,
    group_col: str = "industry",
    figsize: tuple = (8, 4),
    save_path: str | None = None,
) -> plt.Figure:
    """Horizontal bar chart of mean CATE by segment, sorted descending."""
    summary = segment_summary(df, group_col).sort_values("mean_cate")

    fig, ax = plt.subplots(figsize=figsize)
    colors = ["#4C8BB5" if v >= 0 else "#E07B54" for v in summary["mean_cate"]]
    bars = ax.barh(summary[group_col], summary["mean_cate"], color=colors, edgecolor="white")

    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"${x:,.0f}"))
    ax.set_xlabel("Mean Estimated CATE ($)", fontsize=11)
    ax.set_title(f"Average Lift Estimate by {group_col.replace('_', ' ').title()}", fontsize=13, fontweight="bold")
    ax.axvline(0, color="black", linewidth=0.8)
    ax.grid(axis="x", alpha=0.3)

    for bar, val in zip(bars, summary["mean_cate"]):
        ax.text(
            bar.get_width() + 30,
            bar.get_y() + bar.get_height() / 2,
            f"${val:,.0f}",
            va="center", fontsize=9,
        )

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


def plot_feature_importance(
    importances: pd.Series,
    figsize: tuple = (7, 4),
    save_path: str | None = None,
) -> plt.Figure:
    """Horizontal bar chart of Causal Forest feature importances."""
    imp = importances.sort_values()
    labels = {
        "months_in_business": "Months in Business",
        "avg_monthly_txn_pre": "Pre-Period Transactions",
        "num_locations": "Number of Locations",
        "has_pos_system": "Has POS System",
        "onboarding_score": "Onboarding Score",
        "industry_encoded": "Industry",
        "region_encoded": "Region",
    }
    fig, ax = plt.subplots(figsize=figsize)
    ax.barh(
        [labels.get(i, i) for i in imp.index],
        imp.values,
        color="#4C8BB5",
        edgecolor="white",
    )
    ax.set_xlabel("Feature Importance", fontsize=11)
    ax.set_title("Causal Forest — Feature Importances\n(Drivers of Treatment Effect Heterogeneity)", fontsize=12, fontweight="bold")
    ax.grid(axis="x", alpha=0.3)
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig
