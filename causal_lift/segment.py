"""
segment.py
----------
Post-estimation segment analysis: groups merchants by CATE quartile and
by categorical features to surface highest-ROI targeting segments.

CATEs are on the probability scale (percentage-point lift in 30-day activation).
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
    Quartile 4 = highest estimated lift in 30-day activation probability.
    """
    df = df.copy()
    df["cate"] = cate
    df["cate_quartile"] = pd.qcut(
        cate, q=n_quantiles, labels=[f"Q{i}" for i in range(1, n_quantiles + 1)]
    )
    return df


def segment_summary(df: pd.DataFrame, group_col: str = "industry") -> pd.DataFrame:
    """
    Return mean CATE and merchant count by a categorical segment.
    CATE values are in percentage-point terms (activation rate lift).
    """
    summary = (
        df.groupby(group_col)
        .agg(
            n_merchants=("cate", "count"),
            mean_cate_pp=("cate", lambda x: x.mean() * 100),
            median_cate_pp=("cate", lambda x: x.median() * 100),
            activation_rate_treated=("activated_30d", lambda x: x[df.loc[x.index, "treatment"] == 1].mean()),
            activation_rate_control=("activated_30d", lambda x: x[df.loc[x.index, "treatment"] == 0].mean()),
        )
        .sort_values("mean_cate_pp", ascending=False)
        .reset_index()
    )
    return summary


def decile_rank_validation(df: pd.DataFrame) -> pd.DataFrame:
    """
    Uplift model rank validation: sort merchants into CATE deciles and compare
    realized activation rates between treated and matched control within each decile.

    A monotonically decreasing gap (treated - control) from decile 1 to 10
    confirms the model correctly ranks merchant responsiveness.
    """
    df = df.copy()
    df["cate_decile"] = pd.qcut(df["cate"], q=10, labels=False) + 1

    result = (
        df.groupby(["cate_decile", "treatment"])["activated_30d"]
        .mean()
        .unstack("treatment")
        .rename(columns={0: "control_activation", 1: "treated_activation"})
    )
    result["lift_pp"] = (result["treated_activation"] - result["control_activation"]) * 100
    return result.reset_index().sort_values("cate_decile", ascending=False)


def plot_cate_distribution(
    df: pd.DataFrame,
    figsize: tuple = (8, 4),
    save_path: str | None = None,
) -> plt.Figure:
    """Histogram of CATE estimates (pp) with ATE reference line."""
    fig, ax = plt.subplots(figsize=figsize)
    cate_pp = df["cate"] * 100
    ate_pp = cate_pp.mean()

    ax.hist(cate_pp, bins=50, color="#4C8BB5", edgecolor="white", alpha=0.85)
    ax.axvline(ate_pp, color="#E07B54", linewidth=2, linestyle="--",
               label=f"ATE = {ate_pp:+.1f} pp")
    ax.axvline(0, color="black", linewidth=1, linestyle="-", alpha=0.4, label="Zero effect")

    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:+.1f} pp"))
    ax.set_xlabel("Estimated CATE (30-day activation rate lift, pp)", fontsize=11)
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
    """Horizontal bar chart of mean CATE (pp) by segment, sorted descending."""
    summary = segment_summary(df, group_col).sort_values("mean_cate_pp")

    fig, ax = plt.subplots(figsize=figsize)
    colors = ["#4C8BB5" if v >= 0 else "#E07B54" for v in summary["mean_cate_pp"]]
    bars = ax.barh(summary[group_col], summary["mean_cate_pp"], color=colors, edgecolor="white")

    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:+.1f} pp"))
    ax.set_xlabel("Mean Estimated CATE (activation rate lift, pp)", fontsize=11)
    ax.set_title(
        f"Average Activation Lift by {group_col.replace('_', ' ').title()}",
        fontsize=13, fontweight="bold"
    )
    ax.axvline(0, color="black", linewidth=0.8)
    ax.grid(axis="x", alpha=0.3)

    for bar, val in zip(bars, summary["mean_cate_pp"]):
        ax.text(
            bar.get_width() + 0.2,
            bar.get_y() + bar.get_height() / 2,
            f"{val:+.1f} pp",
            va="center", fontsize=9,
        )

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


def plot_decile_rank_validation(
    df: pd.DataFrame,
    figsize: tuple = (9, 4),
    save_path: str | None = None,
) -> plt.Figure:
    """
    Decile rank validation plot: realized activation rates for treated and
    control merchants within each predicted CATE decile (decile 10 = highest lift).

    Monotonically ordered bars confirm the model correctly ranks responsiveness.
    """
    decile_df = decile_rank_validation(df).sort_values("cate_decile")

    fig, ax = plt.subplots(figsize=figsize)
    x = decile_df["cate_decile"].values
    width = 0.35

    ax.bar(x - width / 2, decile_df["treated_activation"] * 100, width,
           label="Treated", color="#E07B54", alpha=0.85)
    ax.bar(x + width / 2, decile_df["control_activation"] * 100, width,
           label="Control", color="#4C8BB5", alpha=0.85)

    ax.set_xlabel("Predicted CATE Decile (1 = lowest, 10 = highest)", fontsize=11)
    ax.set_ylabel("30-Day Activation Rate (%)", fontsize=11)
    ax.set_title("Rank Validation: Activation Rate by Predicted Lift Decile",
                 fontsize=13, fontweight="bold")
    ax.set_xticks(x)
    ax.legend(fontsize=10)
    ax.grid(axis="y", alpha=0.3)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda y, _: f"{y:.0f}%"))
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
    ax.set_title(
        "Causal Forest — Feature Importances\n(Drivers of Treatment Effect Heterogeneity)",
        fontsize=12, fontweight="bold"
    )
    ax.grid(axis="x", alpha=0.3)
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig
