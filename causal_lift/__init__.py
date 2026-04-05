"""
causal_lift
-----------
Causal inference pipeline for estimating heterogeneous treatment effects
on merchant 30-day activation rate following an outbound calling program.

Primary outcome : activated_30d (binary — 30-day activation)
Validation      : 90-day volume lift measured on a separate holdout cohort

Modules
-------
preprocess  : feature engineering + propensity score estimation + PSM matching
balance     : covariate balance diagnostics and love plot
model       : CausalForestDML fitting and CATE extraction (binary outcome)
segment     : segment-level CATE analysis, rank validation, and visualization
"""

from causal_lift.preprocess import engineer_features, estimate_propensity, psm_match
from causal_lift.balance import compute_balance_table, plot_love_plot, plot_overlap
from causal_lift.model import fit_causal_forest, get_feature_importances
from causal_lift.segment import (
    add_cate_segments,
    segment_summary,
    decile_rank_validation,
    plot_cate_distribution,
    plot_cate_by_segment,
    plot_decile_rank_validation,
    plot_feature_importance,
)

__all__ = [
    "engineer_features",
    "estimate_propensity",
    "psm_match",
    "compute_balance_table",
    "plot_love_plot",
    "plot_overlap",
    "fit_causal_forest",
    "get_feature_importances",
    "add_cate_segments",
    "segment_summary",
    "decile_rank_validation",
    "plot_cate_distribution",
    "plot_cate_by_segment",
    "plot_decile_rank_validation",
    "plot_feature_importance",
]
