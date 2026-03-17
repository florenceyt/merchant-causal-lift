"""
causal_lift
-----------
Causal inference pipeline for estimating heterogeneous treatment effects
on merchant 90-day payment volume following a discounted equipment offer.

Modules
-------
preprocess  : feature engineering + propensity score estimation + PSM matching
balance     : covariate balance diagnostics and love plot
model       : CausalForestDML fitting and CATE extraction
segment     : segment-level CATE analysis and visualization
"""

from causal_lift.preprocess import engineer_features, estimate_propensity, psm_match
from causal_lift.balance import compute_balance_table, plot_love_plot
from causal_lift.model import fit_causal_forest, get_feature_importances
from causal_lift.segment import (
    add_cate_segments,
    segment_summary,
    plot_cate_distribution,
    plot_cate_by_segment,
    plot_feature_importance,
)

__all__ = [
    "engineer_features",
    "estimate_propensity",
    "psm_match",
    "compute_balance_table",
    "plot_love_plot",
    "fit_causal_forest",
    "get_feature_importances",
    "add_cate_segments",
    "segment_summary",
    "plot_cate_distribution",
    "plot_cate_by_segment",
    "plot_feature_importance",
]
