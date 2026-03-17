"""
model.py
--------
Causal Forest via EconML's CausalForestDML for heterogeneous treatment effect
(CATE) estimation on matched merchant data.

Two-stage approach:
  Stage 1 — Nuisance models (Y ~ X, T ~ X) residualize outcome and treatment
  Stage 2 — Causal Forest learns CATE from residuals
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from econml.dml import CausalForestDML
from sklearn.ensemble import GradientBoostingRegressor

from causal_lift.preprocess import FEATURE_COLS


def fit_causal_forest(
    df: pd.DataFrame,
    n_estimators: int = 500,
    random_state: int = 42,
) -> tuple[CausalForestDML, np.ndarray]:
    """
    Fit a CausalForestDML model on the matched DataFrame.

    Returns
    -------
    model   : fitted CausalForestDML instance
    cate    : array of individual CATE estimates (same length as df)
    """
    X = df[FEATURE_COLS].values
    T = df["treatment"].values.astype(float)
    Y = df["volume_90d"].values.astype(float)

    model = CausalForestDML(
        model_y=GradientBoostingRegressor(n_estimators=200, max_depth=4, random_state=random_state),
        model_t=GradientBoostingRegressor(n_estimators=200, max_depth=4, random_state=random_state),
        n_estimators=n_estimators,
        min_samples_leaf=10,
        max_depth=None,
        random_state=random_state,
        verbose=0,
    )

    model.fit(Y, T, X=X)
    cate = model.effect(X).flatten()

    ate = cate.mean()
    ci_lo, ci_hi = np.percentile(cate, [2.5, 97.5])
    print(f"ATE estimate:  ${ate:,.0f}")
    print(f"95% CATE interval: [${ci_lo:,.0f}, ${ci_hi:,.0f}]")

    return model, cate


def get_feature_importances(model: CausalForestDML) -> pd.Series:
    """Extract feature importances from the underlying forest."""
    importances = model.feature_importances_
    return pd.Series(importances, index=FEATURE_COLS).sort_values(ascending=False)
