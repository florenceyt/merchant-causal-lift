"""
model.py
--------
Causal Forest via EconML's CausalForestDML for heterogeneous treatment effect
(CATE) estimation on matched merchant data.

Primary outcome: activated_30d (binary — 30-day activation rate).
Using a binary outcome changes the nuisance models from regressors to classifiers.

Two-stage DML approach:
  Stage 1 — Cross-fitted nuisance models residualize outcome and treatment
             on covariates, removing confounding.
  Stage 2 — Causal Forest learns the mapping from covariates to CATE
             from the residuals.

This yields a merchant-level estimate of the treatment effect on 30-day
activation probability. 90-day volume lift is validated separately on a
held-out cohort (see scripts/run_pipeline.py).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from econml.dml import CausalForestDML
from sklearn.ensemble import GradientBoostingClassifier

from causal_lift.preprocess import FEATURE_COLS


def fit_causal_forest(
    df: pd.DataFrame,
    n_estimators: int = 1000,
    random_state: int = 42,
) -> tuple[CausalForestDML, np.ndarray]:
    """
    Fit a CausalForestDML model on the matched DataFrame.

    Both nuisance models (outcome and treatment) use GradientBoostingClassifier
    because activated_30d and treatment are binary variables. Using classifiers
    ensures the residuals fed into Stage 2 are correctly scaled probability
    residuals rather than regression residuals.

    Parameters
    ----------
    df           : matched DataFrame from PSM (must contain activated_30d, treatment, FEATURE_COLS)
    n_estimators : number of trees in the Causal Forest (default 1000 for stable variance estimates)
    random_state : reproducibility seed

    Returns
    -------
    model : fitted CausalForestDML instance
    cate  : array of individual CATE estimates (probability scale), length == len(df)
    """
    X = df[FEATURE_COLS].values
    T = df["treatment"].values.astype(float)
    Y = df["activated_30d"].values.astype(float)

    model = CausalForestDML(
        model_y=GradientBoostingClassifier(n_estimators=200, max_depth=4, random_state=random_state),
        model_t=GradientBoostingClassifier(n_estimators=200, max_depth=4, random_state=random_state),
        n_estimators=n_estimators,
        min_samples_leaf=10,
        max_depth=None,
        inference=True,
        random_state=random_state,
        verbose=0,
    )

    model.fit(Y, T, X=X)
    cate = model.effect(X).flatten()

    ate = cate.mean()
    ci_lo, ci_hi = np.percentile(cate, [2.5, 97.5])
    print(f"ATE estimate (activation rate lift): {ate:+.3f} ({ate*100:+.1f} pp)")
    print(f"95% CATE interval: [{ci_lo:+.3f}, {ci_hi:+.3f}]")

    return model, cate


def get_feature_importances(model: CausalForestDML) -> pd.Series:
    """Extract feature importances from the underlying forest."""
    importances = model.feature_importances_
    return pd.Series(importances, index=FEATURE_COLS).sort_values(ascending=False)
