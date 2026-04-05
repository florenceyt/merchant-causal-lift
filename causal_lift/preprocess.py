"""
preprocess.py
-------------
Feature engineering and propensity-score matching (PSM) for the merchant
discounted-equipment causal lift analysis.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler


FEATURE_COLS = [
    "months_in_business",
    "avg_monthly_txn_pre",
    "num_locations",
    "has_pos_system",
    "onboarding_score",
    "industry_encoded",
    "region_encoded",
]


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Encode categoricals and return a feature-ready DataFrame."""
    df = df.copy()
    df["industry_encoded"] = pd.factorize(df["industry"])[0]
    df["region_encoded"] = pd.factorize(df["region"])[0]
    return df


def estimate_propensity(df: pd.DataFrame) -> pd.DataFrame:
    """
    Fit a logistic regression propensity model.
    Returns df with added columns: propensity_score, logit_score.
    """
    X = df[FEATURE_COLS].values
    y = df["treatment"].values

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    lr = LogisticRegression(max_iter=500, random_state=42)
    lr.fit(X_scaled, y)

    df = df.copy()
    df["propensity_score"] = lr.predict_proba(X_scaled)[:, 1]
    df["logit_score"] = np.log(
        df["propensity_score"] / (1 - df["propensity_score"].clip(1e-6, 1 - 1e-6))
    )
    return df, scaler, lr


def psm_match(
    df: pd.DataFrame,
    caliper: float = 0.05,
    random_state: int = 42,
) -> pd.DataFrame:
    """
    1:1 nearest-neighbor PSM on the logit of the propensity score.

    Parameters
    ----------
    caliper : max allowable logit distance for a valid match (0.05 ≈ 0.2 SD rule)

    Returns
    -------
    Matched DataFrame with balanced treated / control groups.
    """
    treated = df[df["treatment"] == 1].copy()
    control = df[df["treatment"] == 0].copy()

    nn = NearestNeighbors(n_neighbors=1, algorithm="ball_tree")
    nn.fit(control[["logit_score"]].values)

    distances, indices = nn.kneighbors(treated[["logit_score"]].values)

    matched_rows = []
    used_control_idx = set()

    for i, (dist, ctrl_idx) in enumerate(zip(distances[:, 0], indices[:, 0])):
        if dist <= caliper and ctrl_idx not in used_control_idx:
            matched_rows.append(treated.iloc[i])
            matched_rows.append(control.iloc[ctrl_idx])
            used_control_idx.add(ctrl_idx)

    matched_df = pd.DataFrame(matched_rows).reset_index(drop=True)
    n_pairs = len(matched_df) // 2
    print(f"PSM: {n_pairs:,} matched pairs retained (caliper={caliper})")
    return matched_df
