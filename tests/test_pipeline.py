"""
tests/test_pipeline.py
----------------------
Unit tests for the merchant causal lift pipeline.

Run with:
    pytest tests/ -v
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pandas as pd
import pytest

from data.generate_data import generate_merchant_data
from causal_lift.preprocess import engineer_features, estimate_propensity, psm_match, FEATURE_COLS
from causal_lift.balance import compute_balance_table, standardized_mean_diff
from causal_lift.segment import decile_rank_validation


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def raw_dfs():
    """Small synthetic dataset for fast tests."""
    return generate_merchant_data(n=500, seed=0)


@pytest.fixture(scope="module")
def modelling_df(raw_dfs):
    return raw_dfs[0]


@pytest.fixture(scope="module")
def holdout_df(raw_dfs):
    return raw_dfs[1]


@pytest.fixture(scope="module")
def featured_df(modelling_df):
    return engineer_features(modelling_df)


@pytest.fixture(scope="module")
def propensity_df(featured_df):
    df, scaler, lr = estimate_propensity(featured_df)
    return df


@pytest.fixture(scope="module")
def matched_df(propensity_df):
    return psm_match(propensity_df, caliper=0.10)


# ---------------------------------------------------------------------------
# 1. Data generation
# ---------------------------------------------------------------------------

class TestDataGeneration:

    def test_modelling_shape(self, modelling_df):
        assert len(modelling_df) == pytest.approx(450, abs=50)  # ~90% of 500

    def test_holdout_shape(self, holdout_df):
        assert len(holdout_df) == pytest.approx(50, abs=20)  # ~10% of 500

    def test_modelling_required_columns(self, modelling_df):
        expected = {
            "merchant_id", "industry", "region", "months_in_business",
            "avg_monthly_txn_pre", "num_locations", "has_pos_system",
            "onboarding_score", "treatment", "activated_30d",
        }
        assert expected.issubset(set(modelling_df.columns))

    def test_modelling_has_no_volume_90d(self, modelling_df):
        """volume_90d must be absent from modelling cohort — holdout only."""
        assert "volume_90d" not in modelling_df.columns

    def test_holdout_has_volume_90d(self, holdout_df):
        assert "volume_90d" in holdout_df.columns

    def test_treatment_is_binary(self, modelling_df):
        assert set(modelling_df["treatment"].unique()).issubset({0, 1})

    def test_activated_30d_is_binary(self, modelling_df):
        assert set(modelling_df["activated_30d"].unique()).issubset({0, 1})

    def test_volume_non_negative(self, holdout_df):
        assert (holdout_df["volume_90d"] >= 0).all()

    def test_onboarding_score_bounds(self, modelling_df):
        assert modelling_df["onboarding_score"].between(0, 100).all()

    def test_treatment_rate_reasonable(self, modelling_df):
        rate = modelling_df["treatment"].mean()
        assert 0.2 <= rate <= 0.8, f"Treatment rate {rate:.2%} outside expected range"

    def test_no_overlap_between_cohorts(self, modelling_df, holdout_df):
        """Merchant IDs must not appear in both cohorts."""
        shared = set(modelling_df["merchant_id"]) & set(holdout_df["merchant_id"])
        assert len(shared) == 0, f"{len(shared)} merchants appear in both cohorts"

    def test_reproducibility(self):
        df1_mod, df1_hold = generate_merchant_data(n=100, seed=99)
        df2_mod, df2_hold = generate_merchant_data(n=100, seed=99)
        pd.testing.assert_frame_equal(df1_mod, df2_mod)
        pd.testing.assert_frame_equal(df1_hold, df2_hold)


# ---------------------------------------------------------------------------
# 2. Feature engineering
# ---------------------------------------------------------------------------

class TestFeatureEngineering:

    def test_adds_encoded_columns(self, featured_df):
        assert "industry_encoded" in featured_df.columns
        assert "region_encoded" in featured_df.columns

    def test_all_feature_cols_present(self, featured_df):
        for col in FEATURE_COLS:
            assert col in featured_df.columns, f"Missing feature column: {col}"

    def test_does_not_modify_original(self, modelling_df):
        original_cols = set(modelling_df.columns)
        _ = engineer_features(modelling_df)
        assert set(modelling_df.columns) == original_cols

    def test_encoded_values_are_integers(self, featured_df):
        assert featured_df["industry_encoded"].dtype in [np.int64, np.int32, int]
        assert featured_df["region_encoded"].dtype in [np.int64, np.int32, int]


# ---------------------------------------------------------------------------
# 3. Propensity score estimation
# ---------------------------------------------------------------------------

class TestPropensityEstimation:

    def test_propensity_bounds(self, propensity_df):
        assert propensity_df["propensity_score"].between(0, 1).all()

    def test_adds_propensity_columns(self, propensity_df):
        assert "propensity_score" in propensity_df.columns
        assert "logit_score" in propensity_df.columns

    def test_no_nulls(self, propensity_df):
        assert propensity_df["propensity_score"].isna().sum() == 0
        assert propensity_df["logit_score"].isna().sum() == 0

    def test_propensity_varies(self, propensity_df):
        assert propensity_df["propensity_score"].std() > 0.01


# ---------------------------------------------------------------------------
# 4. PSM matching
# ---------------------------------------------------------------------------

class TestPSMMatching:

    def test_balanced_treated_control_counts(self, matched_df):
        n_treated = (matched_df["treatment"] == 1).sum()
        n_control = (matched_df["treatment"] == 0).sum()
        assert n_treated == n_control

    def test_matched_size_smaller_than_original(self, matched_df, propensity_df):
        assert len(matched_df) < len(propensity_df)

    def test_no_duplicate_control_merchants(self, matched_df):
        control_ids = matched_df.loc[matched_df["treatment"] == 0, "merchant_id"]
        assert control_ids.nunique() == len(control_ids)

    def test_caliper_respected(self, matched_df):
        treated = matched_df[matched_df["treatment"] == 1]["logit_score"].values
        control = matched_df[matched_df["treatment"] == 0]["logit_score"].values
        diffs = np.abs(treated - control)
        assert (diffs <= 0.10 + 1e-6).all()


# ---------------------------------------------------------------------------
# 5. Balance table
# ---------------------------------------------------------------------------

class TestBalanceTable:

    def test_balance_table_shape(self, propensity_df, matched_df):
        bt = compute_balance_table(propensity_df, matched_df)
        assert bt.shape == (5, 3)

    def test_balance_table_columns(self, propensity_df, matched_df):
        bt = compute_balance_table(propensity_df, matched_df)
        assert set(bt.columns) == {"covariate", "smd_pre", "smd_post"}

    def test_post_match_smd_below_threshold(self, propensity_df, matched_df):
        """Threshold relaxed to 0.15 for small test sample (n=500)."""
        bt = compute_balance_table(propensity_df, matched_df)
        violations = bt[bt["smd_post"].abs() > 0.15]
        assert len(violations) == 0, f"Balance threshold violated:\n{violations}"

    def test_pre_match_smd_shows_imbalance(self, propensity_df, matched_df):
        bt = compute_balance_table(propensity_df, matched_df)
        avg_pre_smd = bt["smd_pre"].abs().mean()
        assert avg_pre_smd > 0.05


# ---------------------------------------------------------------------------
# 6. Outcome integrity
# ---------------------------------------------------------------------------

class TestOutcomeIntegrity:

    def test_activated_30d_has_both_classes(self, modelling_df):
        """Activation outcome must not be all-zero or all-one."""
        vals = modelling_df["activated_30d"].unique()
        assert 0 in vals and 1 in vals

    def test_treated_activation_rate_higher(self, modelling_df):
        """Treated merchants should activate at a higher raw rate (true by design)."""
        treated_rate = modelling_df[modelling_df.treatment == 1]["activated_30d"].mean()
        control_rate = modelling_df[modelling_df.treatment == 0]["activated_30d"].mean()
        assert treated_rate > control_rate, (
            f"Expected treated rate ({treated_rate:.2%}) > control rate ({control_rate:.2%})"
        )

    def test_holdout_volume_positive(self, holdout_df):
        assert holdout_df["volume_90d"].mean() > 0
