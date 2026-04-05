"""
generate_data.py
----------------
Generates synthetic merchant dataset simulating an outbound calling program
(treatment) aimed at accelerating merchant activation.

Primary outcome : activated_30d  — binary, did merchant process a payment
                                    within 30 days of joining? (used for CATE)
Secondary outcome: volume_90d    — 90-day payment volume, held out for
                                    incremental lift validation only.

Selection bias is intentional: larger, more established merchants were more
likely to receive the outbound call — mimicking real-world non-random assignment.

The dataset is split 90/10 into a modelling cohort and a holdout cohort.
The holdout is used exclusively for the 90-day volume lift validation.
"""

import numpy as np
import pandas as pd


def generate_merchant_data(n: int = 5000, seed: int = 42) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Returns (modelling_df, holdout_df).

    modelling_df (90% of n)
    -----------------------
    merchant_id         : unique identifier
    industry            : merchant industry segment
    months_in_business  : tenure at time of offer
    avg_monthly_txn_pre : average monthly transactions (pre-period)
    num_locations       : number of business locations
    has_pos_system      : whether merchant already had a POS system
    onboarding_score    : composite onboarding quality score (0-100)
    region              : geographic region
    treatment           : 1 = received outbound call, 0 = did not
    activated_30d       : 1 = processed first payment within 30 days

    holdout_df (10% of n)
    ---------------------
    Same columns as modelling_df plus volume_90d.
    Used only for 90-day lift validation; never used to fit PSM or the Causal Forest.
    """
    rng = np.random.default_rng(seed)

    industries = ["Restaurant", "Retail", "Services", "Healthcare", "Entertainment"]
    regions = ["Northeast", "Southeast", "Midwest", "West", "Southwest"]

    industry = rng.choice(industries, size=n, p=[0.30, 0.25, 0.20, 0.15, 0.10])
    region = rng.choice(regions, size=n)
    months_in_business = rng.integers(1, 120, size=n)
    avg_monthly_txn_pre = rng.gamma(shape=3, scale=80, size=n).round(0)
    num_locations = rng.choice([1, 2, 3, 4, 5], size=n, p=[0.55, 0.25, 0.12, 0.05, 0.03])
    has_pos_system = rng.binomial(1, p=0.45, size=n)
    onboarding_score = np.clip(rng.normal(65, 15, size=n), 0, 100).round(1)

    # --- Selection mechanism (non-random treatment assignment) ---
    logit = (
        -3.0
        + 0.015 * months_in_business
        + 0.005 * avg_monthly_txn_pre
        + 0.20 * num_locations
        + 0.25 * has_pos_system
        + 0.010 * onboarding_score
    )
    prop_score = 1 / (1 + np.exp(-logit))
    treatment = rng.binomial(1, p=prop_score)

    # --- Primary outcome: 30-day activation (binary) ---
    # Heterogeneous treatment effect: Restaurant/Retail and no-POS merchants respond most
    industry_effect = np.where(np.isin(industry, ["Restaurant", "Retail"]), 0.18, 0.09)
    pos_penalty = np.where(has_pos_system == 1, -0.06, 0.0)
    baseline_activation_logit = (
        -0.5
        + 0.008 * onboarding_score
        + 0.003 * avg_monthly_txn_pre
        + 0.10 * has_pos_system
    )
    activation_prob = 1 / (1 + np.exp(
        -(baseline_activation_logit + treatment * (industry_effect + pos_penalty))
    ))
    activated_30d = rng.binomial(1, p=activation_prob)

    # --- Secondary outcome: 90-day volume (holdout validation only) ---
    industry_lift = np.where(np.isin(industry, ["Restaurant", "Retail"]), 1800, 900)
    vol_pos_penalty = np.where(has_pos_system == 1, -400, 0)
    true_vol_cate = industry_lift + vol_pos_penalty + rng.normal(0, 200, size=n)
    baseline_volume = (
        1500
        + 4.5 * avg_monthly_txn_pre
        + 300 * num_locations
        + 200 * has_pos_system
        + 8 * onboarding_score
        + rng.normal(0, 800, size=n)
    )
    volume_90d = np.maximum(0, baseline_volume + treatment * true_vol_cate).round(2)

    df = pd.DataFrame({
        "merchant_id": [f"M{str(i).zfill(5)}" for i in range(1, n + 1)],
        "industry": industry,
        "region": region,
        "months_in_business": months_in_business,
        "avg_monthly_txn_pre": avg_monthly_txn_pre,
        "num_locations": num_locations,
        "has_pos_system": has_pos_system,
        "onboarding_score": onboarding_score,
        "treatment": treatment,
        "activated_30d": activated_30d,
        "volume_90d": volume_90d,
    })

    # 90/10 stratified split — preserve treatment balance in both sets
    holdout_idx = (
        df[df.treatment == 1].sample(frac=0.10, random_state=seed).index.tolist()
        + df[df.treatment == 0].sample(frac=0.10, random_state=seed).index.tolist()
    )
    holdout_df = df.loc[holdout_idx].reset_index(drop=True)
    modelling_df = df.drop(index=holdout_idx).reset_index(drop=True)

    # Drop volume_90d from modelling cohort — holdout-only outcome
    modelling_df = modelling_df.drop(columns=["volume_90d"])

    return modelling_df, holdout_df


if __name__ == "__main__":
    modelling_df, holdout_df = generate_merchant_data()
    print(f"Modelling cohort : {len(modelling_df):,} merchants | treatment rate: {modelling_df['treatment'].mean():.1%}")
    print(f"Holdout cohort   : {len(holdout_df):,} merchants  | treatment rate: {holdout_df['treatment'].mean():.1%}")
    print(f"Modelling activation rate (treated):   {modelling_df.loc[modelling_df.treatment==1,'activated_30d'].mean():.1%}")
    print(f"Modelling activation rate (untreated): {modelling_df.loc[modelling_df.treatment==0,'activated_30d'].mean():.1%}")
