"""
generate_data.py
----------------
Generates synthetic merchant dataset simulating a discounted card-reader
equipment offer (treatment) and its effect on 90-day payment volume (outcome).

Selection bias is intentional: larger, more established merchants were more
likely to receive the offer — mimicking real-world non-random assignment.
"""

import numpy as np
import pandas as pd


def generate_merchant_data(n: int = 5000, seed: int = 42) -> pd.DataFrame:
    """
    Returns a DataFrame of synthetic merchant records.

    Features
    --------
    merchant_id         : unique identifier
    industry            : merchant industry segment
    months_in_business  : tenure at time of offer
    avg_monthly_txn_pre : average monthly transactions (pre-period)
    num_locations       : number of business locations
    has_pos_system      : whether merchant already had a POS system
    onboarding_score    : composite onboarding quality score (0–100)
    region              : geographic region
    treatment           : 1 = received discounted equipment offer, 0 = did not
    volume_90d          : 90-day payment volume post-offer (outcome)
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
    # Larger, more established merchants were preferentially targeted
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

    # --- Outcome model ---
    # True treatment effect is heterogeneous:
    #   Restaurants and merchants without existing POS benefit more
    industry_lift = np.where(
        np.isin(industry, ["Restaurant", "Retail"]), 1800, 900
    )
    pos_penalty = np.where(has_pos_system == 1, -400, 0)  # less incremental if already equipped
    true_cate = industry_lift + pos_penalty + rng.normal(0, 200, size=n)

    baseline_volume = (
        1500
        + 4.5 * avg_monthly_txn_pre
        + 300 * num_locations
        + 200 * has_pos_system
        + 8 * onboarding_score
        + rng.normal(0, 800, size=n)
    )

    volume_90d = np.maximum(0, baseline_volume + treatment * true_cate).round(2)

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
        "volume_90d": volume_90d,
    })

    return df


if __name__ == "__main__":
    df = generate_merchant_data()
    out_path = "data/merchants.csv"
    df.to_csv(out_path, index=False)
    print(f"Saved {len(df):,} merchant records to {out_path}")
    print(f"Treatment rate: {df['treatment'].mean():.1%}")
    print(f"Avg 90d volume (treated):   ${df.loc[df.treatment==1,'volume_90d'].mean():,.0f}")
    print(f"Avg 90d volume (untreated): ${df.loc[df.treatment==0,'volume_90d'].mean():,.0f}")
