"""
scripts/run_pipeline.py
-----------------------
End-to-end CLI entrypoint for the merchant causal lift pipeline.

Two-part design:
  Part 1 — PSM + CausalForestDML on the modelling cohort (90% of data)
            Primary outcome: activated_30d (30-day activation rate, binary)
            Produces merchant-level CATEs and segment prioritization.

  Part 2 — Holdout validation on the reserved 10% cohort
            Compares 90-day payment volume between high-CATE merchants
            who received the call vs. matched controls who did not.
            Isolates the incremental 90-day volume lift attributable to
            targeting the high-uplift segments identified in Part 1.

Usage
-----
# Default run (5000 merchants, saves outputs to outputs/)
python scripts/run_pipeline.py

# Custom options
python scripts/run_pipeline.py --n 2000 --caliper 0.08 --n-trees 1000 --out-dir results/

Options
-------
--n          Number of synthetic merchants to generate (default: 5000)
--seed       Random seed (default: 42)
--caliper    PSM caliper on logit scale (default: 0.05)
--n-trees    Number of trees in CausalForestDML (default: 1000)
--out-dir    Directory for output figures and CSV (default: outputs/)
--no-plots   Skip saving figures (useful for CI)
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd

from data.generate_data import generate_merchant_data
from causal_lift import (
    engineer_features,
    estimate_propensity,
    psm_match,
    compute_balance_table,
    plot_love_plot,
    fit_causal_forest,
    get_feature_importances,
    add_cate_segments,
    segment_summary,
    decile_rank_validation,
    plot_cate_distribution,
    plot_cate_by_segment,
    plot_decile_rank_validation,
    plot_feature_importance,
)
from causal_lift.balance import plot_overlap


def parse_args():
    parser = argparse.ArgumentParser(
        description="Merchant Causal Lift Pipeline — PSM + CausalForestDML (30-day activation)"
    )
    parser.add_argument("--n", type=int, default=5000, help="Number of merchants (default: 5000)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
    parser.add_argument("--caliper", type=float, default=0.05, help="PSM caliper (default: 0.05)")
    parser.add_argument("--n-trees", type=int, default=1000, help="CausalForest n_estimators (default: 1000)")
    parser.add_argument("--out-dir", type=str, default="outputs/", help="Output directory (default: outputs/)")
    parser.add_argument("--no-plots", action="store_true", help="Skip saving figures")
    return parser.parse_args()


def holdout_volume_lift(holdout_df: pd.DataFrame, high_cate_merchants: set) -> dict:
    """
    Estimate incremental 90-day volume lift using the holdout cohort.

    Compares 90-day volume between:
      - Treated merchants whose merchant_id was flagged as high-CATE (top 2 quartiles)
      - Matched control merchants in the holdout who were NOT called

    Returns a dict with treated_vol, control_vol, lift_pct, and n counts.
    """
    holdout_treated = holdout_df[
        (holdout_df["treatment"] == 1) &
        (holdout_df["merchant_id"].isin(high_cate_merchants))
    ]
    holdout_control = holdout_df[holdout_df["treatment"] == 0]

    treated_vol = holdout_treated["volume_90d"].mean()
    control_vol = holdout_control["volume_90d"].mean()
    lift_pct = (treated_vol - control_vol) / control_vol * 100

    return {
        "n_treated": len(holdout_treated),
        "n_control": len(holdout_control),
        "treated_avg_volume": treated_vol,
        "control_avg_volume": control_vol,
        "lift_pct": lift_pct,
    }


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    print("=" * 60)
    print("MERCHANT CAUSAL LIFT PIPELINE")
    print("Primary outcome : 30-day activation rate (binary)")
    print("Validation      : 90-day volume lift on holdout cohort")
    print("=" * 60)
    print(f"  n_merchants : {args.n:,}")
    print(f"  seed        : {args.seed}")
    print(f"  caliper     : {args.caliper}")
    print(f"  n_trees     : {args.n_trees}")
    print(f"  output dir  : {args.out_dir}")
    print()

    # ------------------------------------------------------------------
    # Step 1: Data generation — modelling cohort + holdout
    # ------------------------------------------------------------------
    print("[1/7] Generating synthetic merchant data...")
    modelling_df, holdout_df = generate_merchant_data(n=args.n, seed=args.seed)
    print(f"      Modelling cohort: {len(modelling_df):,} merchants | treatment rate: {modelling_df['treatment'].mean():.1%}")
    print(f"      Holdout cohort:   {len(holdout_df):,} merchants  | treatment rate: {holdout_df['treatment'].mean():.1%}")
    naive_diff = (
        modelling_df[modelling_df.treatment == 1]["activated_30d"].mean()
        - modelling_df[modelling_df.treatment == 0]["activated_30d"].mean()
    )
    print(f"      Naive (confounded) activation gap: {naive_diff:+.3f} ({naive_diff*100:+.1f} pp)")

    # ------------------------------------------------------------------
    # Step 2: Feature engineering + propensity
    # ------------------------------------------------------------------
    print("[2/7] Engineering features and estimating propensity scores...")
    df = engineer_features(modelling_df)
    df, scaler, lr = estimate_propensity(df)

    # ------------------------------------------------------------------
    # Step 3: PSM matching
    # ------------------------------------------------------------------
    print(f"[3/7] Running PSM (caliper={args.caliper})...")
    matched_df = psm_match(df, caliper=args.caliper)

    # ------------------------------------------------------------------
    # Step 4: Balance diagnostics
    # ------------------------------------------------------------------
    print("[4/7] Computing covariate balance...")
    bt = compute_balance_table(df, matched_df)
    max_smd = bt["smd_post"].abs().max()
    print(f"      Max post-match |SMD|: {max_smd:.3f} ({'PASS' if max_smd < 0.10 else 'WARN — above 0.10'})")

    balance_path = os.path.join(args.out_dir, "love_plot.png") if not args.no_plots else None
    overlap_path = os.path.join(args.out_dir, "overlap.png") if not args.no_plots else None
    plot_love_plot(bt, save_path=balance_path)
    plot_overlap(df, matched_df, save_path=overlap_path)
    if not args.no_plots:
        print(f"      Saved: {balance_path}, {overlap_path}")

    # ------------------------------------------------------------------
    # Step 5: Causal Forest on 30-day activation
    # ------------------------------------------------------------------
    print(f"[5/7] Fitting CausalForestDML on activated_30d (n_estimators={args.n_trees})...")
    model, cate = fit_causal_forest(matched_df, n_estimators=args.n_trees)
    importances = get_feature_importances(model)
    matched_df = add_cate_segments(matched_df, cate)

    if not args.no_plots:
        plot_cate_distribution(matched_df, save_path=os.path.join(args.out_dir, "cate_distribution.png"))
        plot_cate_by_segment(matched_df, "industry", save_path=os.path.join(args.out_dir, "cate_by_industry.png"))
        plot_decile_rank_validation(matched_df, save_path=os.path.join(args.out_dir, "decile_rank_validation.png"))
        plot_feature_importance(importances, save_path=os.path.join(args.out_dir, "feature_importance.png"))
        print(f"      Saved plots to {args.out_dir}")

    # ------------------------------------------------------------------
    # Step 6: Segment prioritization
    # ------------------------------------------------------------------
    print("[6/7] Segment prioritization results (30-day activation lift):")
    seg = segment_summary(matched_df, "industry")
    print(seg.to_string(index=False))

    # Identify high-CATE merchants (top 2 quartiles) for holdout validation
    high_cate_ids = set(
        matched_df.loc[matched_df["cate_quartile"].isin(["Q3", "Q4"]), "merchant_id"]
    )

    # ------------------------------------------------------------------
    # Step 7: Holdout 90-day volume lift validation
    # ------------------------------------------------------------------
    print("[7/7] Holdout validation — 90-day volume lift...")
    lift_results = holdout_volume_lift(holdout_df, high_cate_ids)
    print(f"      Holdout treated (high-CATE): {lift_results['n_treated']:,} merchants")
    print(f"      Holdout control:              {lift_results['n_control']:,} merchants")
    print(f"      Avg 90d volume (treated):  ${lift_results['treated_avg_volume']:,.0f}")
    print(f"      Avg 90d volume (control):  ${lift_results['control_avg_volume']:,.0f}")
    print(f"      Incremental 90d volume lift: {lift_results['lift_pct']:+.1f}%")

    # Save outputs
    out_csv = os.path.join(args.out_dir, "segment_summary.csv")
    seg.to_csv(out_csv, index=False)
    matched_df[["merchant_id", "industry", "region", "treatment",
                "activated_30d", "cate", "cate_quartile"]].to_csv(
        os.path.join(args.out_dir, "merchant_cate.csv"), index=False
    )
    pd.DataFrame([lift_results]).to_csv(
        os.path.join(args.out_dir, "holdout_lift.csv"), index=False
    )

    print()
    print("=" * 60)
    print("PIPELINE COMPLETE")
    print(f"  ATE (activation rate lift) : {cate.mean()*100:+.1f} pp")
    print(f"  90-day volume lift (holdout): {lift_results['lift_pct']:+.1f}%")
    print(f"  Outputs saved to: {args.out_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()
