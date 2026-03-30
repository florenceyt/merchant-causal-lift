"""
scripts/run_pipeline.py
-----------------------
End-to-end CLI entrypoint for the merchant causal lift pipeline.

Usage
-----
# Default run (5000 merchants, saves outputs to outputs/)
python scripts/run_pipeline.py

# Custom options
python scripts/run_pipeline.py --n 2000 --caliper 0.08 --n-topics 5 --out-dir results/

Options
-------
--n          Number of synthetic merchants to generate (default: 5000)
--seed       Random seed (default: 42)
--caliper    PSM caliper on logit scale (default: 0.05)
--n-trees    Number of trees in CausalForestDML (default: 500)
--out-dir    Directory for output figures and CSV (default: outputs/)
--no-plots   Skip saving figures (useful for CI)
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

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
    plot_cate_distribution,
    plot_cate_by_segment,
    plot_feature_importance,
)
from causal_lift.balance import plot_overlap


def parse_args():
    parser = argparse.ArgumentParser(
        description="Merchant Causal Lift Pipeline — PSM + CausalForestDML"
    )
    parser.add_argument("--n", type=int, default=5000, help="Number of merchants (default: 5000)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
    parser.add_argument("--caliper", type=float, default=0.05, help="PSM caliper (default: 0.05)")
    parser.add_argument("--n-trees", type=int, default=500, help="CausalForest n_estimators (default: 500)")
    parser.add_argument("--out-dir", type=str, default="outputs/", help="Output directory (default: outputs/)")
    parser.add_argument("--no-plots", action="store_true", help="Skip saving figures")
    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    print("=" * 60)
    print("MERCHANT CAUSAL LIFT PIPELINE")
    print("=" * 60)
    print(f"  n_merchants : {args.n:,}")
    print(f"  seed        : {args.seed}")
    print(f"  caliper     : {args.caliper}")
    print(f"  n_trees     : {args.n_trees}")
    print(f"  output dir  : {args.out_dir}")
    print()

    # ------------------------------------------------------------------
    # Step 1: Data generation
    # ------------------------------------------------------------------
    print("[1/6] Generating synthetic merchant data...")
    df = generate_merchant_data(n=args.n, seed=args.seed)
    print(f"      Treatment rate: {df['treatment'].mean():.1%}")
    naive_diff = df[df.treatment==1]["volume_90d"].mean() - df[df.treatment==0]["volume_90d"].mean()
    print(f"      Naive (confounded) diff: ${naive_diff:,.0f}")

    # ------------------------------------------------------------------
    # Step 2: Feature engineering + propensity
    # ------------------------------------------------------------------
    print("[2/6] Engineering features and estimating propensity scores...")
    df = engineer_features(df)
    df, scaler, lr = estimate_propensity(df)

    # ------------------------------------------------------------------
    # Step 3: PSM matching
    # ------------------------------------------------------------------
    print(f"[3/6] Running PSM (caliper={args.caliper})...")
    matched_df = psm_match(df, caliper=args.caliper)

    # ------------------------------------------------------------------
    # Step 4: Balance diagnostics
    # ------------------------------------------------------------------
    print("[4/6] Computing covariate balance...")
    bt = compute_balance_table(df, matched_df)
    max_smd = bt["smd_post"].abs().max()
    print(f"      Max post-match |SMD|: {max_smd:.3f} ({'PASS ✓' if max_smd < 0.10 else 'WARN — above 0.10'})")

    balance_path = os.path.join(args.out_dir, "love_plot.png") if not args.no_plots else None
    overlap_path = os.path.join(args.out_dir, "overlap.png") if not args.no_plots else None
    plot_love_plot(bt, save_path=balance_path)
    plot_overlap(df, matched_df, save_path=overlap_path)
    if not args.no_plots:
        print(f"      Saved: {balance_path}, {overlap_path}")

    # ------------------------------------------------------------------
    # Step 5: Causal Forest
    # ------------------------------------------------------------------
    print(f"[5/6] Fitting CausalForestDML (n_estimators={args.n_trees})...")
    model, cate = fit_causal_forest(matched_df, n_estimators=args.n_trees)
    importances = get_feature_importances(model)
    matched_df = add_cate_segments(matched_df, cate)

    if not args.no_plots:
        plot_cate_distribution(matched_df, save_path=os.path.join(args.out_dir, "cate_distribution.png"))
        plot_cate_by_segment(matched_df, "industry", save_path=os.path.join(args.out_dir, "cate_by_industry.png"))
        plot_feature_importance(importances, save_path=os.path.join(args.out_dir, "feature_importance.png"))
        print(f"      Saved plots to {args.out_dir}")

    # ------------------------------------------------------------------
    # Step 6: Segment summary
    # ------------------------------------------------------------------
    print("[6/6] Segment prioritization results:")
    seg = segment_summary(matched_df, "industry")
    print(seg.to_string(index=False))

    # Save results CSV
    out_csv = os.path.join(args.out_dir, "segment_summary.csv")
    seg.to_csv(out_csv, index=False)
    matched_df[["merchant_id", "industry", "region", "treatment", "volume_90d", "cate", "cate_quartile"]].to_csv(
        os.path.join(args.out_dir, "merchant_cate.csv"), index=False
    )

    print()
    print("=" * 60)
    print("PIPELINE COMPLETE")
    print(f"  ATE estimate : ${cate.mean():,.0f} per merchant")
    print(f"  Outputs saved to: {args.out_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()
