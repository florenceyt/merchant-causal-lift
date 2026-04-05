# merchant-causal-lift

**Heterogeneous treatment effect estimation for a merchant equipment offer using PSM + CausalForestDML.**

A payments company offered discounted card readers to a subset of new merchants. This pipeline estimates the incremental impact on 90-day payment volume — accounting for selection bias — and identifies which merchant segments benefit most.

[![Tests](https://img.shields.io/badge/tests-23%20passed-brightgreen)](#testing)
[![Python](https://img.shields.io/badge/python-3.10%2B-blue)](#installation)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## Why This Matters for Business Decisions

Merchant outreach capacity is finite. Not every merchant can receive a high-touch campaign, a discounted offer, or dedicated onboarding support. This creates a targeting problem:

1. **Incremental lift, not raw outcome.** A merchant who would have activated anyway doesn't benefit from the offer — they just receive it. What matters is the *counterfactual*: how much additional 90-day volume is attributable to the offer, above what would have happened without it?

2. **Selection bias inflates naive estimates.** Larger, more established merchants were preferentially targeted for the equipment offer — so raw comparisons overstate impact. PSM constructs a comparable control group to isolate the true effect.

3. **Segment-level targeting maximizes ROI.** The Causal Forest reveals that Restaurant and Retail merchants respond ~75% more strongly than Entertainment merchants. Concentrating future offers on high-CATE segments captures most of the total lift at a fraction of the campaign cost.

The output of this pipeline is a merchant-level CATE estimate and segment prioritization table — directly actionable for campaign targeting decisions.

---

## Results

| Metric | Value |
|---|---|
| Matched pairs (post-PSM) | ~1,174 |
| ATE estimate | ~$1,200 per merchant |
| Highest-lift segment | Restaurant (~$1,480) |
| Lowest-lift segment | Entertainment (~$850) |
| Max post-match SMD | < 0.05 (all covariates) |

---

## Project Structure

```
merchant-causal-lift/
├── causal_lift/
│   ├── __init__.py
│   ├── preprocess.py     # Feature engineering, propensity estimation, PSM
│   ├── balance.py        # SMD love plot + propensity overlap (common support)
│   ├── model.py          # CausalForestDML fitting + CATE extraction
│   └── segment.py        # Segment-level analysis + visualizations
├── data/
│   └── generate_data.py  # Synthetic data generator (5,000 merchants)
├── notebooks/
│   └── merchant_lift_demo.ipynb
├── scripts/
│   └── run_pipeline.py   # CLI end-to-end entrypoint
├── tests/
│   └── test_pipeline.py  # 23 pytest unit tests
├── requirements.txt
├── requirements-lock.txt
└── README.md
```

---

## Quickstart

### Installation

```bash
git clone https://github.com/florenceyt/merchant-causal-lift.git
cd merchant-causal-lift
pip install -r requirements.txt
```

### Run the full pipeline (CLI)

```bash
python scripts/run_pipeline.py
```

Custom options:

```bash
python scripts/run_pipeline.py --n 2000 --caliper 0.08 --n-trees 300 --out-dir results/
```

| Flag | Default | Description |
|---|---|---|
| `--n` | 5000 | Number of synthetic merchants |
| `--seed` | 42 | Random seed |
| `--caliper` | 0.05 | PSM caliper (logit scale) |
| `--n-trees` | 500 | CausalForest n_estimators |
| `--out-dir` | outputs/ | Directory for figures and CSVs |
| `--no-plots` | False | Skip saving figures |

### Notebook

```bash
jupyter notebook notebooks/merchant_lift_demo.ipynb
```

### Library usage

```python
from data.generate_data import generate_merchant_data
from causal_lift import (
    engineer_features, estimate_propensity, psm_match,
    compute_balance_table, plot_love_plot, plot_overlap,
    fit_causal_forest, add_cate_segments, segment_summary,
)

df = generate_merchant_data(n=5000)
df = engineer_features(df)
df, _, _ = estimate_propensity(df)
matched_df = psm_match(df, caliper=0.05)

plot_love_plot(compute_balance_table(df, matched_df))
plot_overlap(df, matched_df)   # common support check

model, cate = fit_causal_forest(matched_df)
matched_df = add_cate_segments(matched_df, cate)
print(segment_summary(matched_df, "industry"))
```

---

## Testing

```bash
pytest tests/ -v
```

23 tests across 5 classes: data generation shape and reproducibility, feature engineering correctness, propensity score bounds, PSM balance and no-duplicate-control guarantees, balance table shape and threshold checks.

---

## Methodology

### Stage 1 — Propensity Score Matching
Logistic regression estimates each merchant's treatment probability given observed covariates. Merchants are matched 1:1 on the logit of the propensity score within a caliper of 0.05 log-odds units. Balance is validated via:
- **Love plot** — standardized mean differences (SMD) before/after; threshold at |SMD| < 0.10
- **Overlap plot** — propensity score distributions pre/post-match to verify common support

### Stage 2 — CausalForestDML
EconML's `CausalForestDML` uses cross-fitted gradient boosting models to partial out covariate effects on both outcome and treatment, then fits a Causal Forest on the residuals — yielding merchant-level CATE estimates with honest confidence intervals.

### Limitations
PSM removes observed confounding only. Unmeasured confounders cannot be eliminated without a randomized design.

---

## Future Improvements

- Rosenbaum bounds for sensitivity to unmeasured confounding
- TMLE layer for valid subgroup CATE confidence intervals
- SHAP decomposition on CATE outputs
- Policy tree for interpretable targeting rules
- Second outcome: transaction count alongside 90-day volume

---

## Dependencies

- [EconML](https://github.com/py-why/EconML) `>=0.15.0`
- scikit-learn `>=1.3.0`, pandas `>=2.0.0`, numpy `>=1.24.0`
- matplotlib `>=3.7.0`, seaborn `>=0.13.0`

See `requirements-lock.txt` for pinned versions.

---

## Note on Data

All data is **fully synthetic**, generated by `data/generate_data.py`. No real merchant records or proprietary information is included.

---

## License

MIT — see [LICENSE](LICENSE).
