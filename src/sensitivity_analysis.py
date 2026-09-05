"""
sensitivity_analysis.py
========================
The FN/FP cost assumptions (Rs.45,000 / Rs.800) are illustrative simulation
inputs, not derived from real Razorpay data -- so instead of presenting a
single threshold as if it were definitively "correct," we show how the
optimal threshold and its resulting precision/recall shift as those cost
assumptions vary. This is what makes the cost-sensitive decisioning
defensible: we're showing the DECISION LOGIC is sound, not claiming the
specific rupee figures are ground truth.

Uses the VALIDATION set only (consistent with train_evaluate.py's
methodology of never touching the test set for threshold-related decisions).
"""

import pandas as pd
import numpy as np
import joblib
import json
import os

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

DATA_PATH = os.path.join(PROJECT_ROOT, "data", "accounts_with_graph_features.csv")
MODEL_PATH = os.path.join(PROJECT_ROOT, "outputs", "model_bundle.joblib")
OUT_PATH = os.path.join(PROJECT_ROOT, "outputs", "cost_sensitivity.json")

FEATURE_COLS = [
    "account_age_days", "txn_velocity_per_day", "avg_txn_amount", "amount_std",
    "graph_degree", "clustering_coeff", "community_size", "n_distinct_shared_attr_types",
]

# A range of plausible FN:FP cost ratios to test, from conservative (rings
# aren't much worse than false alarms) to aggressive (rings are extremely
# costly relative to review friction).
COST_SCENARIOS = [
    {"label": "Low stakes (FN barely worse than FP, 1.5x)", "fn_cost": 1200, "fp_cost": 800},
    {"label": "Conservative (10x)", "fn_cost": 8000, "fp_cost": 800},
    {"label": "Base case (56x, used elsewhere in this project)", "fn_cost": 45000, "fp_cost": 800},
    {"label": "Aggressive (150x)", "fn_cost": 120000, "fp_cost": 800},
]

MAX_REVIEW_RATE = 0.06


def find_best_threshold(y_true, y_proba, fn_cost, fp_cost, max_review_rate=MAX_REVIEW_RATE):
    n_total = len(y_true)
    max_flags = int(n_total * max_review_rate)
    best_t, best_cost, best_stats = 0.5, float("inf"), None
    for t in np.linspace(0.01, 0.99, 99):
        y_pred = (y_proba >= t).astype(int)
        n_flagged = y_pred.sum()
        if n_flagged > max_flags:
            continue
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        cost = fp * fp_cost + fn * fn_cost
        if cost < best_cost:
            best_cost, best_t = cost, t
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            best_stats = {"precision": round(precision, 3), "recall": round(recall, 3),
                           "fp": int(fp), "fn": int(fn), "tp": int(tp)}
    return best_t, best_cost, best_stats


if __name__ == "__main__":
    bundle = joblib.load(MODEL_PATH)
    model = bundle["model"]

    df = pd.read_csv(DATA_PATH)
    X = df[FEATURE_COLS].fillna(0)
    y = df["is_ring"]

    # Reproduce the SAME split used in train_evaluate.py so we're testing
    # threshold sensitivity on the same validation set (not touching test).
    X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.20, stratify=y, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.25, stratify=y_temp, random_state=42)

    val_proba = model.predict_proba(X_val)[:, 1]

    print(f"{'='*70}\nCOST-SENSITIVITY ANALYSIS (validation set, n={len(y_val)})\n{'='*70}")
    results = []
    for scenario in COST_SCENARIOS:
        t, cost, stats = find_best_threshold(y_val.values, val_proba, scenario["fn_cost"], scenario["fp_cost"])
        print(f"\n{scenario['label']}  (FN=Rs.{scenario['fn_cost']:,}, FP=Rs.{scenario['fp_cost']:,})")
        print(f"  -> Optimal threshold: {t:.2f}  |  Precision: {stats['precision']}  |  Recall: {stats['recall']}")
        print(f"  -> Confusion: FP={stats['fp']}  FN={stats['fn']}  TP={stats['tp']}  |  Simulated cost: Rs.{cost:,}")
        results.append({**scenario, "optimal_threshold": t, "simulated_cost": int(cost), **stats})

    print(f"\n{'='*70}\nTAKEAWAY\n{'='*70}")
    thresholds_seen = [r["optimal_threshold"] for r in results]
    plateau_value = thresholds_seen[-1]
    # Find where the threshold first reaches its final (plateau) value
    plateau_start_idx = next(i for i, t in enumerate(thresholds_seen) if t == plateau_value)
    if plateau_start_idx == 0:
        print(f"The optimal threshold stayed at {plateau_value:.2f} across every cost scenario tested --")
        print("the review-capacity constraint was already binding even at the lowest-stakes scenario.")
    else:
        low_stakes_t = thresholds_seen[0]
        plateau_scenario = COST_SCENARIOS[plateau_start_idx]["label"]
        print(f"The optimal threshold moves from {low_stakes_t:.2f} (low-stakes case) down to "
              f"{plateau_value:.2f} once costs reach the '{plateau_scenario}' scenario, then PLATEAUS at "
              f"{plateau_value:.2f} for all higher FN:FP ratios --")
        print("because the review-capacity constraint becomes the binding limit, not the cost math.")
    print("This is itself a useful, honest finding: beyond a certain cost ratio, review")
    print("capacity -- not the model or the cost assumptions -- becomes the bottleneck, which is a")
    print("real operational insight, not just a math exercise.")

    with open(OUT_PATH, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved -> {OUT_PATH}")
