"""
ablation_study.py
==================
Critical proof-point for the whole project's thesis: "single-account
behavioral models miss coordinated fraud; graph/relational features catch
it." We prove this empirically rather than just asserting it, by training
the SAME model architecture on:
  (A) Behavioral features ONLY (what a traditional single-account risk
      model would use)
  (B) Behavioral + Graph features (our full approach)

and comparing recall/precision/PR-AUC. If (B) doesn't meaningfully beat
(A), the graph approach isn't earning its complexity -- so this is also
an honesty check on ourselves, not just a marketing slide.
"""

import os
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score, average_precision_score, precision_score, recall_score
import json

DATA_PATH = os.path.join(PROJECT_ROOT, "data", "accounts_with_graph_features.csv")
OUT_PATH = os.path.join(PROJECT_ROOT, "outputs", "ablation_results.json")

BEHAVIORAL_ONLY = ["account_age_days", "txn_velocity_per_day", "avg_txn_amount", "amount_std"]
BEHAVIORAL_PLUS_GRAPH = BEHAVIORAL_ONLY + [
    "graph_degree", "clustering_coeff", "community_size", "n_distinct_shared_attr_types"
]


def run_variant(df, feature_cols, name):
    X = df[feature_cols].fillna(0)
    y = df["is_ring"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, stratify=y, random_state=42
    )
    model = HistGradientBoostingClassifier(
        max_iter=200, max_depth=4, learning_rate=0.08,
        class_weight="balanced", random_state=42
    )
    model.fit(X_train, y_train)
    proba = model.predict_proba(X_test)[:, 1]
    pred = (proba >= 0.5).astype(int)

    result = {
        "variant": name,
        "features_used": feature_cols,
        "roc_auc": round(roc_auc_score(y_test, proba), 4),
        "pr_auc": round(average_precision_score(y_test, proba), 4),
        "precision_at_0.5": round(precision_score(y_test, pred, zero_division=0), 4),
        "recall_at_0.5": round(recall_score(y_test, pred, zero_division=0), 4),
    }
    print(f"\n{name}")
    print(f"  Features: {feature_cols}")
    print(f"  ROC-AUC: {result['roc_auc']}  |  PR-AUC: {result['pr_auc']}")
    print(f"  Precision@0.5: {result['precision_at_0.5']}  |  Recall@0.5: {result['recall_at_0.5']}")
    return result


if __name__ == "__main__":
    df = pd.read_csv(DATA_PATH)

    result_a = run_variant(df, BEHAVIORAL_ONLY, "(A) Behavioral features ONLY (traditional approach)")
    result_b = run_variant(df, BEHAVIORAL_PLUS_GRAPH, "(B) Behavioral + Graph features (our approach)")

    pr_auc_lift = result_b["pr_auc"] - result_a["pr_auc"]
    recall_lift = result_b["recall_at_0.5"] - result_a["recall_at_0.5"]

    print(f"\n{'='*60}\nABLATION CONCLUSION\n{'='*60}")
    print(f"PR-AUC lift from adding graph features: {pr_auc_lift:+.4f}")
    print(f"Recall lift at threshold 0.5: {recall_lift:+.4f}")
    if pr_auc_lift > 0.02:
        print("=> Graph features provide MEANINGFUL, non-trivial improvement.")
    else:
        print("=> Graph features provide marginal improvement in this sample -- "
              "reported honestly rather than overstated.")

    with open(OUT_PATH, "w") as f:
        json.dump({"behavioral_only": result_a, "behavioral_plus_graph": result_b,
                    "pr_auc_lift": round(pr_auc_lift, 4), "recall_lift": round(recall_lift, 4)}, f, indent=2)
    print(f"\nSaved -> {OUT_PATH}")
