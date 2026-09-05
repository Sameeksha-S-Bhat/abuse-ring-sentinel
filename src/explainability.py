"""
explainability.py
==================
Generates the AUDIT TRAIL required by the track's "bar": every flagged
account must come with an explanation of WHY it was flagged. We use SHAP
(SHapley Additive exPlanations) to attribute the risk score to individual
features on a per-account basis -- this is what makes the tool usable by
a real risk analyst (who needs to justify an account suspension/review),
not just a black-box score.
"""

import os
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


import pandas as pd
import numpy as np
import joblib
import shap
import json

MODEL_PATH = os.path.join(PROJECT_ROOT, "outputs", "model_bundle.joblib")
DATA_PATH = os.path.join(PROJECT_ROOT, "data", "accounts_with_graph_features.csv")
OUT_PATH = os.path.join(PROJECT_ROOT, "outputs", "sample_audit_trails.json")


def generate_audit_trail(account_row: pd.Series, model, explainer, feature_cols, threshold):
    X_single = pd.DataFrame([account_row[feature_cols].fillna(0).values], columns=feature_cols)
    proba = model.predict_proba(X_single)[0, 1]
    shap_values = explainer(X_single)

    # Rank features by absolute SHAP contribution (impact on this specific score)
    contributions = sorted(
        zip(feature_cols, shap_values.values[0]),
        key=lambda x: abs(x[1]), reverse=True
    )

    reasons = []
    for feat, val in contributions[:4]:
        direction = "increased" if val > 0 else "decreased"
        reasons.append({
            "feature": feat,
            "value": round(float(account_row[feat]), 3),
            "shap_contribution": round(float(val), 4),
            "effect": f"{direction} risk score"
        })

    return {
        "account_id": account_row["account_id"],
        "risk_score": round(float(proba), 4),
        "flagged": bool(proba >= threshold),
        "threshold_used": threshold,
        "ground_truth_is_ring": bool(account_row["is_ring"]),  # only present because this is a demo/eval set
        "top_contributing_factors": reasons,
    }


if __name__ == "__main__":
    bundle = joblib.load(MODEL_PATH)
    model, threshold, feature_cols = bundle["model"], bundle["threshold"], bundle["feature_cols"]

    df = pd.read_csv(DATA_PATH)
    X = df[feature_cols].fillna(0)

    print("Building SHAP explainer (TreeExplainer, fast for gradient boosted trees)...")
    explainer = shap.Explainer(model, X.sample(min(300, len(X)), random_state=42))

    # Generate audit trails for a handful of interesting cases:
    # a true positive (correctly flagged ring), a false positive, and a
    # borderline case -- shows the tool working across scenarios, not
    # just cherry-picking the easiest example.
    df["proba"] = model.predict_proba(X)[:, 1]
    df["pred"] = (df["proba"] >= threshold).astype(int)

    tp_case = df[(df.is_ring == 1) & (df.pred == 1)].iloc[0]
    fp_case_candidates = df[(df.is_ring == 0) & (df.pred == 1)]
    fn_case_candidates = df[(df.is_ring == 1) & (df.pred == 0)]

    trails = [generate_audit_trail(tp_case, model, explainer, feature_cols, threshold)]
    if len(fp_case_candidates) > 0:
        trails.append(generate_audit_trail(fp_case_candidates.iloc[0], model, explainer, feature_cols, threshold))
    if len(fn_case_candidates) > 0:
        trails.append(generate_audit_trail(fn_case_candidates.iloc[0], model, explainer, feature_cols, threshold))

    for t in trails:
        print(f"\nAccount {t['account_id']} | risk_score={t['risk_score']} | flagged={t['flagged']} | "
              f"actually_ring={t['ground_truth_is_ring']}")
        for r in t["top_contributing_factors"]:
            print(f"    - {r['feature']}={r['value']}  ({r['effect']}, SHAP={r['shap_contribution']})")

    with open(OUT_PATH, "w") as f:
        json.dump(trails, f, indent=2)
    print(f"\nSaved sample audit trails -> {OUT_PATH}")
