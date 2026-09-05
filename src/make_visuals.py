"""
make_visuals.py -- generates the plots for README/pitch deck:
  1. Precision-Recall curve comparing behavioral-only vs +graph model
  2. Threshold sweep showing cost vs threshold with capacity constraint
  3. Feature importance (SHAP summary style, using GBT's built-in importance)
  4. A small illustrative network graph snippet showing a detected ring
     vs a legit lookalike cluster (visual proof of the core concept)
"""

import os
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import precision_recall_curve
import networkx as nx
import joblib
import sys
sys.path.append(os.path.join(PROJECT_ROOT, "src"))
from build_graph_features import build_graph, SHARED_ATTRS

OUT_DIR = os.path.join(PROJECT_ROOT, "outputs")
DATA_PATH = os.path.join(PROJECT_ROOT, "data", "accounts_with_graph_features.csv")

plt.rcParams.update({"font.size": 11, "figure.dpi": 130})

BEHAVIORAL_ONLY = ["account_age_days", "txn_velocity_per_day", "avg_txn_amount", "amount_std"]
FULL_FEATURES = BEHAVIORAL_ONLY + ["graph_degree", "clustering_coeff", "community_size", "n_distinct_shared_attr_types"]


def plot_pr_curve(df):
    X = df[FULL_FEATURES].fillna(0)
    y = df["is_ring"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, stratify=y, random_state=42)

    Xb_train, Xb_test = X_train[BEHAVIORAL_ONLY], X_test[BEHAVIORAL_ONLY]

    m_full = HistGradientBoostingClassifier(max_iter=200, max_depth=4, learning_rate=0.08,
                                             class_weight="balanced", random_state=42).fit(X_train, y_train)
    m_behav = HistGradientBoostingClassifier(max_iter=200, max_depth=4, learning_rate=0.08,
                                              class_weight="balanced", random_state=42).fit(Xb_train, y_train)

    p_full, r_full, _ = precision_recall_curve(y_test, m_full.predict_proba(X_test)[:, 1])
    p_behav, r_behav, _ = precision_recall_curve(y_test, m_behav.predict_proba(Xb_test)[:, 1])

    plt.figure(figsize=(7, 5.5))
    plt.plot(r_full, p_full, label="Behavioral + Graph features (ours)", linewidth=2.5, color="#1a5f3f")
    plt.plot(r_behav, p_behav, label="Behavioral features only (traditional)", linewidth=2.5,
             linestyle="--", color="#a83232")
    baseline = y_test.mean()
    plt.axhline(baseline, color="gray", linestyle=":", label=f"Random baseline (base rate={baseline:.1%})")
    plt.xlabel("Recall (% of rings caught)")
    plt.ylabel("Precision (% of flags that are real)")
    plt.title("Precision-Recall: Graph Features vs Behavioral-Only")
    plt.legend(loc="upper right", fontsize=9)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{OUT_DIR}/pr_curve_comparison.png")
    plt.close()
    print("Saved pr_curve_comparison.png")


def plot_threshold_sweep():
    sweep = pd.read_csv(f"{OUT_DIR}/threshold_sweep.csv")
    fig, ax1 = plt.subplots(figsize=(7.5, 5.5))
    ax1.plot(sweep["threshold"], sweep["cost"], color="#a83232", linewidth=2, label="Estimated business cost (Rs.)")
    ax1.set_xlabel("Decision threshold")
    ax1.set_ylabel("Estimated cost (Rs.)", color="#a83232")
    ax1.tick_params(axis="y", labelcolor="#a83232")

    within = sweep[sweep["within_capacity"]]
    best_row = within.loc[within["cost"].idxmin()]
    ax1.axvline(best_row["threshold"], color="#1a5f3f", linestyle="--",
                label=f"Chosen threshold={best_row['threshold']:.2f}\n(within review-capacity limit)")

    ax2 = ax1.twinx()
    ax2.plot(sweep["threshold"], sweep["n_flagged"], color="#2a6fb0", linewidth=1.5, alpha=0.7,
              label="Accounts flagged (count)")
    ax2.set_ylabel("Accounts flagged", color="#2a6fb0")
    ax2.tick_params(axis="y", labelcolor="#2a6fb0")

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper center", fontsize=8)
    plt.title("Cost-Sensitive Threshold Selection (capacity-constrained)")
    plt.tight_layout()
    plt.savefig(f"{OUT_DIR}/threshold_sweep.png")
    plt.close()
    print("Saved threshold_sweep.png")


def plot_feature_importance():
    bundle = joblib.load(f"{OUT_DIR}/model_bundle.joblib")
    model, feature_cols = bundle["model"], bundle["feature_cols"]

    # HistGradientBoostingClassifier doesn't expose feature_importances_
    # directly, so we use permutation importance for an honest measure.
    from sklearn.inspection import permutation_importance
    df = pd.read_csv(DATA_PATH)
    X = df[feature_cols].fillna(0)
    y = df["is_ring"]
    result = permutation_importance(model, X, y, n_repeats=8, random_state=42, scoring="average_precision")

    order = np.argsort(result.importances_mean)
    plt.figure(figsize=(7.5, 5))
    colors = ["#1a5f3f" if feature_cols[i] in ["graph_degree", "clustering_coeff", "community_size",
              "n_distinct_shared_attr_types"] else "#7a7a7a" for i in order]
    plt.barh(range(len(order)), result.importances_mean[order], color=colors)
    plt.yticks(range(len(order)), [feature_cols[i] for i in order])
    plt.xlabel("Permutation importance (drop in PR-AUC when shuffled)")
    plt.title("Feature Importance (green = graph/relational features)")
    plt.tight_layout()
    plt.savefig(f"{OUT_DIR}/feature_importance.png")
    plt.close()
    print("Saved feature_importance.png")


def plot_example_network():
    df = pd.read_csv(DATA_PATH)
    G = build_graph(df)

    # Find one clear ring cluster and one legit lookalike cluster for illustration
    ring_ids = df[df.is_ring == 1]["ring_id"].dropna().unique()
    example_ring_accounts = df[df.ring_id == ring_ids[0]]["account_id"].tolist()

    # a connected component containing only legit accounts, of similar size
    legit_component = None
    for comp in nx.connected_components(G):
        comp_df = df[df.account_id.isin(comp)]
        if comp_df["is_ring"].sum() == 0 and 3 <= len(comp) <= 6:
            legit_component = list(comp)
            break

    fig, axes = plt.subplots(1, 2, figsize=(12, 5.5))

    # Ring subgraph (include 1-hop neighbors for context)
    ring_neighbors = set(example_ring_accounts)
    for acc in example_ring_accounts:
        if acc in G:
            ring_neighbors.update(G.neighbors(acc))
    sub_ring = G.subgraph(ring_neighbors)
    pos1 = nx.spring_layout(sub_ring, seed=42)
    node_colors1 = ["#d94040" if n in example_ring_accounts else "#bbbbbb" for n in sub_ring.nodes()]
    nx.draw(sub_ring, pos1, ax=axes[0], node_color=node_colors1, with_labels=False,
            node_size=350, edge_color="#999999", width=1.2)
    axes[0].set_title(f"Detected Ring Cluster (red = ring members, n={len(example_ring_accounts)})")

    if legit_component:
        sub_legit = G.subgraph(legit_component)
        pos2 = nx.spring_layout(sub_legit, seed=42)
        nx.draw(sub_legit, pos2, ax=axes[1], node_color="#4a90d9", with_labels=False,
                node_size=350, edge_color="#999999", width=1.2)
        axes[1].set_title(f"Legit Lookalike Cluster (blue = innocent, n={len(legit_component)})\n"
                            f"(e.g. family/office sharing) -- similar SIZE, different behavior")

    plt.tight_layout()
    plt.savefig(f"{OUT_DIR}/example_ring_vs_legit.png")
    plt.close()
    print("Saved example_ring_vs_legit.png")


if __name__ == "__main__":
    df = pd.read_csv(DATA_PATH)
    plot_pr_curve(df)
    plot_threshold_sweep()
    plot_feature_importance()
    plot_example_network()
    print("\nAll visuals saved to", OUT_DIR)
