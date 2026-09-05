"""
build_graph_features.py
========================
Core idea of Abuse-Ring Sentinel: a single-transaction/single-account
classifier CANNOT see coordination. Two accounts each individually looking
"normal" become suspicious together if they share infrastructure (device,
IP, payment instrument, address) in a tight cluster with synchronized
behavior.

Approach:
  1. Build a graph: nodes = accounts, edges = shared identifier
     (device_id / ip_subnet / payment_instrument / shipping_address_hash).
  2. Run community detection (Louvain) to find tightly-connected clusters.
  3. Engineer GRAPH features per account (degree, clustering coefficient,
     community size, number of distinct shared identifiers with neighbors)
     ALONGSIDE the original behavioral features.
  4. These combined features feed the classifier in the next stage.

IMPORTANT: shared attributes alone are NOT proof of fraud (families share
cards, offices share IPs). That's why we combine graph structure with
behavioral features rather than using a naive "shared attribute = fraud"
rule -- a naive rule would produce too many false positives on legitimate
shared infrastructure.
"""

import os
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


import pandas as pd
import networkx as nx
import community as community_louvain  # python-louvain package
from collections import defaultdict

DATA_PATH = os.path.join(PROJECT_ROOT, "data", "accounts.csv")
OUT_PATH = os.path.join(PROJECT_ROOT, "data", "accounts_with_graph_features.csv")

SHARED_ATTRS = ["device_id", "ip_subnet", "payment_instrument", "shipping_address_hash"]


def build_graph(df: pd.DataFrame) -> nx.Graph:
    G = nx.Graph()
    G.add_nodes_from(df["account_id"])

    # For each shared attribute, group accounts that share the same value
    # and connect them pairwise (this creates cliques per shared identifier,
    # which is what real "same device / same card" collusion looks like).
    for attr in SHARED_ATTRS:
        groups = defaultdict(list)
        for _, row in df.iterrows():
            groups[row[attr]].append(row["account_id"])
        for value, members in groups.items():
            if len(members) < 2:
                continue
            # only connect if the group isn't absurdly large (e.g. a very
            # common IP subnet used by thousands = not meaningful signal;
            # cap group size used for edges to avoid a "hub" node exploding
            # graph density from incidental large shared infra)
            if len(members) > 25:
                continue
            for i in range(len(members)):
                for j in range(i + 1, len(members)):
                    if G.has_edge(members[i], members[j]):
                        G[members[i]][members[j]]["shared_attrs"].append(attr)
                        G[members[i]][members[j]]["weight"] += 1
                    else:
                        G.add_edge(members[i], members[j], shared_attrs=[attr], weight=1)
    return G


def extract_graph_features(G: nx.Graph, df: pd.DataFrame) -> pd.DataFrame:
    # Community detection (Louvain) -- finds densely-connected clusters
    partition = community_louvain.best_partition(G, random_state=42) if G.number_of_edges() > 0 else {}
    community_sizes = defaultdict(int)
    for node, comm_id in partition.items():
        community_sizes[comm_id] += 1

    degree = dict(G.degree())
    clustering = nx.clustering(G)  # local clustering coefficient

    rows = []
    for acc_id in df["account_id"]:
        comm_id = partition.get(acc_id, -1)
        comm_size = community_sizes.get(comm_id, 1)
        deg = degree.get(acc_id, 0)

        # distinct shared-attribute types with neighbors (evasion often
        # limits this to 1, but rings sharing 2+ types are more obvious)
        distinct_shared_types = set()
        if acc_id in G:
            for neighbor in G.neighbors(acc_id):
                distinct_shared_types.update(G[acc_id][neighbor]["shared_attrs"])

        rows.append({
            "account_id": acc_id,
            "graph_degree": deg,
            "clustering_coeff": clustering.get(acc_id, 0.0),
            "community_id": comm_id,
            "community_size": comm_size,
            "n_distinct_shared_attr_types": len(distinct_shared_types),
            "has_any_connection": int(deg > 0),
        })

    graph_feat_df = pd.DataFrame(rows)
    merged = df.merge(graph_feat_df, on="account_id", how="left")
    return merged, G


if __name__ == "__main__":
    df = pd.read_csv(DATA_PATH)
    G = build_graph(df)
    print(f"Graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

    merged, G = extract_graph_features(G, df)
    merged.to_csv(OUT_PATH, index=False)
    print(f"Saved -> {OUT_PATH}")

    # Quick sanity check: do ring accounts actually have higher graph signal?
    print("\n--- Sanity check: mean feature values, ring vs legit ---")
    print(merged.groupby("is_ring")[
        ["graph_degree", "clustering_coeff", "community_size", "n_distinct_shared_attr_types"]
    ].mean())
