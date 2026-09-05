"""
main.py -- FastAPI serving layer for Abuse-Ring Sentinel
==========================================================
Two endpoints, for two different real use cases:

1. POST /assess         -- score a SINGLE account when you already know its
                            graph features (fast path, e.g. a real-time
                            check during onboarding when you've precomputed
                            a rolling graph snapshot).

2. POST /analyze-network -- the REAL product endpoint. Takes RAW account
                            data (device/IP/payment/address identifiers +
                            behavioral fields) for a batch of accounts,
                            builds the shared-infrastructure graph, extracts
                            graph features, scores every account, AND
                            rolls flagged accounts up into RING-LEVEL
                            reports with evidence -- exactly what a fraud
                            analyst actually wants to see, and what this
                            project's whole thesis (coordinated abuse,
                            not just individual risk) is about.

Run with:  uvicorn main:app --reload --port 8000
Then visit http://localhost:8000/docs for interactive API documentation.

SAFETY NOTE: This is a pure defensive/detection tool. It only SCORES risk
and explains that score -- it takes no autonomous action (no blocking,
no fund movement, no account suspension). Any action based on its output
is left to a human reviewer or a separate, explicitly-authorized system.
This keeps it strictly within the "defense-only" requirement.
"""

import os
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

import sys
sys.path.append(os.path.join(PROJECT_ROOT, "src"))

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import joblib
import pandas as pd
import shap
import numpy as np
import networkx as nx
from typing import List, Dict
from collections import defaultdict

from build_graph_features import build_graph, extract_graph_features, SHARED_ATTRS
from ring_detection import build_ring_reports, RING_RISK_THRESHOLD, MIN_RING_SIZE_TO_REPORT

app = FastAPI(
    title="Abuse-Ring Sentinel API",
    description="Defensive fraud-ring detection with explainable, audited risk scoring. "
                 "Detection only -- takes no autonomous action.",
    version="2.0.0",
)

# CORS: allows the local dashboard.html (opened as a separate file/origin) to
# call this API from a browser. Open (allow_origins=["*"]) is fine for this
# local demo tool running on localhost; a real deployment would restrict this
# to specific trusted origins.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

MODEL_PATH = os.path.join(PROJECT_ROOT, "outputs", "model_bundle.joblib")
DATA_PATH = os.path.join(PROJECT_ROOT, "data", "accounts_with_graph_features.csv")

_bundle = joblib.load(MODEL_PATH)
_model = _bundle["model"]
_threshold = _bundle["threshold"]
_feature_cols = _bundle["feature_cols"]

_background_df = pd.read_csv(DATA_PATH)[_feature_cols].fillna(0).sample(
    min(200, len(pd.read_csv(DATA_PATH))), random_state=42
)
_explainer = shap.Explainer(_model, _background_df)


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------

class AccountFeatures(BaseModel):
    """Pre-computed graph features -- fast path for /assess."""
    account_id: str = Field(..., example="ACC_00001")
    account_age_days: float = Field(..., ge=0, example=15)
    txn_velocity_per_day: float = Field(..., ge=0, example=7.2)
    avg_txn_amount: float = Field(..., ge=0, example=1900)
    amount_std: float = Field(..., ge=0, example=55)
    graph_degree: float = Field(..., ge=0, example=6)
    clustering_coeff: float = Field(..., ge=0, le=1, example=0.65)
    community_size: float = Field(..., ge=1, example=6)
    n_distinct_shared_attr_types: float = Field(..., ge=0, le=4, example=2)


class RawAccount(BaseModel):
    """Raw account record -- the real input a merchant/platform actually has.
    No graph features required; the API computes them internally."""
    account_id: str = Field(..., example="ACC_00001")
    device_id: str = Field(..., example="dev_a1b2c3")
    ip_subnet: str = Field(..., example="ip_9f8e7d")
    payment_instrument: str = Field(..., example="pay_44556")
    shipping_address_hash: str = Field(..., example="addr_11223")
    account_age_days: float = Field(..., ge=0, example=15)
    txn_velocity_per_day: float = Field(..., ge=0, example=7.2)
    avg_txn_amount: float = Field(..., ge=0, example=1900)
    amount_std: float = Field(..., ge=0, example=55)


class ContributingFactor(BaseModel):
    feature: str
    value: float
    shap_contribution: float
    effect: str


class RiskAssessment(BaseModel):
    account_id: str
    risk_score: float
    flagged: bool
    threshold_used: float
    top_contributing_factors: List[ContributingFactor]
    recommendation: str


class RingReport(BaseModel):
    candidate_ring_id: str
    member_count: int
    member_accounts: List[str]
    avg_member_risk_score: float
    internal_edge_density: float
    shared_attribute_evidence: Dict[str, int]
    recommendation: str


class NetworkAnalysisResponse(BaseModel):
    accounts_analyzed: int
    individual_assessments: List[RiskAssessment]
    candidate_rings_detected: int
    ring_reports: List[RingReport]
    methodology_note: str


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _score_and_explain(row: dict) -> RiskAssessment:
    X = pd.DataFrame([[row[c] for c in _feature_cols]], columns=_feature_cols)
    proba = float(_model.predict_proba(X)[0, 1])
    flagged = proba >= _threshold

    shap_values = _explainer(X)
    contributions = sorted(
        zip(_feature_cols, shap_values.values[0]),
        key=lambda x: abs(x[1]), reverse=True
    )
    top_factors = [
        ContributingFactor(
            feature=feat, value=round(float(row[feat]), 3),
            shap_contribution=round(float(val), 4),
            effect="increased risk score" if val > 0 else "decreased risk score",
        )
        for feat, val in contributions[:4]
    ]
    if flagged and proba > 0.7:
        rec = "HIGH PRIORITY: Route to manual fraud review queue immediately."
    elif flagged:
        rec = "MEDIUM PRIORITY: Queue for manual review within standard SLA."
    else:
        rec = "No action needed at current threshold; continue routine monitoring."

    return RiskAssessment(
        account_id=row["account_id"], risk_score=round(proba, 4), flagged=flagged,
        threshold_used=_threshold, top_contributing_factors=top_factors, recommendation=rec,
    )


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/")
def root():
    return {
        "service": "Abuse-Ring Sentinel API",
        "status": "operational",
        "mode": "defense-only (detection + explanation, no autonomous action)",
        "current_threshold": _threshold,
        "endpoints": {
            "/assess": "Score a single account using pre-computed graph features (fast path)",
            "/analyze-network": "RECOMMENDED: submit raw accounts (device/IP/payment/address + "
                                 "behavior); builds the graph, scores every account, and returns "
                                 "ring-level reports with evidence -- the real product.",
        },
    }


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/assess", response_model=RiskAssessment)
def assess_account(account: AccountFeatures):
    """Score a single account when graph features are already known."""
    try:
        return _score_and_explain(account.dict())
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Assessment failed: {str(e)}")


@app.post("/analyze-network", response_model=NetworkAnalysisResponse)
def analyze_network(accounts: List[RawAccount]):
    """
    THE MAIN PRODUCT ENDPOINT.
    Takes a batch of RAW accounts (no graph features required -- you supply
    the identifiers a real system already has: device, IP, payment
    instrument, shipping address, plus basic behavioral stats). This
    endpoint:
      1. Builds the shared-infrastructure graph across the submitted batch
      2. Extracts graph/relational features per account
      3. Scores every account for individual risk
      4. Detects and reports RING-LEVEL clusters with evidence -- the
         actual "coordinated abuse" product, not just per-account scores

    Minimum useful batch size: ~20-30 accounts (graph signal needs enough
    accounts to reveal shared infrastructure). A batch of 2-3 accounts will
    not produce meaningful graph features.
    """
    if len(accounts) < 5:
        raise HTTPException(status_code=400, detail="Submit at least 5 accounts for meaningful "
                             "graph-based analysis -- a tiny batch can't reveal shared infrastructure.")
    if len(accounts) > 2000:
        raise HTTPException(status_code=400, detail="Batch size limited to 2000 accounts per request.")

    df = pd.DataFrame([a.dict() for a in accounts])

    # Build graph from raw shared attributes (same logic as offline pipeline)
    G = build_graph(df)
    df_with_graph, G = extract_graph_features(G, df)

    # Score every account
    X = df_with_graph[_feature_cols].fillna(0)
    df_with_graph["risk_score"] = _model.predict_proba(X)[:, 1]
    df_with_graph["flagged"] = df_with_graph["risk_score"] >= _threshold

    individual_assessments = [
        _score_and_explain(row.to_dict()) for _, row in df_with_graph.iterrows()
    ]

    # Ring-level detection using the SAME graph
    # (build_ring_reports expects an 'is_ring' column for its eval-only
    # diagnostics; since real submitted data has no ground truth, we fill
    # with -1 as a sentinel and strip those fields from the API response)
    df_for_rings = df_with_graph.copy()
    df_for_rings["is_ring"] = -1
    raw_rings = build_ring_reports(df_for_rings, G,
                                    min_ring_size=MIN_RING_SIZE_TO_REPORT,
                                    ring_risk_threshold=RING_RISK_THRESHOLD)

    ring_reports = [
        RingReport(
            candidate_ring_id=r["candidate_ring_id"],
            member_count=r["member_count"],
            member_accounts=r["member_accounts"],
            avg_member_risk_score=r["avg_member_risk_score"],
            internal_edge_density=r["internal_edge_density"],
            shared_attribute_evidence=r["shared_attribute_evidence"],
            recommendation=r["recommendation"],
        )
        for r in raw_rings
    ]

    return NetworkAnalysisResponse(
        accounts_analyzed=len(accounts),
        individual_assessments=individual_assessments,
        candidate_rings_detected=len(ring_reports),
        ring_reports=ring_reports,
        methodology_note=(
            f"Graph built from {len(accounts)} submitted accounts. Ring-level groups require "
            f"{MIN_RING_SIZE_TO_REPORT}+ mutually-connected accounts (cliques) with average risk "
            f"score >= {RING_RISK_THRESHOLD} to be reported. Individual risk scores use a threshold "
            f"of {_threshold} chosen via cost-sensitive analysis on a held-out validation set."
        ),
    )
