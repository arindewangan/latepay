"""LatePay core: TabPFN-based late-payment risk prediction.

Trains (in-context fits) a TabPFNClassifier on the SIMULATED invoice-history
dataset and scores outstanding invoices: P(payment > 30 days past terms).
No API keys, no proprietary models — local CPU inference only.

All outputs are probabilities + rule-based recommendations on SIMULATED data;
see README.md for the honest limitations.
"""
import csv
import os

import numpy as np

DATA_CSV = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "invoice_history.csv")

FEATURES = [
    "amount_log",
    "client_tenure_days",
    "invoices_total",
    "late_rate",
    "avg_days_to_pay",
    "days_since_last_invoice",
    "is_first_invoice",
    "contract_strength",
    "payment_terms_days",
    "client_size",
    "is_international",
    "invoice_month",
]

# (threshold, band, action, detail)
BANDS = [
    (0.60, "HIGH", "Chase now",
     "Call or email today. Pause further billable work until this is settled and add a late fee to the next invoice."),
    (0.35, "MEDIUM", "Require partial upfront",
     "Before the next milestone, ask for 50% upfront. Split the remaining balance into smaller invoices."),
    (0.15, "ELEVATED", "Send a reminder",
     "Send a polite nudge on the due date and again 7 days later. Shorten terms on the next invoice."),
    (0.00, "LOW", "Normal terms",
     "Risk looks low. Keep standard terms; the model sees nothing alarming in this client's history."),
]

_model_cache = {}


def load_dataset():
    """Load SIMULATED training data. Raises FileNotFoundError if not built."""
    with open(DATA_CSV, newline="") as f:
        rows = list(csv.DictReader(f))
    X = np.array([[float(r[c]) for c in FEATURES] for r in rows], dtype=np.float64)
    y = np.array([int(r["late_30"]) for r in rows], dtype=np.int64)
    return X, y


def get_model():
    """Lazily fit the TabPFN classifier (in-context learning; fit = store context).

    Context is a seeded, label-stratified 1000-row subsample of the 4000-row
    simulated dataset: TabPFN's CPU attention is O(n^2) in context size, and
    1000 rows keeps per-request inference in the seconds range on this box.
    n_estimators=2 keeps the ensemble honest while staying CPU-friendly.
    """
    if "clf" not in _model_cache:
        from tabpfn import TabPFNClassifier

        X, y = load_dataset()
        rng = np.random.default_rng(7)
        keep = []
        for cls in (0, 1):
            idx = np.flatnonzero(y == cls)
            keep.append(rng.choice(idx, size=min(500, len(idx)), replace=False))
        keep = np.concatenate(keep)
        rng.shuffle(keep)
        Xc, yc = X[keep], y[keep]
        clf = TabPFNClassifier(device="cpu", n_estimators=2)
        clf.fit(Xc, yc)
        _model_cache["clf"] = clf
        _model_cache["n_train"] = len(Xc)
        _model_cache["late_rate_train"] = float(yc.mean())
    return _model_cache


def coerce_features(raw):
    """Build the feature vector from user input dict; missing history => new-client baseline."""
    amount = float(raw.get("amount_usd", 1000))
    out = {
        "amount_log": float(np.log1p(max(amount, 0.0))),
        "client_tenure_days": float(raw.get("client_tenure_days", 0)),
        "invoices_total": float(raw.get("invoices_total", 0)),
        "late_rate": float(raw.get("late_rate", 0.0)),
        "avg_days_to_pay": float(raw.get("avg_days_to_pay", 30)),
        "days_since_last_invoice": float(raw.get("days_since_last_invoice", 0)),
        "is_first_invoice": int(raw.get("is_first_invoice", 1 if float(raw.get("invoices_total", 0)) == 0 else 0)),
        "contract_strength": int(raw.get("contract_strength", 1)),
        "payment_terms_days": int(raw.get("payment_terms_days", 30)),
        "client_size": int(raw.get("client_size", 1)),
        "is_international": int(raw.get("is_international", 0)),
        "invoice_month": int(raw.get("invoice_month", 10)),
    }
    return np.array([[out[c] for c in FEATURES]], dtype=np.float64)


def recommend(probability):
    for threshold, band, action, detail in BANDS:
        if probability >= threshold:
            return {"band": band, "action": action, "detail": detail}
    return {"band": "LOW", "action": "Normal terms", "detail": BANDS[-1][3]}  # pragma: no cover


def score_invoice(raw):
    """Score one invoice dict. Returns probability, band, recommendation (JSON-safe)."""
    m = get_model()
    X = coerce_features(raw)
    proba = m["clf"].predict_proba(X)[0]
    late_prob = float(proba[1]) if len(proba) > 1 else float(proba[0])
    rec = recommend(late_prob)
    return {
        "late_probability": round(late_prob, 3),
        "late_probability_pct": round(late_prob * 100, 1),
        "risk_band": rec["band"],
        "recommendation": rec["action"],
        "recommendation_detail": rec["detail"],
        "model": "TabPFN (open-source tabular foundation model, local CPU)",
        "training_invoices": m["n_train"],
        "training_late_rate": round(m["late_rate_train"], 3),
        "data_note": "SIMULATED training data — synthetic invoice history generated locally; not real payment records.",
    }
