"""LatePay tests — real TabPFN inference on the simulated dataset; no invented results."""
import io
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from risk import (FEATURES, coerce_features, get_model, load_dataset,
                  recommend, score_invoice)
from app import app as flask_app


@pytest.fixture(scope="module")
def dataset():
    return load_dataset()


@pytest.fixture(scope="module")
def model():
    return get_model()


def low_risk():
    return {"amount_usd": 2000, "client_tenure_days": 500, "invoices_total": 15,
            "late_rate": 0.0, "avg_days_to_pay": 25, "days_since_last_invoice": 20,
            "is_first_invoice": 0, "contract_strength": 2, "payment_terms_days": 30,
            "client_size": 2, "is_international": 0, "invoice_month": 10}


def high_risk():
    return {"amount_usd": 20000, "client_tenure_days": 0, "invoices_total": 0,
            "late_rate": 0.0, "avg_days_to_pay": 30, "days_since_last_invoice": 0,
            "is_first_invoice": 1, "contract_strength": 0, "payment_terms_days": 60,
            "client_size": 0, "is_international": 1, "invoice_month": 12}


def test_dataset_loads_with_expected_shape(dataset):
    X, y = dataset
    assert X.shape[1] == len(FEATURES)
    assert len(X) == len(y) > 3000
    assert set(np.unique(y)) <= {0, 1}


def test_dataset_has_both_classes(dataset):
    _, y = dataset
    rate = y.mean()
    assert 0.05 < rate < 0.6, f"unrealistic late rate {rate}"


def test_model_fits_and_caches(model):
    m2 = get_model()
    assert model is m2
    assert model["n_train"] == 1000  # seeded stratified subsample of the 4000-row dataset


def test_coerce_features_order_and_defaults():
    X = coerce_features({})
    assert X.shape == (1, len(FEATURES))
    assert np.isfinite(X).all()


def test_score_invoice_schema():
    r = score_invoice(low_risk())
    for k in ("late_probability", "late_probability_pct", "risk_band",
              "recommendation", "recommendation_detail", "model",
              "training_invoices", "data_note"):
        assert k in r, k
    assert 0.0 <= r["late_probability"] <= 1.0
    assert r["risk_band"] in ("LOW", "ELEVATED", "MEDIUM", "HIGH")
    assert "SIMULATED" in r["data_note"]


def test_risky_profile_scores_above_safe_profile():
    # Directional sanity on the simulated data (not an invented metric claim).
    assert score_invoice(high_risk())["late_probability"] > \
        score_invoice(low_risk())["late_probability"]


def test_recommendation_bands_cover_unit_interval():
    for p, band in [(0.0, "LOW"), (0.1, "LOW"), (0.2, "ELEVATED"),
                    (0.4, "MEDIUM"), (0.7, "HIGH"), (1.0, "HIGH")]:
        assert recommend(p)["band"] == band


def test_json_serializable():
    import json
    json.dumps(score_invoice(high_risk()))


def test_flask_index_ok():
    c = flask_app.test_client()
    r = c.get("/")
    assert r.status_code == 200
    assert b"LatePay" in r.data


def test_api_risk_ok():
    c = flask_app.test_client()
    r = c.post("/api/risk", json=low_risk())
    assert r.status_code == 200
    body = r.get_json()
    assert body["risk_band"] in ("LOW", "ELEVATED", "MEDIUM", "HIGH")


def test_api_demo_four_results():
    c = flask_app.test_client()
    r = c.get("/api/demo")
    body = r.get_json()
    assert body["simulated"] is True
    assert len(body["results"]) == 4


def test_api_batch_csv_roundtrip():
    c = flask_app.test_client()
    csv_text = ("client_name,invoice_id,amount_usd,late_rate,contract_strength\n"
                "Acme,INV-1,2500,0.0,2\n"
                "Risky,R-2,20000,0.8,0\n")
    r = c.post("/api/batch", data={"file": (io.BytesIO(csv_text.encode()), "in.csv")},
               content_type="multipart/form-data")
    assert r.status_code == 200
    assert "latepay_scored.csv" in r.headers.get("Content-Disposition", "")
    out = r.data.decode()
    assert "risk_band" in out and "INV-1" in out
