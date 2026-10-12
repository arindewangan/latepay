"""Generate a synthetic-but-realistic invoice-history training dataset.

SIMULATED DATA. This dataset is generated locally with a seeded RNG to
mimic plausible freelancer invoice-payment behaviour. It is NOT real
payment data and must never be presented as such. It exists so the
TabPFN classifier has a coherent tabular training set for the demo.

Label: late_30 = 1 when the invoice was paid more than 30 days AFTER the
payment-terms due date (i.e. days_to_pay > payment_terms_days + 30).
"""
import csv
import os
import numpy as np

SEED = 20261012
N_CLIENTS = 400
N_INVOICES = 4000
OUT_CSV = os.path.join(os.path.dirname(os.path.abspath(__file__)), "invoice_history.csv")

# Feature columns used by the model (same order everywhere).
FEATURES = [
    "amount_log",           # log1p(invoice amount, USD)
    "client_tenure_days",   # days since first invoice to this client
    "invoices_total",       # prior invoices to this client
    "late_rate",            # fraction of prior invoices paid >30d late
    "avg_days_to_pay",      # prior average payment time (days)
    "days_since_last_invoice",
    "is_first_invoice",     # 0/1
    "contract_strength",    # 0=none, 1=verbal/email, 2=written contract
    "payment_terms_days",   # 7/15/30/45/60
    "client_size",          # 0=individual, 1=small biz, 2=mid-market, 3=enterprise
    "is_international",     # 0/1
    "invoice_month",        # 1..12 (seasonality)
]

TERMS = [7, 15, 30, 45, 60]


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def generate(seed=SEED, n_clients=N_CLIENTS, n_invoices=N_INVOICES):
    rng = np.random.default_rng(seed)
    # Latent client traits: reliability (higher = pays on time), size, international.
    reliability = rng.normal(0.0, 1.0, size=n_clients)
    client_size = rng.integers(0, 4, size=n_clients)
    is_intl = (rng.random(n_clients) < 0.25).astype(int)

    rows = []
    invoice_idx = 0
    per_client = rng.multinomial(n_invoices, np.ones(n_clients) / n_clients)
    for c in range(n_clients):
        n = int(per_client[c])
        if n == 0:
            continue
        tenure = 0.0
        late_so_far = 0
        days_to_pay_hist = []
        for k in range(n):
            invoice_idx += 1
            amount = float(np.exp(rng.normal(np.log(2500), 1.1))) * rng.choice([1, 1, 1, 5])
            amount = round(min(max(amount, 50.0), 200000.0), 2)
            contract = int(rng.choice([0, 1, 2], p=[0.35, 0.35, 0.30]))
            terms = int(rng.choice(TERMS, p=[0.15, 0.25, 0.40, 0.12, 0.08]))
            month = int(rng.integers(1, 13))
            gap = float(rng.exponential(30)) if k > 0 else 0.0
            tenure += gap
            prior = k
            late_rate = late_so_far / prior if prior else 0.0
            avg_pay = float(np.mean(days_to_pay_hist)) if days_to_pay_hist else 30.0

            # Latent late propensity: unreliable clients + big amounts +
            # weak contracts + generous terms + international + newness = risk.
            logit = (
                -1.2  # base intercept -> ~20% late rate overall
                - 1.6 * reliability[c]
                + 0.35 * (np.log1p(amount) - np.log1p(2500)) / 2.0
                + 1.3 * late_rate
                + 0.5 * int(k == 0)
                - 0.55 * contract
                + 0.012 * (terms - 30)
                + 0.45 * is_intl[c]
                - 0.25 * client_size[c]
                + 0.15 * (month in (12, 1))   # year-end slowdown
                - 0.004 * tenure / 365.0
                + rng.normal(0, 0.4)
            )
            p_late = sigmoid(logit)
            late_30 = int(rng.random() < p_late)

            # Realised payment time consistent-ish with the label.
            if late_30:
                days_to_pay = float(terms + 30 + rng.exponential(25))
            else:
                days_to_pay = float(max(1.0, rng.normal(terms - 3, 8)))
            days_to_pay = round(days_to_pay, 1)

            rows.append({
                "invoice_id": f"INV-{invoice_idx:05d}",
                "client_id": f"C{c:04d}",
                "amount_usd": amount,
                "amount_log": round(float(np.log1p(amount)), 4),
                "client_tenure_days": round(tenure, 1),
                "invoices_total": prior,
                "late_rate": round(late_rate, 3),
                "avg_days_to_pay": round(avg_pay, 1),
                "days_since_last_invoice": round(gap, 1),
                "is_first_invoice": int(k == 0),
                "contract_strength": contract,
                "payment_terms_days": terms,
                "client_size": int(client_size[c]),
                "is_international": int(is_intl[c]),
                "invoice_month": month,
                "days_to_pay": days_to_pay,
                "late_30": late_30,
                "simulated": 1,
            })
            late_so_far += late_30
            days_to_pay_hist.append(days_to_pay)

    rng.shuffle(rows)
    return rows


def main():
    rows = generate()
    with open(OUT_CSV, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    pos = sum(r["late_30"] for r in rows)
    print(f"wrote {OUT_CSV}: {len(rows)} invoices, {pos} late>30d ({pos/len(rows):.1%})")


if __name__ == "__main__":
    main()
