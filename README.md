# LatePay 💸 — late-payment risk predictor for freelancers

*DEV Hacktoberfest AI Challenge — Week 2: Side Hustle. Built Oct 12–18, 2026. This is a NEW project built during the challenge window.*

**Live demo:** https://arindewangan.github.io/latepay/demo/ (precomputed TabPFN outputs, clearly labeled simulated)
**Repo:** https://github.com/arindewangan/latepay
**Video:** *(link added on publish day)*

Freelancers lose real money to late payers. LatePay scores every outstanding invoice with **TabPFN** — an open-source tabular foundation model running locally on CPU, no API keys — estimating the probability the invoice is paid **more than 30 days past terms**. Each score comes with one actionable recommendation: **chase now**, **require partial upfront**, **send a reminder**, or **normal terms**.

## How it works

1. `data/build_dataset.py` — generates a SIMULATED invoice-history dataset (4,000 invoices, seeded RNG; 12 features: amount, client tenure, prior late rate, contract strength, terms, …). Synthetic data, labeled `simulated` in every row and on every UI.
2. `risk.py` — fits `TabPFNClassifier(device="cpu", n_estimators=2)` in-context on a seeded, label-stratified 1000-row subsample of that dataset (documented CPU-speed trade-off; full 4000-row CSV ships as the data artifact) and `score_invoice()` returns the late-payment probability, risk band, and recommendation.
3. `app.py` — Flask dashboard: interactive scorer, CSV batch scoring (`/api/batch` → scored CSV download), `/api/demo` with four real labeled demo profiles.
4. `demo/build_demo.py` — runs real TabPFN inference once over a grid of invoice profiles and bakes the results into `demo/index.html` (static, Pages-ready, build-dated).

## Run it

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python data/build_dataset.py   # generates the simulated dataset
pytest -q                      # real TabPFN inference tests
python app.py                  # dashboard at http://127.0.0.1:5000
python demo/build_demo.py      # rebuilds demo/index.html with fresh predictions
```

TabPFN downloads its pretrained weights on first run (~cached in `~/.cache`) and runs inference on CPU.

## Honest limitations

- The training data is **synthetic** (seeded generator in `data/build_dataset.py`), designed to mimic plausible payment behaviour — it is not real payment records and the model has never seen a real invoice.
- Predictions are risk *estimates* for prioritising follow-ups, **not financial advice** and not a credit score.
- The static demo can only look up precomputed predictions; live scoring needs the Flask backend.

## How it uses open-source AI

LatePay's risk core is **TabPFN v2** (`tabpfn`, open-source tabular foundation model) — `TabPFNClassifier` fits in-context on the simulated invoice history and predicts P(paid >30 days late) per invoice at inference time, all on local CPU. No proprietary models or API keys anywhere in the pipeline.

License: MIT.
