# LatePay — know which invoices will ghost you, before they do 💸

*DEV Hacktoberfest AI Challenge — Week 2: Side Hustle. Cover image: `assets/banner.png`. Tags: `devchallenge, hf26challenge, ai, opensource`.*

---

Every freelancer knows the feeling: the work is delivered, the invoice is sent, and then… silence. Cash flow is the oxygen of a side hustle, and one late payer can choke it for a month.

So I built **LatePay** — a late-payment risk predictor that scores every outstanding invoice with **TabPFN**, the open-source tabular foundation model, and tells you what to do about it: chase now, ask for partial upfront, send a reminder, or relax.

**🔗 Live demo:** https://arindewangan.github.io/latepay/demo/ (precomputed TabPFN outputs, clearly labeled simulated)
**💻 Repo:** https://github.com/arindewangan/latepay
**🎬 Video:** *(link added on publish day)*

## What it does

- **Score an invoice** — enter the amount, client history (tenure, prior late rate, average days to pay), contract strength, and payment terms. TabPFN returns P(paid > 30 days past terms) as a risk band: HIGH / MEDIUM / ELEVATED / LOW.
- **Get one clear action** — HIGH → chase now and pause further billable work; MEDIUM → require 50% upfront before the next milestone; ELEVATED → send a polite nudge on the due date; LOW → normal terms.
- **Batch-score a CSV** — upload your invoice list (`/api/batch`) and download a scored risk report with a recommendation per row. Missing history columns fall back to an honest new-client baseline.
- **Explore offline** — the static demo ships 4 real client profiles plus a what-if grid (late rate × contract × amount), every cell a real precomputed TabPFN prediction, build-dated and labeled simulated.

## How it's built

*(draft — how-it-works + honest-hard-parts to be finished after the model is verified)*

A Python/Flask backend with a vanilla HTML/CSS/JS frontend. The pipeline:

1. `data/build_dataset.py` generates a SIMULATED invoice-history dataset (4,000 invoices, seeded RNG) with 12 features per invoice — amount, client tenure, prior invoice count, prior late rate, average days to pay, contract strength, payment terms, client size, international flag, invoice month.
2. At request time, `TabPFNClassifier(device="cpu")` — fitted in-context on that dataset in `risk.py` — predicts the late-payment probability for the new invoice. No gradient training, no API key, no proprietary models.
3. The GitHub Pages demo bakes real precomputed TabPFN outputs via `demo/build_demo.py` (build date stamped on the page) so it works with zero backend.

## The honest hard parts

*(to be written from real build notes — model verification pending)*

## What I'm proud of

- Real open-source-AI core: local TabPFN inference on real (simulated) tabular data, reproducible end-to-end from the repo — no API keys, no proprietary models, no fake "AI" labels.
- An honest demo: every number on the static page is a real precomputed TabPFN prediction, build-dated and labeled SIMULATED.
- CSV batch scoring that degrades gracefully — missing history columns fall back to a new-client baseline instead of erroring.
- A UI a non-technical freelancer can actually use: one form, one gauge, one action.

## What I learned

*(to be finished after verification — one technical takeaway about TabPFN on invoice data, one process takeaway)*

## What's next

- Let freelancers import real invoice history (Stripe/Invoicely exports) so the context set is their own data, not simulated.
- Per-client portfolio view: total at-risk revenue across all outstanding invoices.
- Email reminder templates wired to the ELEVATED band's nudge schedule.

## How it uses open-source AI

LatePay's risk core is **TabPFN v2** (`tabpfn`, an open-source tabular foundation model). We generate a simulated invoice-history dataset locally, and TabPFN fits in-context on it and predicts P(payment > 30 days past terms) per invoice at inference time — all on local CPU. No proprietary models or API keys are involved anywhere in the pipeline.

---

*Built with: Python, Flask, TabPFN (open-source), JavaScript. License: MIT.*

---

## Writing-quality checklist (run before publishing — writing is the heaviest judging weight)

- [ ] Hook lands in the first 3 lines (pain → project, no throat-clearing).
- [ ] Every AI claim is grounded in code or data (name the file/function/dataset; no "powered by AI" hand-waving).
- [ ] Honest limitations stated somewhere (what the model/data can't do — labeled "honest", not buried).
- [ ] No marketing fluff ("revolutionary", "cutting-edge", "game-changer" banned unless quoted).
- [ ] All four links present and verified live: demo, repo, video, (cover image renders).
- [ ] Tags set: `devchallenge, hf26challenge, ai, opensource`.
- [ ] Read the article aloud once — fix every sentence that stumbles.
- [ ] "How it uses open-source AI" paragraph names the model, license, and data source explicitly.
- [ ] Side-hustle framing present throughout (someone would actually pay for this — freelancer cash-flow protection).
