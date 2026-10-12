"""Bake real TabPFN predictions into demo/index.html (Frostwise pattern).

Every probability shown on the static demo is computed by the actual
TabPFN classifier (local CPU) at build time over a grid of invoice
profiles. The client-side "what-if" explorer only reads this precomputed
lookup table — no model runs in the browser, and no number is invented.

Run: python demo/build_demo.py
"""
import datetime
import json
import os

import numpy as np

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from risk import get_model, FEATURES, recommend

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "index.html")
BUILD_DATE = datetime.date.today().isoformat()

# Curated client cards (same defaults as app.py /api/demo).
CARDS = [
    {"name": "Acme Corp — repeat client, always on time", "amount_usd": 3200,
     "client_tenure_days": 640, "invoices_total": 18, "late_rate": 0.0,
     "avg_days_to_pay": 26, "days_since_last_invoice": 24, "is_first_invoice": 0,
     "contract_strength": 2, "payment_terms_days": 30, "client_size": 2,
     "is_international": 0, "invoice_month": 10},
    {"name": "New overseas client, big first invoice", "amount_usd": 15000,
     "client_tenure_days": 0, "invoices_total": 0, "late_rate": 0.0,
     "avg_days_to_pay": 30, "days_since_last_invoice": 0, "is_first_invoice": 1,
     "contract_strength": 0, "payment_terms_days": 30, "client_size": 1,
     "is_international": 1, "invoice_month": 10},
    {"name": "Local studio, patchy payment history", "amount_usd": 1800,
     "client_tenure_days": 210, "invoices_total": 6, "late_rate": 0.5,
     "avg_days_to_pay": 52, "days_since_last_invoice": 40, "is_first_invoice": 0,
     "contract_strength": 1, "payment_terms_days": 15, "client_size": 1,
     "is_international": 0, "invoice_month": 12},
    {"name": "Enterprise client, written contract", "amount_usd": 45000,
     "client_tenure_days": 900, "invoices_total": 30, "late_rate": 0.1,
     "avg_days_to_pay": 38, "days_since_last_invoice": 18, "is_first_invoice": 0,
     "contract_strength": 2, "payment_terms_days": 45, "client_size": 3,
     "is_international": 0, "invoice_month": 10},
]


def row_for(profile):
    return [float(profile.get(c, 0)) for c in FEATURES]


def base_row():
    return {"amount_log": float(np.log1p(2500)), "client_tenure_days": 180.0,
            "invoices_total": 8.0, "late_rate": 0.0, "avg_days_to_pay": 35.0,
            "days_since_last_invoice": 20.0, "is_first_invoice": 0,
            "contract_strength": 1, "payment_terms_days": 30, "client_size": 1,
            "is_international": 0, "invoice_month": 10}


def main():
    m = get_model()
    # What-if grid: prior late-rate x contract strength x invoice amount.
    # Every cell = one real TabPFN prediction; the page only looks these up.
    late_rates = [0.0, 0.2, 0.4, 0.6, 0.8]
    contracts = [0, 1, 2]
    amounts = [500, 2500, 10000, 40000]
    profiles, keys = [], []
    for card in CARDS:
        profiles.append(card); keys.append(("card", len(keys)))
    for lr in late_rates:
        for c in contracts:
            for a in amounts:
                p = base_row()
                p.update({"late_rate": lr, "contract_strength": c,
                          "amount_log": float(np.log1p(a)), "amount_usd": a})
                profiles.append(p); keys.append(("grid", lr, c, a))

    X = np.array([[float(pr.get(c, 0)) for c in FEATURES] for pr in profiles],
                 dtype=np.float64)
    proba = m["clf"].predict_proba(X)[:, 1]
    probs = [float(p) for p in proba]

    cards, grid = [], {}
    for (kind, *rest), pr, prob in zip(keys, profiles, probs):
        rec = recommend(prob)
        entry = {"prob": round(prob, 3), "pct": round(prob * 100, 1),
                 "band": rec["band"], "action": rec["action"], "detail": rec["detail"]}
        if kind == "card":
            entry["name"] = pr["name"]; entry["amount_usd"] = pr["amount_usd"]
            cards.append(entry)
        else:
            lr, c, a = rest
            grid.setdefault(str(lr), {}).setdefault(str(c), {})[str(a)] = entry

    data = {"build_date": BUILD_DATE, "model": "TabPFN (open-source, local CPU)",
            "n_train": m["n_train"], "simulated": True,
            "cards": cards, "grid": grid}
    html = PAGE.replace("__DATA__", json.dumps(data))
    with open(OUT, "w") as f:
        f.write(html)
    print(f"wrote {OUT}: {len(cards)} cards + {len(probs)-len(cards)} grid cells, build {BUILD_DATE}")


PAGE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>LatePay — late-payment risk demo (precomputed TabPFN)</title>
<style>
:root{--bg:#0e1116;--card:#171c24;--ink:#eef2f7;--muted:#9aa4b2;--acc:#4cc38a;--warn:#f5b841;--bad:#ef6461}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);font-family:system-ui,-apple-system,Segoe UI,Roboto,sans-serif}
.wrap{max-width:980px;margin:0 auto;padding:24px}
.sim{background:#3a2b12;border:1px solid var(--warn);color:var(--warn);border-radius:8px;padding:10px 14px;margin:16px 0;font-size:14px}
.card{background:var(--card);border-radius:12px;padding:20px;margin:16px 0}
.cards{display:grid;grid-template-columns:repeat(auto-fit,minmax(220px,1fr));gap:12px}
.client{border:1px solid #2a3340;border-radius:10px;padding:14px;cursor:pointer}
.client.sel{border-color:var(--acc)}
.band{font-weight:800;font-size:20px}
.gauge{height:14px;border-radius:7px;background:#222a35;overflow:hidden;margin:8px 0}
.gauge>div{height:100%}
.rec{font-weight:700}.detail{color:var(--muted);font-size:14px}
label{font-size:12px;color:var(--muted);display:block;margin:10px 0 4px}
input[type=range]{width:100%}
.note{font-size:12px;color:var(--muted)}
.row{display:flex;gap:24px;flex-wrap:wrap}.row>div{flex:1;min-width:240px}
a{color:var(--acc)}
</style>
</head>
<body><div class="wrap">
<h1>LatePay 💸 — interactive demo</h1>
<p class="sub">Every number below is a <b>real TabPFN prediction</b> (open-source tabular foundation model), precomputed on <b id="bdate"></b> from simulated invoice data. No backend needed — the page only looks up precomputed results.</p>
<div class="sim">⚠️ <b>SIMULATED / PRECOMPUTED:</b> TabPFN was trained on a locally generated synthetic invoice history (4,000 invoices, seeded RNG). These are illustrative risk estimates, not financial advice.</div>

<div class="card"><h3>1 · Four real client profiles</h3>
<div class="cards" id="cards"></div>
<div id="carddetail" style="margin-top:12px"></div></div>

<div class="card"><h3>2 · What-if explorer (precomputed grid)</h3>
<p class="note">Move the sliders — each combination reads a real precomputed TabPFN prediction. Other features are held at typical values (repeat client, 30-day terms, email agreement).</p>
<div class="row"><div>
<label>Prior late rate: <b id="lrv">0.2</b></label>
<input type="range" id="lr" min="0" max="4" value="1">
<label>Contract strength: <b id="ctv">verbal/email</b></label>
<input type="range" id="ct" min="0" max="2" value="1">
<label>Invoice amount (USD): <b id="amv">2,500</b></label>
<input type="range" id="am" min="0" max="3" value="1">
</div><div>
<div class="band" id="gband"></div>
<div class="gauge"><div id="gbar"></div></div>
<div id="gpct" style="font-size:20px"></div>
<div class="rec" id="grec" style="margin-top:8px"></div>
<div class="detail" id="gdet"></div>
</div></div></div>

<div class="card"><h3>3 · Batch scoring</h3>
<p class="note">Upload a CSV of invoices in the <a href="https://github.com/arindewangan/latepay">full app</a> (Flask, local TabPFN inference) to score hundreds of invoices at once and download the risk report.</p></div>

<p class="note">Built Oct 12–18, 2026 · DEV Hacktoberfest AI Challenge, Week 2: Side Hustle · MIT · Model: TabPFN (open-source) · Repo: <a href="https://github.com/arindewangan/latepay">github.com/arindewangan/latepay</a></p>
</div>
<script>
const D=__DATA__;
document.getElementById("bdate").textContent=D.build_date;
function color(b){return b==="HIGH"?"var(--bad)":b==="MEDIUM"?"var(--warn)":"var(--acc)"}
const cardsEl=document.getElementById("cards"),det=document.getElementById("carddetail");
D.cards.forEach((c,i)=>{const d=document.createElement("div");d.className="client"+(i===0?" sel":"");d.innerHTML=`<b>${c.name}</b><div class="gauge"><div style="width:${c.pct}%;background:${color(c.band)}"></div></div><span class="band" style="color:${color(c.band)}">${c.band}</span> · ${c.pct}%<div class="rec">→ ${c.action}</div>`;d.onclick=()=>{document.querySelectorAll(".client").forEach(e=>e.classList.remove("sel"));d.classList.add("sel");det.innerHTML=`<div class="band" style="color:${color(c.band)}">${c.band} RISK — ${c.pct}%</div><div class="rec">→ ${c.action}</div><div class="detail">${c.detail}</div><p class="note">$${c.amount_usd.toLocaleString()} invoice · TabPFN, ${D.n_train} simulated training invoices · precomputed ${D.build_date}</p>`};cardsEl.appendChild(d);if(i===0)d.onclick()});
const LRS=["0.0","0.2","0.4","0.6","0.8"],CTS=["0","1","2"],AMS=["500","2500","10000","40000"];
const CTL=["no contract","verbal/email","written contract"];
function upd(){const lr=LRS[+document.getElementById("lr").value],ct=CTS[+document.getElementById("ct").value],am=AMS[+document.getElementById("am").value];
document.getElementById("lrv").textContent=lr;document.getElementById("ctv").textContent=CTL[+ct];document.getElementById("amv").textContent=(+am).toLocaleString();
const e=D.grid[lr][ct][am];const b=document.getElementById("gband");b.textContent=e.band+" RISK";b.style.color=color(e.band);
const g=document.getElementById("gbar");g.style.width=e.pct+"%";g.style.background=color(e.band);
document.getElementById("gpct").textContent=e.pct+"% chance of being paid >30 days past terms";
document.getElementById("grec").textContent="→ "+e.action;document.getElementById("gdet").textContent=e.detail}
["lr","ct","am"].forEach(id=>document.getElementById(id).oninput=upd);upd();
</script>
</body></html>
"""

if __name__ == "__main__":
    main()
