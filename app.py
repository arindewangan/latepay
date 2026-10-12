"""LatePay Flask app: invoice risk dashboard + JSON API."""
import csv
import io

from flask import Flask, jsonify, render_template_string, request, Response

from risk import score_invoice, coerce_features, get_model

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 2 * 1024 * 1024  # 2 MB CSV cap

DEMO_CLIENTS = [
    {"name": "Acme Corp (repeat client, always on time)", "amount_usd": 3200,
     "client_tenure_days": 640, "invoices_total": 18, "late_rate": 0.0,
     "avg_days_to_pay": 26, "days_since_last_invoice": 24, "is_first_invoice": 0,
     "contract_strength": 2, "payment_terms_days": 30, "client_size": 2,
     "is_international": 0, "invoice_month": 10},
    {"name": "New overseas client, big first invoice", "amount_usd": 15000,
     "client_tenure_days": 0, "invoices_total": 0, "late_rate": 0.0,
     "avg_days_to_pay": 30, "days_since_last_invoice": 0, "is_first_invoice": 1,
     "contract_strength": 0, "payment_terms_days": 30, "client_size": 1,
     "is_international": 1, "invoice_month": 10},
    {"name": "Local studio, patchy history", "amount_usd": 1800,
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


@app.route("/")
def index():
    return render_template_string(INDEX_HTML)


@app.route("/api/risk", methods=["POST"])
def api_risk():
    data = request.get_json(force=True) or {}
    return jsonify(score_invoice(data))


@app.route("/api/demo")
def api_demo():
    """Scored sample invoices — real TabPFN inference, labeled SIMULATED."""
    out = []
    for c in DEMO_CLIENTS:
        r = score_invoice(c)
        r["client_name"] = c["name"]
        out.append(r)
    return jsonify({"simulated": True, "results": out})


@app.route("/api/batch", methods=["POST"])
def api_batch():
    """Score a CSV of invoices. Unknown/missing history columns fall back to new-client baseline."""
    f = request.files.get("file")
    if f is None:
        return jsonify({"error": "no file uploaded (field name 'file')"}), 400
    rows = list(csv.DictReader(io.StringIO(f.read().decode("utf-8-sig"))))
    scored = []
    for row in rows:
        payload = {k: v for k, v in row.items() if v not in (None, "")}
        r = score_invoice(payload)
        r["input"] = {k: row.get(k) for k in ("client_name", "invoice_id", "amount_usd")}
        scored.append(r)
    buf = io.StringIO()
    w = csv.DictWriter(buf, fieldnames=["client_name", "invoice_id", "amount_usd",
                                       "late_probability", "risk_band", "recommendation"])
    w.writeheader()
    for s in scored:
        w.writerow({"client_name": s["input"].get("client_name", ""),
                    "invoice_id": s["input"].get("invoice_id", ""),
                    "amount_usd": s["input"].get("amount_usd", ""),
                    "late_probability": s["late_probability"],
                    "risk_band": s["risk_band"],
                    "recommendation": s["recommendation"]})
    return Response(buf.getvalue(), mimetype="text/csv",
                    headers={"Content-Disposition": "attachment; filename=latepay_scored.csv"})


INDEX_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>LatePay — Late-payment risk predictor for freelancers</title>
<style>
:root{--bg:#0e1116;--card:#171c24;--ink:#eef2f7;--muted:#9aa4b2;--acc:#4cc38a;--warn:#f5b841;--bad:#ef6461}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);font-family:system-ui,-apple-system,Segoe UI,Roboto,sans-serif}
.wrap{max-width:960px;margin:0 auto;padding:24px}
h1{margin:0 0 4px}.sub{color:var(--muted);margin:0 0 20px}
.sim{background:#3a2b12;border:1px solid var(--warn);color:var(--warn);border-radius:8px;padding:10px 14px;margin:16px 0;font-size:14px}
.card{background:var(--card);border-radius:12px;padding:20px;margin:16px 0}
.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(200px,1fr));gap:12px}
label{font-size:12px;color:var(--muted);display:block;margin-bottom:4px}
input,select{width:100%;padding:8px;border-radius:8px;border:1px solid #2a3340;background:#0e1116;color:var(--ink)}
button{background:var(--acc);border:0;border-radius:8px;padding:10px 18px;font-weight:700;cursor:pointer;color:#06281a}
button.ghost{background:transparent;border:1px solid #2a3340;color:var(--ink)}
.gauge{height:22px;border-radius:11px;background:#222a35;overflow:hidden;margin:12px 0}
.gauge>div{height:100%;border-radius:11px;transition:width .5s}
.band{font-size:28px;font-weight:800}
.rec{font-size:18px;font-weight:700;margin:8px 0}
.detail{color:var(--muted)}
.clients button{margin:4px 8px 4px 0}
table{width:100%;border-collapse:collapse;font-size:14px}
th,td{text-align:left;padding:8px;border-bottom:1px solid #2a3340}
.note{font-size:12px;color:var(--muted)}
</style>
</head>
<body><div class="wrap">
<h1>LatePay 💸</h1>
<p class="sub">Will this invoice be paid late? TabPFN — an open-source tabular foundation model — scores every outstanding invoice, so you can act before cash flow hurts.</p>
<div class="sim">⚠️ <b>SIMULATED training data:</b> the model was trained on a locally generated synthetic invoice history (4,000 invoices, seeded RNG; 1,000-row stratified context at inference). Scores are illustrative, not financial advice.</div>

<div class="card">
<h3>Score an invoice</h3>
<div class="clients" id="samples"></div>
<div class="grid">
<div><label>Amount (USD)</label><input id="amount_usd" type="number" value="2500"></div>
<div><label>Client tenure (days)</label><input id="client_tenure_days" type="number" value="120"></div>
<div><label>Prior invoices</label><input id="invoices_total" type="number" value="4"></div>
<div><label>Prior late rate (0–1)</label><input id="late_rate" type="number" step="0.01" value="0.25"></div>
<div><label>Avg days to pay (prior)</label><input id="avg_days_to_pay" type="number" value="41"></div>
<div><label>Contract</label><select id="contract_strength"><option value="0">None</option><option value="1" selected>Verbal/email</option><option value="2">Written</option></select></div>
<div><label>Payment terms (days)</label><select id="payment_terms_days"><option>7</option><option>15</option><option selected>30</option><option>45</option><option>60</option></select></div>
<div><label>Client size</label><select id="client_size"><option value="0">Individual</option><option value="1" selected>Small business</option><option value="2">Mid-market</option><option value="3">Enterprise</option></select></div>
<div><label>International client?</label><select id="is_international"><option value="0" selected>No</option><option value="1">Yes</option></select></div>
</div>
<p><button onclick="score()">Score invoice</button></p>
<div id="result" style="display:none">
<div class="band" id="band"></div>
<div class="gauge"><div id="bar"></div></div>
<div id="pct" style="font-size:20px"></div>
<div class="rec" id="rec"></div>
<div class="detail" id="det"></div>
<p class="note" id="meta"></p>
</div>
</div>

<div class="card">
<h3>Batch scoring — upload a CSV</h3>
<p class="note">Columns: client_name, invoice_id, amount_usd, plus any of client_tenure_days, invoices_total, late_rate, avg_days_to_pay, contract_strength, payment_terms_days, client_size, is_international. Missing history columns default to a new-client baseline.</p>
<form id="csvform"><input type="file" id="csvfile" accept=".csv"><button type="submit">Score CSV</button></form>
</div>

<div class="card">
<h3>How it works</h3>
<ol>
<li><b>Invoice features</b> — amount, client history, contract strength, terms.</li>
<li><b>TabPFN</b> (open-source tabular foundation model, local CPU, no API key) estimates P(paid &gt; 30 days past terms) from 4,000 simulated invoices.</li>
<li><b>Action</b> — chase now / partial upfront / reminder / normal terms.</li>
</ol>
<p class="note">Built Oct 12–18, 2026 for the DEV Hacktoberfest AI Challenge (Week 2: Side Hustle). MIT licensed.</p>
</div>
</div>
<script>
const FIELDS=["amount_usd","client_tenure_days","invoices_total","late_rate","avg_days_to_pay","contract_strength","payment_terms_days","client_size","is_international"];
const SAMPLES=[
 {label:"Reliable repeat client",v:{amount_usd:3200,client_tenure_days:640,invoices_total:18,late_rate:0,avg_days_to_pay:26,contract_strength:2,payment_terms_days:30,client_size:2,is_international:0}},
 {label:"Risky new overseas client",v:{amount_usd:15000,client_tenure_days:0,invoices_total:0,late_rate:0,avg_days_to_pay:30,contract_strength:0,payment_terms_days:30,client_size:1,is_international:1}},
 {label:"Patchy local studio",v:{amount_usd:1800,client_tenure_days:210,invoices_total:6,late_rate:0.5,avg_days_to_pay:52,contract_strength:1,payment_terms_days:15,client_size:1,is_international:0}}
];
const sc=document.getElementById("samples");
SAMPLES.forEach(s=>{const b=document.createElement("button");b.className="ghost";b.textContent=s.label;b.onclick=()=>{FIELDS.forEach(f=>{document.getElementById(f).value=s.v[f]??document.getElementById(f).value});score()};sc.appendChild(b)});
function color(band){return band==="HIGH"?"var(--bad)":band==="MEDIUM"?"var(--warn)":"var(--acc)"}
async function score(){
 const body={};FIELDS.forEach(f=>{body[f]=document.getElementById(f).value});
 const r=await fetch("/api/risk",{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify(body)}).then(r=>r.json());
 document.getElementById("result").style.display="block";
 const b=document.getElementById("band");b.textContent=r.risk_band+" RISK";b.style.color=color(r.risk_band);
 const bar=document.getElementById("bar");bar.style.width=r.late_probability_pct+"%";bar.style.background=color(r.risk_band);
 document.getElementById("pct").textContent=r.late_probability_pct+"% chance of being paid >30 days past terms";
 document.getElementById("rec").textContent="→ "+r.recommendation;
 document.getElementById("det").textContent=r.recommendation_detail;
 document.getElementById("meta").textContent="TabPFN, local CPU · trained on "+r.training_invoices+" simulated invoices · "+r.data_note;
}
document.getElementById("csvform").onsubmit=async e=>{e.preventDefault();const f=document.getElementById("csvfile").files[0];if(!f)return alert("Pick a CSV first");const fd=new FormData();fd.append("file",f);const r=await fetch("/api/batch",{method:"POST",body:fd});if(!r.ok){alert(await r.text());return}const blob=await r.blob();const a=document.createElement("a");a.href=URL.createObjectURL(blob);a.download="latepay_scored.csv";a.click()};
</script>
</body></html>
"""


if __name__ == "__main__":
    app.run(debug=True)
