"""Local review page linking real native renders and measured experiment artifacts."""

import json
from pathlib import Path

root = Path("artifacts/generalization-20260905")
rows = json.loads((root / "resolution-throughput.json").read_text())
sections = []
for family in ("utility_plant", "data_center", "forest"):
    items = [r for r in rows if r["family"] == family]
    cards = "".join(
        f'<figure><img src="{family}-{r["resolution"][0]}.png"><figcaption>{r["resolution"][0]} × {r["resolution"][1]} · {r["frames_per_s"]:.1f} frames/s</figcaption></figure>'
        for r in items
    )
    sections.append(
        f'<section><h2>{family.replace("_", " ").title()}</h2><div class="grid">{cards}</div></section>'
    )
html = (
    """<!doctype html><html><meta charset="utf-8"><title>FlightRL environment review</title><style>
body{margin:0;background:#111820;color:#e2eaf0;font:16px system-ui;padding:40px;max-width:1400px;margin:auto}h1{font-size:32px}p{max-width:850px;line-height:1.6;color:#adbdc9}h2{font-size:24px;margin-top:40px}.grid{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:18px}figure{margin:0;background:#1c2730;border:1px solid #344550;border-radius:10px;overflow:hidden}img{width:100%;display:block}figcaption{padding:14px}a{color:#83d4de}@media(max-width:700px){body{padding:16px}.grid{grid-template-columns:1fr}}</style>
<h1>Environment and resolution review</h1><p>Actual native RGB-D camera renders at the same pose. The data center has artificial lighting and no windows. The forest uses analytic trees and rocks, procedural materials and direct sunlight. These are procedural research scenes, not photorealistic assets. Collision remains conservative; forest navigation is bounded to the authored plot.</p><p><a href="baseline-throughput.json">Baseline stage timings</a> · <a href="resolution-throughput.json">Resolution timings</a> · <a href="training/evaluation.json">Generalization results</a></p>"""
    + "".join(sections)
    + "</html>"
)
report = json.loads((root / "training/evaluation.json").read_text())
summary = "<h2>Held-out navigation</h2><p>One unseen seed per family; targets reached out of three. No collisions in these tests. Both policies used the same classical mission planner and safety supervisor.</p><table><tr><th>Environment</th><th>Classical</th><th>Plant-only</th><th>Mixed indoor</th></tr>"
for family in ("utility_plant", "data_center", "forest"):
    counts = [
        round(report["results"][f"{arm}_test/{family}/100"]["coverage"] * 3)
        for arm in ("classical", "plant_only", "mixed_indoor")
    ]
    summary += (
        f"<tr><td>{family.replace('_', ' ')}</td>"
        + "".join(f"<td>{count}/3</td>" for count in counts)
        + "</tr>"
    )
summary += '</table><p>Mixed training did not improve this pilot. Forest was excluded from training and validation; partial transfer is not proof of robust generalization.</p><p><a href="summary.md">Full findings and throughput</a> · <a href="current-throughput.json">Current stage timings</a> · <a href="training/stress-results.json">Visibility and recovery probes</a></p>'
html = html.replace(
    "</style>",
    "table{border-collapse:collapse}th,td{padding:12px 22px;text-align:left;border-bottom:1px solid #344550}</style>",
).replace("<section>", summary + "<section>", 1)
quality = """<section><h2>Detailed forest renderer</h2><p><a href="/forest.html">Open interactive forest camera and observer view</a>. New WebGPU geometry, textured soil, canopy shadows and animated leaves. These 1536 × 1152 RGB renders use recorded poses; the training results below used the original native camera.</p><div class="grid"><figure><img src="/forest-quality/camera-0.png"><figcaption>Recorded drone camera · departure</figcaption></figure><figure><img src="/forest-quality/observer-0.png"><figcaption>Interactive observer view</figcaption></figure></div></section>"""
html = html.replace("<h2>Held-out navigation</h2>", quality + "<h2>Held-out navigation</h2>")
(root / "review.html").write_text(html)
