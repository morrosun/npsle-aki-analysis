# -*- coding: utf-8 -*-
"""v18: redraw Panel B (prevalence) of Figure 1.

Change vs v17: NWICU is retained for descriptive comparison but is no longer
entered into any pooled estimate (random-effects with k=3 gave a 49-stay
cohort ~24% of the pooling weight). Its bars are therefore drawn in grey
(carrying the same light/dark ordering as the narrow/broad pair) and marked
with an asterisk, and a footnote states the pooling rule.

Layout, colours, fonts and bar geometry are kept identical to analyze.py so the
panel still matches the v10/v17 look. Backups: prevalence_v17.png,
figure1_combined_v17.png.
"""
import json, os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

BASE = os.environ.get("SLECI_BASE", ".")
ANA = os.path.join(BASE, "analysis")

with open(os.path.join(ANA, "results.json"), encoding="utf-8") as f:
    R = json.load(f)
prev = R["prevalence"]

C_MIMIC = "#2166ac"
C_BROAD = "#d6604d"
# descriptive (not pooled): grey ramp, same light/dark ordering as the
# narrow/broad pair above so the two bars stay tellable apart
C_DESC_N = "#8f8f8f"
C_DESC_B = "#c9c9c9"

dbs = ["MIMIC-IV", "NWICU", "eICU"]
narr = [prev[d]["narrow_pct"] for d in dbs]
barr = [prev[d]["broad_pct"] for d in dbs]
x = np.arange(len(dbs))
w = 0.36

fig, ax = plt.subplots(figsize=(7.0, 4.2))

for i, d in enumerate(dbs):
    desc = (d == "NWICU")
    cn, cb = (C_DESC_N, C_DESC_B) if desc else (C_MIMIC, C_BROAD)
    lab_n = "Narrow (cognitive/delirium)" if i == 0 else None
    lab_b = "Broad (full NPSLE)" if i == 0 else None
    ax.bar(x[i] - w / 2, narr[i], w, color=cn, label=lab_n)
    ax.bar(x[i] + w / 2, barr[i], w, color=cb, label=lab_b)
    ax.text(i - w / 2, narr[i] + 0.6, f"{narr[i]}%", ha="center", fontsize=8.5)
    ax.text(i + w / 2, barr[i] + 0.6, f"{barr[i]}%", ha="center", fontsize=8.5)

labels = [f"{d}{'*' if d == 'NWICU' else ''}\n(n={prev[d]['n']})" for d in dbs]
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.set_ylabel("% of SLE ICU stays")
ax.legend()
ax.set_ylim(0, max(barr) + 6)
ax.set_title("Prevalence of neuropsychiatric involvement in SLE ICU stays",
             fontweight="bold")

pn = R["prevalence_pooled_narrow"]
pb = R["prevalence_pooled_broad"]
fig.text(0.5, 0.02,
         f"* NWICU is descriptive only and is not included in the pooled estimates. "
         f"Pooled (MIMIC-IV + eICU, random effects): narrow {pn['pct']}% "
         f"({pn['lo']}\u2013{pn['hi']}), broad {pb['pct']}% ({pb['lo']}\u2013{pb['hi']}).",
         ha="center", fontsize=7.8, color="#444444")

plt.tight_layout()
plt.subplots_adjust(bottom=0.22)
plt.savefig(os.path.join(ANA, "prevalence.png"), dpi=300, facecolor="white",
            bbox_inches="tight")
plt.close()
print("wrote prevalence.png (v18, NWICU descriptive)")
