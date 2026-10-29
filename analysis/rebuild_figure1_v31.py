# -*- coding: utf-8 -*-
"""v31: redraw Panel B (prevalence) of Figure 1 and rebuild the composite.

Change vs v18: the broad pooled prevalence is no longer shown. Broad lists are
not harmonized between MIMIC-IV and eICU (eICU adds CNS infection), so broad
proportions are reported by database only and the footnote says so. The narrow
pooled estimate is unchanged (11.7%, 5.3-24.1).

Panel A (flow.png) is produced by analysis/redraw_flow.py, which was rerun with
the same change. This script only redraws Panel B and re-stitches the composite,
so figure2_aki_core.png is deliberately NOT touched.

Backup of the previous Panel B: prevalence_v18.png (made by the v18 script).
"""
import json
import os
import shutil

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw, ImageFont

BASE = os.environ.get("SLECI_BASE", os.environ.get("SLECI_BASE", "."))
ANA = os.path.join(BASE, "analysis")

with open(os.path.join(ANA, "results.json"), encoding="utf-8") as f:
    R = json.load(f)
prev = R["prevalence"]

C_MIMIC = "#2166ac"
C_BROAD = "#d6604d"
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
    lab_n = "Narrow coded involvement" if i == 0 else None
    lab_b = "Broad coded involvement" if i == 0 else None
    ax.bar(x[i] - w / 2, narr[i], w, color=cn, label=lab_n)
    ax.bar(x[i] + w / 2, barr[i], w, color=cb, label=lab_b)
    ax.text(i - w / 2, narr[i] + 0.6, f"{narr[i]}%", ha="center", fontsize=8.5)
    ax.text(i + w / 2, barr[i] + 0.6, f"{barr[i]}%", ha="center", fontsize=8.5)

labels = [f"{d}{'*' if d == 'NWICU' else ''}\n(n={prev[d]['n']})" for d in dbs]
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.set_ylabel("% of SLE ICU stays")
ax.legend(loc="upper right", fontsize=8)
ax.set_ylim(0, max(barr) + 10)
ax.set_title("Prevalence of neuropsychiatric involvement in SLE ICU stays",
             fontweight="bold")

pn = R["prevalence_pooled_narrow"]
fig.text(0.5, 0.02,
         f"* NWICU is descriptive only and is not included in any pooled estimate. "
         f"Pooled (MIMIC-IV + eICU, random effects), narrow definition only: {pn['pct']}% "
         f"({pn['lo']}\u2013{pn['hi']}). Broad proportions are shown by database and are not "
         f"pooled, because the broad code lists are not harmonized.",
         ha="center", fontsize=7.4, color="#444444")

plt.tight_layout()
plt.subplots_adjust(bottom=0.24)
plt.savefig(os.path.join(ANA, "prevalence.png"), dpi=300, facecolor="white",
            bbox_inches="tight")
plt.close()
print("wrote prevalence.png (v31, no pooled broad)")

# ---- re-stitch Panel A + Panel B, same geometry as make_figures_v10.py ----
flow = Image.open(os.path.join(ANA, "flow.png")).convert("RGB")
previmg = Image.open(os.path.join(ANA, "prevalence.png")).convert("RGB")


def rescale(img, target_w):
    if img.width == target_w:
        return img
    r = target_w / img.width
    return img.resize((target_w, int(round(img.height * r))), Image.LANCZOS)


def add_label(img, letter, fs):
    out = img.copy()
    d = ImageDraw.Draw(out)
    try:
        fnt = ImageFont.truetype("C:/Windows/Fonts/arialbd.ttf", fs)
    except Exception:
        fnt = ImageFont.load_default()
    bbox = d.textbbox((0, 0), letter, font=fnt)
    tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
    pad = int(fs * 0.18)
    off = int(fs * 0.12)
    d.rectangle([off, off, off + tw + 2 * pad, off + th + 2 * pad], fill="white")
    d.text((off + pad, off + pad), letter, fill="black", font=fnt)
    return out


W = max(flow.width, previmg.width)
flow_r = rescale(flow, W)
prev_r = rescale(previmg, W)
fs = max(60, int(W * 0.055))
flow_l = add_label(flow_r, "A", fs)
prev_l = add_label(prev_r, "B", fs)

pad = 24
H = flow_l.height + prev_l.height + pad
combined = Image.new("RGB", (W, H), (255, 255, 255))
combined.paste(flow_l, (0, 0))
combined.paste(prev_l, (0, flow_l.height + pad))
combined.save(os.path.join(ANA, "figure1_combined.png"), dpi=(300, 300))
print("wrote figure1_combined.png", combined.size)
