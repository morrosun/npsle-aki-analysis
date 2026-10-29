# -*- coding: utf-8 -*-
"""Build v10 figures:
 1) figure1_combined.png  -> Panel A cohort flow, Panel B prevalence
 2) figure2_aki_core.png   -> Panel A MIMIC-IV core estimates, Panel B external/pooled AKI
Both written to analysis/ (where md_to_docx.py reads images).
"""
import os, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from PIL import Image
import numpy as np

ANA = "analysis"
OUT = ANA

# ---------------------------------------------------------------
# 1) Combined Figure 1 (composite of existing flow + prevalence)
# ---------------------------------------------------------------
flow = Image.open(f"{ANA}/flow.png").convert("RGB")
prev = Image.open(f"{ANA}/prevalence.png").convert("RGB")

from PIL import ImageDraw, ImageFont
def rescale(img, target_w):
    if img.width == target_w:
        return img
    r = target_w / img.width
    return img.resize((target_w, int(round(img.height * r))), Image.LANCZOS)

def add_label(img, letter, fs):
    """Return a copy of img with a bold black letter (A/B) at top-left,
    using a caller-supplied font size so both panels match exactly."""
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
    d.rectangle([off, off, off + tw + 2*pad, off + th + 2*pad], fill="white")
    d.text((off + pad, off + pad), letter, fill="black", font=fnt)
    return out

# scale both panels to the same width FIRST, then apply one shared label size
W = max(flow.width, prev.width)
flow_r = rescale(flow, W)
prev_r = rescale(prev, W)
fs = max(60, int(W * 0.055))   # identical font size for A and B
flow_l = add_label(flow_r, "A", fs)
prev_l = add_label(prev_r, "B", fs)

# stack vertically
pad = 24
H = flow_l.height + prev_l.height + pad
combined = Image.new("RGB", (W, H), (255, 255, 255))
combined.paste(flow_l, (0, 0))
combined.paste(prev_l, (0, flow_l.height + pad))
combined.save(f"{OUT}/figure1_combined.png", dpi=(300, 300))
print("wrote figure1_combined.png", combined.size)

# ---------------------------------------------------------------
# 2) Enhanced Figure 2 (2-panel forest)
# ---------------------------------------------------------------
NAVY = "#2166ac"

def forest_panel(ax, rows, xlim):
    """rows: list of (label, OR, lo, hi). Plots on log x-scale."""
    ys = list(range(len(rows)))[::-1]  # top-to-bottom
    for y, (lab, o, lo, hi) in zip(ys, rows):
        ax.plot([lo, hi], [y, y], color="#444444", lw=1.4, zorder=2)
        ax.scatter([o], [y], s=42, color=NAVY, zorder=3, marker="s")
        ax.annotate(f"{o:.2f} ({lo:.2f}\u2013{hi:.2f})",
                    (hi, y), textcoords="offset points", xytext=(8, 0),
                    va="center", fontsize=8.5, color="#222222")
        ax.text(-0.02, y, lab, ha="right", va="center", fontsize=9,
                transform=ax.get_yaxis_transform())
    ax.axvline(1.0, color="#999999", ls="--", lw=1.0, zorder=1)
    ax.set_yticks(ys)
    ax.set_yticklabels([])
    ax.set_xscale("log")
    ax.set_xlim(xlim)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", length=0)
    ax.set_xticks([0.5, 1, 2, 4, 8])
    ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax.set_xticklabels(["0.5", "1", "2", "4", "8"])

panelA = [
    ("Primary adjusted", 2.97, 1.58, 5.59),
    ("PSM matched", 3.44, 1.64, 7.23),
    ("OASIS-adjusted", 2.34, 1.21, 4.53),
    ("Severe AKI (KDIGO \u22652)", 3.07, 1.85, 5.12),
    ("Early eGFR <60 excluded", 3.27, 1.45, 7.37),
]
panelB = [
    ("MIMIC-IV (age/sex aOR)", 2.94, 1.59, 5.44),
    ("eICU (age/sex aOR)", 3.17, 1.01, 10.00),
    ("Pooled (random-effects)", 2.99, 1.74, 5.14),
]

fig, (a1, a2) = plt.subplots(1, 2, figsize=(7.4, 4.6))
forest_panel(a1, panelA, (0.5, 12))
forest_panel(a2, panelB, (0.4, 10))
fig.text(0.5, 0.02, "Odds ratio (log scale); 95% CI. Reference line at OR = 1.",
         ha="center", fontsize=8, color="#555555")
plt.tight_layout(rect=[0, 0.07, 1, 0.93])

# Place A/B at the TOP-LEFT corner OUTSIDE each subplot, using the actual
# rendered bounding box of each axes so the letters never drift to the middle.
fig.canvas.draw()
renderer = fig.canvas.get_renderer()
inv = fig.transFigure.inverted()
for ax, letter in ((a1, "A"), (a2, "B")):
    bb = ax.get_tightbbox(renderer)
    x0, y1 = inv.transform((bb.x0, bb.y1))
    fig.text(x0, y1 + 0.012, letter, fontsize=16, fontweight="bold",
             va="bottom", ha="left", color="#111111")

plt.savefig(f"{OUT}/figure2_aki_core.png", dpi=300, bbox_inches="tight", facecolor="white")
plt.close()
print("wrote figure2_aki_core.png")
