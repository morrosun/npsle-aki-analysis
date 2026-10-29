# -*- coding: utf-8 -*-
"""v33：重画 Figure S3（敏感性森林图）。

只改图内标签与留白，数字沿用正文已定稿值：
  primary 2.97 (1.58-5.59)；KDIGO >=2 3.07 (1.85-5.12)；
  early eGFR <60 excluded 3.27 (1.45-7.37)
原图把主模型标为 "full-adj (9-covariate)"，与正文六协变量主模型不符（9 是 PSM 协变量数），
此处改为 "six covariates"。
"""
import os
import shutil
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE = Path(os.environ.get("SLECI_BASE", "."))
OUT = BASE / "analysis"

ROWS = [
    ("AKI (KDIGO \u22651), primary\nsix covariates", 2.97, 1.58, 5.59, "#b2182b", "solid", False),
    ("AKI (KDIGO \u22652)\nsix covariates", 3.07, 1.85, 5.12, "#2166ac", "solid", False),
    ("AKI (\u22651), early eGFR <60 excluded\nsix covariates", 3.27, 1.45, 7.37, "#7b3294", ":", True),
]

src = OUT / "forest_sens.png"
bak = OUT / "forest_sens_v32.png"
if src.exists() and not bak.exists():
    shutil.copyfile(src, bak)

fig, ax = plt.subplots(figsize=(7.4, 3.1))
yy = list(range(len(ROWS)))[::-1]
for i, (lab, o, lo, hi, col, ls, open_m) in enumerate(ROWS):
    ax.plot([lo, hi], [yy[i], yy[i]], color=col, lw=1.6, ls=ls)
    if open_m:
        ax.plot([o], [yy[i]], "D", color="none", markeredgecolor=col, ms=9, mew=1.6)
    else:
        ax.plot([o], [yy[i]], "D", color=col, ms=9)
    ax.text(hi * 1.06, yy[i], f"{o:.2f} ({lo:.2f}\u2013{hi:.2f})",
            va="center", fontsize=8.5, color=col)

ax.text(0.02, 0.13, "[wide CI \u2014 see text]", fontsize=7.5,
        color="#7b3294", style="italic", transform=ax.transAxes)

ax.axvline(1, color="gray", ls="--", lw=1)
ax.set_yticks(yy)
ax.set_yticklabels([r[0] for r in ROWS])
ax.set_xscale("log")
ax.set_xticks([0.5, 1, 2, 4, 8])
ax.set_xticklabels(["0.5", "1", "2", "4", "8"])
ax.set_xlim(0.33, 20)
ax.set_xlabel("Adjusted odds ratio (95% CI), MIMIC-IV derivation cohort")
ax.set_title("Sensitivity analyses for the coded-involvement\u2013AKI association",
             fontweight="bold", fontsize=11)
plt.tight_layout()
plt.savefig(OUT / "forest_sens.png", dpi=150, bbox_inches="tight")
plt.close()
print("[saved]", OUT / "forest_sens.png")
