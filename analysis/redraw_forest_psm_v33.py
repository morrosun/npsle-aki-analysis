# -*- coding: utf-8 -*-
"""v33：重画 Figure 4（PSM 森林图），使图内数字与最终版 Table 3 一致。

不重跑 analyze.py（其结果 json 与正文的 v28/v29 审计口径不同），
本脚本只把已定稿的 Table 3 数值画出来。

Table 3（89 pairs，matched-pair conditional OR，Miettinen CI）：
  AKI KDIGO >=1   3.44 (1.64-7.23)  n10/n01 = 31/9
  AKI KDIGO >=2   3.08 (1.61-5.91)  n10/n01 = 37/12
  IMV             2.07 (1.09-3.92)
  Vasopressors    1.25 (0.65-2.41)
  Mortality       1.55 (0.72-3.30)
  RRT             描述性，不检验，无 OR -> 不画
"""
import os
import shutil
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE = Path(os.environ.get("SLECI_BASE", "."))
OUT = BASE / "analysis"
C_PRIMARY = "#b2182b"

ROWS = [
    ("AKI, KDIGO stage \u22651 (primary)", 3.44, 1.64, 7.23, C_PRIMARY),
    ("AKI, KDIGO stage \u22652", 3.08, 1.61, 5.91, C_PRIMARY),
    ("Invasive mechanical ventilation", 2.07, 1.09, 3.92, "#2166ac"),
    ("Vasopressors", 1.25, 0.65, 2.41, "#2166ac"),
    ("In-hospital mortality", 1.55, 0.72, 3.30, "#2166ac"),
]

src = OUT / "forest_psm.png"
bak = OUT / "forest_psm_v32.png"
if src.exists() and not bak.exists():
    shutil.copyfile(src, bak)

fig, ax = plt.subplots(figsize=(7.6, 3.5))
yy = list(range(len(ROWS)))[::-1]
for i, (lab, o, lo, hi, col) in enumerate(ROWS):
    ax.plot([lo, hi], [yy[i], yy[i]], color=col, lw=1.7)
    ax.plot([o], [yy[i]], "s", color=col, ms=7)
    ax.text(hi * 1.12, yy[i], f"{o:.2f} ({lo:.2f}\u2013{hi:.2f})", va="center", fontsize=8.5)

ax.axvline(1, color="gray", ls="--", lw=1)
ax.set_yticks(yy)
ax.set_yticklabels([r[0] for r in ROWS])
ax.set_xscale("log")
ax.set_xticks([0.5, 1, 2, 4, 8])
ax.set_xticklabels(["0.5", "1", "2", "4", "8"])
ax.set_xlim(0.4, 18)
ax.set_xlabel("Odds ratio (95% CI), propensity-matched cohort (89 pairs)")
ax.set_title("Narrow coded involvement and outcomes after propensity-score matching",
             fontweight="bold", fontsize=10.5)
plt.tight_layout()
plt.savefig(OUT / "forest_psm.png", dpi=150, bbox_inches="tight")
plt.close()
print("[saved]", OUT / "forest_psm.png")
print("[backup]", OUT / "forest_psm_v32.png")
