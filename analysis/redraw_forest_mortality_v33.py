# -*- coding: utf-8 -*-
"""v33：重画 Figure S2 的年龄/性别调整版（forest_mortality_adj.png）。

修三处与正文冲突的地方：
  1) 标题超出画布被截断；
  2) 图内印有异质性统计量（0.0%），而正文明确"two databases，不报告异质性统计量"；
  3) 主模型行写作 "full-adj (9-covariate)"，正文为六协变量（9 是 PSM 协变量数）。

数字沿用正文：MIMIC-IV 2.06 (1.09-3.89)；eICU 2.25 (0.56-9.08)；
pooled 2.09 (1.17-3.73)；MIMIC-IV 六协变量 1.85 (0.95-3.60)。
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
    ("MIMIC-IV (n = 550)", 2.06, 1.09, 3.89, "#2166ac", "o", False),
    ("eICU (n = 210)", 2.25, 0.56, 9.08, "#1b7837", "o", False),
    ("Pooled, random effects", 2.09, 1.17, 3.73, "#b2182b", "D", True),
    ("MIMIC-IV, six covariates", 1.85, 0.95, 3.60, "#666666", "o", False),
]

src = OUT / "forest_mortality_adj.png"
bak = OUT / "forest_mortality_adj_v32.png"
if src.exists() and not bak.exists():
    shutil.copyfile(src, bak)

fig, ax = plt.subplots(figsize=(7.4, 3.4))
yy = list(range(len(ROWS)))[::-1]
for i, (lab, o, lo, hi, col, mk, big) in enumerate(ROWS):
    ax.plot([lo, hi], [yy[i], yy[i]], color=col, lw=1.8 if big else 1.5)
    ax.plot([o], [yy[i]], mk, color=col, ms=10 if big else 7)
    ax.text(hi * 1.06, yy[i], f"{o:.2f} ({lo:.2f}\u2013{hi:.2f})",
            va="center", fontsize=8.5, color=col)

ax.axvline(1, color="gray", ls="--", lw=1)
ax.set_yticks(yy)
ax.set_yticklabels([r[0] for r in ROWS])
ax.set_xscale("log")
ax.set_xticks([0.5, 1, 2, 4, 8])
ax.set_xticklabels(["0.5", "1", "2", "4", "8"])
ax.set_xlim(0.3, 34)
ax.set_xlabel("Adjusted odds ratio (95% CI) for in-hospital mortality")
ax.set_title("Narrow NPSLE and in-hospital mortality (age/sex-adjusted)",
             fontweight="bold", fontsize=11)
plt.tight_layout()
plt.savefig(OUT / "forest_mortality_adj.png", dpi=150, bbox_inches="tight")
plt.close()
print("[saved]", OUT / "forest_mortality_adj.png")
