# -*- coding: utf-8 -*-
"""Figure: 时序结构 + 双向时变 Cox（v21 新增图 3）"""
import os, warnings, json
import numpy as np, pandas as pd, psycopg2
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

warnings.filterwarnings("ignore")
mpl.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8.2,
                     "axes.linewidth": 0.8, "axes.edgecolor": "#333333",
                     "savefig.dpi": 300})

BASE = os.environ.get("SLECI_BASE", os.environ.get("SLECI_BASE", "."))
ANA = os.path.join(BASE, "analysis")
PG = dict(host="localhost", port=5432, user="postgres",
          password=os.environ.get("SLECI_PGPASSWORD", ""), dbname="mimiciv")

df = pd.read_csv(os.path.join(BASE, "data", "mimiciv_sle.csv"))
ids = df["stay_id"].astype("int64").tolist()
con = psycopg2.connect(**PG)
icu = pd.read_sql("SELECT stay_id,intime,outtime FROM mimiciv_icu.icustays WHERE stay_id=ANY(%s)", con, params=(ids,))
kd = pd.read_sql("SELECT stay_id,charttime,aki_stage FROM mimiciv_derived.kdigo_stages WHERE stay_id=ANY(%s)", con, params=(ids,))
cam = pd.read_sql("SELECT stay_id,charttime,value FROM mimiciv_icu.chartevents WHERE stay_id=ANY(%s) AND itemid=228332", con, params=(ids,))
con.close()
for d in (kd, cam): d["charttime"] = pd.to_datetime(d["charttime"])
icu["intime"] = pd.to_datetime(icu["intime"])

kd = kd.merge(icu[["stay_id", "intime"]], on="stay_id", how="left")
kd["h"] = (kd["charttime"] - kd["intime"]).dt.total_seconds() / 3600
t_aki = kd[(kd["aki_stage"] >= 1) & (kd["h"] >= 0)].groupby("stay_id")["h"].min()

cam["h"] = (cam["charttime"] - cam["stay_id"].map(icu.set_index("stay_id")["intime"])).dt.total_seconds() / 3600
cam["v"] = cam["value"].astype(str).str.strip().str.lower()
# v28：CAM 记录限制在 intime–outtime（与结局定义一致；此前未限制）
icu["outtime"] = pd.to_datetime(icu["outtime"])
_te = ((icu["outtime"] - icu["intime"]).dt.total_seconds() / 3600)
_te.index = icu["stay_id"].values
cam["t_end"] = cam["stay_id"].map(_te)
cam = cam[(cam["h"] >= 0) & (cam["h"] <= cam["t_end"])]
t_pos = cam[cam.v == "positive"].groupby("stay_id")["h"].min()

d = pd.DataFrame({"t_aki": t_aki, "t_cam": t_pos}).dropna()
d["gap"] = d["t_aki"] - d["t_cam"]
# 排序分析限于入 ICU 前无 AKI 记录者：否则 25 例的 ICU 内记录不是 onset
_pre = set(kd[(kd["aki_stage"] >= 1) & (kd["h"] < 0)].stay_id)
d_all = d.copy()
d = d[~d.index.isin(_pre)]
print("共现 %d 例（全部）；其中入 ICU 前无 AKI %d 例（作图用）" % (len(d_all), len(d)))
print("  全部：谵妄在前 %d，同期 %d，AKI 在前 %d"
      % ((d_all.gap > 6).sum(), (d_all.gap.abs() <= 6).sum(), (d_all.gap < -6).sum()))
print("  作图：谵妄在前 %d，同期 %d，AKI 在前 %d"
      % ((d.gap > 6).sum(), (d.gap.abs() <= 6).sum(), (d.gap < -6).sum()))

C_AKI = "#d6604d"; C_CAM = "#2166ac"; C_SAME = "#9e9e9e"; C_TXT = "#1a1a1a"

fig = plt.figure(figsize=(7.2, 3.9))
gs = fig.add_gridspec(1, 2, width_ratios=[1.15, 1], wspace=0.36,
                      left=0.085, right=0.975, top=0.90, bottom=0.28)

# ---- Panel A: gap distribution
ax = fig.add_subplot(gs[0, 0])
bins = np.arange(-240, 265, 12)
ax.axvspan(-240, -6, color=C_AKI, alpha=0.13, lw=0)
ax.axvspan(-6, 6, color=C_SAME, alpha=0.18, lw=0)
ax.axvspan(6, 264, color=C_CAM, alpha=0.13, lw=0)
ax.hist(d.gap, bins=bins, color="#4d4d4d", edgecolor="white", linewidth=0.6)
ax.axvline(0, color="#1a1a1a", lw=1.1, ls="--")
ax.set_xlim(-240, 264)
ax.set_xlabel("From first positive CAM-ICU to first qualifying AKI record (h)", fontsize=8.2)
ax.set_ylabel("Number of ICU stays", fontsize=8.2)
ax.set_xticks([-240, -180, -120, -60, 0, 60, 120, 180, 240])
ax.tick_params(labelsize=7.6)

n_aki = int((d.gap < -6).sum()); n_same = int((d.gap.abs() <= 6).sum()); n_cam = int((d.gap > 6).sum())
tot = len(d)
ax.annotate("", xy=(-172, 13.6), xytext=(-26, 13.6),
            arrowprops=dict(arrowstyle="->", color=C_AKI, lw=1.3,
                            shrinkA=0, shrinkB=0))
ax.text(-98, 14.6, f"AKI first\n{n_aki}/{tot} ({100*n_aki/tot:.0f}%)",
        ha="center", va="bottom", fontsize=7.6, color=C_AKI, fontweight="bold")
ax.text(132, 14.6, f"Delirium first\n{n_cam}/{tot} ({100*n_cam/tot:.0f}%)",
        ha="center", va="bottom", fontsize=7.6, color=C_CAM, fontweight="bold")
ax.set_ylim(0, 21.5)
for s in ("top", "right"): ax.spines[s].set_visible(False)
ax.set_title("A   Recorded sequence, no AKI before admission", loc="left",
             fontsize=9.0, fontweight="bold", color=C_TXT, pad=6)

# ---- Panel B: bidirectional time-varying Cox
ax2 = fig.add_subplot(gs[0, 1])
rows = [
    ("Delirium → AKI", 1.07, 0.74, 1.54, C_CAM),
    ("  + vasopressor / MV", 0.87, 0.60, 1.27, C_CAM),
    ("  + sepsis", 0.83, 0.56, 1.21, C_CAM),
    ("  KDIGO ≥2", 1.50, 1.03, 2.17, C_CAM),
    ("AKI → delirium", 1.84, 1.05, 3.21, C_AKI),
    ("  + vasopressor / MV", 1.48, 0.83, 2.65, C_AKI),
    ("  + sepsis", 1.41, 0.79, 2.50, C_AKI),
    ("  First 48 h", 2.41, 1.29, 4.50, C_AKI),
    ("  After 48 h", 0.87, 0.35, 2.21, C_AKI),
]
y = np.arange(len(rows))[::-1]
for (lab, hr, lo, hi, c), yy in zip(rows, y):
    ax2.plot([lo, hi], [yy, yy], color=c, lw=1.5, solid_capstyle="round")
    ax2.plot([hr], [yy], "s", color=c, ms=5.2, mec="white", mew=0.8, zorder=3)
ax2.axvline(1, color="#555555", lw=0.9, ls=":")
ax2.set_yticks(y); ax2.set_yticklabels([r[0] for r in rows], fontsize=7.6)
ax2.set_xscale("log")
ax2.set_xlim(0.22, 5.2)
ax2.set_xticks([0.25, 0.5, 1, 2, 4])
ax2.set_xticklabels(["0.25", "0.5", "1", "2", "4"], fontsize=7.4)
ax2.xaxis.set_minor_locator(mpl.ticker.NullLocator())
ax2.xaxis.set_minor_formatter(mpl.ticker.NullFormatter())
ax2.set_xlabel("Hazard ratio (95% CI, log scale)", fontsize=8.2)
for s in ("top", "right"): ax2.spines[s].set_visible(False)
ax2.set_title("B   Time-varying Cox, both directions", loc="left",
              fontsize=9.0, fontweight="bold", color=C_TXT, pad=6)
ax2.set_ylim(-0.65, len(rows) - 0.35)
ax2.grid(axis="x", color="#e6e6e6", lw=0.6)
ax2.set_axisbelow(True)

fig.text(0.085, 0.055,
         "Panel A: horizontal axis is the first qualifying ICU-window AKI record minus the first positive CAM-ICU; negative values indicate that AKI came first. Stays with AKI documented before admission are excluded.\n"
         "Panel B: exposure switches from 0 to 1 at the first recorded event; censoring at ICU discharge (forward) or at the last CAM-ICU assessment (reverse).",
         fontsize=6.4, color="#555555", va="bottom", linespacing=1.5)

out = os.path.join(ANA, "figure3_temporal.png")
fig.savefig(out, facecolor="white", bbox_inches="tight", pad_inches=0.06)
print("[saved]", out)
