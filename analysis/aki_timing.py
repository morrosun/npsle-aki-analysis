# -*- coding: utf-8 -*-
"""
AKI 时序分层分析 v2（回应审稿意见 R1-C1：反向因果 / 时序不可判定）

关键设计
--------
1. kdigo_stages 覆盖**住院全程**（首评估中位在入 ICU 前 6.5 h），因此必须区分：
     - 入 ICU 前已发生（pre-ICU AKI）
     - 入 ICU 后 24/48/72 h 内
     - 其后新发
2. 为把「时间窗效应」与「亚组限制效应」分开，同时报告：
     (a) 全队列 任意 AKI     —— 复现主稿 aOR
     (b) 全队列 入 ICU 后 AKI
     (c) LOS>24h 亚组 任意 AKI  —— 单独看限制条件造成的影响
     (d) LOS>24h 亚组 ≤24h AKI
     (e) LOS>24h 且 24h 无 AKI → 其后新发  —— 真正的 landmark
3. CAM-ICU 带时间戳，做双向检验（谵妄→AKI / AKI→谵妄），仅限 24h 内确有评估者
"""
import io
import json
import os
import warnings

import numpy as np
import pandas as pd
import psycopg2
import statsmodels.formula.api as smf
from scipy import stats as sps
from statsmodels.duration.hazard_regression import PHReg

warnings.filterwarnings("ignore")

BASE = os.environ.get("SLECI_BASE", os.environ.get("SLECI_BASE", "."))
ANA = os.path.join(BASE, "analysis")
DATA = os.path.join(BASE, "data")
PG = dict(host="localhost", port=5432, user="postgres",
          password=os.environ.get("SLECI_PGPASSWORD", ""), dbname="mimiciv")

COV = ["age", "female", "charlson_norheum", "lupus_nephritis", "renal_disease", "diabetes"]
EXPO = "npsle_narrow"

log_lines = []


def log(s=""):
    print(s)
    log_lines.append(str(s))


R = {}

# ---------------------------------------------------------------- 1
df = pd.read_csv(os.path.join(DATA, "mimiciv_sle.csv"))
ids = df["stay_id"].astype("int64").tolist()
assert len(df) == 550
log(f"队列 n = {len(df)}；暴露(+) {int(df[EXPO].sum())}；AKI(+) {int(df['aki'].sum())} "
    f"({100*df['aki'].mean():.1f}%)")

con = psycopg2.connect(**PG)
kd = pd.read_sql("SELECT stay_id, charttime, aki_stage FROM mimiciv_derived.kdigo_stages "
                 "WHERE stay_id = ANY(%s) ORDER BY stay_id, charttime", con, params=(ids,))
kd["charttime"] = pd.to_datetime(kd["charttime"])
icu = pd.read_sql("SELECT stay_id, intime FROM mimiciv_icu.icustays WHERE stay_id = ANY(%s)",
                  con, params=(ids,))
cam = pd.read_sql("SELECT stay_id, charttime, value FROM mimiciv_icu.chartevents "
                  "WHERE stay_id = ANY(%s) AND itemid = 228332", con, params=(ids,))
con.close()
cam["charttime"] = pd.to_datetime(cam["charttime"])
log(f"kdigo 明细 {len(kd)} 行覆盖 {kd['stay_id'].nunique()}/550；CAM 评估 {len(cam)} 次覆盖 "
    f"{cam['stay_id'].nunique()}/550")

# ---------------------------------------------------------------- 2 合并
m = df.merge(icu, on="stay_id", how="left")
m["intime"] = pd.to_datetime(m["intime"])
m = m.merge(kd, on="stay_id", how="left", suffixes=("", "_k"))
m["h"] = (m["charttime"] - m["intime"]).dt.total_seconds() / 3600.0
m["icu_h"] = m["los_icu"] * 24.0

pos = m[m["aki_stage"] >= 1]
g = pos.groupby("stay_id")["h"].agg(["min", "max"])
m = m.drop(columns=["charttime", "aki_stage"]).drop_duplicates("stay_id")
m = m.merge(g.rename(columns={"min": "onset_h", "max": "last_aki_h"}), on="stay_id", how="left")

# ICU 后首达
pos_icu = pos[pos["h"] >= 0]
g2 = pos_icu.groupby("stay_id")["h"].min().rename("onset_icu_h")
m = m.merge(g2, on="stay_id", how="left")

# 入 ICU 前是否已 AKI
pre = pos[pos["h"] < 0].groupby("stay_id").size().rename("n_pre_aki")
m = m.merge(pre, on="stay_id", how="left")
m["pre_aki"] = m["n_pre_aki"].notna().astype(int)

log(f"\n入 ICU 前已有 KDIGO≥1 记录: {int(m['pre_aki'].sum())}/550 "
    f"({100*m['pre_aki'].mean():.1f}%)")
log(f"AKI 首达时点中位 {m['onset_h'].median():.1f} h；ICU 后首达中位 "
    f"{m['onset_icu_h'].median():.1f} h")

# ---------------------------------------------------------------- 3 表 1 分布


def band(row):
    if pd.isna(row["onset_h"]):
        return "0 无 AKI"
    if row["onset_h"] < 0:
        return "1 入科前已发生"
    if row["onset_h"] <= 6:
        return "2 ≤6 h"
    if row["onset_h"] <= 24:
        return "3 6–24 h"
    if row["onset_h"] <= 48:
        return "4 24–48 h"
    if row["onset_h"] <= 72:
        return "5 48–72 h"
    return "6 >72 h"


m["band"] = m.apply(band, axis=1)
order = ["0 无 AKI", "1 入科前已发生", "2 ≤6 h", "3 6–24 h", "4 24–48 h", "5 48–72 h", "6 >72 h"]
tab = pd.crosstab(m["band"], m[EXPO]).reindex(order).fillna(0).astype(int)
tab.columns = ["暴露(−)", "暴露(+)"]
tab["合计"] = tab.sum(axis=1)
tab["暴露(+)%"] = (100 * tab["暴露(+)"] / tab["合计"].replace(0, np.nan)).round(1)
log("\n" + "=" * 80)
log("表 1. AKI 首达时点分布（相对 ICU 入科）")
log("=" * 80)
log(tab.to_string())
R["dist"] = {k: [int(v["暴露(−)"]), int(v["暴露(+)"])] for k, v in tab.iterrows()}

# ---------------------------------------------------------------- 4 表 2 分解
def fit(d, outcome, expo=EXPO, covs=COV):
    d = d.dropna(subset=[outcome] + covs).copy()
    ne = int(d[outcome].sum())
    if ne < 5 or d[outcome].nunique() < 2 or d[expo].nunique() < 2:
        return {"insufficient": True, "n": int(len(d)), "events": ne}
    try:
        r = smf.logit(f"{outcome} ~ {expo} + " + " + ".join(covs), data=d).fit(disp=0)
    except Exception as e:
        return {"error": str(e)[:60]}
    ci = r.conf_int().loc[expo]
    return {"OR": round(float(np.exp(r.params[expo])), 2),
            "lo": round(float(np.exp(ci[0])), 2), "hi": round(float(np.exp(ci[1])), 2),
            "p": float(r.pvalues[expo]), "n": int(len(d)), "events": ne,
            "event_exp": int(d.loc[d[expo] == 1, outcome].sum()),
            "n_exp": int((d[expo] == 1).sum()),
            "event_unexp": int(d.loc[d[expo] == 0, outcome].sum()),
            "n_unexp": int((d[expo] == 0).sum())}


m["aki_any"] = m["onset_h"].notna().astype(int)
m["aki_icu"] = m["onset_icu_h"].notna().astype(int)
m["aki_24"] = ((m["onset_h"].notna()) & (m["onset_h"] >= 0) & (m["onset_h"] <= 24)).astype(int)
m["aki_48"] = ((m["onset_h"].notna()) & (m["onset_h"] >= 0) & (m["onset_h"] <= 48)).astype(int)

long24 = m[m["icu_h"] > 24].copy()
free24 = long24[(long24["onset_h"].isna()) | (long24["onset_h"] > 24)].copy()
free24["aki_after24"] = ((free24["onset_h"].notna()) & (free24["onset_h"] > 24)).astype(int)
long48 = m[m["icu_h"] > 48].copy()
free48 = long48[(long48["onset_h"].isna()) | (long48["onset_h"] > 48)].copy()
free48["aki_after48"] = ((free48["onset_h"].notna()) & (free48["onset_h"] > 48)).astype(int)

specs = [
    ("A 全队列 · 任意 AKI（复现主稿）", m, "aki_any"),
    ("B 全队列 · 入 ICU 后 AKI", m, "aki_icu"),
    ("C 全队列 · ≤24 h AKI", m, "aki_24"),
    ("D LOS>24h · 任意 AKI（仅限制条件）", long24, "aki_any"),
    ("E LOS>24h · ≤24 h AKI", long24, "aki_24"),
    ("F LOS>24h 且 24h 无 AKI → 其后新发", free24, "aki_after24"),
    ("G LOS>48h 且 48h 无 AKI → 其后新发", free48, "aki_after48"),
]
log("\n" + "=" * 80)
log("表 2. 分解：时间窗效应 vs 亚组限制效应（同一协变量集）")
log("=" * 80)
rows = []
for name, d, oc in specs:
    r = fit(d, oc)
    R[name.split()[0]] = r
    if r.get("insufficient"):
        rows.append([name, "样本不足", "—", "—", "—", f"{r['n']}（事件 {r['events']}）"])
    elif "error" in r:
        rows.append([name, "拟合失败", "—", "—", "—", r["error"]])
    else:
        rows.append([name, f"{r['OR']:.2f} ({r['lo']:.2f}–{r['hi']:.2f})", f"{r['p']:.3f}",
                     f"{r['event_exp']}/{r['n_exp']}", f"{r['event_unexp']}/{r['n_unexp']}",
                     f"{r['n']}（事件 {r['events']}）"])
log(pd.DataFrame(rows, columns=["模型", "aOR (95% CI)", "p", "暴露(+)", "暴露(−)", "n"]).to_string(index=False))

# ---------------------------------------------------------------- 4b 稳健性诊断
log("\n" + "=" * 80)
log("表 2b. 主结果稳健性诊断：是否由少数短停留患者支撑")
log("=" * 80)
short = m[m["icu_h"] <= 24]
long_ = m[m["icu_h"] > 24]
log(f"LOS ≤24 h: n={len(short)}（暴露(+) {int(short[EXPO].sum())}, "
    f"AKI {int(short['aki_any'].sum())}, 死亡 {int(short['mortality'].sum())}）")
log(f"LOS  >24 h: n={len(long_)}（暴露(+) {int(long_[EXPO].sum())}, "
    f"AKI {int(long_['aki_any'].sum())}, 死亡 {int(long_['mortality'].sum())}）")
log("\n短停留亚组内（n=%d）：" % len(short))
tab_s = pd.crosstab(short[EXPO], short["aki_any"])
log(tab_s.to_string())
if tab_s.shape == (2, 2):
    a_, b_ = tab_s.loc[1, 1], tab_s.loc[1, 0]
    c2, d2 = tab_s.loc[0, 1], tab_s.loc[0, 0]
    log(f"  短停留内粗 OR = {(a_*d2)/(b_*c2):.2f}（暴露(+) {a_}/{a_+b_}, 暴露(−) {c2}/{c2+d2}）")

log("\n影响分析（leave-out）：")
r_loo = fit(m[~((m["icu_h"] <= 24) & (m[EXPO] == 1))], "aki_any")
if "OR" in r_loo:
    log(f"  剔除 7 例「短停留且暴露(+)」后，全队列 aOR = {r_loo['OR']:.2f} "
        f"({r_loo['lo']:.2f}–{r_loo['hi']:.2f}), p={r_loo['p']:.3f}, n={r_loo['n']}")
    R["leaveout_short_exposed"] = r_loo
r_loo2 = fit(m[(m["icu_h"] > 24)], "aki_any")
log(f"  剔除全部短停留者（=模型 D）aOR = {r_loo2.get('OR')} "
    f"({r_loo2.get('lo')}–{r_loo2.get('hi')}), p={r_loo2.get('p')}")

log("\n改用更严结局（KDIGO ≥2，降低基线肌酐口径的敏感度）：")
kd2 = pd.read_sql("SELECT stay_id, MIN(charttime) AS t2 FROM mimiciv_derived.kdigo_stages "
                  "WHERE stay_id = ANY(%s) AND aki_stage >= 2 GROUP BY stay_id",
                  psycopg2.connect(**PG), params=(ids,))
kd2["t2"] = pd.to_datetime(kd2["t2"])
m2 = m.merge(kd2, on="stay_id", how="left")
m2["aki2_any"] = m2["t2"].notna().astype(int)
m2["h2"] = (m2["t2"] - m2["intime"]).dt.total_seconds() / 3600.0
r2a = fit(m2, "aki2_any")
log(f"  全队列 KDIGO≥2: aOR = {r2a.get('OR')} ({r2a.get('lo')}–{r2a.get('hi')}), "
    f"p={r2a.get('p')}, 事件 {r2a.get('events')}")
m2["aki2_after24"] = ((m2["h2"].notna()) & (m2["h2"] > 24) & (m2["icu_h"] > 24)
                      & ((m2["onset_h"].isna()) | (m2["onset_h"] > 24))).astype(int)
r2b = fit(m2[(m2["icu_h"] > 24) & ((m2["onset_h"].isna()) | (m2["onset_h"] > 24))], "aki2_after24")
log(f"  landmark 24h 的 KDIGO≥2 新发: aOR = {r2b.get('OR')} ({r2b.get('lo')}–{r2b.get('hi')}), "
    f"p={r2b.get('p')}, 事件 {r2b.get('events')}")
R.update({"kd2_all": r2a, "kd2_landmark24": r2b})

# ---------------------------------------------------------------- 5 Cox
log("\n" + "=" * 80)
log("表 3. Cox（出 ICU 删失）")
log("=" * 80)
for L in (0, 24):
    d = m[m["icu_h"] > L].copy()
    if L > 0:
        d = d[(d["onset_h"].isna()) | (d["onset_h"] > L)]
    d["time"] = np.where(d["onset_h"].notna(), d["onset_h"], d["icu_h"]) - L
    d["event"] = ((d["onset_h"].notna()) & (d["onset_h"] > L)).astype(int)
    d = d[d["time"] > 0].dropna(subset=["time", "event"] + COV)
    if d["event"].sum() < 5:
        log(f"  landmark {L:2d}h：事件 {int(d['event'].sum())}，略")
        continue
    X = d[[EXPO] + COV].astype(float).values
    ph = PHReg(d["time"].values, X, status=d["event"].values, ties="efron").fit()
    hr = float(np.exp(ph.params[0]))
    lo, hi = np.exp(ph.conf_int()[0])
    log(f"  landmark {L:2d}h: n={len(d):3d}, 事件={int(d['event'].sum()):3d}, "
        f"HR={hr:.2f} ({lo:.2f}–{hi:.2f}), p={ph.pvalues[0]:.3f}")
    R[f"cox{L}"] = {"HR": round(hr, 2), "lo": round(float(lo), 2), "hi": round(float(hi), 2),
                    "p": float(ph.pvalues[0]), "n": int(len(d)), "events": int(d["event"].sum())}

# ---------------------------------------------------------------- 6 CAM 双向
log("\n" + "=" * 80)
log("表 4. CAM-ICU 时间戳双向检验（仅限 24 h 内确有评估者）")
log("=" * 80)
cam["h"] = (cam["charttime"] - cam["stay_id"].map(m.set_index("stay_id")["intime"])).dt.total_seconds() / 3600.0
c1 = cam.groupby("stay_id")["h"].min().rename("assess_h")
c2 = (cam[cam["value"].astype(str).str.strip().str.lower() == "positive"]
      .groupby("stay_id")["h"].min().rename("pos_h"))
mm = m.merge(c1, on="stay_id", how="left").merge(c2, on="stay_id", how="left")
mm["assessed24"] = ((mm["assess_h"].notna()) & (mm["assess_h"] <= 24)).astype(int)
mm["cam_pos24"] = ((mm["pos_h"].notna()) & (mm["pos_h"] <= 24)).astype(int)
mm["cam_pos_late"] = ((mm["pos_h"].notna()) & (mm["pos_h"] > 24)).astype(int)
log(f"CAM 24h 内有评估 {int(mm['assessed24'].sum())} 例；其中 24h 内阳性 "
    f"{int(((mm['assessed24']==1)&(mm['cam_pos24']==1)).sum())} 例；"
    f"24h 后阳性 {int(mm['cam_pos_late'].sum())} 例")

dA = mm[(mm["assessed24"] == 1) & (mm["icu_h"] > 24)].copy()
dA = dA[(dA["onset_h"].isna()) | (dA["onset_h"] > 24)]
dA["out"] = ((dA["onset_h"].notna()) & (dA["onset_h"] > 24)).astype(int)
rA = fit(dA, "out", expo="cam_pos24")
log(f"\n方向 A（24h 内谵妄 → 其后 AKI）: n={len(dA)}, 事件={int(dA['out'].sum())}, "
    f"谵妄(+) {int(dA['cam_pos24'].sum())}")
if "OR" in rA:
    log(f"  aOR = {rA['OR']:.2f} ({rA['lo']:.2f}–{rA['hi']:.2f}), p={rA['p']:.3f}")
else:
    log(f"  {rA}")
R["camA"] = rA

dB = mm[(mm["assessed24"] == 1) & (mm["icu_h"] > 24) & (mm["cam_pos24"] == 0)].copy()
dB["out"] = dB["cam_pos_late"]
dB["early_aki"] = ((dB["onset_h"].notna()) & (dB["onset_h"] >= 0) & (dB["onset_h"] <= 24)).astype(int)
rB = fit(dB, "out", expo="early_aki")
log(f"\n方向 B（24h 内 AKI → 其后谵妄）: n={len(dB)}, 事件={int(dB['out'].sum())}, "
    f"早期 AKI {int(dB['early_aki'].sum())}")
if "OR" in rB:
    log(f"  aOR = {rB['OR']:.2f} ({rB['lo']:.2f}–{rB['hi']:.2f}), p={rB['p']:.3f}")
else:
    log(f"  {rB}")
R["camB"] = rB

# ---------------------------------------------------------------- 7 落盘
R["_meta"] = {"n": int(len(df)), "exposed": int(df[EXPO].sum()), "n_aki": int(m["aki_any"].sum()),
              "n_aki_icu": int(m["aki_icu"].sum()), "pre_icu_aki": int(m["pre_aki"].sum()),
              "median_onset_h": float(m["onset_h"].median()),
              "n_after48": int((m["onset_h"] > 48).sum()),
              "cam_assessed": int(cam["stay_id"].nunique())}
with io.open(os.path.join(ANA, "results_aki_timing.json"), "w", encoding="utf-8") as f:
    json.dump(R, f, ensure_ascii=False, indent=1, default=str)
with io.open(os.path.join(ANA, "aki_timing_report.txt"), "w", encoding="utf-8") as f:
    f.write("\n".join(log_lines))
print("\n[已写出] analysis/results_aki_timing.json, analysis/aki_timing_report.txt")
