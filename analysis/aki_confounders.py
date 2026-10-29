# -*- coding: utf-8 -*-
"""
v22 急性混杂专项：脓毒症、休克/升压药、机械通气。
时间窗前置界定：
  - 基线（入 ICU 6 h 内）：sepsis-3、升压药启动、有创通气启动
  - 随访中：升压药与通气按启动时点作为**时变协变量**进入 Cox，避免用结局后的状态做基线校正
"""
import os
import warnings

import numpy as np
import pandas as pd
import psycopg2
import statsmodels.formula.api as smf
from statsmodels.duration.hazard_regression import PHReg

warnings.filterwarnings("ignore")

BASE = os.environ.get("SLECI_BASE", os.environ.get("SLECI_BASE", "."))
DATA = os.path.join(BASE, "data")
OUT = os.path.join(BASE, "analysis", "confounder_v22_report.txt")
PG = dict(host="localhost", port=5432, user="postgres",
          password=os.environ.get("SLECI_PGPASSWORD", ""), dbname="mimiciv")
COV = ["age", "female", "charlson_norheum", "lupus_nephritis", "renal_disease", "diabetes"]
ACUTE = ["sepsis_early", "vaso_early", "imv_early"]

buf = []


def log(s=""):
    print(s)
    buf.append(str(s))


m = pd.read_csv(os.path.join(DATA, "mimiciv_sle.csv"))
ids = m["stay_id"].astype("int64").tolist()
con = psycopg2.connect(**PG)
icu = pd.read_sql("SELECT stay_id, intime, outtime FROM mimiciv_icu.icustays WHERE stay_id = ANY(%s)",
                  con, params=(ids,))
sep = pd.read_sql("SELECT stay_id, suspected_infection_time, sofa_time, sofa_score, sepsis3 "
                  "FROM mimiciv_derived.sepsis3 WHERE stay_id = ANY(%s)", con, params=(ids,))
vaso = pd.read_sql("SELECT stay_id, starttime, endtime FROM mimiciv_derived.vasoactive_agent "
                   "WHERE stay_id = ANY(%s)", con, params=(ids,))
vent = pd.read_sql("SELECT stay_id, starttime, endtime, ventilation_status "
                   "FROM mimiciv_derived.ventilation WHERE stay_id = ANY(%s)", con, params=(ids,))
kd = pd.read_sql("SELECT stay_id, charttime, aki_stage FROM mimiciv_derived.kdigo_stages "
                 "WHERE stay_id = ANY(%s)", con, params=(ids,))
cam = pd.read_sql("SELECT stay_id, charttime, value FROM mimiciv_icu.chartevents "
                  "WHERE stay_id = ANY(%s) AND itemid = 228332", con, params=(ids,))
con.close()

for d in (icu, sep, vaso, vent, kd, cam):
    for c in d.columns:
        if "time" in c or c in ("intime", "outtime", "charttime"):
            d[c] = pd.to_datetime(d[c], errors="coerce")

m = m.merge(icu, on="stay_id", how="left")
m["t_end"] = (m["outtime"] - m["intime"]).dt.total_seconds() / 3600.0
m["t_end"] = m["t_end"].fillna(m["los_icu"] * 24.0)

# 早期（≤6 h）暴露
def first_hour(df, cond=None):
    d = df if cond is None else df[cond]
    return ((d["starttime"] - d["stay_id"].map(icu.set_index("stay_id")["intime"]))
            .dt.total_seconds() / 3600.0).groupby(d["stay_id"]).min()


h_vaso = first_hour(vaso)
h_vent = first_hour(vent, vent["ventilation_status"] == "InvasiveVent")
m = m.merge(h_vaso.rename("h_vaso"), on="stay_id", how="left").merge(
    h_vent.rename("h_vent"), on="stay_id", how="left")
m["vaso_early"] = ((m.h_vaso >= 0) & (m.h_vaso <= 6)).astype(int)
m["imv_early"] = ((m.h_vent >= 0) & (m.h_vent <= 6)).astype(int)

sep["t_sep"] = (sep["suspected_infection_time"] - sep["stay_id"].map(
    icu.set_index("stay_id")["intime"])).dt.total_seconds() / 3600.0
m = m.merge(sep[["stay_id", "sepsis3", "t_sep", "sofa_score"]], on="stay_id", how="left")
m["sepsis_flag"] = m["sepsis3"].fillna(False).astype(bool)
m["sepsis_early"] = (m["sepsis_flag"] & (m["t_sep"].fillna(1e9) <= 6)).astype(int)
m["sepsis_any"] = m["sepsis_flag"].astype(int)

log("=" * 82)
log("急性混杂变量的可得性与分布（MIMIC-IV, n=550）")
log("=" * 82)
for v in ["sepsis_any", "sepsis_early", "vaso_early", "imv_early"]:
    log(f"  {v:14s} {int(m[v].sum()):3d} ({100*m[v].mean():.1f}%)")
log(f"  sepsis3 有记录 {int(m.sepsis3.notna().sum())}；SOFA 中位 "
    f"{m.sofa_score.median():.0f}" if m.sofa_score.notna().any() else "  SOFA 无")
for v in ACUTE:
    for k, g in m.groupby(v):
        pass
log("\n按暴露分组的急性事件比例：")
for k, g in m.groupby("npsle_narrow"):
    log(f"  暴露={k}: n={len(g)}, sepsis_early {100*g.sepsis_early.mean():.1f}%, "
        f"vaso_early {100*g.vaso_early.mean():.1f}%, imv_early {100*g.imv_early.mean():.1f}%")

# ---------------- 1 逻辑回归
log("\n" + "=" * 82)
log("表 1. AKI：逐步加入急性混杂因素（MIMIC-IV, n=550）")
log("=" * 82)
m["aki01"] = m["aki"].astype(int)
models = [
    ("(1) 主模型（6 协变量）", COV),
    ("(2) + 早期脓毒症", COV + ["sepsis_early"]),
    ("(3) + 早期脓毒症 + 早期升压药 + 早期有创通气", COV + ACUTE),
    ("(4) 参照：+ 全程脓毒症（非时序，仅对照）", COV + ["sepsis_any"]),
]
for lab, covs in models:
    d = m.dropna(subset=covs + ["aki01", "npsle_narrow"])
    r = smf.logit("aki01 ~ npsle_narrow + " + " + ".join(covs), data=d).fit(disp=0)
    ci = r.conf_int().loc["npsle_narrow"]
    log(f"  {lab}")
    log(f"      aOR {np.exp(r.params['npsle_narrow']):.2f} "
        f"({np.exp(ci[0]):.2f}–{np.exp(ci[1]):.2f}), p={r.pvalues['npsle_narrow']:.3f}, n={len(d)}")

# ---------------- 2 时变 Cox（升压药/通气为时变协变量）
log("\n" + "=" * 82)
log("表 2. 时变 Cox：主模型 vs 追加时变升压药/通气")
log("=" * 82)
kd = kd.merge(icu[["stay_id", "intime"]], on="stay_id", how="left")
kd["h"] = (kd["charttime"] - kd["intime"]).dt.total_seconds() / 3600.0
pos = kd[kd.aki_stage >= 1]
m2 = (m.merge(pos[pos.h >= 0].groupby("stay_id")["h"].min().rename("t_aki"), on="stay_id", how="left")
       .merge(pos[pos.h < 0].groupby("stay_id").size().rename("n_pre"), on="stay_id", how="left"))
m2["pre_aki"] = m2["n_pre"].notna().astype(int)

cam["h"] = (cam["charttime"] - cam["stay_id"].map(icu.set_index("stay_id")["intime"])).dt.total_seconds() / 3600
cam["v"] = cam["value"].astype(str).str.strip().str.lower()
n_ass = cam[cam.v.isin(["positive", "negative"])].groupby("stay_id").size().rename("n_assess")
t_pos = cam[cam.v == "positive"].groupby("stay_id")["h"].min().rename("t_pos")
t_first = cam[cam.v.isin(["positive", "negative"])].groupby("stay_id")["h"].min().rename("t_first")
t_last = cam[cam.v.isin(["positive", "negative"])].groupby("stay_id")["h"].max().rename("t_last")
m2 = (m2.merge(n_ass, on="stay_id", how="left").merge(t_pos, on="stay_id", how="left")
        .merge(t_first, on="stay_id", how="left").merge(t_last, on="stay_id", how="left"))
m2["n_assess"] = m2["n_assess"].fillna(0).astype(int)


def tvc(d, t_expo, t_out, t_censor, extra_tv=None, label=""):
    rows = []
    for _, r in d.iterrows():
        cand = [x for x in (r[t_out], r[t_censor]) if pd.notna(x) and np.isfinite(x)]
        if not cand:
            continue
        end = min(cand)
        if end <= 0:
            continue
        ev_t = r[t_out] if pd.notna(r[t_out]) else np.inf
        exp_t = r[t_expo] if pd.notna(r[t_expo]) else np.inf
        exp_t = max(exp_t, 0.0)
        cuts = {0.0, end}
        if exp_t < end:
            cuts.add(exp_t)
        tv_cuts = {}
        if extra_tv:
            for name in extra_tv:
                v = r.get(name, np.nan)
                tv_cuts[name] = None if pd.isna(v) else float(v)
                if tv_cuts[name] is not None and 0 < tv_cuts[name] < end:
                    cuts.add(tv_cuts[name])
        cuts = sorted(cuts)
        for a, b in zip(cuts[:-1], cuts[1:]):
            if b <= a:
                continue
            row = {"entry": a, "stop": b,
                   "event": 1 if (a < ev_t <= b) else 0,
                   "expo": 1 if a >= exp_t else 0,
                   **{c: r[c] for c in COV}}
            if extra_tv:
                for name in extra_tv:
                    st = tv_cuts[name]
                    row[name] = 0 if (st is None or a < st) else 1
            rows.append(row)
    e = pd.DataFrame(rows)
    e = e[e.stop > e.entry]
    X = e[["expo"] + COV + (extra_tv or [])].astype(float).values
    ph = PHReg(e.stop.values, X, status=e.event.values, entry=e.entry.values, ties="efron").fit()
    hr = np.exp(ph.params[0])
    lo, hi = np.exp(ph.conf_int()[0])
    log(f"  {label}: HR {hr:.2f} ({lo:.2f}–{hi:.2f}), p={ph.pvalues[0]:.3f}; "
        f"n={len(d)}, 事件={int(e.event.sum())}")
    return hr, lo, hi


base = m2[(m2.n_assess > 0) & (m2.pre_aki == 0)].copy()
fwd = base[(base.t_aki.isna()) | (base.t_aki > 0)]
rev = base[(base.t_pos.isna()) | (base.t_pos > 0)].copy()
rev["t_cam_end"] = rev[["t_last", "t_end"]].min(axis=1)

log("前向 谵妄 → AKI")
tvc(fwd, "t_pos", "t_aki", "t_end", label="  主模型")
tvc(fwd, "t_pos", "t_aki", "t_end", extra_tv=["h_vaso", "h_vent"],
    label="  + 时变升压药/有创通气")
log("反向 AKI → 谵妄")
tvc(rev, "t_aki", "t_pos", "t_cam_end", label="  主模型")
tvc(rev, "t_aki", "t_pos", "t_cam_end", extra_tv=["h_vaso", "h_vent"],
    label="  + 时变升压药/有创通气")

import io
io.open(OUT, "w", encoding="utf-8").write("\n".join(buf))
print("\nwritten:", OUT)
