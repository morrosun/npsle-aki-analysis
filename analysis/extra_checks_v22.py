# -*- coding: utf-8 -*-
"""v22 其余核查：比例风险（时间分割）、单一受试者贡献、统一匹配队列下的严重 AKI"""
import os
import warnings

import numpy as np
import pandas as pd
import psycopg2
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.duration.hazard_regression import PHReg

warnings.filterwarnings("ignore")
BASE = os.environ.get("SLECI_BASE", os.environ.get("SLECI_BASE", "."))
DATA = os.path.join(BASE, "data")
OUT = os.path.join(BASE, "analysis", "extra_v22_report.txt")
PG = dict(host="localhost", port=5432, user="postgres",
          password=os.environ.get("SLECI_PGPASSWORD", ""), dbname="mimiciv")
COV = ["age", "female", "charlson_norheum", "lupus_nephritis", "renal_disease", "diabetes"]
buf = []


def log(s=""):
    print(s)
    buf.append(str(s))


m = pd.read_csv(os.path.join(DATA, "mimiciv_sle.csv"))
sens = pd.read_csv(os.path.join(DATA, "mimiciv_sle_sens.csv"))
ids = m["stay_id"].astype("int64").tolist()
con = psycopg2.connect(**PG)
icu = pd.read_sql("SELECT stay_id, intime, outtime FROM mimiciv_icu.icustays WHERE stay_id = ANY(%s)",
                  con, params=(ids,))
kd = pd.read_sql("SELECT stay_id, charttime, aki_stage FROM mimiciv_derived.kdigo_stages "
                 "WHERE stay_id = ANY(%s)", con, params=(ids,))
cam = pd.read_sql("SELECT stay_id, charttime, value FROM mimiciv_icu.chartevents "
                  "WHERE stay_id = ANY(%s) AND itemid = 228332", con, params=(ids,))
con.close()
for d in (icu, kd, cam):
    for c in d.columns:
        if "time" in c or c in ("intime", "outtime", "charttime"):
            d[c] = pd.to_datetime(d[c], errors="coerce")

m = m.merge(icu, on="stay_id", how="left")
m["t_end"] = (m["outtime"] - m["intime"]).dt.total_seconds() / 3600
m["t_end"] = m["t_end"].fillna(m["los_icu"] * 24)

kd = kd.merge(icu[["stay_id", "intime"]], on="stay_id", how="left")
kd["h"] = (kd["charttime"] - kd["intime"]).dt.total_seconds() / 3600
pos = kd[kd.aki_stage >= 1]
m = (m.merge(pos[pos.h >= 0].groupby("stay_id")["h"].min().rename("t_aki"), on="stay_id", how="left")
      .merge(pos[pos.h < 0].groupby("stay_id").size().rename("n_pre"), on="stay_id", how="left"))
m["pre_aki"] = m.n_pre.notna().astype(int)
cam["h"] = (cam["charttime"] - cam["stay_id"].map(icu.set_index("stay_id")["intime"])).dt.total_seconds() / 3600
cam["v"] = cam["value"].astype(str).str.strip().str.lower()
n_ass = cam[cam.v.isin(["positive", "negative"])].groupby("stay_id").size().rename("n_assess")
t_pos = cam[cam.v == "positive"].groupby("stay_id")["h"].min().rename("t_pos")
t_first = cam[cam.v.isin(["positive", "negative"])].groupby("stay_id")["h"].min().rename("t_first")
t_last = cam[cam.v.isin(["positive", "negative"])].groupby("stay_id")["h"].max().rename("t_last")
m = (m.merge(n_ass, on="stay_id", how="left").merge(t_pos, on="stay_id", how="left")
      .merge(t_first, on="stay_id", how="left").merge(t_last, on="stay_id", how="left"))
m["n_assess"] = m.n_assess.fillna(0).astype(int)

log("=" * 80)
log("一、重复入院与单一受试者影响")
log("=" * 80)
vc = m.subject_id.value_counts()
log(f"550 stay / {m.subject_id.nunique()} 受试者；>1 stay 的受试者 {int((vc>1).sum())}，最多 {int(vc.max())} 次")
top = vc.index[0]
log(f"贡献最多的受试者：{int(vc.iloc[0])} 次住院，暴露 {int(m.loc[m.subject_id==top,'npsle_narrow'].sum())}，"
    f"AKI {int(m.loc[m.subject_id==top,'aki'].sum())}")


def fit(d, extra=""):
    d = d.dropna(subset=COV + ["aki", "npsle_narrow"])
    f = "aki ~ npsle_narrow + " + " + ".join(COV) + extra
    r = smf.logit(f, data=d).fit(disp=0)
    ci = r.conf_int().loc["npsle_narrow"]
    return np.exp(r.params["npsle_narrow"]), np.exp(ci[0]), np.exp(ci[1]), r.pvalues["npsle_narrow"], len(d)


for lab, d in [("全部", m), (f"剔除该受试者", m[m.subject_id != top])]:
    a, lo, hi, p, n = fit(d)
    log(f"  {lab}: aOR {a:.2f} ({lo:.2f}–{hi:.2f}), p={p:.3f}, n={n}")

log("\n" + "=" * 80)
log("二、比例风险检查（时间分割：0–48 h vs >48 h）")
log("=" * 80)


def tvc(d, t_expo, t_out, t_censor, entry=0.0, stop=None, label=""):
    rows = []
    for _, r in d.iterrows():
        cand = [x for x in (r[t_out], r[t_censor]) if pd.notna(x) and np.isfinite(x)]
        if not cand:
            continue
        end = min(cand)
        if stop is not None:
            end = min(end, stop)
        if end <= entry:
            continue
        ev_t = r[t_out] if pd.notna(r[t_out]) else np.inf
        exp_t = r[t_expo] if pd.notna(r[t_expo]) else np.inf
        exp_t = max(exp_t, 0.0)
        segs = []
        if exp_t <= entry:
            segs.append((entry, end, 1))
        elif exp_t < end and exp_t < ev_t:
            segs.append((entry, exp_t, 0))
            segs.append((exp_t, end, 1))
        else:
            segs.append((entry, end, 0))
        for a, b, x in segs:
            if b <= a:
                continue
            rows.append({"entry": a, "stop": b, "event": 1 if (a < ev_t <= b) else 0,
                         "expo": x, **{c: r[c] for c in COV}})
    e = pd.DataFrame(rows)
    e = e[e.stop > e.entry]
    if e.empty or e.event.sum() < 5:
        log(f"  {label}: 事件不足（{int(e.event.sum()) if len(e) else 0}）")
        return
    X = e[["expo"] + COV].astype(float).values
    ph = PHReg(e.stop.values, X, status=e.event.values, entry=e.entry.values, ties="efron").fit()
    hr = np.exp(ph.params[0])
    lo, hi = np.exp(ph.conf_int()[0])
    log(f"  {label}: HR {hr:.2f} ({lo:.2f}–{hi:.2f}), p={ph.pvalues[0]:.3f}; "
        f"事件 {int(e.event.sum())}, 风险集 {len(e)}")


base = m[(m.n_assess > 0) & (m.pre_aki == 0)].copy()
fwd = base[(base.t_aki.isna()) | (base.t_aki > 0)]
rev = base[(base.t_pos.isna()) | (base.t_pos > 0)].copy()
rev["t_cam_end"] = rev[["t_last", "t_end"]].min(axis=1)
log("前向 谵妄→AKI")
tvc(fwd, "t_pos", "t_aki", "t_end", entry=0, stop=48, label="  0–48 h")
tvc(fwd, "t_pos", "t_aki", "t_end", entry=48, label="  >48 h（左截断）")
log("反向 AKI→谵妄")
tvc(rev, "t_aki", "t_pos", "t_cam_end", entry=0, stop=48, label="  0–48 h")
tvc(rev, "t_aki", "t_pos", "t_cam_end", entry=48, label="  >48 h（左截断）")

log("\n" + "=" * 80)
log("三、统一 89 对匹配队列下评价 KDIGO≥2（避免 90 对/89 对不一致）")
log("=" * 80)
d = m.merge(sens[["stay_id", "aki_stage_max"]], on="stay_id", how="left")
d["severe"] = (d.aki_stage_max >= 2).astype(int)
# 重建主队列匹配（与 analyze.py 同一算法）
PSCOV = COV + ["cerebrovascular_disease", "chf", "copd"]
dd = d.dropna(subset=PSCOV + ["aki"]).copy().reset_index(drop=True)
X = sm.add_constant(dd[PSCOV].astype(float))
ps = smf.logit("npsle_narrow ~ " + " + ".join(PSCOV), data=dd).fit(disp=0).predict(X)
dd["lps"] = np.log(ps / (1 - ps))
cal = 0.2 * dd.lps.std()
tr = dd[dd.npsle_narrow == 1].sort_values("lps")
ct = dd[dd.npsle_narrow == 0].sort_values("lps")
used, pairs = set(), []
cl = ct.lps.values
for _, r in tr.iterrows():
    dif = np.abs(cl - r.lps)
    for j in np.argsort(dif):
        if dif[j] > cal:
            break
        if j in used:
            continue
        used.add(j)
        pairs.append((r.stay_id, ct.iloc[j].stay_id))
        break
mm = dd.set_index("stay_id")
a = mm.loc[[p[0] for p in pairs]]
b = mm.loc[[p[1] for p in pairs]]
n10 = int(((a.severe.values == 1) & (b.severe.values == 0)).sum())
n01 = int(((a.severe.values == 0) & (b.severe.values == 1)).sum())
orr = n10 / n01 if n01 else np.nan
se = np.sqrt(1 / n10 + 1 / n01) if n10 and n01 else np.nan
log(f"  匹配 {len(pairs)} 对；严重 AKI 暴露组 {int(a.severe.sum())}/{len(a)} "
    f"({100*a.severe.mean():.1f}%) vs 对照 {int(b.severe.sum())}/{len(b)} ({100*b.severe.mean():.1f}%)")
log(f"  discordant n10={n10}, n01={n01}；matched OR {orr:.2f} "
    f"({np.exp(np.log(orr)-1.96*se):.2f}–{np.exp(np.log(orr)+1.96*se):.2f})")

import io
io.open(OUT, "w", encoding="utf-8").write("\n".join(buf))
print("\nwritten:", OUT)
