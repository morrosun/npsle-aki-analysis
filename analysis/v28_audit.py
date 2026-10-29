# -*- coding: utf-8 -*-
"""v28 审计：①G93.4x 限制暴露的一致性 2×2（严格 CAM 窗口）+ Clopper–Pearson 区间
②Table 5 主表 CP 区间核对 ③分段暴露（0–48 h / >48 h）的 p 值 ④G93 集合计数复核。

前置：无（直接读 data/mimiciv_sle.csv + PostgreSQL）。
输出：analysis/v28_audit_report.txt
"""
import os
import io, os, json, warnings
import numpy as np, pandas as pd, psycopg2
from scipy import stats
from statsmodels.duration.hazard_regression import PHReg

warnings.filterwarnings("ignore")
BASE = os.environ.get("SLECI_BASE", os.environ.get("SLECI_BASE", "."))
DATA, ANA = os.path.join(BASE, "data"), os.path.join(BASE, "analysis")
PG = dict(host="localhost", port=5432, user="postgres",
          password=os.environ.get("SLECI_PGPASSWORD", ""), dbname="mimiciv")
COV = ["age", "female", "charlson_norheum", "lupus_nephritis", "renal_disease", "diabetes"]
L = []


def log(s=""):
    print(s); L.append(str(s))


def cp_ci(k, n, alpha=0.05):
    """Clopper–Pearson 精确区间（与 binom.test / proportion_confint(method='beta') 一致）。"""
    lo = 0.0 if k == 0 else stats.beta.ppf(alpha / 2, k, n - k + 1)
    hi = 1.0 if k == n else stats.beta.ppf(1 - alpha / 2, k + 1, n - k)
    return float(lo), float(hi)


m = pd.read_csv(os.path.join(DATA, "mimiciv_sle.csv"))
ids = m.stay_id.astype("int64").tolist()
con = psycopg2.connect(**PG)
icu = pd.read_sql("SELECT stay_id,subject_id,intime,outtime,hadm_id FROM mimiciv_icu.icustays "
                  "WHERE stay_id=ANY(%s)", con, params=(ids,))
cam = pd.read_sql("SELECT stay_id,charttime,value FROM mimiciv_icu.chartevents "
                  "WHERE stay_id=ANY(%s) AND itemid=228332", con, params=(ids,))
dx = pd.read_sql("SELECT hadm_id,icd_code,icd_version FROM mimiciv_hosp.diagnoses_icd "
                 "WHERE hadm_id=ANY(%s)",
                 con, params=(icu.hadm_id.dropna().astype("int64").unique().tolist(),))
con.close()
for d in (cam,):
    d["charttime"] = pd.to_datetime(d["charttime"])
icu["intime"] = pd.to_datetime(icu.intime); icu["outtime"] = pd.to_datetime(icu.outtime)
imap = icu.set_index("stay_id")["intime"]

m = m.merge(icu[["stay_id", "intime", "outtime", "hadm_id"]], on="stay_id", how="left",
            suffixes=("", "_icu"))
m["t_end"] = (m.outtime - m.intime).dt.total_seconds() / 3600.0
m["t_end"] = m.t_end.fillna(m.los_icu * 24.0)

# ---------- CAM（严格窗） ----------
cam["h"] = (cam.charttime - cam.stay_id.map(imap)).dt.total_seconds() / 3600.0
cam["v"] = cam.value.astype(str).str.strip().str.lower()
cam["t_end"] = cam.stay_id.map(m.set_index("stay_id")["t_end"])
n0 = len(cam)
cam = cam[(cam.h >= 0) & (cam.h <= cam.t_end)].copy()
ass = cam[cam.v.isin(["positive", "negative"])]
m = m.merge(cam[cam.v == "positive"].groupby("stay_id")["h"].min().rename("t_pos"),
            on="stay_id", how="left")
m = m.merge(ass.groupby("stay_id").size().rename("n_assess"), on="stay_id", how="left")
m["n_assess"] = m.n_assess.fillna(0).astype(int)
m["assessable"] = (m.n_assess > 0).astype(int)
m["cam_pos"] = m.t_pos.notna().astype(int)
log(f"[CAM] 记录 {n0} -> {len(cam)}（严格 intime–outtime）；可判读 {int(m.assessable.sum())}；"
    f"阳性 {int(m.cam_pos.sum())}")

# ---------- 暴露：窄定义 / G93.4x 限制 ----------
dx["code"] = dx.icd_code.astype(str).str.strip().str.upper()
dx["v"] = dx.icd_version.astype(int)


def is_narrow(df):
    c10 = df.v == 10
    c9 = df.v == 9
    p10 = df.code.str.startswith(("F05", "R41", "F06", "G93", "F01", "F02", "F03", "G30", "G31"))
    p9 = (df.code.str.startswith(("293", "294", "310")) | (df.code == "3483")
          | df.code.str.startswith("7800"))
    return (c10 & p10) | (c9 & p9)


def is_restr(df):
    c10 = df.v == 10
    c9 = df.v == 9
    p10 = df.code.str.startswith(("F05", "R41", "F06", "G934", "F01", "F02", "F03", "G30", "G31"))
    p9 = (df.code.str.startswith(("293", "294", "310")) | (df.code == "3483")
          | df.code.str.startswith("7800"))
    return (c10 & p10) | (c9 & p9)


dx["narrow"] = is_narrow(dx); dx["restr"] = is_restr(dx)
hn = set(dx.loc[dx.narrow, "hadm_id"]); hr = set(dx.loc[dx.restr, "hadm_id"])
m["e92"] = m.hadm_id.isin(hn).astype(int)
m["e80"] = m.hadm_id.isin(hr).astype(int)
log(f"[暴露] 窄定义 {int(m.e92.sum())}（CSV npsle_narrow={int(m.npsle_narrow.sum())}）；"
    f"G93.4x 限制 {int(m.e80.sum())}；限制后剔除 {int((m.e92 - m.e80).sum())} 例")

# ---------- G93 集合计数 ----------
v10 = dx[dx.v == 10]
g93 = set(v10[v10.code.str.startswith("G93")].hadm_id) & hn
g934 = set(v10[v10.code.str.startswith("G934")].hadm_id) & hn
s348 = set(v10[v10.code.str.startswith("3483")].hadm_id) & hn
non934 = g93 - g934
log("")
log("=" * 78)
log("1. G93 集合（限 92 例窄定义）")
log("=" * 78)
log(f"  任意 G93 代码                  {len(g93)}")
log(f"  有 G93.4x                      {len(g934)}")
log(f"  有任意非 G93.4 子码            {len(non934)}")
log(f"  同时有 G93.4x 及非 G93.4 子码  {len(g934 & non934)}")
log(f"  有非 G93.4 子码、但无 G93.4x    {len(non934 - g934)}")
lose = sorted(non934 - g934 - (hn - hr))  # 该组中仍保留暴露者
keep_in_13 = sorted((non934 - g934) & hr)
log(f"  —— 其中另有保留暴露码（仍为暴露） {len(keep_in_13)}  {keep_in_13}")
log(f"  —— 限制后失去暴露资格           {len((non934 - g934) - hr)}")

# ---------- 2×2 与 CP 区间 ----------
log("")
log("=" * 78)
log("2. 与实测谵妄的一致性（376 例可判读，严格 CAM 窗口）")
log("=" * 78)


def agree(col, lab):
    d = m[m.assessable == 1]
    TP = int(((d[col] == 1) & (d.cam_pos == 1)).sum())
    FP = int(((d[col] == 1) & (d.cam_pos == 0)).sum())
    FN = int(((d[col] == 0) & (d.cam_pos == 1)).sum())
    TN = int(((d[col] == 0) & (d.cam_pos == 0)).sum())
    n = TP + FP + FN + TN
    assert n == 376, n
    se = TP / (TP + FN); sp = TN / (TN + FP); ppv = TP / (TP + FP); npv = TN / (TN + FN)
    po = (TP + TN) / n
    pe = ((TP + FP) * (TP + FN) + (FN + TN) * (FP + TN)) / n ** 2
    k = (po - pe) / (1 - pe)
    rows = [("Sensitivity", TP, TP + FN), ("Specificity", TN, TN + FP),
            ("PPV", TP, TP + FP), ("NPV", TN, TN + FN)]
    log(f"  [{lab}] TP {TP} / FP {FP} / FN {FN} / TN {TN}   合计 {n}")
    log(f"    点估计: sens {se*100:.1f}  spec {sp*100:.1f}  PPV {ppv*100:.1f}  "
        f"NPV {npv*100:.1f}  agree {po*100:.1f}  kappa {k:.2f}")
    for nm, a, b in rows:
        lo, hi = cp_ci(a, b)
        log(f"    {nm:12s} {a}/{b} = {a/b*100:5.1f}%  CP {lo*100:4.1f}–{hi*100:4.1f}")
    log(f"    Agreement    {TP+TN}/{n} = {po*100:.1f}%  CP "
        f"{cp_ci(TP+TN, n)[0]*100:.1f}–{cp_ci(TP+TN, n)[1]*100:.1f}")
    return dict(TP=TP, FP=FP, FN=FN, TN=TN, sens=round(se*100, 1), spec=round(sp*100, 1),
                ppv=round(ppv*100, 1), npv=round(npv*100, 1), agree=round(po*100, 1),
                kappa=round(k, 2))


A92 = agree("e92", "窄定义 92")
log("")
A80 = agree("e80", "G93.4x 限制 80")

# ---------- 分段暴露 p 值 ----------
log("")
log("=" * 78)
log("3. 分段暴露（0–48 h / >48 h）的 p 值")
log("=" * 78)
kd = None
con = psycopg2.connect(**PG)
kd = pd.read_sql("SELECT stay_id,charttime,aki_stage FROM mimiciv_derived.kdigo_stages "
                 "WHERE stay_id=ANY(%s)", con, params=(ids,))
con.close()
kd["charttime"] = pd.to_datetime(kd.charttime)
kd["h"] = (kd.charttime - kd.stay_id.map(imap)).dt.total_seconds() / 3600.0
pos = kd[kd.aki_stage >= 1]
m = m.merge(pos[pos.h >= 0].groupby("stay_id")["h"].min().rename("t_aki"), on="stay_id", how="left")
m = m.merge(pos[pos.h < 0].groupby("stay_id").size().rename("n_pre"), on="stay_id", how="left")
m["pre_aki"] = m.n_pre.notna().astype(int)
m = m.merge(ass.groupby("stay_id")["h"].max().rename("t_last"), on="stay_id", how="left")
m = m.merge(ass.groupby("stay_id")["h"].min().rename("t_first"), on="stay_id", how="left")


def build_cp(d, t_out, t_censor, expo_col, extra_cps=()):
    rows = []
    for _, r in d.iterrows():
        cand = [x for x in (r[t_out], r[t_censor]) if pd.notna(x) and np.isfinite(x)]
        if not cand:
            continue
        end = min(cand)
        if end <= 0:
            continue
        ev = (r[t_out] - 0.0) if pd.notna(r[t_out]) else np.inf
        cps = set(x for x in extra_cps if 0 < x < end)
        t = r[expo_col]
        if pd.notna(t) and np.isfinite(t) and 0 < t < end:
            cps.add(float(t))
        bounds = [0.0] + sorted(cps) + [end]
        for a, b in zip(bounds[:-1], bounds[1:]):
            if b <= a:
                continue
            rows.append({"entry": a, "stop": b,
                         "event": 1 if (ev > a and ev <= b) else 0,
                         "stay_id": int(r.stay_id), "subject_id": int(r.subject_id),
                         "expo": 1 if (pd.notna(t) and np.isfinite(t) and t <= a) else 0,
                         **{c: float(r[c]) for c in COV}})
    return pd.DataFrame(rows)


base = m[(m.assessable == 1) & (m.pre_aki == 0)].copy()
base = base[(base.t_aki.isna()) | (base.t_aki > 0)]
rev = m[(m.assessable == 1) & (m.pre_aki == 0) & m.t_first.notna()].copy()
rev["t_cam_end"] = rev[["t_last", "t_end"]].min(axis=1)
rev = rev[(rev.t_pos.isna()) | (rev.t_pos > 0)]

PR = {}
for name, d, t_out, t_cen, expo in [
        ("forward", base, "t_aki", "t_end", "t_pos"),
        ("reverse", rev, "t_pos", "t_cam_end", "t_aki")]:
    ep = build_cp(d, t_out, t_cen, expo, extra_cps=(48.0,))
    ep["period"] = np.where(ep.entry >= 48 - 1e-9, 2, 1)
    ep["expo_p1"] = ep.expo * (ep.period == 1)
    ep["expo_p2"] = ep.expo * (ep.period == 2)
    cols = ["expo_p1", "expo_p2"] + COV
    ph = PHReg(ep.stop.values, ep[cols].astype(float).values, status=ep.event.values,
               entry=ep.entry.values, ties="efron").fit()
    ci = ph.conf_int(); b1, b2 = ph.params[0], ph.params[1]
    V = ph.cov_params(); se_d = float(np.sqrt(V[0, 0] + V[1, 1] - 2 * V[0, 1]))
    p_int = float(2 * (1 - stats.norm.cdf(abs((b2 - b1) / se_d))))
    log(f"  {name}: n={len(d)} 事件 {int(ep.event.sum())} 行 {len(ep)}")
    for lab, b, se, c in [("0–48 h", b1, np.sqrt(V[0, 0]), ci[0]),
                          (">48 h", b2, np.sqrt(V[1, 1]), ci[1])]:
        p = float(2 * (1 - stats.norm.cdf(abs(b / se))))
        log(f"    {lab:8s} HR {np.exp(b):.3f} (Wald {np.exp(c[0]):.3f}–{np.exp(c[1]):.3f}) "
            f"p={p:.4f}   [log-HR {b:.4f}, SE {se:.4f}, z {b/se:.3f}]")
    log(f"    期间差异 ratio {np.exp(b2-b1):.3f} ({np.exp(b2-b1-1.96*se_d):.3f}–"
        f"{np.exp(b2-b1+1.96*se_d):.3f}) p={p_int:.4f}")
    PR[name] = dict(hr1=round(float(np.exp(b1)), 3), hr2=round(float(np.exp(b2)), 3),
                    p1=round(float(2 * (1 - stats.norm.cdf(abs(b1 / np.sqrt(V[0, 0]))))), 4),
                    p2=round(float(2 * (1 - stats.norm.cdf(abs(b2 / np.sqrt(V[1, 1]))))), 4),
                    p_int=round(p_int, 4), events=int(ep.event.sum()),
                    e1=int(ep.loc[ep.period == 1, "event"].sum()),
                    e2=int(ep.loc[ep.period == 2, "event"].sum()))

out = dict(agree92=A92, agree80=A80, period=PR)
with io.open(os.path.join(ANA, "results_v28_audit.json"), "w", encoding="utf-8") as f:
    json.dump(out, f, ensure_ascii=False, indent=1)
io.open(os.path.join(ANA, "v28_audit_report.txt"), "w", encoding="utf-8").write("\n".join(L))
print("\nwritten")
