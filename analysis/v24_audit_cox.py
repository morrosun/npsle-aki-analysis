# -*- coding: utf-8 -*-
"""v24 审计 C：48 h 分段改用「两个分段时间变暴露」；导出数据供 R survival::coxph 交叉验证。"""
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


m = pd.read_csv(os.path.join(DATA, "mimiciv_sle.csv"))
ids = m.stay_id.astype("int64").tolist()
con = psycopg2.connect(**PG)
icu = pd.read_sql("SELECT stay_id,subject_id,intime,outtime FROM mimiciv_icu.icustays "
                  "WHERE stay_id=ANY(%s)", con, params=(ids,))
kd = pd.read_sql("SELECT stay_id,charttime,aki_stage FROM mimiciv_derived.kdigo_stages "
                 "WHERE stay_id=ANY(%s)", con, params=(ids,))
cam = pd.read_sql("SELECT stay_id,charttime,value FROM mimiciv_icu.chartevents "
                  "WHERE stay_id=ANY(%s) AND itemid=228332", con, params=(ids,))
con.close()
for d in (kd, cam):
    d["charttime"] = pd.to_datetime(d["charttime"])
icu["intime"] = pd.to_datetime(icu.intime); icu["outtime"] = pd.to_datetime(icu.outtime)
imap = icu.set_index("stay_id")["intime"]

has_subj = "subject_id" in m.columns
m = m.merge(icu[["stay_id", "intime", "outtime"] +
               ([] if has_subj else ["subject_id"])], on="stay_id", how="left")
m["t_end"] = (m.outtime - m.intime).dt.total_seconds() / 3600.0
m["t_end"] = m.t_end.fillna(m.los_icu * 24.0)

kd["h"] = (kd.charttime - kd.stay_id.map(imap)).dt.total_seconds() / 3600.0
pos = kd[kd.aki_stage >= 1]
m = m.merge(pos[pos.h >= 0].groupby("stay_id")["h"].min().rename("t_aki"), on="stay_id", how="left")
m = m.merge(pos[pos.h < 0].groupby("stay_id").size().rename("n_pre"), on="stay_id", how="left")
m["pre_aki"] = m.n_pre.notna().astype(int)

cam["h"] = (cam.charttime - cam.stay_id.map(imap)).dt.total_seconds() / 3600.0
cam["v"] = cam.value.astype(str).str.strip().str.lower()
ass = cam[cam.v.isin(["positive", "negative"])]
m = m.merge(cam[cam.v == "positive"].groupby("stay_id")["h"].min().rename("t_pos"), on="stay_id", how="left")
m = m.merge(ass.groupby("stay_id").size().rename("n_assess"), on="stay_id", how="left")
m = m.merge(ass.groupby("stay_id")["h"].min().rename("t_first"), on="stay_id", how="left")
m = m.merge(ass.groupby("stay_id")["h"].max().rename("t_last"), on="stay_id", how="left")
m["n_assess"] = m.n_assess.fillna(0).astype(int)
m["assessable"] = (m.n_assess > 0).astype(int)


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


def cox_sandwich(e, cols):
    """Lin–Wei 聚类稳健三明治方差；点估计与方差统一用 Breslow ties。"""
    X = e[cols].astype(float).values
    entry, stop = e.entry.values.astype(float), e.stop.values.astype(float)
    event = e.event.values.astype(int)
    beta = PHReg(stop, X, status=event, entry=entry, ties="breslow").fit().params
    w = np.exp(X @ beta)
    p = X.shape[1]
    I = np.zeros((p, p)); sc = np.zeros_like(X)
    idx_ev = np.where(event == 1)[0]
    for t in np.unique(stop[event == 1]):
        risk = (entry < t) & (stop >= t)
        if not risk.any():
            continue
        S0 = w[risk].sum(); S1 = (X[risk] * w[risk, None]).sum(0)
        S2 = (X[risk][:, :, None] * X[risk][:, None, :] * w[risk, None, None]).sum(0)
        xb = S1 / S0
        d = int(((stop == t) & (event == 1)).sum())
        I += d * (S2 / S0 - np.outer(xb, xb))
        for i in idx_ev[stop[idx_ev] == t]:
            sc[i] = X[i] - xb
    Iinv = np.linalg.pinv(I)
    g = e.subject_id.values
    Ug = {}
    for i in idx_ev:
        Ug[g[i]] = Ug.get(g[i], np.zeros(p)) + sc[i]
    # Lin–Wei：聚类数 G 取数据中全部分组数（含无事件者），无事件组贡献 0 但计入 G
    allg = pd.unique(g)
    for gg in allg:
        Ug.setdefault(gg, np.zeros(p))
    meat = sum(np.outer(u, u) for u in Ug.values())
    G, n = len(allg), len(e)
    c = G / (G - 1) * (n - 1) / (n - p)
    return beta, c * Iinv @ meat @ Iinv, G


def report(e, cols, label, cluster=False):
    X = e[cols].astype(float).values
    ph = PHReg(e.stop.values, X, status=e.event.values, entry=e.entry.values,
               ties="efron").fit()
    hr = float(np.exp(ph.params[0])); lo, hi = np.exp(ph.conf_int()[0])
    p = float(ph.pvalues[0]); tag = ""
    if cluster:
        beta, V, G = cox_sandwich(e, cols)
        hr = float(np.exp(beta[0])); se = float(np.sqrt(V[0, 0]))
        lo, hi = np.exp(beta[0] - 1.96 * se), np.exp(beta[0] + 1.96 * se)
        p = float(2 * (1 - stats.norm.cdf(abs(beta[0] / se))))
        tag = f" [聚类 G={G}]"
    log(f"  {label:42s} 事件 {int(e.event.sum()):4d} 行 {len(e):5d}  "
        f"HR {hr:.2f} ({lo:.2f}–{hi:.2f}) p={p:.3f}{tag}")
    return dict(HR=round(hr, 3), lo=round(float(lo), 3), hi=round(float(hi), 3),
                p=round(p, 4), events=int(e.event.sum()), rows=int(len(e)))


base = m[(m.assessable == 1) & (m.pre_aki == 0)].copy()
base = base[(base.t_aki.isna()) | (base.t_aki > 0)]
rev = m[(m.assessable == 1) & (m.pre_aki == 0) & m.t_first.notna()].copy()
rev["t_cam_end"] = rev[["t_last", "t_end"]].min(axis=1)
rev = rev[(rev.t_pos.isna()) | (rev.t_pos > 0)]

R = {}
for name, d, t_out, t_cen, expo in [
        ("forward", base, "t_aki", "t_end", "t_pos"),
        ("reverse", rev, "t_pos", "t_cam_end", "t_aki")]:
    log("=" * 84)
    log(f"{name}：n={len(d)}，事件 {int(d[t_out].notna().sum())}")
    log("=" * 84)
    e = build_cp(d, t_out, t_cen, expo)
    # 「每位患者首次合格入住」标记：在已通过可评估/入科无 AKI 筛选的人群内按 intime 取首个
    first = d.sort_values(["subject_id", "intime"]).drop_duplicates("subject_id", keep="first")
    e["first_stay"] = e.stay_id.isin(first.stay_id).astype(int)
    e.to_csv(os.path.join(ANA, f"_cp_{name}.csv"), index=False)
    r = {}
    r["main"] = report(e, ["expo"] + COV, "主模型（Efron，常规 SE）")
    r["main_cluster"] = report(e, ["expo"] + COV, "主模型 + 按患者聚类稳健 SE", cluster=True)
    r["first_stay"] = report(e[e.first_stay == 1], ["expo"] + COV,
                             f"仅首次合格入住（n={len(first)}）")
    # 分段：两个分段时间变暴露，行在 48 h 强制切分
    ep = build_cp(d, t_out, t_cen, expo, extra_cps=(48.0,))
    ep["period"] = np.where(ep.entry >= 48 - 1e-9, 2, 1)
    ep["expo_p1"] = ep.expo * (ep.period == 1)
    ep["expo_p2"] = ep.expo * (ep.period == 2)
    ep.to_csv(os.path.join(ANA, f"_cp_{name}_period.csv"), index=False)
    cols = ["expo_p1", "expo_p2"] + COV
    X = ep[cols].astype(float).values
    ph = PHReg(ep.stop.values, X, status=ep.event.values, entry=ep.entry.values,
               ties="efron").fit()
    ci = ph.conf_int()
    b1, b2 = ph.params[0], ph.params[1]
    V = ph.cov_params()
    se_d = float(np.sqrt(V[0, 0] + V[1, 1] - 2 * V[0, 1]))
    z = (b2 - b1) / se_d
    p_int = float(2 * (1 - stats.norm.cdf(abs(z))))
    log(f"  {'分段暴露（0–48 h）':42s} 事件 {int(ep.event.sum()):4d} 行 {len(ep):5d}  "
        f"HR {np.exp(b1):.2f} ({np.exp(ci[0][0]):.2f}–{np.exp(ci[0][1]):.2f})")
    log(f"  {'分段暴露（> 48 h）':42s} 事件 {int(ep.event.sum()):4d} 行 {len(ep):5d}  "
        f"HR {np.exp(b2):.2f} ({np.exp(ci[1][0]):.2f}–{np.exp(ci[1][1]):.2f})")
    log(f"  {'期间差异（>48h vs 0–48h）':42s} 比值 {np.exp(b2 - b1):.2f} "
        f"({np.exp(b2-b1-1.96*se_d):.2f}–{np.exp(b2-b1+1.96*se_d):.2f}), p={p_int:.3f}")
    r["period"] = {"HR_0_48": round(float(np.exp(b1)), 3),
                   "lo_0_48": round(float(np.exp(ci[0][0])), 3),
                   "hi_0_48": round(float(np.exp(ci[0][1])), 3),
                   "HR_gt48": round(float(np.exp(b2)), 3),
                   "lo_gt48": round(float(np.exp(ci[1][0])), 3),
                   "hi_gt48": round(float(np.exp(ci[1][1])), 3),
                   "ratio": round(float(np.exp(b2 - b1)), 3),
                   "p_interaction": round(p_int, 4)}
    R[name] = r
    log("")

with io.open(os.path.join(ANA, "results_v24_cox.json"), "w", encoding="utf-8") as f:
    json.dump(R, f, ensure_ascii=False, indent=1)
io.open(os.path.join(ANA, "v24_audit_cox_report.txt"), "w", encoding="utf-8").write("\n".join(L))
print("\nwritten")
