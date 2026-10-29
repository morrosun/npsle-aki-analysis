# -*- coding: utf-8 -*-
"""
v23 审计 B：PSM 单一匹配队列 / PH 交互检验 / 聚类稳健 SE / 仅首次入住

评审要求：
  (1) 89、90、88 三套匹配必须收敛为一套固定、可复现的方案，并保存匹配对；
  (2) 48 h 前后差异不能直接宣称「PH 假设不成立」，应在联合模型中检验交互项；
  (3) Cox 模型也需按患者聚类稳健 SE，并做「每位患者仅首次入住」敏感性；
  (4) 摘要称时变 Cox 校正了 sepsis，但表中只有升压药与通气 —— 需补 sepsis 时变项。
"""
import io
import json
import os
import warnings

import numpy as np
import pandas as pd
import psycopg2
import statsmodels.api as sm
from scipy import stats
from statsmodels.duration.hazard_regression import PHReg

warnings.filterwarnings("ignore")

BASE = os.environ.get("SLECI_BASE", os.environ.get("SLECI_BASE", "."))
DATA = os.path.join(BASE, "data")
ANA = os.path.join(BASE, "analysis")
PG = dict(host="localhost", port=5432, user="postgres",
          password=os.environ.get("SLECI_PGPASSWORD", ""), dbname="mimiciv")
COV = ["age", "female", "charlson_norheum", "lupus_nephritis", "renal_disease", "diabetes"]
COV9 = COV + ["cerebrovascular_disease", "chf", "copd"]   # PSM 用的 9 协变量（与 analyze.py 一致）

buf = []


def log(s=""):
    print(s)
    buf.append(str(s))


R = {}

# ---------------------------------------------------------------- 数据
m = pd.read_csv(os.path.join(DATA, "mimiciv_sle.csv"))
ids = m["stay_id"].astype("int64").tolist()

con = psycopg2.connect(**PG)
icu = pd.read_sql("SELECT stay_id, subject_id, hadm_id, intime, outtime FROM mimiciv_icu.icustays "
                  "WHERE stay_id = ANY(%s)", con, params=(ids,))
kd = pd.read_sql("SELECT stay_id, charttime, aki_stage FROM mimiciv_derived.kdigo_stages "
                 "WHERE stay_id = ANY(%s) ORDER BY stay_id, charttime", con, params=(ids,))
cam = pd.read_sql("SELECT stay_id, charttime, value FROM mimiciv_icu.chartevents "
                  "WHERE stay_id = ANY(%s) AND itemid = 228332", con, params=(ids,))
sep = pd.read_sql("SELECT stay_id, suspected_infection_time, sofa_time, sepsis3 "
                  "FROM mimiciv_derived.sepsis3 WHERE stay_id = ANY(%s)", con, params=(ids,))
vaso = pd.read_sql("SELECT stay_id, starttime FROM mimiciv_derived.vasoactive_agent "
                   "WHERE stay_id = ANY(%s)", con, params=(ids,))
vent = pd.read_sql("SELECT stay_id, starttime, ventilation_status FROM mimiciv_derived.ventilation "
                   "WHERE stay_id = ANY(%s)", con, params=(ids,))
con.close()

kd["charttime"] = pd.to_datetime(kd["charttime"])
cam["charttime"] = pd.to_datetime(cam["charttime"])
icu["intime"] = pd.to_datetime(icu["intime"])
icu["outtime"] = pd.to_datetime(icu["outtime"])
sep["suspected_infection_time"] = pd.to_datetime(sep["suspected_infection_time"])
vaso["starttime"] = pd.to_datetime(vaso["starttime"])
vent["starttime"] = pd.to_datetime(vent["starttime"])

imap = icu.set_index("stay_id")["intime"]

m = m.merge(icu[["stay_id", "intime", "outtime"]], on="stay_id", how="left")
m["t_end"] = (m["outtime"] - m["intime"]).dt.total_seconds() / 3600.0
m["t_end"] = m["t_end"].fillna(m["los_icu"] * 24.0)

kd["h"] = (kd["charttime"] - kd["stay_id"].map(imap)).dt.total_seconds() / 3600.0
pos = kd[kd.aki_stage >= 1]
m = m.merge(pos[pos.h >= 0].groupby("stay_id")["h"].min().rename("t_aki"), on="stay_id", how="left")
m = m.merge(pos[pos.aki_stage >= 2].groupby("stay_id")["h"].min().rename("t_aki2"),
            on="stay_id", how="left")
m = m.merge(pos[pos.h < 0].groupby("stay_id").size().rename("n_pre"), on="stay_id", how="left")
m["pre_aki"] = m["n_pre"].notna().astype(int)

cam["h"] = (cam["charttime"] - cam["stay_id"].map(imap)).dt.total_seconds() / 3600.0
cam["v"] = cam["value"].astype(str).str.strip().str.lower()
cam["t_end"] = cam["stay_id"].map(m.set_index("stay_id")["t_end"])
cam = cam[(cam.h >= 0) & (cam.h <= cam.t_end)].copy()   # v27 严格窗口
ass = cam[cam.v.isin(["positive", "negative"])]
m = m.merge(cam[cam.v == "positive"].groupby("stay_id")["h"].min().rename("t_pos"),
            on="stay_id", how="left")
m = m.merge(ass.groupby("stay_id").size().rename("n_assess"), on="stay_id", how="left")
m = m.merge(ass.groupby("stay_id")["h"].min().rename("t_first"), on="stay_id", how="left")
m = m.merge(ass.groupby("stay_id")["h"].max().rename("t_last"), on="stay_id", how="left")
m["n_assess"] = m["n_assess"].fillna(0).astype(int)
m["assessable"] = (m["n_assess"] > 0).astype(int)

sep["t_sep"] = (sep["suspected_infection_time"] - sep["stay_id"].map(imap)
                ).dt.total_seconds() / 3600.0
m = m.merge(sep[["stay_id", "sepsis3", "t_sep"]], on="stay_id", how="left")
m["sepsis3"] = m["sepsis3"].fillna(False).astype(bool)

vaso["h"] = (vaso["starttime"] - vaso["stay_id"].map(imap)).dt.total_seconds() / 3600.0
m = m.merge(vaso.groupby("stay_id")["h"].min().rename("t_vaso"), on="stay_id", how="left")

vent = vent[vent.ventilation_status.isin(["InvasiveVent", "Tracheostomy"])].copy()
vent["h"] = (vent["starttime"] - vent["stay_id"].map(imap)).dt.total_seconds() / 3600.0
m = m.merge(vent.groupby("stay_id")["h"].min().rename("t_vent"), on="stay_id", how="left")

# ---------------------------------------------------------------- 1 PSM


# ---------------------------------------------------------------- 2 时变 Cox
def build_cp(d, t_out, t_censor, tv_list, covs=COV, landmark=0.0, horizon=None,
             extra_cps=()):
    """counting-process 数据集；tv_list=[(name, t_start_col)]，保留 stay_id/subject_id。"""
    LM = float(landmark)
    rows, n_skip = [], 0
    for _, r in d.iterrows():
        cand = [x for x in (r[t_out], r[t_censor]) if pd.notna(x) and np.isfinite(x)]
        if horizon is not None:
            cand.append(horizon)
        if not cand:
            n_skip += 1
            continue
        end = min(cand) - LM
        if end <= 0:
            n_skip += 1
            continue
        ev_t = (r[t_out] - LM) if pd.notna(r[t_out]) else np.inf
        cps = set(x for x in extra_cps if 0 < x < end)
        for _, col in tv_list:
            t = r[col]
            if pd.notna(t) and np.isfinite(t):
                t = max(t - LM, 0.0)
                if 0 < t < end:
                    cps.add(t)
        bounds = [0.0] + sorted(cps) + [end]
        for a, b in zip(bounds[:-1], bounds[1:]):
            if b <= a:
                continue
            seg = {"entry": a, "stop": b,
                   "event": 1 if (ev_t > a and ev_t <= b) else 0,
                   "stay_id": r["stay_id"], "subject_id": r["subject_id"],
                   **{c: r[c] for c in covs}}
            for name, col in tv_list:
                t = r[col]
                seg[name] = 1 if (pd.notna(t) and np.isfinite(t) and (t - LM) <= a) else 0
            rows.append(seg)
    return pd.DataFrame(rows), n_skip


def cox_sandwich(e, cols):
    """按 subject_id 聚类的稳健三明治方差（Breslow 型近似）。"""
    X = e[cols].astype(float).values
    entry = e["entry"].values.astype(float)
    stop = e["stop"].values.astype(float)
    event = e["event"].values.astype(int)
    beta = PHReg(stop, X, status=event, entry=entry, ties="efron").fit().params
    w = np.exp(X @ beta)
    p = X.shape[1]
    I = np.zeros((p, p))
    score_rows = np.zeros_like(X)
    idx_event = np.where(event == 1)[0]
    for t in np.unique(stop[event == 1]):
        risk = (entry < t) & (stop >= t)
        if risk.sum() == 0:
            continue
        S0 = w[risk].sum()
        S1 = (X[risk] * w[risk, None]).sum(0)
        S2 = (X[risk][:, :, None] * X[risk][:, None, :] * w[risk, None, None]).sum(0)
        xbar = S1 / S0
        d = int(((stop == t) & (event == 1)).sum())
        I += d * (S2 / S0 - np.outer(xbar, xbar))
        for i2 in idx_event[stop[idx_event] == t]:
            score_rows[i2] = X[i2] - xbar
    Iinv = np.linalg.pinv(I)
    g = e["subject_id"].values
    Ug = {}
    for i2 in idx_event:
        Ug.setdefault(g[i2], np.zeros(p))
        Ug[g[i2]] = Ug[g[i2]] + score_rows[i2]
    meat = sum(np.outer(u, u) for u in Ug.values())
    G, n = len(Ug), len(e)
    c = G / (G - 1) * (n - 1) / (n - p)
    return beta, c * Iinv @ meat @ Iinv


def fit_cox(e, tv_names=None, cluster=False, label=""):
    cols = ["expo"] + (tv_names or []) + COV
    X = e[cols].astype(float).values
    ph = PHReg(e.stop.values, X, status=e.event.values, entry=e.entry.values,
               ties="efron").fit()
    hr, p = float(np.exp(ph.params[0])), float(ph.pvalues[0])
    lo, hi = np.exp(ph.conf_int()[0])
    tag = ""
    if cluster:
        beta, V = cox_sandwich(e, cols)
        hr = float(np.exp(beta[0]))
        se = float(np.sqrt(V[0, 0]))
        lo, hi = float(np.exp(beta[0] - 1.96 * se)), float(np.exp(beta[0] + 1.96 * se))
        p = float(2 * (1 - stats.norm.cdf(abs(beta[0] / se))))
        tag = " [聚类稳健]"
    n_sub = e.subject_id.nunique()
    log(f"  {label}: 事件 {int(e.event.sum())}, 患者 {n_sub}, 行 {len(e)}；"
        f"HR {hr:.2f} ({lo:.2f}–{hi:.2f}), p={p:.4f}{tag}")
    return {"HR": round(hr, 3), "lo": round(float(lo), 3), "hi": round(float(hi), 3),
            "p": p, "events": int(e.event.sum()), "n_subjects": int(n_sub)}



base = m[(m.assessable == 1) & (m.pre_aki == 0)].copy()
base = base[(base.t_aki.isna()) | (base.t_aki > 0)]
rev = m[(m.assessable == 1) & (m.pre_aki == 0) & m.t_first.notna()].copy()
rev["t_cam_end"] = rev[["t_last", "t_end"]].min(axis=1)
rev = rev[(rev.t_pos.isna()) | (rev.t_pos > 0)]

log("\n" + "=" * 92)
log("2. 时变 Cox：主模型 / 时变 sepsis+升压药+通气 / 聚类稳健 SE / 仅首次入住")
log("=" * 92)
log(f"前向 n={len(base)}（事件 {int(base.t_aki.notna().sum())}）；"
    f"反向 n={len(rev)}（事件 {int(rev.t_pos.notna().sum())}）")

TV = [("expo", None)]  # placeholder


def run_models(d, t_out, t_cen, direction):
    log(f"\n【{direction}】")
    # 主模型
    e, _ = build_cp(d, t_out, t_cen, [("expo", expo_col)])
    r["main"] = fit_cox(e, label="主模型（6 协变量）")
    r["main_cl"] = fit_cox(e, cluster=True, label="主模型 + 按患者聚类稳健 SE")
    # 时变 sep/vaso/vent
    e2, _ = build_cp(d, t_out, t_cen, [("expo", expo_col), ("sep_tv", "t_sep"),
                                       ("vaso_tv", "t_vaso"), ("vent_tv", "t_vent")])
    r["tv3"] = fit_cox(e2, ["sep_tv", "vaso_tv", "vent_tv"],
                       label="+ 时变 sepsis / 升压药 / 有创通气")
    r["tv2"] = fit_cox(
        build_cp(d, t_out, t_cen, [("expo", expo_col), ("vaso_tv", "t_vaso"),
                                   ("vent_tv", "t_vent")])[0],
        ["vaso_tv", "vent_tv"], label="+ 时变升压药 / 有创通气（v22 口径）")
    # 仅首次入住
    d1 = d.sort_values(["subject_id", "intime"]).drop_duplicates("subject_id", keep="first")
    r["first_stay"] = fit_cox(build_cp(d1, t_out, t_cen, [("expo", expo_col)])[0],
                              label=f"仅首次入住（n={len(d1)}）")
    # PH 交互检验：暴露 × 48 h 期（单一模型，行在 48 h 处强制切分）
    e3, _ = build_cp(d, t_out, t_cen, [("expo", expo_col)], extra_cps=(48.0,))
    e3["late"] = (e3.entry >= 48 - 1e-9).astype(float)
    e3["expo_x_late"] = e3["expo"] * e3["late"]
    X = e3[["expo", "late", "expo_x_late"] + COV].astype(float).values
    ph = PHReg(e3.stop.values, X, status=e3.event.values, entry=e3.entry.values,
               ties="efron").fit()
    i = 2
    log(f"  PH 交互检验（exposure×48h 期，单一模型）：交互 HR "
        f"{np.exp(ph.params[i]):.2f} ({np.exp(ph.conf_int()[i][0]):.2f}–"
        f"{np.exp(ph.conf_int()[i][1]):.2f}), p={ph.pvalues[i]:.3f}")
    r["ph_interaction"] = {"HR": round(float(np.exp(ph.params[i])), 3),
                           "p": float(ph.pvalues[i])}


r = {}
expo_col = "t_pos"
R["forward"] = r
run_models(base, "t_aki", "t_end", "前向：谵妄 → AKI")
expo_col = "t_aki"
r = {}
R["reverse"] = r
run_models(rev, "t_pos", "t_cam_end", "反向：AKI → 谵妄")

with io.open(os.path.join(ANA, "results_v27_tv.json"), "w", encoding="utf-8") as f:
    json.dump(R, f, ensure_ascii=False, indent=1, default=str)
io.open(os.path.join(ANA, "v27_tv_report.txt"), "w", encoding="utf-8").write("\n".join(buf))
print("\nwritten: analysis/v27_tv_report.txt")
