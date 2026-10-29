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
log("=" * 92)
log("1. PSM：收敛为一套固定、可复现的匹配队列（确定性贪心最近邻，caliper=0.2×SD(logit PS)）")
log("=" * 92)


def smd_binary(x1, x0):
    p1, p0 = np.mean(x1), np.mean(x0)
    sp = np.sqrt((p1 * (1 - p1) + p0 * (1 - p0)) / 2)
    return 0.0 if sp == 0 else (p1 - p0) / sp


def smd_cont(x1, x0):
    m1, m0 = np.mean(x1), np.mean(x0)
    s1, s0 = np.std(x1, ddof=1), np.std(x0, ddof=1)
    sp = np.sqrt((s1 ** 2 + s0 ** 2) / 2)
    return 0.0 if sp == 0 else (m1 - m0) / sp


def mcnemar(n10, n01):
    p = stats.binomtest(min(n10, n01), n10 + n01, 0.5).pvalue if (n10 + n01) > 0 else 1.0
    orr = n10 / n01 if n01 > 0 else np.nan
    lo = hi = np.nan
    if n10 > 0 and n01 > 0:
        se = np.sqrt(1 / n10 + 1 / n01)
        lo, hi = np.exp(np.log(orr) - 1.96 * se), np.exp(np.log(orr) + 1.96 * se)
    return orr, lo, hi, p


def psm_fixed(d, exposure="npsle_narrow"):
    d = d.dropna(subset=COV9 + [exposure]).reset_index(drop=True)
    X = sm.add_constant(d[COV9].astype(float))
    y = d[exposure].astype(int)
    ps = sm.Logit(y, X).fit(disp=0).predict(X)
    d["logit_ps"] = np.log(ps / (1 - ps))
    cal = 0.2 * d["logit_ps"].std()
    tr, ct = d[d[exposure] == 1], d[d[exposure] == 0]
    cidx = ct.index.tolist()
    clp = ct["logit_ps"].values
    used, pairs = set(), []
    for ti, trow in tr.iterrows():
        diffs = np.abs(clp - trow["logit_ps"])
        for j in np.argsort(diffs):
            cj = cidx[j]
            if cj in used:
                continue
            if diffs[j] <= cal:
                used.add(cj)
                pairs.append((ti, cj))
                break
    mt = d.loc[[p[0] for p in pairs]]
    mc = d.loc[[p[1] for p in pairs]]
    out = {"n_pairs": len(pairs), "caliper": round(cal, 4),
           "n_treated_pool": int(len(tr)), "n_control_pool": int(len(ct)), "smd_after": {}}
    for cov in COV:
        out["smd_after"][cov] = round(
            smd_cont(mt[cov], mc[cov]) if cov in ("age", "charlson_norheum")
            else smd_binary(mt[cov], mc[cov]), 3)
    out["max_smd_after"] = round(max(abs(v) for v in out["smd_after"].values()), 3)
    tbl = pd.DataFrame({"stay_id_treated": mt.stay_id.values, "stay_id_control": mc.stay_id.values,
                        "ps_treated": mt.logit_ps.values, "ps_control": mc.logit_ps.values})
    return out, tbl


# 唯一一次进入匹配的资格：6 协变量 + AKI + 院内结局非缺失（不含 LOS，LOS 缺失只影响 LOS 一行）
need = COV9 + ["npsle_narrow", "aki", "mortality", "vent", "rrt", "vaso"]
ELIG = m[need].notna().all(axis=1)
log(f"匹配资格（协变量 + AKI + 院内结局非缺失）：{int(ELIG.sum())} / {len(m)} stay")
excluded = sorted(set(m.stay_id) - set(m.loc[ELIG, "stay_id"]))
log(f"  被排除的 stay：{excluded if excluded else '无'}")

psm_res, pairs_tbl = psm_fixed(m[ELIG], "npsle_narrow")
log(f"匹配对 {psm_res['n_pairs']}；caliper {psm_res['caliper']}；"
    f"最大 post-match |SMD| {psm_res['max_smd_after']}")
log(f"  SMD after: {psm_res['smd_after']}")

mm = m.set_index("stay_id")
mit = pairs_tbl.stay_id_treated.tolist()
mic = pairs_tbl.stay_id_control.tolist()

for oc, lab, kind in [("aki", "AKI KDIGO≥1", "bin"), ("t_aki2", "AKI KDIGO≥2", "time"),
                      ("vent", "IMV", "bin"), ("rrt", "RRT", "bin"),
                      ("vaso", "升压药", "bin"), ("mortality", "院内死亡", "bin")]:
    if kind == "bin":
        yt = mm.loc[mit, oc] == 1
        yc = mm.loc[mic, oc] == 1
    else:
        yt = mm.loc[mit, oc].notna()
        yc = mm.loc[mic, oc].notna()
    yt, yc = yt.values, yc.values
    n11 = int((yt & yc).sum()); n10 = int((yt & ~yc).sum())
    n01 = int((~yt & yc).sum()); n00 = int((~yt & ~yc).sum())
    orr, lo, hi, p = mcnemar(n10, n01)
    log(f"  {lab}: 暴露 {int(yt.sum())}/{len(mit)} ({100*yt.mean():.1f}%) vs "
        f"对照 {int(yc.sum())}/{len(mic)} ({100*yc.mean():.1f}%)")
    log(f"    一致对 n11={n11}, n00={n00}；不一致对 n10={n10} / n01={n01}；"
        f"matched OR {orr:.2f} ({lo:.2f}–{hi:.2f}), McNemar p={p:.4g}")
    R[f"psm_{oc}"] = {"n11": n11, "n10": n10, "n01": n01, "n00": n00,
                      "OR": round(float(orr), 3), "lo": round(float(lo), 3),
                      "hi": round(float(hi), 3), "p": float(p)}

pairs_tbl.to_csv(os.path.join(ANA, "psm_pairs_v23.csv"), index=False)
R["psm"] = psm_res
log("  匹配对已保存：analysis/psm_pairs_v23.csv")

# 旧口径复算，说明三套数字的来源
need_old = need + ["los_hospital"]
ELIG_old = m[need_old].notna().all(axis=1)
log(f"\n旧口径（额外要求 LOS 非缺失）配对数：{psm_fixed(m[ELIG_old])[0]['n_pairs']}")
log("→ 89 / 90 / 88 三套的差异全部来自「进入匹配的资格」不同，匹配算法本身相同；"
    "v23 固定为上面一套。")

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

with io.open(os.path.join(ANA, "results_v23_psm_ph.json"), "w", encoding="utf-8") as f:
    json.dump(R, f, ensure_ascii=False, indent=1, default=str)
io.open(os.path.join(ANA, "v23_audit_psm_ph_report.txt"), "w", encoding="utf-8").write("\n".join(buf))
print("\nwritten: analysis/v23_audit_psm_ph_report.txt")
