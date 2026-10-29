# -*- coding: utf-8 -*-
"""
方案 B：以 CAM-ICU 时间戳构建**时变暴露**，用时变 Cox 检验
「谵妄记录是否先于新发 AKI」以及反向「AKI 是否先于谵妄」。

为什么需要这一步
----------------
主稿的暴露是 ICD 编码（无时间戳），因此无法回应 R1-C1「尿毒症/代谢紊乱本身
可致脑病，关联可能是反向因果」。CAM-ICU（itemid 228332）带 charttime，
可把暴露做成随时间切换的状态变量，直接检验时序。

模型设定
--------
模型 1（主方向，谵妄 → AKI）
  人群：≥1 次可判读 CAM（Negative/Positive）+ 入 ICU 前无 AKI
  时间轴：t=0 为 ICU intime；暴露在首次 CAM 阳性时点由 0 切为 1
  结局：ICU 内首次 KDIGO≥1；删失：出 ICU
  时变 Cox（counting process，entry/stop 区间）
  约定：CAM 阳性必须**严格早于** AKI 才计为暴露，同期判为未暴露（保守）

模型 2（反向，AKI → 谵妄）
  人群：同上
  结局：首次 CAM 阳性；暴露：AKI 发生时点切换
  删失：末次 CAM 评估时点（此后结局不可观测）与出 ICU 取早

敏感性
------
S1 结局改 KDIGO≥2
S2 限制前 72 h
S3 限制 24 h 内即完成首次 CAM 评估者（降低「评估强度差异」偏倚）
S4 暴露定义改用「CAM 阳性或 UTA」——UTA 多因深镇静/昏迷，作为对照
"""
import io
import json
import os
import warnings

import numpy as np
import pandas as pd
import psycopg2
from statsmodels.duration.hazard_regression import PHReg

warnings.filterwarnings("ignore")

BASE = os.environ.get("SLECI_BASE", os.environ.get("SLECI_BASE", "."))
ANA = os.path.join(BASE, "analysis")
DATA = os.path.join(BASE, "data")
PG = dict(host="localhost", port=5432, user="postgres",
          password=os.environ.get("SLECI_PGPASSWORD", ""), dbname="mimiciv")

COV = ["age", "female", "charlson_norheum", "lupus_nephritis", "renal_disease", "diabetes"]

log_lines = []


def log(s=""):
    print(s)
    log_lines.append(str(s))


R = {}

# ---------------------------------------------------------------- 1 数据
df = pd.read_csv(os.path.join(DATA, "mimiciv_sle.csv"))
ids = df["stay_id"].astype("int64").tolist()
assert len(df) == 550

con = psycopg2.connect(**PG)
icu = pd.read_sql("SELECT stay_id, intime, outtime FROM mimiciv_icu.icustays "
                  "WHERE stay_id = ANY(%s)", con, params=(ids,))
kd = pd.read_sql("SELECT stay_id, charttime, aki_stage FROM mimiciv_derived.kdigo_stages "
                 "WHERE stay_id = ANY(%s) ORDER BY stay_id, charttime", con, params=(ids,))
cam = pd.read_sql("SELECT stay_id, charttime, value FROM mimiciv_icu.chartevents "
                  "WHERE stay_id = ANY(%s) AND itemid = 228332", con, params=(ids,))
con.close()

for d in (kd, cam):
    d["charttime"] = pd.to_datetime(d["charttime"])
icu["intime"] = pd.to_datetime(icu["intime"])
icu["outtime"] = pd.to_datetime(icu["outtime"])

m = df.merge(icu, on="stay_id", how="left")
m["t_end"] = (m["outtime"] - m["intime"]).dt.total_seconds() / 3600.0
m["t_end"] = m["t_end"].fillna(m["los_icu"] * 24.0)

# 首达 KDIGO≥1（相对 intime）
kd = kd.merge(icu[["stay_id", "intime"]], on="stay_id", how="left")
kd["h"] = (kd["charttime"] - kd["intime"]).dt.total_seconds() / 3600.0
pos = kd[kd["aki_stage"] >= 1]
t_aki_all = pos.groupby("stay_id")["h"].min().rename("t_aki_all")
t_aki_icu = pos[pos["h"] >= 0].groupby("stay_id")["h"].min().rename("t_aki")
t_aki2 = (pos[pos["aki_stage"] >= 2].groupby("stay_id")["h"].min()
          .rename("t_aki2"))
pre = pos[pos["h"] < 0].groupby("stay_id").size().rename("n_pre")
m = (m.merge(t_aki_all, on="stay_id", how="left")
      .merge(t_aki_icu, on="stay_id", how="left")
      .merge(t_aki2, on="stay_id", how="left")
      .merge(pre, on="stay_id", how="left"))
m["pre_aki"] = m["n_pre"].notna().astype(int)

# CAM 时点
cam["h"] = (cam["charttime"] - cam["stay_id"].map(icu.set_index("stay_id")["intime"])
            ).dt.total_seconds() / 3600.0
cam["v"] = cam["value"].astype(str).str.strip().str.lower()
# v27：严格限制在 intime–outtime（评审：CAM 记录必须用与结局定义一致的窗口）
_end = m.set_index("stay_id")["t_end"]
cam["t_end_cam"] = cam["stay_id"].map(_end)
_n0 = len(cam)
cam = cam[(cam["h"] >= 0) & (cam["h"] <= cam["t_end_cam"])].copy()
log(f"[v27 CAM 窗口] {_n0} -> {len(cam)} 条")
cam_pos = cam[cam["v"] == "positive"]
t_pos = cam_pos.groupby("stay_id")["h"].min().rename("t_pos")
t_uta = cam[cam["v"] == "uta"].groupby("stay_id")["h"].min().rename("t_uta")
n_ass = cam[cam["v"].isin(["positive", "negative"])].groupby("stay_id").size().rename("n_assess")
t_first = cam[cam["v"].isin(["positive", "negative"])].groupby("stay_id")["h"].min().rename("t_first_assess")
t_last = cam[cam["v"].isin(["positive", "negative"])].groupby("stay_id")["h"].max().rename("t_last_assess")

m = (m.merge(t_pos, on="stay_id", how="left")
      .merge(t_uta, on="stay_id", how="left")
      .merge(n_ass, on="stay_id", how="left")
      .merge(t_first, on="stay_id", how="left")
      .merge(t_last, on="stay_id", how="left"))
m["n_assess"] = m["n_assess"].fillna(0).astype(int)
m["assessable"] = (m["n_assess"] > 0).astype(int)

log("=" * 84)
log("队列概况")
log("=" * 84)
log(f"总队列 n={len(m)}；暴露(ICD 窄定义) {int(m['npsle_narrow'].sum())}；"
    f"任意 AKI {int(m['t_aki_all'].notna().sum())}")
log(f"可判读 CAM（≥1 次 Negative/Positive）: {int(m['assessable'].sum())}/550 "
    f"({100*m['assessable'].mean():.1f}%)")
log(f"  CAM 阳性至少 1 次: {int(m['t_pos'].notna().sum())}")
log(f"  UTA 至少 1 次: {int(m['t_uta'].notna().sum())}")
log(f"入 ICU 前已 AKI（prevalent）: {int(m['pre_aki'].sum())}")
log(f"ICU 内 AKI: {int(m['t_aki'].notna().sum())}；其中 AKI≥2: {int(m['t_aki2'].notna().sum())}")
log(f"CAM 评估强度：中位 {m.loc[m['assessable']==1,'n_assess'].median():.0f} 次/"
    f"中位停留 {m['t_end'].median():.1f} h")

# ---------------------------------------------------------------- 2 时序次序
both = m[m["t_pos"].notna() & m["t_aki"].notna()].copy()
both["gap"] = both["t_aki"] - both["t_pos"]
def order(r):
    if r["gap"] > 6:
        return "1 谵妄在前（>6h）"
    if r["gap"] < -6:
        return "3 AKI 在前（>6h）"
    return "2 6h 内同期"
both["order"] = both.apply(order, axis=1)
tab = both["order"].value_counts().reindex(
    ["1 谵妄在前（>6h）", "2 6h 内同期", "3 AKI 在前（>6h）"]).fillna(0).astype(int)
log("\n" + "=" * 84)
log("表 1. 同时具备 CAM 阳性与 ICU 内 AKI 者的时序次序（n=%d）" % len(both))
log("=" * 84)
for k, v in tab.items():
    log(f"  {k:20s} {v:3d}  ({100*v/max(len(both),1):.1f}%)")
log(f"  谵妄→AKI 中位间隔 {both.loc[both['gap']>0,'gap'].median():.1f} h")
log(f"  AKI→谵妄 中位间隔 {-both.loc[both['gap']<0,'gap'].median():.1f} h")
R["order"] = {k: int(v) for k, v in tab.items()}

# ---------------------------------------------------------------- 3 时变 Cox
def tvc_cox(d, t_expo, t_out, t_censor, covs=COV, label="", tag="", landmark=0.0):
    """counting-process 时变 Cox：暴露在 t_expo 由 0→1。可指定 landmark（左截断）。"""
    LM = float(landmark)
    d = d.dropna(subset=covs).copy()
    rows, meta = [], []
    n_skip = 0
    for _, r in d.iterrows():
        # 注意：t_aki 为 NaN 的删失者，min(nan, x) 会返回 nan 而整例被丢弃。
        # 必须剔除 NaN 后再取最小值，否则风险集只剩病例。
        cand = [x for x in (r[t_out], r[t_censor])
                if pd.notna(x) and np.isfinite(x)]
        if not cand:
            n_skip += 1
            continue
        end = min(cand) - LM
        if end <= 0:
            n_skip += 1
            continue
        ev_t = (r[t_out] - LM) if pd.notna(r[t_out]) else np.inf
        exp_t = (r[t_expo] - LM) if pd.notna(r[t_expo]) else np.inf
        if not np.isfinite(exp_t):
            exp_t = np.inf
        exp_t = max(exp_t, 0.0)          # 早于 landmark 的暴露时点按 landmark 即暴露处理
        # 分段：暴露前 / 暴露后
        segs = []
        if exp_t <= 0:
            segs.append((0.0, end, 1))
        elif exp_t < end and exp_t < ev_t:
            segs.append((0.0, exp_t, 0))
            segs.append((exp_t, end, 1))
        else:
            segs.append((0.0, end, 0))
        for (a, b, x) in segs:
            if b <= a:
                continue
            ev = 1 if (ev_t > a and ev_t <= b) else 0
            rows.append({"entry": a, "stop": b, "event": ev, "expo": x,
                         **{c: r[c] for c in covs}})
        meta.append(1)
    e = pd.DataFrame(rows)
    if e.empty or e["event"].sum() < 5 or e["expo"].nunique() < 2:
        log(f"  {label}: 事件 {int(e['event'].sum()) if len(e) else 0}，样本不足 — 略")
        return {"insufficient": True, "n": int(len(d)),
                "events": int(e["event"].sum()) if len(e) else 0}
    e = e[e["stop"] > e["entry"]]
    X = e[["expo"] + covs].astype(float).values
    ph = PHReg(e["stop"].values, X, status=e["event"].values,
               entry=e["entry"].values, ties="efron").fit()
    hr = float(np.exp(ph.params[0]))
    lo, hi = np.exp(ph.conf_int()[0])
    ne = int(e["event"].sum())
    # 暴露人时 / 未暴露人时
    pt1 = float(e.loc[e["expo"] == 1, "stop"].sub(e.loc[e["expo"] == 1, "entry"]).sum())
    pt0 = float(e.loc[e["expo"] == 0, "stop"].sub(e.loc[e["expo"] == 0, "entry"]).sum())
    log(f"  {label}")
    log(f"    n={int(len(d))}（无随访 {n_skip}）, 事件={ne}, 行={len(e)}, "
        f"风险集内人数={len(d) - n_skip}; HR={hr:.2f} ({lo:.2f}–{hi:.2f}), p={ph.pvalues[0]:.4f}")
    log(f"    暴露人时 {pt1:.0f} h / 未暴露人时 {pt0:.0f} h")
    return {"HR": round(hr, 2), "lo": round(float(lo), 2), "hi": round(float(hi), 2),
            "p": float(ph.pvalues[0]), "n": int(len(d)), "events": ne,
            "person_time_exposed": round(pt1, 1), "person_time_unexposed": round(pt0, 1)}


log("\n" + "=" * 84)
log("表 2. 模型 1（主方向）：时变 CAM 阳性 → 新发 AKI")
log("=" * 84)
base = m[(m["assessable"] == 1) & (m["pre_aki"] == 0)].copy()
base = base[(base["t_aki"].isna()) | (base["t_aki"] > 0)]
log(f"分析人群：可判读 CAM 且入 ICU 时无 AKI，n={len(base)}；"
    f"其中 CAM 阳性 {int(base['t_pos'].notna().sum())}；"
    f"ICU 内新发 AKI {int(base['t_aki'].notna().sum())}")
R["m1"] = tvc_cox(base, "t_pos", "t_aki", "t_end", label="M1 主分析（全 ICU 停留）", tag="m1")

log("\n  —— 敏感性 ——")
d1 = base.copy()
d1["t_aki"] = d1["t_aki2"]
R["s1"] = tvc_cox(d1, "t_pos", "t_aki", "t_end", label="S1 结局改 KDIGO≥2")

d2 = base.copy()
d2["t_end"] = d2["t_end"].clip(upper=72)
R["s2"] = tvc_cox(d2, "t_pos", "t_aki", "t_end", label="S2 限制前 72 h")

d3 = base[base["t_first_assess"] <= 24].copy()
R["s3"] = tvc_cox(d3, "t_pos", "t_aki", "t_end", label="S3 24 h 内完成首次 CAM 评估")

d4 = base.copy()
d4["t_pos_uta"] = d4[["t_pos", "t_uta"]].min(axis=1)
R["s4"] = tvc_cox(d4, "t_pos_uta", "t_aki", "t_end", label="S4 暴露=CAM 阳性或 UTA")

# S5 landmark 24 h：排除 24 h 内已 AKI 或已谵妄者，纯看其后新发
d5 = base[base["t_end"] > 24].copy()
d5 = d5[(d5["t_aki"].isna()) | (d5["t_aki"] > 24)]
d5 = d5[(d5["t_pos"].isna()) | (d5["t_pos"] > 24)]
log(f"  [S5 人群] 24 h 内未发生 AKI 且未谵妄者 n={len(d5)}；"
    f"其后新发 AKI {int(d5['t_aki'].notna().sum())}；其后首次谵妄 "
    f"{int(d5['t_pos'].notna().sum())}")
R["s5"] = tvc_cox(d5, "t_pos", "t_aki", "t_end",
                  label="S5 landmark 24 h（24 h 后新发）", landmark=24)

log("\n" + "=" * 84)
log("表 3. 模型 2（反向）：时变 AKI → 首次谵妄记录（删失于末次 CAM 评估）")
log("=" * 84)
rev = m[(m["assessable"] == 1) & (m["pre_aki"] == 0)].copy()
rev = rev[rev["t_first_assess"].notna()]
rev["t_cam_end"] = rev[["t_last_assess", "t_end"]].min(axis=1)
# 首次评估即阳性者，事件时点=首次评估时点；要求 t>0
rev = rev[(rev["t_pos"].isna()) | (rev["t_pos"] > 0)]
log(f"分析人群 n={len(rev)}；CAM 阳性 {int(rev['t_pos'].notna().sum())}；"
    f"AKI {int(rev['t_aki'].notna().sum())}")
R["m2"] = tvc_cox(rev, "t_aki", "t_pos", "t_cam_end", label="M2 反向（全 ICU 停留）")

log("\n  —— 反向敏感性 ——")
r2 = rev.copy()
r2["t_cam_end"] = r2["t_cam_end"].clip(upper=72)
R["r_s2"] = tvc_cox(r2, "t_aki", "t_pos", "t_cam_end", label="R-S2 限制前 72 h")
r3 = rev[rev["t_first_assess"] <= 24].copy()
R["r_s3"] = tvc_cox(r3, "t_aki", "t_pos", "t_cam_end", label="R-S3 24 h 内完成首评")

# 检测偏倚：AKI 患者 CAM 评估更密，可能抬高反向 HR
log("\n  —— 反向模型的检测偏倚专项 ——")
rev["log_ass"] = np.log1p(rev["n_assess"])
R["r_s4"] = tvc_cox(rev, "t_aki", "t_pos", "t_cam_end",
                    covs=COV + ["log_ass"], label="R-S4 校正 CAM 评估次数")
r5 = rev[rev["n_assess"] >= 4].copy()
R["r_s5"] = tvc_cox(r5, "t_aki", "t_pos", "t_cam_end",
                    label="R-S5 仅纳入评估≥4 次者")

# ---------------------------------------------------------------- 4 检测偏倚
log("\n" + "=" * 84)
log("表 4. 评估强度与检出偏倚检查")
log("=" * 84)
at = base.copy()
at["aki_any"] = at["t_aki"].notna().astype(int)
at["rate"] = at["n_assess"] / (at["t_end"] / 24.0).clip(lower=0.1)
log("按 AKI 状态比较 CAM 评估强度（次/24h）：")
for k, g in at.groupby("aki_any"):
    log(f"  AKI={k}: n={len(g):3d}, 评估次数中位 {g['n_assess'].median():.0f}, "
        f"评估密度中位 {g['rate'].median():.1f}/24h, 阳性率 "
        f"{100*g['t_pos'].notna().mean():.1f}%")
log("\n按 ICD 暴露状态比较：")
for k, g in base.groupby("npsle_narrow"):
    log(f"  暴露={k}: n={len(g):3d}, 评估次数中位 {g['n_assess'].median():.0f}, "
        f"CAM 阳性率 {100*g['t_pos'].notna().mean():.1f}%")

# ---------------------------------------------------------------- 5 落盘
R["_meta"] = {"n": int(len(m)), "assessable": int(m["assessable"].sum()),
              "cam_pos": int(m["t_pos"].notna().sum()),
              "pre_aki": int(m["pre_aki"].sum()),
              "n_both": int(len(both)),
              "m1_n": int(len(base)), "m1_events": int(base["t_aki"].notna().sum())}
with io.open(os.path.join(ANA, "results_v27_tvc.json"), "w", encoding="utf-8") as f:
    json.dump(R, f, ensure_ascii=False, indent=1, default=str)
with io.open(os.path.join(ANA, "v27_tvc_report.txt"), "w", encoding="utf-8") as f:
    f.write("\n".join(log_lines))
print("\n[已写出] analysis/results_v27_tvc.json, analysis/v27_tvc_report.txt")
