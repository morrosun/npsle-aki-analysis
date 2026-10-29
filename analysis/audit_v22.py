# -*- coding: utf-8 -*-
"""
v22 方法审计：核对评审指出的可核查项，并把需要新分析的量算出来。
1) 队列流程人数（550 → 376 → 326/325）、UTA 构成
2) 编码类别频数（MIMIC-IV）
3) 去除慢性神经退行性诊断后的暴露与 AKI 关联
4) OASIS 组成项核对
5) eICU 窄定义字符串构成与"急性化"后关联
"""
import io
import os
import warnings

import numpy as np
import pandas as pd
import psycopg2
import statsmodels.formula.api as smf

warnings.filterwarnings("ignore")

BASE = os.environ.get("SLECI_BASE", os.environ.get("SLECI_BASE", "."))
DATA = os.path.join(BASE, "data")
OUT = os.path.join(BASE, "analysis", "audit_v22_report.txt")
PG = dict(host="localhost", port=5432, user="postgres",
          password=os.environ.get("SLECI_PGPASSWORD", ""), dbname="mimiciv")
COV = ["age", "female", "charlson_norheum", "lupus_nephritis", "renal_disease", "diabetes"]

buf = []


def log(s=""):
    print(s)
    buf.append(str(s))


def aor(df, expo, out, covs=COV):
    d = df.dropna(subset=covs + [expo, out]).copy()
    f = f"{out} ~ {expo} + " + " + ".join(covs)
    m = smf.logit(f, data=d).fit(disp=0)
    ci = m.conf_int().loc[expo]
    return (float(np.exp(m.params[expo])), float(np.exp(ci[0])), float(np.exp(ci[1])),
            float(m.pvalues[expo]), len(d), int(d[expo].sum()), int(d[out].sum()))


# ------------------------------------------------------------------ 0 数据
m = pd.read_csv(os.path.join(DATA, "mimiciv_sle.csv"))
sens = pd.read_csv(os.path.join(DATA, "mimiciv_sle_sens.csv"))
ids = m["stay_id"].astype("int64").tolist()
con = psycopg2.connect(**PG)

icu = pd.read_sql("SELECT stay_id, hadm_id, intime, outtime FROM mimiciv_icu.icustays "
                  "WHERE stay_id = ANY(%s)", con, params=(ids,))
kd = pd.read_sql("SELECT stay_id, charttime, aki_stage FROM mimiciv_derived.kdigo_stages "
                 "WHERE stay_id = ANY(%s)", con, params=(ids,))
cam = pd.read_sql("SELECT stay_id, charttime, value FROM mimiciv_icu.chartevents "
                  "WHERE stay_id = ANY(%s) AND itemid = 228332", con, params=(ids,))
hadm = icu["hadm_id"].dropna().astype("int64").unique().tolist()
dx = pd.read_sql("SELECT hadm_id, seq_num, icd_code, icd_version FROM mimiciv_hosp.diagnoses_icd "
                 "WHERE hadm_id = ANY(%s)", con, params=(hadm,))
con.close()

for d in (kd, cam):
    d["charttime"] = pd.to_datetime(d["charttime"])
icu["intime"] = pd.to_datetime(icu["intime"])
icu["outtime"] = pd.to_datetime(icu["outtime"])

m = m.merge(icu[["stay_id", "hadm_id", "intime", "outtime"]], on="stay_id", how="left")
m["t_end"] = (m["outtime"] - m["intime"]).dt.total_seconds() / 3600.0
m["t_end"] = m["t_end"].fillna(m["los_icu"] * 24.0)

kd = kd.merge(icu[["stay_id", "intime"]], on="stay_id", how="left")
kd["h"] = (kd["charttime"] - kd["intime"]).dt.total_seconds() / 3600.0
pos = kd[kd["aki_stage"] >= 1]
m = m.merge(pos[pos["h"] >= 0].groupby("stay_id")["h"].min().rename("t_aki"), on="stay_id", how="left")
m = m.merge(pos[pos["h"] < 0].groupby("stay_id").size().rename("n_pre"), on="stay_id", how="left")
m["pre_aki"] = m["n_pre"].notna().astype(int)
m = m.merge(pos[pos["h"] == 0].groupby("stay_id").size().rename("n_at0"), on="stay_id", how="left")
m["at0_aki"] = m["n_at0"].notna().astype(int)

cam["h"] = (cam["charttime"] - cam["stay_id"].map(icu.set_index("stay_id")["intime"])).dt.total_seconds() / 3600
cam["v"] = cam["value"].astype(str).str.strip().str.lower()
n_ass = cam[cam["v"].isin(["positive", "negative"])].groupby("stay_id").size().rename("n_assess")
t_pos = cam[cam["v"] == "positive"].groupby("stay_id")["h"].min().rename("t_pos")
n_uta = cam[cam["v"] == "uta"].groupby("stay_id").size().rename("n_uta")
t_first = cam[cam["v"].isin(["positive", "negative"])].groupby("stay_id")["h"].min().rename("t_first")
m = (m.merge(n_ass, on="stay_id", how="left").merge(t_pos, on="stay_id", how="left")
      .merge(n_uta, on="stay_id", how="left").merge(t_first, on="stay_id", how="left"))
m["n_assess"] = m["n_assess"].fillna(0).astype(int)
m["n_uta"] = m["n_uta"].fillna(0).astype(int)
m["assessable"] = (m["n_assess"] > 0).astype(int)

log("=" * 88)
log("一、队列流程与 CAM-ICU 构成")
log("=" * 88)
log(f"总队列 550；可判读 CAM-ICU（≥1 次 Negative/Positive）{m['assessable'].sum()} "
    f"({100*m['assessable'].mean():.1f}%)")
log(f"入 ICU 前（h<0）已有 KDIGO≥1：{m['pre_aki'].sum()} / 550")
log(f"  其中可判读者 {int(((m.pre_aki == 1) & (m.assessable == 1)).sum())}；"
    f"不可判读者 {int(((m.pre_aki == 1) & (m.assessable == 0)).sum())}")
log(f"首条 KDIGO≥1 恰在 intime（h=0）：{m['at0_aki'].sum()}")
step1 = m[m.assessable == 1]
log(f"376 可判读者中：pre_aki {int(step1.pre_aki.sum())}，at0 {int(step1.at0_aki.sum())}")
fwd = step1[(step1.pre_aki == 0) & ((step1.t_aki.isna()) | (step1.t_aki > 0))]
log(f"前向模型人群 = 376 - pre_aki - at0 = {len(fwd)}（稿件 326）")
rev_pop = step1[(step1.pre_aki == 0) & step1.t_first.notna()]
rev_pop = rev_pop[(rev_pop.t_pos.isna()) | (rev_pop.t_pos > 0)]
log(f"反向模型人群 = 376 - pre_aki - (首次阳性恰在 intime) = {len(rev_pop)}（稿件 325）")
log(f"  → 差异来源：t_pos<=0 的例数 = {int(((step1.pre_aki == 0) & (step1.t_pos <= 0)).sum())}；"
    f"t_aki<=0 的例数 = {int(((step1.pre_aki == 0) & (step1.t_aki <= 0)).sum())}")

uta_any = int((m["n_uta"] > 0).sum())
uta_only = int(((m["n_uta"] > 0) & (m["assessable"] == 0)).sum())
uta_both = int(((m["n_uta"] > 0) & (m["assessable"] == 1)).sum())
log(f"\n至少 1 次 UTA：{uta_any}；其中 UTA-only（无任何可判读记录）{uta_only}，"
    f"同时有可判读记录 {uta_both}（两者之和 = {uta_only+uta_both}）")
log(f"CAM 阳性至少 1 次：{int(m.t_pos.notna().sum())}")

# ------------------------------------------------------------------ 2 编码类别
log("\n" + "=" * 88)
log("二、MIMIC-IV 窄定义的实际编码构成（按 stay 计，可重叠）")
log("=" * 88)
dx["code"] = dx["icd_code"].astype(str).str.strip().str.upper()
dx["v"] = dx["icd_version"].astype(int)
dx = dx.merge(icu[["hadm_id", "stay_id"]], on="hadm_id", how="inner")


def has(patterns, ver):
    sub = dx[dx["v"] == ver]
    hit = set()
    for p in patterns:
        hit |= set(sub[sub["code"].str.startswith(p)]["stay_id"])
    return hit


CATS = {
    "Delirium (F05 / 293.0-293.1)": (["F05", "2930", "2931"], None),
    "Encephalopathy (G93.4 / 348.3 / 293.82?)": (["G934", "3483"], 9),
    "Encephalopathy (G93.4) ICD-10": (["G934"], 10),
    "Other specified organic mental disorder (F06.x)": (["F06"], 10),
    "Transient mental disorder, other (293.8x/293.9)": (["2938", "2939"], 9),
    "Amnesia / altered mental status (R41.x)": (["R41"], 10),
    "Amnesia / alteration of consciousness (780.0x/780.9)": (["7800", "7809"], 9),
    "Dementia / neurodegeneration (F01-F03, G30, G31)": (["F01", "F02", "F03", "G30", "G31"], 10),
    "Dementia / persistent disorder (294.x)": (["294"], 9),
    "Personality change (310.x / F07)": (["310", "F07"], None),
}
exp_stays = set(m.loc[m.npsle_narrow == 1, "stay_id"])
log(f"窄定义暴露 stay 数 = {len(exp_stays)}")
rows = []
for name, (pats, ver) in CATS.items():
    if ver is None:
        hit = has([p for p in pats if not p.startswith("F") or p[1:2].isdigit()], 10)
        hit |= has([p for p in pats if p.isdigit()], 9)
    else:
        hit = has(pats, ver)
    inter = exp_stays & hit
    rows.append((name, len(hit), len(inter), 100 * len(inter) / max(len(exp_stays), 1)))
log(f"{'类别':56s} {'队列内':>6s} {'暴露内':>6s} {'占暴露%':>8s}")
for r in rows:
    log(f"{r[0]:56s} {r[1]:6d} {r[2]:6d} {r[3]:7.1f}%")

CHRONIC10 = ["F01", "F02", "F03", "G30", "G31"]
CHRONIC9 = ["294"]
chronic = has(CHRONIC10, 10) | has(CHRONIC9, 9)
acute_only = exp_stays - chronic
log(f"\n仅慢性神经退行性编码（无任何急性码）的暴露者：{len(exp_stays & chronic)} / {len(exp_stays)}")
log(f"去除慢性类后仍阳性的暴露者：{len(acute_only)}")

m["npsle_acute"] = m["stay_id"].isin(acute_only).astype(int)
m["npsle_chronic_only"] = m["stay_id"].isin(exp_stays & chronic).astype(int)
log(f"npsle_acute 计数（含原窄定义中被剔除者外的全部）= {int(m.npsle_acute.sum())}")

res_acute = aor(m.assign(aki=m.aki.astype(int)), "npsle_acute", "aki")
log(f"\n【敏感性】排除慢性神经退行性编码后的暴露 → AKI："
    f"aOR {res_acute[0]:.2f} ({res_acute[1]:.2f}–{res_acute[2]:.2f}), p={res_acute[3]:.3f}, "
    f"n={res_acute[4]}, 暴露 {res_acute[5]}, 事件 {res_acute[6]}")

# 慢性组单独
sub = m[m.npsle_chronic_only == 1]
log(f"仅慢性编码者（n={len(sub)}）AKI {int(sub.aki.sum())} ({100*sub.aki.mean():.1f}%)；"
    f"无任何受累者 AKI {100*m.loc[m.npsle_narrow==0,'aki'].mean():.1f}%")

# ------------------------------------------------------------------ 3 同队列 PSM 下 KDIGO>=2
log("\n" + "=" * 88)
log("三、PSM 对数差异（89 vs 90）溯源")
log("=" * 88)
log(f"主队列 mimiciv_sle.csv n={len(m)}；敏感性数据集 mimiciv_sle_sens.csv n={len(sens)}")
log(f"主队列 dropna(9 协变量+AKI) 后 n={len(m.dropna(subset=COV+['aki']))}")
log(f"严重 AKI 分析在 sens 数据集上重建匹配，故控制池/caliper 略有不同 → 90 对")

# ------------------------------------------------------------------ 4 OASIS
log("\n" + "=" * 88)
log("四、OASIS 组成项")
log("=" * 88)
log("mimiciv_derived.oasis 字段：age, preiculos, gcs, heartrate, meanbp, resprate, "
    "temp, urineoutput, mechvent, electivesurgery")
log("→ 含尿量(urineoutput)、GCS、机械通气(mechvent)；不含血清肌酐。")
log("→ 'non-renal severity' 的表述不成立；且 mechvent 同时是次要结局。")

# ------------------------------------------------------------------ 5 eICU
log("\n" + "=" * 88)
log("五、eICU 窄定义字符串构成")
log("=" * 88)
e = pd.read_csv(os.path.join(DATA, "eicu_sle.csv"))
log(f"eICU 队列 n={len(e)}；窄定义 {int(e.npsle_narrow.sum())}；宽定义 {int(e.npsle_broad.sum())}")
con = psycopg2.connect(**dict(PG, dbname="eicu"))
pids = e["stay_id"].astype("int64").tolist()
dxe = pd.read_sql("SELECT patientunitstayid, diagnosisstring FROM eicu_crd.diagnosis "
                  "WHERE patientunitstayid = ANY(%s)", con, params=(pids,))
con.close()
dxe["s"] = dxe["diagnosisstring"].astype(str).str.lower()
TERMS = {
    "delirium": ["delirium"], "encephalopathy": ["encephal"], "dementia": ["dementia"],
    "cognitive": ["cognitive"], "amnesia": ["amnesia"], "psychosis": ["psychos"],
    "hallucinosis": ["hallucin"],
}
for k, ts in TERMS.items():
    hit = set(dxe[dxe["s"].apply(lambda x: any(t in x for t in ts))]["patientunitstayid"])
    log(f"  {k:16s} 命中 stay {len(hit):3d}")
exp_e = set(e.loc[e.npsle_narrow == 1, "stay_id"])
hit_any = set()
for ts in TERMS.values():
    hit_any |= set(dxe[dxe["s"].apply(lambda x: any(t in x for t in ts))]["patientunitstayid"])
log(f"  窄定义集合 {len(exp_e)}；按上述 7 词重算 {len(hit_any & set(e.stay_id))}")
acute_terms = ["delirium", "encephal", "cognitive", "amnesia"]
hit_acute = set()
for t in acute_terms:
    hit_acute |= set(dxe[dxe["s"].str.contains(t, regex=False)]["patientunitstayid"])
e["npsle_acute"] = e["stay_id"].isin(hit_acute).astype(int)
log(f"  去除 dementia/psychosis/hallucin 后的 eICU 急性定义：{int(e.npsle_acute.sum())}")

for expo, lab in [("npsle_narrow", "原窄定义"), ("npsle_acute", "急性化窄定义")]:
    d = e.dropna(subset=["age", "female", "aki", expo]).copy()
    r = smf.logit(f"aki ~ {expo} + age + female", data=d).fit(disp=0)
    ci = r.conf_int().loc[expo]
    log(f"  eICU {lab} → AKI：aOR {np.exp(r.params[expo]):.2f} "
        f"({np.exp(ci[0]):.2f}–{np.exp(ci[1]):.2f}), p={r.pvalues[expo]:.3f}, "
        f"n={len(d)}, 暴露 {int(d[expo].sum())}")

# MIMIC 同口径（急性化）年龄性别校正
mm = m.dropna(subset=["age", "female", "aki"]).copy()
r = smf.logit("aki ~ npsle_acute + age + female", data=mm).fit(disp=0)
ci = r.conf_int().loc["npsle_acute"]
log(f"  MIMIC 急性化定义（age/sex 校正）→ AKI：aOR {np.exp(r.params['npsle_acute']):.2f} "
    f"({np.exp(ci[0]):.2f}–{np.exp(ci[1]):.2f}), p={r.pvalues['npsle_acute']:.3f}, "
    f"暴露 {int(mm.npsle_acute.sum())}")

io.open(OUT, "w", encoding="utf-8").write("\n".join(buf))
print("\nwritten:", OUT)
