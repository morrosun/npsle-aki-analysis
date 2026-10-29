# -*- coding: utf-8 -*-
"""
v23 审计 A：编码构成可加性核对 + KDIGO / sepsis 实现细节核查

评审核心质疑：Table 6 各类别频数之和 91 ≠ 92 例暴露。
本脚本按「与暴露定义完全相同的匹配规则」逐 stay 打标签，
输出：①每个暴露 stay 的原始编码；②类别归属；③类别并集是否恰为 92；
      ④未被任何类别覆盖的 stay 及其编码（若有）；⑤G93/293/780.0 亚码分布。

另核查评审关于 KDIGO 与 sepsis 实现的问题：
  - kdigo_stages 的时间覆盖范围（入 ICU 前多少小时开始有记录）
  - aki_stage 是否由 crrt 驱动（字段 aki_stage_crrt 存在）
  - 首次 AKI 记录由肌酐还是尿量标准达成
  - sepsis3 的 sofa_time 是否晚于 intime+6h（即「6 小时内脓毒症」是否用了之后的信息）
"""
import io
import os
import warnings

import numpy as np
import pandas as pd
import psycopg2

warnings.filterwarnings("ignore")

BASE = os.environ.get("SLECI_BASE", os.environ.get("SLECI_BASE", "."))
DATA = os.path.join(BASE, "data")
PG = dict(host="localhost", port=5432, user="postgres",
          password=os.environ.get("SLECI_PGPASSWORD", ""), dbname="mimiciv")

buf = []


def log(s=""):
    print(s)
    buf.append(str(s))


m = pd.read_csv(os.path.join(DATA, "mimiciv_sle.csv"))
ids = m["stay_id"].astype("int64").tolist()
exp_stays = set(m.loc[m.npsle_narrow == 1, "stay_id"])

con = psycopg2.connect(**PG)
icu = pd.read_sql("SELECT stay_id, hadm_id, intime, outtime FROM mimiciv_icu.icustays "
                  "WHERE stay_id = ANY(%s)", con, params=(ids,))
hadm = icu["hadm_id"].dropna().astype("int64").unique().tolist()
dx = pd.read_sql("SELECT hadm_id, seq_num, icd_code, icd_version FROM mimiciv_hosp.diagnoses_icd "
                 "WHERE hadm_id = ANY(%s)", con, params=(hadm,))
kd = pd.read_sql("SELECT stay_id, charttime, creat, aki_stage, aki_stage_creat, "
                 "aki_stage_uo, aki_stage_crrt FROM mimiciv_derived.kdigo_stages "
                 "WHERE stay_id = ANY(%s)", con, params=(ids,))
sep = pd.read_sql("SELECT stay_id, suspected_infection_time, sofa_time, sofa_score, sepsis3 "
                  "FROM mimiciv_derived.sepsis3 WHERE stay_id = ANY(%s)", con, params=(ids,))
rrt = pd.read_sql("SELECT stay_id, charttime, dialysis_active, dialysis_present "
                  "FROM mimiciv_derived.rrt WHERE stay_id = ANY(%s)", con, params=(ids,))
con.close()

icu["intime"] = pd.to_datetime(icu["intime"])
icu["outtime"] = pd.to_datetime(icu["outtime"])
kd["charttime"] = pd.to_datetime(kd["charttime"])
for c in ("suspected_infection_time", "sofa_time"):
    sep[c] = pd.to_datetime(sep[c])
rrt["charttime"] = pd.to_datetime(rrt["charttime"])

dx["code"] = dx["icd_code"].astype(str).str.strip().str.upper()
dx["v"] = dx["icd_version"].astype(int)
dx = dx.merge(icu[["hadm_id", "stay_id"]], on="hadm_id", how="inner")

# ------------------------------------------------ 类别定义：与 extract_mimiciv.sql 完全一致
# 窄定义实际 SQL：
#   v10: F05% R41% F06% G93% F01% F02% F03% G30% G31%
#   v9 : 293% 294% 3483 310% 7800%
CATS = [
    ("Delirium",                       ["F05"], 10),
    ("Delirium",                       ["2930", "2931"], 9),
    ("Encephalopathy, other/unspecified", ["G934"], 10),
    ("Brain disorder, other specified", ["G938", "G939"], 10),
    ("Anoxic/hypoxic brain damage",    ["G931"], 10),
    ("Benign intracranial hypertension", ["G932"], 10),
    ("Intracranial hypotension",       ["G933", "G935", "G936", "G937"], 10),
    ("Encephalopathy, ICD-9",          ["3483"], 9),
    ("Organic mental disorder, other specified", ["F06"], 10),
    ("Transient mental disorder, other", ["2938", "2939"], 9),
    ("Amnesia / cognitive symptoms",   ["R41"], 10),
    ("Alteration of consciousness",    ["7800"], 9),
    ("Dementia / neurodegeneration",   ["F01", "F02", "F03", "G30", "G31"], 10),
    ("Persistent mental disorder",     ["294"], 9),
    ("Personality change",             ["310"], 9),
    ("Personality change",             ["F07"], 10),
]

stay_codes = {}
for sid, g in dx.groupby("stay_id"):
    stay_codes[int(sid)] = [(str(r.code), int(r.v)) for r in g.itertuples()]


def cat_hits(codes, pats, ver):
    s = set()
    for c, v in codes:
        if v != ver:
            continue
        for p in pats:
            if c.startswith(p):
                s.add(p)
    return s


log("=" * 92)
log("A1. 类别并集核对（类别定义与窄定义 SQL 匹配规则完全一致）")
log("=" * 92)
log(f"窄定义暴露 stay 数（来自 mimiciv_sle.csv）= {len(exp_stays)}")

cat_of_stay = {s: [] for s in exp_stays}
rows = []
for name, pats, ver in CATS:
    hit = set()
    for s in exp_stays:
        if cat_hits(stay_codes.get(s, []), pats, ver):
            hit.add(s)
            cat_of_stay[s].append(name)
    rows.append((name, ", ".join(pats), ver, len(hit)))

covered = set(s for s in exp_stays if cat_of_stay[s])
uncov = sorted(exp_stays - covered)
log(f"被至少一个类别覆盖的暴露 stay：{len(covered)}")
log(f"未被任何类别覆盖的暴露 stay：{len(uncov)}")
if uncov:
    for s in uncov:
        log(f"  stay {s} 编码：{stay_codes.get(s)}")

log("")
log(f"{'类别':46s} {'规则':28s} {'版本':>4s} {'暴露内':>6s}")
for r in rows:
    log(f"{r[0]:46s} {r[1]:28s} {r[2]:4d} {r[3]:6d}")

# 一个 stay 是否可以被同一 code family 计多次
log("\n各类别命中数之和（重叠不去重）= %d" % sum(r[3] for r in rows))
multi = {s: v for s, v in cat_of_stay.items() if len(v) > 1}
log(f"归属 ≥2 个类别的暴露 stay：{len(multi)}")

# ------------------------------------------------ A2 G93 / 293 / 780.0 亚码分布
log("\n" + "=" * 92)
log("A2. 关键亚码分布（全部 550 stay 中，按 stay 计）")
log("=" * 92)
for pfx, ver, lab in [("G93", 10, "ICD-10 G93.x"), ("293", 9, "ICD-9 293.x"),
                      ("7800", 9, "ICD-9 780.0x"), ("F06", 10, "ICD-10 F06.x"),
                      ("310", 9, "ICD-9 310.x"), ("F07", 10, "ICD-10 F07"),
                      ("R41", 10, "ICD-10 R41.x")]:
    sub = dx[(dx.v == ver) & dx.code.str.startswith(pfx)]
    log(f"\n{lab}：{sub.code.nunique()} 个不同编码，涉及 {sub.stay_id.nunique()} stay")
    g = sub.groupby("code").agg(n_stay=("stay_id", "nunique"))
    g["exposed"] = [len(set(sub[sub.code == c].stay_id) & exp_stays) for c in g.index]
    g = g.sort_values("n_stay", ascending=False)
    for c, r in g.iterrows():
        log(f"    {c:8s} stay {int(r.n_stay):3d}  其中暴露 {int(r.exposed):3d}")

# ------------------------------------------------ A3 以 stay 为单位还原「5 类」说法
log("\n" + "=" * 92)
log("A3. 按「族」去重后的暴露构成（族 = 临床概念，而非编码前缀）")
log("=" * 92)
FAM = {
    "Delirium (F05; 293.0–293.1)": ([("F05", 10), ("2930", 9), ("2931", 9)]),
    "Encephalopathy (G93.x; 348.3)": ([("G93", 10), ("3483", 9)]),
    "Amnesia / cognitive symptoms (R41.x)": ([("R41", 10)]),
    "Alteration of consciousness (780.0x)": ([("7800", 9)]),
    "Organic mental disorder, other (F06.x; 293.8x, 293.9)": ([("F06", 10), ("2938", 9), ("2939", 9)]),
    "Dementia / neurodegeneration (F01–F03, G30, G31)": ([("F01", 10), ("F02", 10), ("F03", 10), ("G30", 10), ("G31", 10)]),
    "Persistent mental disorder (294.x)": ([("294", 9)]),
    "Personality change (310.x; F07)": ([("310", 9), ("F07", 10)]),
}
fam_hit = {}
for name, pats in FAM.items():
    hit = set()
    for s in exp_stays:
        for p, v in pats:
            if cat_hits(stay_codes.get(s, []), [p], v):
                hit.add(s)
                break
    fam_hit[name] = hit
    log(f"{name:56s} {len(hit):3d}  ({100*len(hit)/len(exp_stays):4.1f}%)")
u = set().union(*fam_hit.values())
log(f"\n族并集 = {len(u)}；暴露 = {len(exp_stays)}；差集 = {sorted(exp_stays - u)}")

# 慢性-only
chronic_only = set()
for s in exp_stays:
    fams = [k for k, v in fam_hit.items() if s in v]
    if fams and all(k.startswith(("Dementia", "Persistent")) for k in fams):
        chronic_only.add(s)
log(f"仅慢性类（痴呆/持续精神障碍）而无任何急性类：{len(chronic_only)}")

# ------------------------------------------------ B1 KDIGO 时间覆盖
log("\n" + "=" * 92)
log("B1. kdigo_stages 时间覆盖与 AKI 判定来源")
log("=" * 92)
k = kd.merge(icu[["stay_id", "intime", "outtime"]], on="stay_id", how="left")
k["h"] = (k["charttime"] - k["intime"]).dt.total_seconds() / 3600.0
log(f"kdigo_stages 记录总行数 {len(k)}，覆盖 {k.stay_id.nunique()} / 550 stay")
log(f"相对 intime 的小时数：min {k.h.min():.1f}, P1 {k.h.quantile(.01):.1f}, "
    f"中位 {k.h.median():.1f}, P99 {k.h.quantile(.99):.1f}, max {k.h.max():.1f}")
log(f"记录早于 intime 者 {int((k.h < 0).sum())} 行，涉及 {k.loc[k.h<0,'stay_id'].nunique()} stay；"
    f"最早 {-k.h.min():.1f} h 前")
log(f"记录晚于 outtime 者 {int((k.h > (k.outtime-k.intime).dt.total_seconds()/3600).sum())} 行")

pos = k[k.aki_stage >= 1].sort_values(["stay_id", "charttime"])
first = pos.groupby("stay_id").first().reset_index()
log(f"\nICU 内（h≥0）首次 aki_stage≥1 的 stay：{first.pipe(lambda d: d[d.h>=0]).stay_id.nunique()}")
f0 = first[first.h >= 0]
log("首次达标记录的判定来源（该行各分项是否为阳性）：")
log(f"  aki_stage_creat ≥1：{int((f0.aki_stage_creat >= 1).sum())} / {len(f0)}")
log(f"  aki_stage_uo    ≥1：{int((f0.aki_stage_uo >= 1).sum())} / {len(f0)}")
log(f"  aki_stage_crrt  ≥1：{int((f0.aki_stage_crrt >= 1).sum())} / {len(f0)}")
both = int(((f0.aki_stage_creat >= 1) & (f0.aki_stage_uo >= 1)).sum())
log(f"  仅肌酐达标 {int(((f0.aki_stage_creat>=1)&(f0.aki_stage_uo<1)).sum())}；"
    f"仅尿量达标 {int(((f0.aki_stage_uo>=1)&(f0.aki_stage_creat<1)).sum())}；"
    f"两者同时 {both}；仅 CRRT {int(((f0.aki_stage_crrt>=1)&(f0.aki_stage_creat<1)&(f0.aki_stage_uo<1)).sum())}")
log(f"任意时点 aki_stage_crrt ≥1 的 stay：{k.loc[k.aki_stage_crrt>=1,'stay_id'].nunique()}")

# 维持性透析：rrt 表 dialysis_present
log(f"\nmimiciv_derived.rrt：{rrt.stay_id.nunique()} stay 有记录")
if len(rrt):
    present = rrt.groupby("stay_id")["dialysis_present"].max()
    active = rrt.groupby("stay_id")["dialysis_active"].max()
    log(f"  dialysis_present=1 的 stay {int(present.sum())}；dialysis_active=1 的 stay {int(active.sum())}")
    r0 = rrt.merge(icu[["stay_id", "intime"]], on="stay_id", how="left")
    r0["h"] = (r0.charttime - r0.intime).dt.total_seconds() / 3600.0
    firstrrt = r0.groupby("stay_id")["h"].min()
    log(f"  首次 rrt 记录相对 intime：<0 h 的 stay {int((firstrrt<0).sum())}；"
        f"≤6 h 的 stay {int((firstrrt<=6).sum())}")

# ------------------------------------------------ B2 sepsis 时间实现
log("\n" + "=" * 92)
log("B2. sepsis-3 的实现与「首 6 小时」的定义")
log("=" * 92)
s = sep.merge(icu[["stay_id", "intime"]], on="stay_id", how="left")
s["h_inf"] = (s.suspected_infection_time - s.intime).dt.total_seconds() / 3600.0
s["h_sofa"] = (s.sofa_time - s.intime).dt.total_seconds() / 3600.0
log(f"sepsis3 表覆盖 {len(sep)} / 550 stay；sepsis3=True 者 {int(sep.sepsis3.fillna(False).sum())}")
log(f"suspected_infection_time 相对 intime：中位 {s.h_inf.median():.1f} h，"
    f"P10 {s.h_inf.quantile(.1):.1f}，P90 {s.h_inf.quantile(.9):.1f}")
log(f"sofa_time 相对 intime：中位 {s.h_sofa.median():.1f} h，P90 {s.h_sofa.quantile(.9):.1f}，"
    f"max {s.h_sofa.max():.1f}")
early = s[(s.sepsis3 == True) & (s.h_inf <= 6)]
log(f"「早期脓毒症」（sepsis3=True 且感染疑诊 ≤6 h）：{len(early)}")
log(f"  其中 sofa_time 晚于 intime+6 h：{int((early.h_sofa > 6).sum())}"
    f"（SOFA 窗口含感染疑诊后的数据，故该变量并非纯 6 h 内信息）")
log(f"  其中 sofa_time 早于 intime：{int((early.h_sofa < 0).sum())}")
log(f"  其中感染疑诊早于 intime：{int((early.h_inf < 0).sum())}")

io.open(os.path.join(BASE, "analysis", "v23_audit_codes_report.txt"), "w",
        encoding="utf-8").write("\n".join(buf))
print("\nwritten: analysis/v23_audit_codes_report.txt")
