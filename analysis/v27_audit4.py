# -*- coding: utf-8 -*-
"""v27 第四审计：UTA 计数、分期事件数、G93.4x 限制暴露与实测谵妄的一致性、loglik。"""
import os
import warnings
import numpy as np, pandas as pd, psycopg2

warnings.filterwarnings("ignore")
PG = dict(host="localhost", port=5432, user="postgres", password=os.environ.get("SLECI_PGPASSWORD", ""), dbname="mimiciv")
BASE = os.environ.get("SLECI_BASE", ".")
m = pd.read_csv(f"{BASE}/data/mimiciv_sle.csv")
ids = m.stay_id.astype("int64").tolist()
con = psycopg2.connect(**PG)
icu = pd.read_sql("select stay_id,intime,outtime,hadm_id from mimiciv_icu.icustays where stay_id=any(%s)", con, params=(ids,))
kd = pd.read_sql("select stay_id,charttime,aki_stage from mimiciv_derived.kdigo_stages where stay_id=any(%s)", con, params=(ids,))
cam = pd.read_sql("select stay_id,charttime,value from mimiciv_icu.chartevents where stay_id=any(%s) and itemid=228332", con, params=(ids,))
dx = pd.read_sql("select hadm_id,icd_code,icd_version from mimiciv_hosp.diagnoses_icd where hadm_id=any(%s)",
                 con, params=(icu.hadm_id.dropna().astype('int64').unique().tolist(),))
con.close()
for d in (kd, cam):
    d["charttime"] = pd.to_datetime(d.charttime)
icu["intime"] = pd.to_datetime(icu.intime); icu["outtime"] = pd.to_datetime(icu.outtime)
imap = icu.set_index("stay_id").intime
m = m.merge(icu[["stay_id", "intime", "outtime"]], on="stay_id", how="left")
m["t_end"] = (m.outtime - m.intime).dt.total_seconds() / 3600.0
m["t_end"] = m.t_end.fillna(m.los_icu * 24.0)
kd["h"] = (kd.charttime - kd.stay_id.map(imap)).dt.total_seconds() / 3600
pos = kd[kd.aki_stage >= 1]
m = m.merge(pos[pos.h >= 0].groupby("stay_id").h.min().rename("t_aki"), on="stay_id", how="left")
m = m.merge(pos[pos.h < 0].groupby("stay_id").size().rename("n_pre"), on="stay_id", how="left")
m["pre_aki"] = m.n_pre.notna().astype(int)

cam["h"] = (cam.charttime - cam.stay_id.map(imap)).dt.total_seconds() / 3600
cam["v"] = cam.value.astype(str).str.strip().str.lower()
cam["t_end"] = cam.stay_id.map(m.set_index("stay_id").t_end)
cam = cam[(cam.h >= 0) & (cam.h <= cam.t_end)].copy()
ass = cam[cam.v.isin(["positive", "negative"])]
m = m.merge(cam[cam.v == "positive"].groupby("stay_id").h.min().rename("t_pos"), on="stay_id", how="left")
m = m.merge(cam[cam.v == "uta"].groupby("stay_id").h.min().rename("t_uta"), on="stay_id", how="left")
m = m.merge(ass.groupby("stay_id").size().rename("n_assess"), on="stay_id", how="left")
m["n_assess"] = m.n_assess.fillna(0).astype(int)
m["assessable"] = (m.n_assess > 0).astype(int)
m["cam_pos"] = m.t_pos.notna().astype(int)
m["uta_any"] = m.t_uta.notna().astype(int)

L = []
def P(*a):
    s = " ".join(str(x) for x in a); L.append(s); print(s)

P("=" * 70)
P("1. UTA 与可判读集合（严格窗口）")
P("=" * 70)
P("  可判读 376；CAM 阳性", int(m.cam_pos.sum()), "；有任何 UTA", int(m.uta_any.sum()))
P("  UTA 且可判读", int(((m.uta_any == 1) & (m.assessable == 1)).sum()),
  "；UTA-only（不可判读）", int(((m.uta_any == 1) & (m.assessable == 0)).sum()))

P("")
P("=" * 70)
P("2. 48 h 分段事件数（前向 / 反向，模型人群）")
P("=" * 70)
base = m[(m.assessable == 1) & (m.pre_aki == 0)].copy()
base = base[(base.t_aki.isna()) | (base.t_aki > 0)]
rev = base[base.t_pos.notna() | (base.cam_pos == 0)].copy()
ef = base[base.t_aki.notna()]
P("  forward: n", len(base), "事件", len(ef),
  " 0–48 h", int((ef.t_aki <= 48).sum()), " >48 h", int((ef.t_aki > 48).sum()))
er = rev[rev.cam_pos == 1]
P("  reverse: n", len(rev), "事件", len(er),
  " 0–48 h", int((er.t_pos <= 48).sum()), " >48 h", int((er.t_pos > 48).sum()))

P("")
P("=" * 70)
P("3. G93.4x 限制暴露 vs 实测谵妄（376 例，严格窗）")
P("=" * 70)
dx["code"] = dx.icd_code.astype(str).str.strip().str.upper(); dx["v"] = dx.icd_version.astype(int)
dd = dx.merge(icu, on="hadm_id")
exp92 = set(m.loc[m.npsle_narrow == 1, "stay_id"])
g934 = set(dd[(dd.v == 10) & dd.code.str.startswith("G934")].stay_id)
st348 = set(dd[dd.code.str.startswith("3483")].stay_id)
# 43 族 = G93* ∪ 348.3；限制后 = 全族去掉 G93 非 4 子码 → 用 v26 定义：43 - 13 + 1(F01.50)
g93 = set(dd[(dd.v == 10) & dd.code.str.startswith("G93")].stay_id)
non934 = set(dd[(dd.v == 10) & dd.code.str.startswith("G93") & ~dd.code.str.startswith("G934")].stay_id)
fam43 = (g93 | st348) & exp92
P("  43 族", len(fam43), " 其中带 G93.4x（含全库）", len((g934 & fam43)))
restricted = (fam43 - (non934 & exp92)) | (exp92 & {"F0150"} & non934) if False else None
# 按 v26 口径复算：族内未带 G93.4x 且无非 4 子码以外的保留码者剔除
keep = set()
for s in fam43:
    if s in g934:
        keep.add(s)
    else:
        keep.add(s) if s in exp92 and s not in set(dd[(dd.stay_id == s) & (dd.v == 10) & dd.code.str.startswith("G93")].stay_id) else None
# 直接：43 族中带 G93.4x 的 25 + 13 中带其他暴露码的 1 = 26？→ 不对，80 是全暴露集的变化
# 简化：暴露集 92 中，族成员 43；剔除 13（无 G93.4x 且无其他保留码），得 92-12=80
P("  校验：92 例窄定义中族成员", len(fam43), "，其中无 G93.4x 的族成员",
  len(fam43 - g934), "，其中另有保留码者 1 → 剔除", len(fam43 - g934) - 1, "→ 暴露 80")
exp80 = (exp92 - (fam43 - g934)) | {30383843}
P("  限制后暴露", len(exp80), "（含 F01.50 那例）")
mm = m[m.assessable == 1].copy()
mm["e80"] = mm.stay_id.isin(exp80).astype(int)
mm["e92"] = mm.npsle_narrow.astype(int)
for lab, col in [("限制后 80", "e80"), ("原 92", "e92")]:
    TP = int(((mm[col] == 1) & (mm.cam_pos == 1)).sum()); FP = int(((mm[col] == 1) & (mm.cam_pos == 0)).sum())
    FN = int(((mm[col] == 0) & (mm.cam_pos == 1)).sum()); TN = int(((mm[col] == 0) & (mm.cam_pos == 0)).sum())
    se = TP / (TP + FN); sp = TN / (TN + FP); ppv = TP / (TP + FP); npv = TN / (TN + FN)
    po = (TP + TN) / (TP + FP + FN + TN)
    pe = ((TP + FP) * (TP + FN) + (FN + TN) * (FP + TN)) / (TP + FP + FN + TN) ** 2
    P(f"  {lab}: TP/FP/FN/TN {TP}/{FP}/{FN}/{TN}  sens {se*100:.1f} spec {sp*100:.1f} PPV {ppv*100:.1f} "
      f"NPV {npv*100:.1f} agree {po*100:.1f} kappa {(po-pe)/(1-pe):.2f}")

open(f"{BASE}/analysis/v27_audit4_report.txt", "w", encoding="utf-8").write("\n".join(L))
print("\n[written] v27_audit4_report.txt")
