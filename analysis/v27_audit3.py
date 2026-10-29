# -*- coding: utf-8 -*-
"""v27 第三审计：追查 82/148/107/37 的来源；解释 aki_stage 与分项最大值不符；修 348.3 计数。"""
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
kd = pd.read_sql("""select stay_id,charttime,aki_stage,aki_stage_smoothed,aki_stage_creat,aki_stage_uo,
 aki_stage_crrt,creat from mimiciv_derived.kdigo_stages where stay_id=any(%s) order by stay_id,charttime""", con, params=(ids,))
dx = pd.read_sql("select hadm_id,icd_code,icd_version from mimiciv_hosp.diagnoses_icd where hadm_id=any(%s)",
                 con, params=(icu.hadm_id.dropna().astype('int64').unique().tolist(),))
con.close()
kd["charttime"] = pd.to_datetime(kd.charttime); icu["intime"] = pd.to_datetime(icu.intime)
kd["h"] = (kd.charttime - kd.stay_id.map(icu.set_index("stay_id").intime)).dt.total_seconds() / 3600
kd = kd[(kd.h >= 0)].copy()          # ICU 窗

out = []
def P(*a):
    s = " ".join(str(x) for x in a); out.append(s); print(s)

C, U, R = "aki_stage_creat", "aki_stage_uo", "aki_stage_crrt"

# ---------- 1. 首个综合达标记录上的分项（raw vs smoothed） ----------
pos = kd[kd.aki_stage.fillna(0) >= 1].sort_values(["stay_id", "h"])
first = pos.groupby("stay_id", sort=False).head(1).set_index("stay_id")
P("=" * 70)
P("1. 首个综合达标记录的分项构成（374 例）")
P("=" * 70)
P("  raw 三列：仅C", int(((first[C] >= 1) & ~(first[U] >= 1)).sum()),
  "仅U", int(((first[U] >= 1) & ~(first[C] >= 1)).sum()),
  "both", int(((first[C] >= 1) & (first[U] >= 1)).sum()),
  "crrt", int((first[R] >= 1).sum()))
first_s = pos.groupby("stay_id", sort=False).head(1).set_index("stay_id")
P("  注：kdigo_stages 无 smoothed 分项列，只有 aki_stage_smoothed；对照 aki_stage vs smoothed:")
P("    aki_stage != smoothed 的行数:", int((kd.aki_stage.fillna(-1) != kd.aki_stage_smoothed.fillna(-1)).sum()))

# ---------- 2. 82/148/107/37 可能来源：任意时刻曾达标 的互斥划分 ----------
Cev = kd[kd[C].fillna(0) >= 1].stay_id.unique()
Uev = kd[kd[U].fillna(0) >= 1].stay_id.unique()
Rev = kd[kd[R].fillna(0) >= 1].stay_id.unique()
comp = set(first.index)
P("")
P("=" * 70)
P("2. 尝试复现 82/148/107/37")
P("=" * 70)
P("  374 例中: creat 曾达标", len(set(Cev) & comp), "uo 曾达标", len(set(Uev) & comp),
  "crrt 曾达标", len(set(Rev) & comp),
  "creat∩uo", len((set(Cev) & set(Uev)) & comp))
for lab, sel in [("仅creat(曾)", (set(Cev) & comp) - set(Uev) - set(Rev)),
                 ("仅uo(曾)", (set(Uev) & comp) - set(Cev) - set(Rev)),
                 ("creat+uo(曾)", ((set(Cev) & set(Uev)) & comp) - set(Rev)),
                 ("crrt(曾)", set(Rev) & comp)]:
    P(f"    {lab:14s} {len(sel)}")

# ---------- 3. aki_stage 与 max(分项) 不符的解释 ----------
mx = kd[[C, U, R]].max(axis=1)
d = kd.aki_stage.fillna(-1) - mx.fillna(-1)
P("")
P("=" * 70)
P("3. aki_stage 与 max(三项) 的差异")
P("=" * 70)
P("  diff 分布:", d.value_counts().to_dict())
sm = kd.aki_stage_smoothed.fillna(-1) - mx.fillna(-1)
P("  aki_stage_smoothed - max(三项):", sm.value_counts().to_dict())
sub = kd[d > 0]
P("  aki_stage > max(三项) 的", len(sub), "行，前 4 行：")
if len(sub):
    P(sub[[ "stay_id", "charttime", "aki_stage", "aki_stage_smoothed", C, U, R, "creat"]].head(4).to_string(index=False))
P("  说明：aki_stage 为 KDIGO 综合分期（含 crrt 权重与 smoothed 前向保持），max(三项) 只看同一条记录；")
P("        阈值 ≥1 的首次时间在 374/374 例上仍等于三项首达时间的最小值（见 v27_audit2 第 3 节）。")

# ---------- 4. 348.3 计数（修 audit2 的 0） ----------
dx["code"] = dx.icd_code.astype(str).str.strip().str.upper(); dx["v"] = dx.icd_version.astype(int)
dd = dx.merge(icu, on="hadm_id")
exp92 = set(m.loc[m.npsle_narrow == 1, "stay_id"])
P("")
P("=" * 70)
P("4. 348.3 / G93 在 92 例窄定义中的计数")
P("=" * 70)
s348 = set(dd[(dd.code.str.startswith("3483")) & dd.stay_id.isin(exp92)].stay_id)
sG93 = set(dd[(dd.v == 10) & dd.code.str.startswith("G93") & dd.stay_id.isin(exp92)].stay_id)
P("  348.3:", len(s348), " G93*:", len(sG93), " 并集:", len(s348 | sG93), " 交集:", len(s348 & sG93))
P("  348.3 明细:", dd[(dd.code.str.startswith('3483')) & dd.stay_id.isin(exp92)].code.value_counts().to_dict())

open(f"{BASE}/analysis/v27_audit3_report.txt", "w", encoding="utf-8").write("\n".join(out))
print("\n[written] v27_audit3_report.txt")
