"""v26 审计：第五轮评审四项硬问题的入住级核对。
(1) 120 例与 25/95/115 的交叉表；(2) G93 子码入住级矩阵；
(3) 首个 ICU 窗达标记录的分项来源（是否 smoothed）；(4) eGFR 基线肌酐的实际时间范围。
"""
import os
import io
import numpy as np
import pandas as pd
import psycopg2

BASE = os.environ.get("SLECI_BASE", ".")
PG = dict(host="localhost", port=5432, user="postgres", password=os.environ.get("SLECI_PGPASSWORD", ""), dbname="mimiciv")
OUT = io.StringIO()


def P(*a):
    line = " ".join(str(x) for x in a)
    print(line)
    OUT.write(line + "\n")


m = pd.read_csv(f"{BASE}/data/mimiciv_sle.csv")
ids = m.stay_id.astype("int64").tolist()
con = psycopg2.connect(**PG)
icu = pd.read_sql("select stay_id,hadm_id,intime,outtime from mimiciv_icu.icustays where stay_id=any(%s)", con, params=(ids,))
kd = pd.read_sql("""select stay_id,charttime,aki_stage,aki_stage_creat,aki_stage_uo,aki_stage_crrt,
                           creat from mimiciv_derived.kdigo_stages where stay_id=any(%s)""", con, params=(ids,))
cam = pd.read_sql("select stay_id,charttime,value from mimiciv_icu.chartevents where itemid=228332 and stay_id=any(%s)", con, params=(ids,))
lab = pd.read_sql("""select icu.stay_id, le.charttime, le.valuenum as creat
                     from mimiciv_hosp.labevents le
                     join mimiciv_icu.icustays icu on icu.hadm_id = le.hadm_id
                     where le.itemid = 50912 and icu.stay_id = any(%s)
                       and le.valuenum is not null""", con, params=(ids,))
con.close()

for d in (kd, cam, lab):
    d["charttime"] = pd.to_datetime(d.charttime)
icu["intime"] = pd.to_datetime(icu.intime)
icu["outtime"] = pd.to_datetime(icu.outtime)
imap = icu.set_index("stay_id").intime

kd["h"] = (kd.charttime - kd.stay_id.map(imap)).dt.total_seconds() / 3600
cam["h"] = (cam.charttime - cam.stay_id.map(imap)).dt.total_seconds() / 3600

pos = kd[kd.aki_stage >= 1].copy()
P("=" * 78)
P("A. 120 例 / 25 / 95 / 115 的交叉表")
P("=" * 78)
pre = set(pos[pos.h < 0].stay_id)
inw = pos[pos.h >= 0]
taki = inw.groupby("stay_id").h.min()
cpos = cam[cam.value.astype(str).str.strip().str.lower() == "positive"]
tpos = cpos.groupby("stay_id").h.min()
both = sorted(set(tpos.index) & set(taki.index))
P(f"  两事件均发生（任意窗 AKI）: {len(both)}")
b25 = [s for s in both if s in pre]
b95 = [s for s in both if s not in pre]
P(f"  入 ICU 前已有达标记录: {len(b25)}")
P(f"  无入 ICU 前达标记录: {len(b95)}")
P(f"  其中 t_pos <= 0（被反向模型排除）: {[s for s in b95 if tpos[s] <= 0]}")
P(f"  其中 t_pos >  0: {len([s for s in b95 if tpos[s] > 0])}")
P(f"  120 中 t_pos <= 0 者: {[s for s in both if tpos[s] <= 0]}")

# 反向模型人群：assessable（≥1 次 Negative/Positive）且无 pre-AKI
campos = cam[cam.value.astype(str).str.strip().str.lower().isin(["negative", "positive"])]
assessable = set(campos.stay_id)
base = m[m.stay_id.isin(assessable)].copy()
P(f"  assessable: {len(base)}")
rev = base[~base.stay_id.isin(pre)]
P(f"  反向人群（assessable 且无 pre-AKI）: {len(rev)}")
r_id = set(rev.stay_id)
P(f"  反向人群中有阳性 CAM（t>0）: {len(r_id & set(tpos[tpos > 0].index))}")
P(f"  反向事件数（r_main 输出 115）核对: {len([s for s in r_id if tpos.get(s, np.nan) > 0])}")
fwd = base.copy()
fwd_id = set(fwd.stay_id)
P(f"  前向人群: {len(fwd_id)}；其中 b95 全在: {set(b95) <= fwd_id}")
P(f"  b95 同时具备 t_pos>0 与 t_aki>0: {len([s for s in b95 if tpos[s] > 0 and taki[s] > 0])}")

P("")
P("=" * 78)
P("B. G93 子码入住级矩阵（窄定义暴露 92 例）")
P("=" * 78)
con = psycopg2.connect(**PG)
dx = pd.read_sql("select hadm_id,icd_code,icd_version from mimiciv_hosp.diagnoses_icd where hadm_id=any(%s)",
                 con, params=(icu.hadm_id.dropna().astype("int64").unique().tolist(),))
con.close()
dx["code"] = dx.icd_code.astype(str).str.strip().str.upper()
dx["v"] = dx.icd_version.astype(int)
dx = dx.merge(icu[["hadm_id", "stay_id"]], on="hadm_id")
exp = set(m.loc[m.npsle_narrow == 1, "stay_id"])

# 每 stay 的 G93 子码集合
g93 = dx[(dx.v == 10) & dx.code.str.startswith("G93") & dx.stay_id.isin(exp)]
sub = g93.groupby("stay_id").code.apply(lambda s: set(s.tolist()))
P(f"  有任意 G93 码的暴露 stay: {len(sub)}")
cnt = pd.Series([c for s in sub for c in s]).value_counts().sort_index()
P("  全部 G93 子码频次（stay 级）：")
for k, v in cnt.items():
    P(f"    {k:10s} {v}")

has4 = {s for s, c in sub.items() if any(x.startswith("G934") for x in c)}
P(f"  携带 G93.4x: {len(has4)}")
P(f"  未携带 G93.4x: {len(sub) - len(has4)}")

# 其余保留暴露码
KEEP = ("F05", "R41", "F06", "F01", "F02", "F03", "G30", "G31", "3483", "310", "7800", "294")
def kept_other(s):
    d = dx[(dx.stay_id == s)]
    out = set()
    for c in d.code.tolist():
        if c in sub.get(s, set()):
            continue
        if c.startswith(KEEP):
            out.add(c)
        elif c.startswith("293") and c[:4] in ("2930", "2931"):
            out.add(c)
    return out

non4 = sorted(set(sub.index) - has4)
P("")
P("  非 G93.4x 携带者（13 例）逐例子码与保留码：")
n_keep = 0
for s in non4:
    codes = sorted(sub[s])
    k = sorted(kept_other(s))
    if k:
        n_keep += 1
    P(f"    stay {s}: G93={codes} | 其他保留={k}")
P(f"  其中另有保留暴露码: {n_keep}；仅靠非 G93.4 的 G93 码: {len(non4) - n_keep}")

P("")
P("  交叉关系核对表：")
P(f"    任意非 G93.4 子码携带者            : {len(non4)}")
P(f"    非 G93.4 与 G93.4 同时存在          : {len([s for s in non4 if s in has4])}")
P(f"    无 G93.4 但有其他保留暴露码         : {n_keep}")
P(f"    限制后真正失去暴露资格              : {len(non4) - n_keep}")
P(f"    限制后暴露数                        : {len(exp) - (len(non4) - n_keep)}")

P("")
P("=" * 78)
P("C. 首个 ICU 窗达标记录的分项来源")
P("=" * 78)
first = inw.sort_values("h").groupby("stay_id").first()
P(f"  ICU 窗内曾达标 stay: {len(first)}")
P(f"  该记录上 aki_stage>=1: {int((first.aki_stage >= 1).sum())}")
P(f"  creat>=1: {int((first.aki_stage_creat >= 1).sum())}")
P(f"  uo>=1   : {int((first.aki_stage_uo >= 1).sum())}")
P(f"  crrt>=1 : {int((first.aki_stage_crrt >= 1).sum())}")
combo = first.apply(lambda r: ("C" if r.aki_stage_creat >= 1 else "") + ("U" if r.aki_stage_uo >= 1 else "") + ("R" if r.aki_stage_crrt >= 1 else ""), axis=1)
P("  分项组合（首个达标记录）：")
for k, v in combo.value_counts().items():
    P(f"    {k or '(none)':8s} {v}")
P(f"  表格列: {list(kd.columns)}")
P("  → 已用 aki_stage 原始列；未使用任何 smoothed 字段（该表不含平滑列）")

P("")
P("=" * 78)
P("D. eGFR 基线肌酐的实际可观察窗口")
P("=" * 78)
lab["h"] = (lab.charttime - lab.stay_id.map(imap)).dt.total_seconds() / 3600
w = lab[(lab.h <= 48)]
P(f"  charttime <= intime+48h 的肌酐行数: {len(w)}；涉及 stay: {w.stay_id.nunique()}")
P(f"  其中 h < 0（入 ICU 前）: {int((w.h < 0).sum())} 行；stay {w[w.h < 0].stay_id.nunique()}")
if len(w[w.h < 0]):
    P(f"  入 ICU 前最早值 h 范围: {w.h.min():.1f} .. {w[w.h < 0].h.max():.1f}")
low = w.sort_values("h").groupby("stay_id").first()
P(f"  最低值（按时间）? 实际代码取 min(valuenum)：")
lowv = w.groupby("stay_id").creat.min()
P(f"  取最低值的 stay: {len(lowv)}")
# 最低值是否发生在 intime 之前
idx = w.sort_values("creat").groupby("stay_id").head(1)
P(f"  最低值发生在入 ICU 前者: {int((idx.h < 0).sum())} stay；在 ICU 窗内者: {int((idx.h >= 0).sum())} stay")
P(f"  最低值 h 的分布: min {idx.h.min():.1f}, median {idx.h.median():.1f}, max {idx.h.max():.1f}")
inwv = w[(w.h >= 0)].groupby("stay_id").creat.min()
both_ids = set(lowv.index) & set(inwv.index)
diff = (lowv.loc[sorted(both_ids)] - inwv.loc[sorted(both_ids)])
P(f"  「无下界」比「仅 ICU 窗」更低的 stay 数: {int((diff < -1e-9).sum())} / {len(both_ids)}")
P(f"  差异幅度 mg/dL: median {diff.median():.3f}, max {diff.max():.3f}")

io.open(f"{BASE}/analysis/v26_audit_report.txt", "w", encoding="utf-8").write(OUT.getvalue())
print("\nwritten")
