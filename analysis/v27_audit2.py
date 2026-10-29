# -*- coding: utf-8 -*-
"""v27 审计 2：Table 5 区间、Table 4 Panel A 的 95 例排序、G93 集合表、AKI 分项。"""
import io
import os

import numpy as np
import pandas as pd
import psycopg2
from scipy import stats

BASE = os.environ.get("SLECI_BASE", os.environ.get("SLECI_BASE", "."))
PG = dict(host="localhost", port=5432, user="postgres",
          password=os.environ.get("SLECI_PGPASSWORD", ""), dbname="mimiciv")
OUT = io.StringIO()


def P(*a):
    line = " ".join(str(x) for x in a)
    print(line); OUT.write(line + "\n")


def cp(k, n):
    lo, hi = stats.beta.ppf([0.025, 0.975], k + 0.5, n - k + 0.5)
    return 100 * k / n, 100 * lo, 100 * hi


P("=" * 84)
P("1. Table 5 Panel B（严格 CAM 窗口，n=376；51/24/89/212）")
P("=" * 84)
TP, FN, FP, TN = 51, 24, 89, 212
for lab, k, n in [("Sensitivity", TP, TP + FN), ("Specificity", TN, TN + FP),
                  ("PPV", TP, TP + FP), ("NPV", TN, TN + FN)]:
    v, lo, hi = cp(k, n)
    P(f"  {lab:12s} {v:5.1f}% ({lo:.1f}–{hi:.1f})  {k}/{n}")
po = (TP + TN) / (TP + FN + FP + TN)
pe = ((TP + FP) * (TP + FN) + (FN + TN) * (FP + TN)) / (TP + FN + FP + TN) ** 2
P(f"  Overall agreement {100*po:.1f}%   Cohen κ {(po-pe)/(1-pe):.2f}")
v, lo, hi = cp(TP + TN, 376)
P(f"  Overall agreement CI {lo:.1f}–{hi:.1f}")
P(f"  McNemar: discordant {FP} vs {FN}, exact p = "
  f"{stats.binomtest(min(FP, FN), FP + FN, 0.5).pvalue:.4f}")

# ------------------------------------------------------------------ 数据
m = pd.read_csv(f"{BASE}/data/mimiciv_sle.csv")
ids = m.stay_id.astype("int64").tolist()
con = psycopg2.connect(**PG)
icu = pd.read_sql("select stay_id,hadm_id,intime,outtime from mimiciv_icu.icustays "
                  "where stay_id=any(%s)", con, params=(ids,))
kd = pd.read_sql("""select stay_id,charttime,aki_stage,aki_stage_creat,aki_stage_uo,
                           aki_stage_crrt from mimiciv_derived.kdigo_stages
                    where stay_id=any(%s) order by stay_id,charttime""",
                 con, params=(ids,))
cam = pd.read_sql("select stay_id,charttime,value from mimiciv_icu.chartevents "
                  "where itemid=228332 and stay_id=any(%s)", con, params=(ids,))
con.close()
for d in (kd, cam):
    d["charttime"] = pd.to_datetime(d.charttime)
icu["intime"] = pd.to_datetime(icu.intime); icu["outtime"] = pd.to_datetime(icu.outtime)
icu["t_end"] = (icu.outtime - icu.intime).dt.total_seconds() / 3600
imap = icu.set_index("stay_id")["intime"]
kd["h"] = (kd.charttime - kd.stay_id.map(imap)).dt.total_seconds() / 3600
cam["h"] = (cam.charttime - cam.stay_id.map(imap)).dt.total_seconds() / 3600
cam["t_end"] = cam.stay_id.map(icu.set_index("stay_id")["t_end"])
cam["v"] = cam.value.astype(str).str.strip().str.lower()
cam = cam[(cam.h >= 0) & (cam.h <= cam.t_end)].copy()      # 严格窗口
P(f"\n[cam] 严格窗口后记录 {len(cam)} 条 / {cam.stay_id.nunique()} 例")

pos = kd[kd.aki_stage >= 1]
pre = set(pos.loc[pos.h < 0, "stay_id"])
taki = pos[pos.h >= 0].groupby("stay_id").h.min()
tpos = cam[cam.v == "positive"].groupby("stay_id").h.min()

P("")
P("=" * 84)
P("2. Table 4 Panel A：记录次序（95 例，严格窗口）")
P("=" * 84)
both = sorted(set(taki.index) & set(tpos.index))
b25 = [s for s in both if s in pre]
b95 = [s for s in both if s not in pre]
P(f"  两事件均为 ICU 内记录: {len(both)}；其中入 ICU 前已达标 {len(b25)}；剩 {len(b95)}")
gap = pd.Series({s: taki[s] - tpos[s] for s in b95})
for lab, sel in [("AKI 先（>6h）", gap > 6), ("±6h 同期", gap.abs() <= 6),
                 ("谵妄先（>6h）", gap < -6)]:
    P(f"  {lab:14s} {int(sel.sum()):3d}  ({100*sel.mean():.1f}%)")
P(f"  仅不一致组中位间隔：谵妄先 {gap[gap < -6].abs().median():.1f} h；"
  f"AKI 先 {gap[gap > 6].median():.1f} h")
gap120 = pd.Series({s: taki[s] - tpos[s] for s in both})
P(f"  120 例：{(gap120>6).sum()}/{(gap120.abs()<=6).sum()}/{(gap120<-6).sum()}；"
  f"仅不一致组中位 {gap120[gap120<-6].abs().median():.1f} / {gap120[gap120>6].median():.1f} h")
P(f"  95 例中 t_pos<=0 者: {[s for s in b95 if tpos[s] <= 0]}")

P("")
P("=" * 84)
P("3. AKI 分项（首个综合达标记录 = 该行的原始三列，不做跨行填充）")
P("=" * 84)
kdi = kd[(kd.h >= 0) & (kd.h <= kd.stay_id.map(icu.set_index('stay_id')["t_end"]))]
first = kdi[kdi.aki_stage >= 1].sort_values("h").groupby("stay_id").head(1).set_index("stay_id")
c1 = first.aki_stage_creat >= 1; u1 = first.aki_stage_uo >= 1; r1 = first.aki_stage_crrt >= 1
P(f"  ICU 窗内曾达标 {len(first)} 例")
P(f"    仅 creat {int((c1&~u1&~r1).sum())} / 仅 uo {int((u1&~c1&~r1).sum())} / "
  f"both {int((c1&u1).sum())} / crrt {int(r1.sum())}")
P(f"  任意时刻曾满足各分项：creat {int((kdi.aki_stage_creat>=1).any() if False else kdi[kdi.aki_stage_creat>=1].stay_id.nunique())} / "
  f"uo {int(kdi[kdi.aki_stage_uo>=1].stay_id.nunique())} / "
  f"crrt {int(kdi[kdi.aki_stage_crrt>=1].stay_id.nunique())}")
comp = kdi[kdi.aki_stage>=1].groupby("stay_id").h.min()
tcr = kdi[kdi.aki_stage_creat>=1].groupby("stay_id").h.min()
tuo = kdi[kdi.aki_stage_uo>=1].groupby("stay_id").h.min()
trr = kdi[kdi.aki_stage_crrt>=1].groupby("stay_id").h.min()
B = pd.concat([comp.rename("t_comp"), tcr.rename("t_cr"), tuo.rename("t_uo"),
               trr.rename("t_crrt")], axis=1)
P(f"  t_comp == min(t_cr,t_uo,t_crrt): "
  f"{int((B.t_comp == B[['t_cr','t_uo','t_crrt']].min(axis=1)).sum())}/{len(B)}")

# 前向 217 事件
ass = cam[cam.v.isin(["positive", "negative"])].groupby("stay_id").size()
fwd_ids = [s for s in ass.index if s not in pre and s in taki.index and taki[s] > 0]
fe = first.loc[first.index.intersection(fwd_ids)]
fc, fu, fr = fe.aki_stage_creat >= 1, fe.aki_stage_uo >= 1, fe.aki_stage_crrt >= 1
P(f"\n  前向模型的 {len(fe)} 个 AKI 事件（按首个综合达标记录）")
P(f"    仅 creat {int((fc&~fu&~fr).sum())} / 仅 uo {int((fu&~fc&~fr).sum())} / "
  f"both {int((fc&fu).sum())} / crrt {int(fr.sum())}")

P("")
P("=" * 84)
P("4. G93 集合表（窄定义暴露 92 例）")
P("=" * 84)
con = psycopg2.connect(**PG)
dx = pd.read_sql("select hadm_id,icd_code,icd_version from mimiciv_hosp.diagnoses_icd "
                 "where hadm_id=any(%s)", con,
                 params=(icu.hadm_id.dropna().astype("int64").unique().tolist(),))
con.close()
dx["code"] = dx.icd_code.astype(str).str.strip().str.upper()
dx["v"] = dx.icd_version.astype(int)
dx = dx.merge(icu[["hadm_id", "stay_id"]], on="hadm_id")
exp = set(m.loc[m.npsle_narrow == 1, "stay_id"])
g93 = dx[(dx.v == 10) & dx.code.str.startswith("G93") & dx.stay_id.isin(exp)]
sub = g93.groupby("stay_id").code.apply(lambda s: set(s.tolist()))
has4 = {s for s, c in sub.items() if any(x.startswith("G934") for x in c)}
hasnon4 = {s for s, c in sub.items() if any(not x.startswith("G934") for x in c)}
P(f"  任意 G93 码            : {len(sub)}")
P(f"  有 G93.4x              : {len(has4)}")
P(f"  有任意非 G93.4 子码    : {len(hasnon4)}")
P(f"  两者兼具               : {len(has4 & hasnon4)}")
P(f"  非 G93.4 但无 G93.4x   : {len(hasnon4 - has4)}")
KEEP = ("F05", "R41", "F06", "F01", "F02", "F03", "G30", "G31", "3483", "310", "7800", "294")
non4only = sorted(hasnon4 - has4)
n_keep = 0
for s in non4only:
    d = dx[dx.stay_id == s]
    k = [c for c in d.code.tolist() if c not in sub.get(s, set())
         and (c.startswith(KEEP) or (c.startswith("293") and c[:4] in ("2930", "2931")))]
    if k:
        n_keep += 1
        P(f"    保留: stay {s} → {k}")
P(f"  上述 {len(non4only)} 例中另有保留暴露码: {n_keep}")
P(f"  限制后失去资格: {len(non4only) - n_keep}；限制后暴露: {len(exp) - (len(non4only) - n_keep)}")
P(f"\n  G93.4x 与 G93.x/348.3 族的 43 例关系：整族 {len(set(sub.index) | set())} 例有 G93 + "
  f"348.3 独有需另计")
# 348.3 单列
n348 = int(dx[(dx.v == 9) & (dx.code == "3483") & dx.stay_id.isin(exp)].stay_id.nunique())
P(f"  348.3 携带者: {n348}；G93 或 348.3 并集: {len(set(sub.index) | set(dx[(dx.v==9)&(dx.code=='3483')&dx.stay_id.isin(exp)].stay_id))}")

with io.open(f"{BASE}/analysis/v27_audit2_report.txt", "w", encoding="utf-8") as f:
    f.write(OUT.getvalue())
print("\n[saved] analysis/v27_audit2_report.txt")
