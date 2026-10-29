# -*- coding: utf-8 -*-
"""eICU AKI 按 KDIGO 时间窗重算（48 h 内 +0.3 mg/dL；7 d 内 1.5×），与原「峰值/最低值」规则对比"""
import os
import warnings

import numpy as np
import pandas as pd
import psycopg2
import statsmodels.formula.api as smf

warnings.filterwarnings("ignore")
BASE = os.environ.get("SLECI_BASE", os.environ.get("SLECI_BASE", "."))
DATA = os.path.join(BASE, "data")
OUT = os.path.join(BASE, "analysis", "eicu_kdigo_report.txt")
e = pd.read_csv(os.path.join(DATA, "eicu_sle.csv"))
ids = e["stay_id"].astype("int64").tolist()
con = psycopg2.connect(host="localhost", port=5432, user="postgres",
                       password=os.environ.get("SLECI_PGPASSWORD", ""), dbname="eicu")
lab = pd.read_sql("SELECT patientunitstayid, labresult, labresultoffset FROM eicu_crd.lab "
                  "WHERE labname='creatinine' AND patientunitstayid = ANY(%s) "
                  "AND labresult IS NOT NULL AND labresult > 0 AND labresult < 30",
                  con, params=(ids,))
con.close()
buf = []


def log(s=""):
    print(s)
    buf.append(str(s))


lab = lab.rename(columns={"patientunitstayid": "stay_id"})
lab = lab.sort_values(["stay_id", "labresultoffset"])
rows = []
for sid, g in lab.groupby("stay_id"):
    v = g["labresult"].values.astype(float)
    t = g["labresultoffset"].values.astype(float)  # minutes
    aki = 0
    for i in range(len(v)):
        m7 = v[(t >= t[i] - 7 * 1440) & (t <= t[i])]
        m48 = v[(t >= t[i] - 48 * 60) & (t <= t[i])]
        if (len(m7) and v[i] >= 1.5 * m7.min()) or (len(m48) and (v[i] - m48.min()) >= 0.3):
            aki = 1
            break
    rows.append((sid, aki, len(v)))
k = pd.DataFrame(rows, columns=["stay_id", "aki_kdigo_win", "n_creat"])
e = e.merge(k, on="stay_id", how="left")
log("=" * 78)
log("eICU AKI 定义对比（n=%d）" % len(e))
log("=" * 78)
log(f"原规则（峰值/最低值，无时间窗）：AKI {int(e.aki.sum())} ({100*e.aki.mean():.1f}%)")
log(f"KDIGO 时间窗规则：AKI {int(e.aki_kdigo_win.sum())} ({100*e.aki_kdigo_win.mean():.1f}%)")
log(f"有肌酐记录者 {int(e.n_creat.notna().sum())}；中位测量次数 {e.n_creat.median():.0f}")
both = e.dropna(subset=["aki", "aki_kdigo_win"])
log(f"两者一致 {int((both.aki == both.aki_kdigo_win).sum())}/{len(both)}")

for expo, lab_ in [("npsle_narrow", "窄定义"), ("npsle_broad", "宽定义")]:
    for out, outlab in [("aki", "原规则"), ("aki_kdigo_win", "KDIGO 时间窗")]:
        d = e.dropna(subset=["age", "female", out, expo])
        r = smf.logit(f"{out} ~ {expo} + age + female", data=d).fit(disp=0)
        ci = r.conf_int().loc[expo]
        log(f"  {lab_} → AKI（{outlab}）：aOR {np.exp(r.params[expo]):.2f} "
            f"({np.exp(ci[0]):.2f}–{np.exp(ci[1]):.2f}), p={r.pvalues[expo]:.3f}, n={len(d)}, "
            f"暴露 {int(d[expo].sum())}, 事件 {int(d[out].sum())}")

# 无透析版本的纯肌酐口径（已隐含，因本规则不含透析）
log("\n注：KDIGO 时间窗规则仅用肌酐，不含透析/尿量；原规则含透析。")

import io
io.open(OUT, "w", encoding="utf-8").write("\n".join(buf))
print("written:", OUT)
