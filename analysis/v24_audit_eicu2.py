# -*- coding: utf-8 -*-
"""v24 审计 4：eICU 8 例单次肌酐 / 6 例无肌酐的归属与敏感性。"""
import os
import io, os, warnings, numpy as np, pandas as pd, psycopg2
import statsmodels.formula.api as smf
warnings.filterwarnings("ignore")
BASE = os.environ.get("SLECI_BASE", ".")
L = []
def log(s=""):
    L.append(str(s)); print(s)

e = pd.read_csv(os.path.join(BASE, "data", "eicu_sle.csv"))
ids = e.stay_id.astype("int64").tolist()
con = psycopg2.connect(host="localhost", port=5432, user="postgres",
                       password=os.environ.get("SLECI_PGPASSWORD", ""), dbname="eicu")
lab = pd.read_sql("select patientunitstayid,labresult,labresultoffset from eicu_crd.lab "
                  "where labname='creatinine' and patientunitstayid=any(%s) "
                  "and labresult is not null and labresult>0 and labresult<30",
                  con, params=(ids,))
dx = pd.read_sql("select patientunitstayid,diagnosisstring from eicu_crd.diagnosis "
                 "where patientunitstayid=any(%s)", con, params=(ids,))
con.close()
lab = lab.rename(columns={"patientunitstayid": "stay_id"}).sort_values(
    ["stay_id", "labresultoffset"])
nobs = lab.groupby("stay_id").size()
e["n_creat"] = e.stay_id.map(nobs).fillna(0).astype(int)

rows = []
for sid, g in lab.groupby("stay_id"):
    v = g.labresult.values.astype(float); t = g.labresultoffset.values.astype(float)
    aki = 0
    for i in range(len(v)):
        m7 = v[(t >= t[i] - 7*1440) & (t <= t[i])]
        m48 = v[(t >= t[i] - 48*60) & (t <= t[i])]
        if (len(m7) and v[i] >= 1.5*m7.min()) or (len(m48) and (v[i]-m48.min()) >= 0.3):
            aki = 1; break
    rows.append((sid, aki))
k = pd.DataFrame(rows, columns=["stay_id", "aki_kdigo_win"])
e = e.merge(k, on="stay_id", how="left")

log("=" * 74)
log("eICU：肌酐检测次数分布与结局可判定性")
log("=" * 74)
log(e.n_creat.value_counts().sort_index().to_string())
log("")
z = e[e.n_creat == 0]
one = e[e.n_creat == 1]
log(f"  无合格肌酐（0 次）: {len(z)} 例；其中窄暴露 {int(z.npsle_narrow.sum())}、"
    f"宽暴露 {int(z.npsle_broad.sum()) if 'npsle_broad' in z else 'n/a'}")
log(f"  仅 1 次肌酐: {len(one)} 例；其中窄暴露 {int(one.npsle_narrow.sum())}、"
    f"AKI 判定 {int(one.aki_kdigo_win.fillna(0).sum())} 例")
log(f"  AKI 可判定（>=2 次）: {int((e.n_creat >= 2).sum())}；"
    f"其中 AKI {int(e[e.n_creat >= 2].aki_kdigo_win.fillna(0).sum())}")

def fit(d, oc="aki_kdigo_win", expo="npsle_narrow"):
    d = d.dropna(subset=["age", "female", expo, oc])
    r = smf.logit(f"{oc} ~ {expo} + age + female", data=d).fit(disp=0)
    ci = r.conf_int().loc[expo]
    return (f"n={len(d)}, 暴露 {int(d[expo].sum())}, 事件 {int(d[oc].sum())}, "
            f"aOR {np.exp(r.params[expo]):.2f} ({np.exp(ci[0]):.2f}-{np.exp(ci[1]):.2f}), "
            f"p={r.pvalues[expo]:.3f}")

log("")
log("=" * 74)
log("敏感性：排除仅 1 次肌酐的 8 例（其 AKI 状态不可靠）")
log("=" * 74)
log("  主分析（含 8 例）      : " + fit(e))
log("  排除 8 例单次肌酐       : " + fit(e[e.n_creat >= 2]))

io.open(os.path.join(BASE, "analysis", "v24_audit_eicu2_report.txt"), "w",
        encoding="utf-8").write("\n".join(L))
print("\nwritten")
