"""v31 审计 1：eICU 两例「仅病毒性脑炎」暴露的 AKI 状态与重新分类后的估计。

正文 eICU 的 AKI 结局采用 KDIGO 时间窗定义（48 h 内 +0.3 mg/dL 或 7 d 内 1.5×），
对应 aOR 3.17 (1.01–10.00)。因此本审计在同一套定义下重算：
把 3162003（West Nile）与 3229333（herpes）——仅由 %encephal% 词干经
encephalitis 术语进入者——归为未暴露，保留在同一 AKI 分析队列，报告
其 AKI 状态、修订后暴露数与事件数、年龄/性别校正 OR 与 CI。
"""
import json
import os

import numpy as np
import pandas as pd
import psycopg2
import statsmodels.formula.api as smf

BASE = os.environ.get("SLECI_BASE", os.environ.get("SLECI_BASE", "."))
OUT = []
def log(*a):
    OUT.append(" ".join(str(x) for x in a))

ENS = [3162003, 3229333]

e = pd.read_csv(os.path.join(BASE, "data", "eicu_sle.csv"))
ids = e["stay_id"].astype("int64").tolist()

# ---- 与 analysis/eicu_kdigo_windows.py 完全相同的 KDIGO 时间窗口径 ----
con = psycopg2.connect(host="localhost", port=5432, user="postgres",
                       password=os.environ.get("SLECI_PGPASSWORD", ""), dbname="eicu")
lab = pd.read_sql("SELECT patientunitstayid, labresult, labresultoffset FROM eicu_crd.lab "
                  "WHERE labname='creatinine' AND patientunitstayid = ANY(%s) "
                  "AND labresult IS NOT NULL AND labresult > 0 AND labresult < 30",
                  con, params=(ids,))
con.close()
lab = lab.rename(columns={"patientunitstayid": "stay_id"}).sort_values(["stay_id", "labresultoffset"])

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

log("== eICU 总表 ==")
log("rows", len(e), "narrow", int(e.npsle_narrow.sum()), "broad", int(e.npsle_broad.sum()))
log("KDIGO 时间窗 AKI:", int(e.aki_kdigo_win.sum()), "of", int(e.aki_kdigo_win.notna().sum()),
    "(%.1f%%)" % (100 * e.aki_kdigo_win.mean()))
log("原规则（无时间窗）AKI:", int(e.aki.sum()))

OUTCOME = "aki_kdigo_win"
sub = e.dropna(subset=[OUTCOME, "npsle_narrow", "age", "female"]).copy()
log("\nAKI 分析队列 n =", len(sub), " 暴露", int(sub.npsle_narrow.sum()),
    " 事件", int(sub[OUTCOME].sum()))

log("\n== 两例 encephalitis-only 停留 ==")
for s in ENS:
    r = e[e.stay_id == s]
    if len(r) == 0:
        log(s, "NOT FOUND"); continue
    r = r.iloc[0]
    log(f"  {s}: aki_kdigo_win={r[OUTCOME]} 原规则aki={r['aki']} narrow={r['npsle_narrow']} "
        f"age={r['age']} female={r['female']} n_creat={r['n_creat']}")

exp_rows = sub[sub.npsle_narrow == 1]
enc_aki = int(exp_rows[exp_rows.stay_id.isin(ENS)][OUTCOME].sum())
log("\n原暴露组 n=%d, AKI=%d" % (len(exp_rows), int(exp_rows[OUTCOME].sum())))
log("暴露组中脑炎-only 两例的 AKI 合计 =", enc_aki)

def fit(d, label):
    m = smf.logit(f"{OUTCOME} ~ npsle_narrow + age + female", data=d).fit(disp=0)
    b = m.params["npsle_narrow"]; ci = m.conf_int().loc["npsle_narrow"]
    mu = smf.logit(f"{OUTCOME} ~ npsle_narrow", data=d).fit(disp=0)
    cu = mu.conf_int().loc["npsle_narrow"]
    a = int(((d.npsle_narrow == 1) & (d[OUTCOME] == 1)).sum())
    b2 = int(((d.npsle_narrow == 1) & (d[OUTCOME] == 0)).sum())
    c = int(((d.npsle_narrow == 0) & (d[OUTCOME] == 1)).sum())
    d2 = int(((d.npsle_narrow == 0) & (d[OUTCOME] == 0)).sum())
    log(f"{label}: n={len(d)} 暴露={int(d.npsle_narrow.sum())} 事件={int(d[OUTCOME].sum())} "
        f"2x2={a}/{b2}/{c}/{d2}")
    log(f"    aOR={np.exp(b):.2f} ({np.exp(ci[0]):.2f}-{np.exp(ci[1]):.2f}) p={m.pvalues['npsle_narrow']:.4g}"
        f" | 未调整 OR={np.exp(mu.params['npsle_narrow']):.2f} "
        f"({np.exp(cu[0]):.2f}-{np.exp(cu[1]):.2f})")
    return dict(n=len(d), exposed=int(d.npsle_narrow.sum()), events=int(d[OUTCOME].sum()),
                tp=a, fp=b2, fn=c, tn=d2,
                aOR=round(float(np.exp(b)), 2), aLO=round(float(np.exp(ci[0])), 2),
                aHI=round(float(np.exp(ci[1])), 2), ap=round(float(m.pvalues["npsle_narrow"]), 4),
                uOR=round(float(np.exp(mu.params["npsle_narrow"])), 2),
                uLO=round(float(np.exp(cu[0])), 2), uHI=round(float(np.exp(cu[1])), 2))

log("\n== KDIGO 时间窗定义下 ==")
orig = fit(sub, "原 narrow (16 暴露)")
sub2 = sub.copy()
sub2.loc[sub2.stay_id.isin(ENS), "npsle_narrow"] = 0
rec = fit(sub2, "重新分类 (14 暴露)")
sub3 = sub[~sub.stay_id.isin(ENS)]
drop = fit(sub3, "对照：删除两例 (n=202)")

log("\n== 附：无时间窗定义下的同一对比 ==")
sub_old = e.dropna(subset=["aki", "npsle_narrow", "age", "female"]).copy()
o1 = smf.logit("aki ~ npsle_narrow + age + female", data=sub_old).fit(disp=0)
sub_old2 = sub_old.copy()
sub_old2.loc[sub_old2.stay_id.isin(ENS), "npsle_narrow"] = 0
o2 = smf.logit("aki ~ npsle_narrow + age + female", data=sub_old2).fit(disp=0)
log("  原 16 暴露 aOR=%.2f; 修订 14 暴露 aOR=%.2f" % (np.exp(o1.params["npsle_narrow"]),
                                                   np.exp(o2.params["npsle_narrow"])))

log("\n== 两例在 broad 中的角色 ==")
for s in ENS:
    r = e[e.stay_id == s].iloc[0]
    log(f"  {s}: broad={r['npsle_broad']}")

res = dict(definition="KDIGO time-window (creatinine only)",
           original=orig, recoded=rec, dropped=drop,
           encephalitis_only=[int(x) for x in ENS],
           enc_only_aki=enc_aki,
           full_cohort_aki=int(e[OUTCOME].sum()),
           full_cohort_n=int(e[OUTCOME].notna().sum()))
json.dump(res, open(f"{BASE}/analysis/v31_encephalitis_only.json", "w"), indent=2)
open(f"{BASE}/analysis/v31_encephalitis_only_report.txt", "w", encoding="utf-8").write("\n".join(OUT))
print("\n".join(OUT))
