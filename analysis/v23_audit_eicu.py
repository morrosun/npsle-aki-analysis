# -*- coding: utf-8 -*-
"""
v23 审计 C：eICU 结局口径切换后的完整结果重算

评审指出：
  (1) eICU 主 AKI 已改为 KDIGO 时间窗，但 Table S2 的 APACHE 调整结果仍是旧口径 0.97；
  (2) 正文称「主模型无缺失」，但 eICU 只有 204/210 进入模型，需说明 6 例为何排除；
  (3) 新定义下的未调整 OR 与 2×2 计数缺失；
  (4) 只测一次肌酐的入住如何处理。
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
ANA = os.path.join(BASE, "analysis")
buf = []


def log(s=""):
    print(s)
    buf.append(str(s))


e = pd.read_csv(os.path.join(DATA, "eicu_sle.csv"))
ids = e["stay_id"].astype("int64").tolist()

con = psycopg2.connect(host="localhost", port=5432, user="postgres",
                       password=os.environ.get("SLECI_PGPASSWORD", ""), dbname="eicu")
lab = pd.read_sql("SELECT patientunitstayid, labresult, labresultoffset FROM eicu_crd.lab "
                  "WHERE labname='creatinine' AND patientunitstayid = ANY(%s) "
                  "AND labresult IS NOT NULL AND labresult > 0 AND labresult < 30",
                  con, params=(ids,))
dly = pd.read_sql("SELECT patientunitstayid FROM eicu_crd.patient "
                  "WHERE patientunitstayid = ANY(%s) AND 1=0", con, params=(ids,))
con.close()

lab = lab.rename(columns={"patientunitstayid": "stay_id"}).sort_values(
    ["stay_id", "labresultoffset"])

rows = []
for sid, g in lab.groupby("stay_id"):
    v = g["labresult"].values.astype(float)
    t = g["labresultoffset"].values.astype(float)   # minutes from ICU admission
    aki, t_first = 0, np.nan
    for i in range(len(v)):
        m7 = v[(t >= t[i] - 7 * 1440) & (t <= t[i])]
        m48 = v[(t >= t[i] - 48 * 60) & (t <= t[i])]
        hit = (len(m7) and v[i] >= 1.5 * m7.min()) or (len(m48) and (v[i] - m48.min()) >= 0.3)
        if hit:
            aki, t_first = 1, t[i]
            break
    rows.append((sid, aki, len(v), t_first))
k = pd.DataFrame(rows, columns=["stay_id", "aki_kdigo_win", "n_creat", "t_first_aki"])
e = e.merge(k, on="stay_id", how="left")

log("=" * 84)
log("C1. eICU 队列与 AKI 结局可得性")
log("=" * 84)
log(f"队列 n={len(e)}；窄定义暴露 {int(e.npsle_narrow.sum())}；宽定义 {int(e.npsle_broad.sum())}")
no_creat = e[e.n_creat.isna()]
log(f"无任何合格肌酐记录（0<Cr<30 mg/dL）的 stay：{len(no_creat)}")
log(f"  其中窄定义暴露 {int(no_creat.npsle_narrow.sum())}，宽定义暴露 {int(no_creat.npsle_broad.sum())}")
log(f"  这些 stay 的 stay_id：{no_creat.stay_id.tolist()}")
log(f"  其原规则 aki 取值：{no_creat.aki.tolist()}")
log(f"仅 1 次肌酐测定的 stay：{int((e.n_creat == 1).sum())}（其中窄暴露 "
    f"{int(((e.n_creat == 1) & (e.npsle_narrow == 1)).sum())}）—— 无第二次测定无法观察升高，"
    f"按「无 AKI」处理，属结局错分而非缺失")
log(f"肌酐测定次数：中位 {e.n_creat.median():.0f} [IQR {e.n_creat.quantile(.25):.0f}–"
    f"{e.n_creat.quantile(.75):.0f}]，最少 {int(e.n_creat.min())}，最多 {int(e.n_creat.max())}")
log(f"\nKDIGO 时间窗 AKI：{int(e.aki_kdigo_win.sum())} / {int(e.n_creat.notna().sum())} "
    f"({100*e.aki_kdigo_win.sum()/e.n_creat.notna().sum():.1f}%)")
log(f"原规则（含透析、无时间窗）AKI：{int(e.aki.sum())} / {int(e.aki.notna().sum())} "
    f"({100*e.aki.sum()/e.aki.notna().sum():.1f}%)")

log("\n" + "=" * 84)
log("C2. 新定义下的 2×2 与未调整 OR")
log("=" * 84)
for expo, lab_ in [("npsle_narrow", "窄定义"), ("npsle_broad", "宽定义")]:
    d = e.dropna(subset=[expo, "aki_kdigo_win"])
    a = int(((d[expo] == 1) & (d.aki_kdigo_win == 1)).sum())
    b = int(((d[expo] == 1) & (d.aki_kdigo_win == 0)).sum())
    c = int(((d[expo] == 0) & (d.aki_kdigo_win == 1)).sum())
    dd = int(((d[expo] == 0) & (d.aki_kdigo_win == 0)).sum())
    orr = (a * dd) / (b * c) if b * c else np.nan
    se = np.sqrt(1/a + 1/b + 1/c + 1/dd) if min(a, b, c, dd) > 0 else np.nan
    log(f"{lab_}（KDIGO 时间窗）：暴露 {a}/{a+b} ({100*a/(a+b):.1f}%) vs "
        f"对照 {c}/{c+dd} ({100*c/(c+dd):.1f}%)；uOR {orr:.2f} "
        f"({np.exp(np.log(orr)-1.96*se):.2f}–{np.exp(np.log(orr)+1.96*se):.2f})；"
        f"2×2 a={a} b={b} c={c} d={dd}")

log("\n" + "=" * 84)
log("C3. 校正模型（新定义 vs 旧定义）")
log("=" * 84)
for expo, lab_ in [("npsle_narrow", "窄定义"), ("npsle_broad", "宽定义")]:
    for out, outlab in [("aki_kdigo_win", "KDIGO 时间窗（主）"), ("aki", "原规则（替代）")]:
        d = e.dropna(subset=["age", "female", expo, out])
        r = smf.logit(f"{out} ~ {expo} + age + female", data=d).fit(disp=0)
        ci = r.conf_int().loc[expo]
        log(f"  {lab_} → {outlab}: aOR {np.exp(r.params[expo]):.2f} "
            f"({np.exp(ci[0]):.2f}–{np.exp(ci[1]):.2f}), p={r.pvalues[expo]:.3f}, "
            f"n={len(d)}, 暴露 {int(d[expo].sum())}, 事件 {int(d[out].sum())}")

log("\n  —— APACHE IV 校正 ——")
log(f"APACHE IV 可得：{int(e.apache_iv.notna().sum())} / {len(e)}；"
    f"其中窄暴露 {int(((e.apache_iv.notna()) & (e.npsle_narrow == 1)).sum())}")
for expo, lab_ in [("npsle_narrow", "窄定义"), ("npsle_broad", "宽定义")]:
    for out, outlab in [("aki_kdigo_win", "KDIGO 时间窗（主）"), ("aki", "原规则（替代）")]:
        d = e.dropna(subset=["age", "female", expo, out, "apache_iv"])
        if len(d) < 30 or d[expo].sum() < 3:
            log(f"  {lab_} → {outlab}: 样本不足（n={len(d)}）")
            continue
        try:
            r = smf.logit(f"{out} ~ {expo} + age + female + apache_iv", data=d).fit(disp=0)
            ci = r.conf_int().loc[expo]
            log(f"  {lab_} → {outlab}: aOR {np.exp(r.params[expo]):.2f} "
                f"({np.exp(ci[0]):.2f}–{np.exp(ci[1]):.2f}), p={r.pvalues[expo]:.3f}, "
                f"n={len(d)}, 暴露 {int(d[expo].sum())}, 事件 {int(d[out].sum())}")
        except Exception as ex:
            log(f"  {lab_} → {outlab}: 拟合失败 {ex}")

log("\n" + "=" * 84)
log("C4. 合并估计（MIMIC-IV 年龄/性别校正 + eICU 年龄/性别校正）")
log("=" * 84)
m = pd.read_csv(os.path.join(DATA, "mimiciv_sle.csv"))
d = m.dropna(subset=["age", "female", "npsle_narrow", "aki"])
r = smf.logit("aki ~ npsle_narrow + age + female", data=d).fit(disp=0)
ci = r.conf_int().loc["npsle_narrow"]
mim_b = float(r.params["npsle_narrow"])
mim_se = float((ci[1] - ci[0]) / (2 * 1.96))
eic_d = e.dropna(subset=["age", "female", "npsle_narrow", "aki_kdigo_win"])
r2 = smf.logit("aki_kdigo_win ~ npsle_narrow + age + female", data=eic_d).fit(disp=0)
ci2 = r2.conf_int().loc["npsle_narrow"]
eic_b = float(r2.params["npsle_narrow"])
eic_se = float((ci2[1] - ci2[0]) / (2 * 1.96))
log(f"MIMIC-IV 年龄/性别校正：{np.exp(mim_b):.2f} ({np.exp(ci[0]):.2f}–{np.exp(ci[1]):.2f}), "
    f"n={len(d)}, 暴露 {int(d.npsle_narrow.sum())}")
log(f"eICU（新定义）：{np.exp(eic_b):.2f} ({np.exp(ci2[0]):.2f}–{np.exp(ci2[1]):.2f}), "
    f"n={len(eic_d)}, 暴露 {int(eic_d.npsle_narrow.sum())}")
# DerSimonian-Laird
w = np.array([1 / mim_se**2, 1 / eic_se**2])
b = np.array([mim_b, eic_b])
fe = (w * b).sum() / w.sum()
Q = (w * (b - fe) ** 2).sum()
C = w.sum() - (w ** 2).sum() / w.sum()
tau2 = max(0.0, (Q - 1) / C) if C > 0 else 0.0
wr = 1 / (np.array([mim_se, eic_se]) ** 2 + tau2)
re = (wr * b).sum() / wr.sum()
re_se = float(np.sqrt(1 / wr.sum()))
I2 = max(0.0, (Q - 1) / Q) * 100 if Q > 0 else 0.0
log(f"DL 随机效应合并：{np.exp(re):.2f} ({np.exp(re - 1.96*re_se):.2f}–"
    f"{np.exp(re + 1.96*re_se):.2f})；Q={Q:.2f}, I²={I2:.1f}%, tau²={tau2:.4f}")

log("\n" + "=" * 84)
log("C5. 严重程度与 AKI 的关系（解释 APACHE 校正后点估计为何偏向 1）")
log("=" * 84)
d = e.dropna(subset=["apache_iv", "aki_kdigo_win", "npsle_narrow"])
log(f"APACHE IV：暴露组中位 {d.loc[d.npsle_narrow==1,'apache_iv'].median():.0f} vs "
    f"非暴露 {d.loc[d.npsle_narrow==0,'apache_iv'].median():.0f}")
log(f"APACHE IV 与 AKI 的相关：AKI 组中位 {d.loc[d.aki_kdigo_win==1,'apache_iv'].median():.0f} vs "
    f"无 AKI {d.loc[d.aki_kdigo_win==0,'apache_iv'].median():.0f}")
log("注：eICU 的 dialysis 记录位于 intakeOutput / treatment 表，本队列未用于结局定义。")

with io.open(os.path.join(ANA, "v23_audit_eicu_report.txt"), "w", encoding="utf-8") as f:
    f.write("\n".join(buf))
print("\nwritten: analysis/v23_audit_eicu_report.txt")
