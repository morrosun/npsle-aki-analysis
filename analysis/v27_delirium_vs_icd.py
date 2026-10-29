# -*- coding: utf-8 -*-
"""v27：ICD 编码暴露 vs 实测 CAM 谵妄（严格 CAM 窗口 intime–outtime）。"""
import os, warnings, json, io
import numpy as np, pandas as pd, psycopg2
import statsmodels.formula.api as smf
from statsmodels.duration.hazard_regression import PHReg

warnings.filterwarnings("ignore")
BASE = os.environ.get("SLECI_BASE", os.environ.get("SLECI_BASE", "."))
ANA = os.path.join(BASE, "analysis"); DATA = os.path.join(BASE, "data")
PG = dict(host="localhost", port=5432, user="postgres",
          password=os.environ.get("SLECI_PGPASSWORD", ""), dbname="mimiciv")
COV = ["age", "female", "charlson_norheum", "lupus_nephritis", "renal_disease", "diabetes"]
L = []
def log(s=""):
    print(s); L.append(str(s))

df = pd.read_csv(os.path.join(DATA, "mimiciv_sle.csv"))
ids = df["stay_id"].astype("int64").tolist()
con = psycopg2.connect(**PG)
icu = pd.read_sql("SELECT stay_id,intime,outtime FROM mimiciv_icu.icustays WHERE stay_id=ANY(%s)", con, params=(ids,))
kd = pd.read_sql("SELECT stay_id,charttime,aki_stage FROM mimiciv_derived.kdigo_stages WHERE stay_id=ANY(%s)", con, params=(ids,))
cam = pd.read_sql("SELECT stay_id,charttime,value FROM mimiciv_icu.chartevents WHERE stay_id=ANY(%s) AND itemid=228332", con, params=(ids,))
con.close()
for d in (kd, cam): d["charttime"] = pd.to_datetime(d["charttime"])
icu["intime"] = pd.to_datetime(icu["intime"]); icu["outtime"] = pd.to_datetime(icu["outtime"])

m = df.merge(icu, on="stay_id", how="left")
m["t_end"] = (m["outtime"]-m["intime"]).dt.total_seconds()/3600
kd = kd.merge(icu[["stay_id","intime"]], on="stay_id", how="left")
kd["h"] = (kd["charttime"]-kd["intime"]).dt.total_seconds()/3600
pos = kd[kd["aki_stage"]>=1]
m = m.merge(pos[pos["h"]>=0].groupby("stay_id")["h"].min().rename("t_aki"), on="stay_id", how="left")
m = m.merge(pos[pos["h"]<0].groupby("stay_id").size().rename("npre"), on="stay_id", how="left")
m["pre_aki"] = m["npre"].notna().astype(int)
cam["h"] = (cam["charttime"]-cam["stay_id"].map(icu.set_index("stay_id")["intime"])).dt.total_seconds()/3600
cam["v"] = cam["value"].astype(str).str.strip().str.lower()
cam["t_end"] = cam["stay_id"].map(m.set_index("stay_id")["t_end"])
cam = cam[(cam["h"]>=0)&(cam["h"]<=cam["t_end"])].copy()   # v27 严格窗口
m = m.merge(cam[cam["v"]=="positive"].groupby("stay_id")["h"].min().rename("t_pos"), on="stay_id", how="left")
m = m.merge(cam[cam["v"].isin(["positive","negative"])].groupby("stay_id").size().rename("n_ass"), on="stay_id", how="left")
m["n_ass"] = m["n_ass"].fillna(0).astype(int)
m["assessable"] = (m["n_ass"]>0).astype(int)
m["cam_any"] = m["t_pos"].notna().astype(int)
m["icd"] = m["npsle_narrow"].astype(int)

log("="*80); log("表 A. ICD 窄定义暴露 vs CAM 实测谵妄 的一致性（可判读 CAM 者，n=%d）"%int(m["assessable"].sum())); log("="*80)
a = m[m["assessable"]==1]
ct = pd.crosstab(a["icd"], a["cam_any"])
ct.index = ["ICD(−)","ICD(+)"][:len(ct)]; ct.columns = ["CAM(−)","CAM(+)"][:ct.shape[1]]
log(ct.to_string())
if ct.shape == (2,2):
    # 行 = ICD(0/1)，列 = CAM(0/1)；FN 是「编码阴性但实测阳性」= 行0列1
    tp = ct.iloc[1,1]; fn = ct.iloc[0,1]; fp = ct.iloc[1,0]; tn = ct.iloc[0,0]
    log(f"\nICD 编码对 CAM 谵妄的敏感性 = {100*tp/(tp+fn):.1f}%（漏诊 {100*fn/(tp+fn):.1f}%）"
        f"  特异度 = {100*tn/(tn+fp):.1f}%")
    log(f"阳性预测值 = {100*tp/(tp+fp):.1f}%  阴性预测值 = {100*tn/(tn+fn):.1f}%")
    log(f"符合率 = {100*(tp+tn)/len(a):.1f}%  Kappa = "
        f"{(lambda po,pe:(po-pe)/(1-pe))( (tp+tn)/len(a), ((tp+fp)*(tp+fn)+(fn+tn)*(fp+tn))/len(a)**2 ):3f}")
    log(f"原始计数: TP={tp} FN={fn} FP={fp} TN={tn}")

base = m[(m["assessable"]==1)&(m["pre_aki"]==0)].copy()
base = base[(base["t_aki"].isna())|(base["t_aki"]>0)]
base["aki"] = base["t_aki"].notna().astype(int)
base["t"] = np.where(base["t_aki"].notna(), base["t_aki"], base["t_end"])
base = base[base["t"]>0]
log("\n"+"="*80); log("表 B. 同一人群（可判读 CAM、入科无 AKI，n=%d）内两种暴露对 AKI 的估计"%len(base)); log("="*80)

def lg(expo, d=base):
    d = d.dropna(subset=[expo]+COV)
    r = smf.logit("aki ~ %s + %s"%(expo, " + ".join(COV)), data=d).fit(disp=0)
    ci = r.conf_int().loc[expo]
    return f"{np.exp(r.params[expo]):.2f} ({np.exp(ci[0]):.2f}–{np.exp(ci[1]):.2f}), p={r.pvalues[expo]:.3f}"

log(f"  ICD 编码暴露  → AKI  aOR = {lg('icd')}")
log(f"  CAM 实测谵妄   → AKI  aOR = {lg('cam_any')}")
r = smf.logit("aki ~ icd + cam_any + "+" + ".join(COV), data=base).fit(disp=0)
for v in ("icd","cam_any"):
    ci = r.conf_int().loc[v]
    log(f"  同时纳入模型: {v:8s} aOR = {np.exp(r.params[v]):.2f} ({np.exp(ci[0]):.2f}–{np.exp(ci[1]):.2f}), p={r.pvalues[v]:.3f}")

log("\n  按 CAM 状态分层（ICD 编码 → AKI）：")
for k,g in base.groupby("cam_any"):
    if g['icd'].nunique()<2 or g['aki'].sum()<5:
        log(f"    CAM={k}: n={len(g)}, 事件 {int(g['aki'].sum())} — 不足"); continue
    log(f"    CAM={k}: n={len(g)}, 事件 {int(g['aki'].sum())}, aOR = {lg('icd', g)}")

log("\n"+"="*80); log("表 C. 时变 Cox 检验效能：可排除的效应量"); log("="*80)
log("  模型 1 暴露人时 1356 h、事件 217；95% CI 0.71–1.52")
log("  即：可排除 CAM 谵妄使后续 AKI 风险升高 >52%，但无法排除中等以下效应(<1.5)")

io.open(os.path.join(ANA,"delirium_vs_icd_report.txt"),"w",encoding="utf-8").write("\n".join(L))
print("\n[已写出] analysis/delirium_vs_icd_report.txt")
