# -*- coding: utf-8 -*-
"""v24 审计 1b：G93 子类构成 + G93.4 限制分析（修正前缀，icd_code 无小数点）+ CAM 同口径交叉表
   + ICU 时间窗内 AKI 敏感性 + 慢性码两种处理（重分类 vs 删除）。"""
import os
import io, os, warnings, numpy as np, pandas as pd, psycopg2
import statsmodels.formula.api as smf
warnings.filterwarnings("ignore")

BASE = os.environ.get("SLECI_BASE", ".")
OUT = os.path.join(BASE, "analysis", "v24_audit_codes2_report.txt")
L = []


def log(s=""):
    L.append(str(s)); print(s)


m = pd.read_csv(os.path.join(BASE, "data", "mimiciv_sle.csv"))
ids = m.stay_id.astype("int64").tolist()
con = psycopg2.connect(host="localhost", port=5432, user="postgres",
                       password=os.environ.get("SLECI_PGPASSWORD", ""), dbname="mimiciv")
icu = pd.read_sql("select stay_id,hadm_id,intime,outtime from mimiciv_icu.icustays "
                  "where stay_id=any(%s)", con, params=(ids,))
hadm = icu.hadm_id.dropna().astype("int64").unique().tolist()
dx = pd.read_sql("select hadm_id,icd_code,icd_version from mimiciv_hosp.diagnoses_icd "
                 "where hadm_id=any(%s)", con, params=(hadm,))
kd = pd.read_sql("select stay_id,charttime,aki_stage,aki_stage_creat,aki_stage_uo,"
                 "aki_stage_crrt from mimiciv_derived.kdigo_stages "
                 "where stay_id=any(%s)", con, params=(ids,))
cam = pd.read_sql("select stay_id,charttime,value from mimiciv_icu.chartevents "
                  "where itemid=228332 and stay_id=any(%s)", con, params=(ids,))
con.close()

dx["code"] = dx.icd_code.astype(str).str.strip().str.upper()
dx["v"] = dx.icd_version.astype(int)
dx = dx.merge(icu[["hadm_id", "stay_id"]], on="hadm_id")
exp = set(m.loc[m.npsle_narrow == 1, "stay_id"])
COV = ["age", "female", "charlson_norheum", "lupus_nephritis", "renal_disease", "diabetes"]


def hit(pats, ver):
    s = set()
    for p in pats:
        s |= set(dx[(dx.v == ver) & (dx.code.str.startswith(p))].stay_id)
    return s


FAM = [
    ("脑病 G93.x", hit(["G93"], 10)),
    ("脑病 348.3", hit(["3483"], 9)),
    ("谵妄 F05/293.0-1", hit(["F05"], 10) | hit(["2930", "2931"], 9)),
    ("遗忘/认知 R41.x", hit(["R41"], 10)),
    ("意识改变 780.0x", hit(["7800"], 9)),
    ("痴呆 F01-F03/G30/G31", hit(["F01", "F02", "F03", "G30", "G31"], 10)),
    ("持久性精神障碍 294.x", hit(["294"], 9)),
    ("其他器质性 F06.x/293.8x/293.9", hit(["F06"], 10) | hit(["2938", "2939"], 9)),
    ("人格改变 310.x", hit(["310"], 9)),
]
fam_sets = [(n, s & exp) for n, s in FAM]

log("=" * 78)
log("一、Table 6 家族命中交叉表（限定 92 例暴露入住）")
log("=" * 78)
for n, s in fam_sets:
    log(f"  {n:34s} {len(s):4d}")
tot = sum(len(s) for _, s in fam_sets)
union = set().union(*[s for _, s in fam_sets])
cnt = {}
for sid in exp:
    k = sum(1 for _, s in fam_sets if sid in s)
    cnt[k] = cnt.get(k, 0) + 1
log(f"  {'族计数之和':34s} {tot:4d}")
log(f"  {'并集':34s} {len(union):4d}")
log("  每例命中族数：" + "；".join(f"{k} 个族 {v} 例" for k, v in sorted(cnt.items())))
log(f"  → 命中 ≥2 个族的入住 = {sum(v for k, v in cnt.items() if k >= 2)} 例"
    f"（其中 2 个族 {cnt.get(2,0)} 例、3 个族 {cnt.get(3,0)} 例）")
log(f"  → 额外计数 = {sum((k-1)*v for k, v in cnt.items() if k >= 2)}"
    f"（= 2 族各 1 次 × {cnt.get(2,0)} + 3 族多出 2 次 × {cnt.get(3,0)}）")
log(f"  → 校验：单族 {cnt.get(1,0)} + 多族 {sum(v for k,v in cnt.items() if k>=2)} = "
    f"{cnt.get(1,0) + sum(v for k, v in cnt.items() if k >= 2)}")

log("")
log("=" * 78)
log("二、G93.x 实际命中的子类（92 例暴露入住内；icd_code 无小数点）")
log("=" * 78)
LAB = {"G931": "G93.1 缺氧性脑损伤", "G9340": "G93.40 脑病，未特指",
       "G9341": "G93.41 代谢性脑病", "G9349": "G93.49 其他特指脑病",
       "G935": "G93.5 脑受压", "G936": "G93.6 脑水肿",
       "G9381": "G93.81 颞叶癫痫", "G9382": "G93.82 脑死亡",
       "G9389": "G93.89 其他特指脑病"}
g93 = dx[(dx.v == 10) & (dx.code.str.startswith("G93")) & (dx.stay_id.isin(exp))]
sub = g93.groupby("code").stay_id.nunique().sort_values(ascending=False)
for c, n in sub.items():
    log(f"  {c:8s} {LAB.get(c, ''):28s} {n:3d}")
log(f"  G93.x 合计 distinct stays: {g93.stay_id.nunique()}")
g934 = set(g93[g93.code.str.startswith("G934")].stay_id)
log(f"  其中 G93.4x: {len(g934)} 例；非 G93.4x: {len(set(g93.stay_id) - g934)} 例")
# 仅靠 G93 非 .4 子类入组的
others = set().union(*[s for n, s in fam_sets if not n.startswith("脑病 G93")])
log(f"  无其他窄定义码、仅靠 G93 非 .4 入组: "
    f"{len((set(g93.stay_id) - g934) - others)} 例")

log("")
log("=" * 78)
log("三、敏感性：暴露定义中 G93.x 限制为 G93.4x（其余不变）")
log("=" * 78)
exp_g934 = (exp - set(g93.stay_id)) | g934
log(f"  原暴露 {len(exp)} → 限制后 {len(exp_g934)}（减少 {len(exp) - len(exp_g934)}）")
m2 = m.copy()
m2["np_g934"] = m2.stay_id.isin(exp_g934).astype(int)
for lab, col in [("原暴露（G93.x）", "npsle_narrow"), ("限制 G93.4x", "np_g934")]:
    d = m2.dropna(subset=COV + ["aki", col])
    r = smf.logit(f"aki ~ {col} + " + "+".join(COV), data=d).fit(disp=0)
    ci = r.conf_int().loc[col]
    s1 = d[d[col] == 1]
    log(f"  {lab:12s}: 暴露 {int(d[col].sum()):3d}, AKI 事件 {int(d.aki.sum())}, "
        f"暴露组 AKI {int(s1.aki.sum())}/{len(s1)} ({100*s1.aki.mean():.1f}%); "
        f"aOR {np.exp(r.params[col]):.2f} ({np.exp(ci[0]):.2f}-{np.exp(ci[1]):.2f}), "
        f"p={r.pvalues[col]:.4f}")

log("")
log("  与 CAM-ICU 的一致性（可评估 = 至少一条 positive/negative 记录）")
cam["v"] = cam.value.astype(str).str.strip().str.lower()
ass = cam[cam.v.isin(["positive", "negative"])]
assess = set(ass.stay_id)
pos = set(cam.loc[cam.v == "positive", "stay_id"])
log(f"    可评估 stay {len(assess)}；CAM 阳性 {len(pos & assess)}")
for lab, s in [("原暴露", exp), ("限制 G93.4x", exp_g934)]:
    a = m2[m2.stay_id.isin(assess)].copy()
    a["e"] = a.stay_id.isin(s).astype(int)      # 指标（编码）
    a["c"] = a.stay_id.isin(pos).astype(int)    # 参照（CAM-ICU）
    tp = int(((a.e == 1) & (a.c == 1)).sum())   # 编码+ 且 CAM+
    fp = int(((a.e == 1) & (a.c == 0)).sum())   # 编码+ 且 CAM-
    fn = int(((a.e == 0) & (a.c == 1)).sum())   # 编码- 且 CAM+
    tn = int(((a.e == 0) & (a.c == 0)).sum())   # 编码- 且 CAM-
    po = (tp + tn) / len(a)
    pe = ((tp + fp) * (tp + fn) + (fn + tn) * (fp + tn)) / len(a) ** 2
    log(f"    {lab:12s}: 编码+ {tp+fp}, CAM+ {tp+fn}, n={len(a)}")
    log(f"      TP={tp} FP={fp} FN={fn} TN={tn}; "
        f"敏感性 {100*tp/(tp+fn):.1f}%, 特异度 {100*tn/(tn+fp):.1f}%, "
        f"PPV {100*tp/(tp+fp):.1f}%, NPV {100*tn/(tn+fn):.1f}%, "
        f"准确率 {100*po:.1f}%, κ={(po-pe)/(1-pe):.3f}")

log("")
log("=" * 78)
log("四、AKI 判定时间窗")
log("=" * 78)
icu["intime"] = pd.to_datetime(icu.intime); icu["outtime"] = pd.to_datetime(icu.outtime)
kd["charttime"] = pd.to_datetime(kd.charttime)
kd = kd.merge(icu[["stay_id", "intime", "outtime"]], on="stay_id", how="left")
kd["phase"] = np.where(kd.charttime < kd.intime, "pre",
                       np.where(kd.charttime > kd.outtime, "post", "during"))
log("  记录分布：" + kd.phase.value_counts().to_dict().__str__())
mx_all = kd.groupby("stay_id").aki_stage.max()
during = kd[kd.phase == "during"]
mx_dur = during.groupby("stay_id").aki_stage.max()
log(f"  任意时间窗 MAX>=1: {int((mx_all >= 1).sum())}/550（= cohort 的 aki 变量）")
log(f"  仅 ICU 内 MAX>=1:  {int((mx_dur >= 1).sum())}/{len(mx_dur)}")
onlypre = set(mx_all[mx_all >= 1].index) - set(mx_dur[mx_dur >= 1].index)
log(f"  仅凭入 ICU 前记录判为 AKI: {len(onlypre)} 例")

m3 = m.copy()
m3["np"] = m3.stay_id.isin(exp).astype(int)
m3["aki_icu"] = m3.stay_id.map(mx_dur).fillna(0).clip(upper=1).astype(int)
for lab, oc in [("任意时间窗（原）", "aki"), ("仅 ICU 内", "aki_icu")]:
    d = m3.dropna(subset=COV + [oc, "np"])
    r = smf.logit(f"{oc} ~ np + " + "+".join(COV), data=d).fit(disp=0)
    ci = r.conf_int().loc["np"]
    log(f"  {lab:16s}: n={len(d)}, 事件 {int(d[oc].sum())}, "
        f"aOR {np.exp(r.params.np):.2f} ({np.exp(ci[0]):.2f}-{np.exp(ci[1]):.2f})")

log("")
log("=" * 78)
log("五、慢性码的两种处理")
log("=" * 78)
chronic = hit(["F01", "F02", "F03", "G30", "G31"], 10) | hit(["294"], 9)
acute = (hit(["F05", "F06", "R41", "G93"], 10)
         | hit(["2930", "2931", "2938", "2939", "3483", "7800"], 9))
only_chronic = exp & chronic - acute
any_chronic = exp & chronic
log(f"  携带任何慢性码的暴露入住: {len(any_chronic)}")
log(f"  仅慢性码（无任何急性码）: {len(only_chronic)}")
for lab, new_exp in [("重分类：剔除 16 例携带慢性码者（n 仍 550）",
                      exp - any_chronic),
                     ("重分类：仅剔除 11 例慢性-only（n 仍 550）",
                      exp - only_chronic),
                     ("删除：11 例慢性-only 整例移出队列（n 539）",
                      None)]:
    if new_exp is not None:
        mm = m3.copy()
        mm["np2"] = mm.stay_id.isin(new_exp).astype(int)
        d = mm.dropna(subset=COV + ["aki", "np2"])
    else:
        mm = m3[~m3.stay_id.isin(only_chronic)].copy()
        mm["np2"] = mm.stay_id.isin(exp - only_chronic).astype(int)
        d = mm.dropna(subset=COV + ["aki", "np2"])
    r = smf.logit("aki ~ np2 + " + "+".join(COV), data=d).fit(disp=0)
    ci = r.conf_int().loc["np2"]
    log(f"  {lab}: n={len(d)}, 暴露 {int(d.np2.sum())}, "
        f"aOR {np.exp(r.params.np2):.2f} ({np.exp(ci[0]):.2f}-{np.exp(ci[1]):.2f})")

io.open(OUT, "w", encoding="utf-8").write("\n".join(L))
print("\n写入", OUT)
