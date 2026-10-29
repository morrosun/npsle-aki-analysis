# -*- coding: utf-8 -*-
"""v25 审计 1：回应第四轮评审的计数闭合问题
   A 家族数（8 vs 9）与重叠分布重算
   B G93.4x 限制：12 还是 13（含核对表）
   C G93 子码完整枚举
   D AKI 时间窗：84 / 374 / 388 / 304 与首次达标标准的归属
   E ICU 窗 logistic 的实际样本量（547 还是 550）
   F 120 例排序队列 与 Cox 人群（326/325/217/115）的交叉表
   G ESRD 在时序模型人群中的分布
   H 匹配后 LOS 的配对检验
"""
import os
import io, os, warnings
import numpy as np, pandas as pd, psycopg2
import statsmodels.formula.api as smf
from scipy import stats

warnings.filterwarnings("ignore")
BASE = os.environ.get("SLECI_BASE", os.environ.get("SLECI_BASE", "."))
OUT = os.path.join(BASE, "analysis", "v25_audit_report.txt")
L = []


def log(s=""):
    L.append(str(s))
    print(s)


m = pd.read_csv(os.path.join(BASE, "data", "mimiciv_sle.csv"))
ids = m.stay_id.astype("int64").tolist()
con = psycopg2.connect(host="localhost", port=5432, user="postgres",
                       password=os.environ.get("SLECI_PGPASSWORD", ""), dbname="mimiciv")
icu = pd.read_sql("SELECT stay_id,hadm_id,intime,outtime FROM mimiciv_icu.icustays "
                  "WHERE stay_id=ANY(%s)", con, params=(ids,))
hadm = icu.hadm_id.dropna().astype("int64").unique().tolist()
dx = pd.read_sql("SELECT hadm_id,icd_code,icd_version FROM mimiciv_hosp.diagnoses_icd "
                 "WHERE hadm_id=ANY(%s)", con, params=(hadm,))
kd = pd.read_sql("SELECT stay_id,charttime,aki_stage,aki_stage_creat,aki_stage_uo,aki_stage_crrt "
                 "FROM mimiciv_derived.kdigo_stages WHERE stay_id=ANY(%s)", con, params=(ids,))
cam = pd.read_sql("SELECT stay_id,charttime,value FROM mimiciv_icu.chartevents "
                  "WHERE itemid=228332 AND stay_id=ANY(%s)", con, params=(ids,))
con.close()

dx["code"] = dx.icd_code.astype(str).str.strip().str.upper()
dx["v"] = dx.icd_version.astype(int)
dx = dx.merge(icu[["hadm_id", "stay_id"]], on="hadm_id")
exp = set(m.loc[m.npsle_narrow == 1, "stay_id"])
COV = ["age", "female", "charlson_norheum", "lupus_nephritis", "renal_disease", "diabetes"]

icu["intime"] = pd.to_datetime(icu.intime); icu["outtime"] = pd.to_datetime(icu.outtime)
kd["charttime"] = pd.to_datetime(kd.charttime); cam["charttime"] = pd.to_datetime(cam.charttime)
imap = icu.set_index("stay_id")["intime"]


def hit(pats, ver):
    s = set()
    for p in pats:
        s |= set(dx[(dx.v == ver) & (dx.code.str.startswith(p))].stay_id)
    return s


# ---- 各前缀集合（用于 B 的核对表与 C 的枚举）----
S = dict(
    F05=hit(["F05"], 10), R41=hit(["R41"], 10), F06=hit(["F06"], 10),
    G93=hit(["G93"], 10), F01_03=hit(["F01", "F02", "F03"], 10),
    G30=hit(["G30"], 10), G31=hit(["G31"], 10),
    c293=hit(["293"], 9), c294=hit(["294"], 9), c3483=hit(["3483"], 9),
    c310=hit(["310"], 9), c7800=hit(["7800"], 9),
)
# 与定义逐字对齐：ICD-9 的 293.x 拆为 293.0–293.1（谵妄）与 293.8x/293.9（其他器质性）
D293 = dx[(dx.v == 9) & dx.code.str.startswith("293")]
S["d293"] = set(D293[D293.code.str.startswith(("2930", "2931"))].stay_id)
S["o293"] = set(D293[D293.code.str.startswith(("2938", "2939"))].stay_id)
g934 = set(dx[(dx.v == 10) & dx.code.str.startswith("G934")].stay_id)
g93other = S["G93"] - g934

log("=" * 80)
log("A. 家族数与重叠分布（顶层家族 = 临床类别，ICD-9/10 对应码合并）")
log("=" * 80)
# 8 族：G93+348.3 合并；dementia(F01-03) 与 neurodegeneration(G30,G31) 合并
FAM8 = [
    ("Other brain disorders / encephalopathy", S["G93"] | S["c3483"]),
    ("Delirium", S["F05"] | S["d293"]),
    ("Amnesia / cognitive symptoms", S["R41"]),
    ("Alteration of consciousness", S["c7800"]),
    ("Dementia / neurodegeneration", S["F01_03"] | S["G30"] | S["G31"]),
    ("Persistent mental disorder", S["c294"]),
    ("Other specified organic mental disorder", S["F06"] | S["o293"]),
    ("Personality change due to brain disease", S["c310"]),
]
# 9 族：额外把 dementia 与 neurodegeneration 拆成两族
FAM9 = FAM8[:4] + [
    ("Dementia (F01-F03)", S["F01_03"]),
    ("Neurodegeneration (G30, G31)", S["G30"] | S["G31"]),
] + FAM8[5:]


def overlap(fams, exp_set, tag):
    fs = [(n, s & exp_set) for n, s in fams]
    for n, s in fs:
        log(f"   {tag} {n:44s} {len(s):4d}")
    tot = sum(len(s) for _, s in fs)
    uni = set().union(*[s for _, s in fs])
    cnt = {k: 0 for k in range(1, 6)}
    for sid in exp_set:
        k = sum(1 for _, s in fs if sid in s)
        cnt[k] = cnt.get(k, 0) + 1
    log(f"   {tag} {'族计数之和':44s} {tot:4d}")
    log(f"   {tag} {'并集':44s} {len(uni):4d}")
    log(f"   {tag} 命中族数分布：" + "；".join(f"{k} 族 {v} 例" for k, v in cnt.items() if v))
    n_matched = sum(1 for _, s in fs if len(s) > 0)
    log(f"   {tag} 定义了 {len(fams)} 族，其中 {n_matched} 族至少命中 1 例；"
        f"并集 {len(uni)}；计数盈余 {tot - len(uni)}")
    return dict(tot=tot, uni=len(uni), cnt=cnt, n_fam=len(fams), n_match=n_matched)


r8 = overlap(FAM8, exp, "[8族]")
log("")
r9 = overlap(FAM9, exp, "[9族]")

log("")
log("=" * 80)
log("B. G93 限制为 G93.4x：暴露减少 12 还是 13（入住级核对表）")
log("=" * 80)
log(f"  暴露总数 {len(exp)}")
log(f"  含 G93.x（ICD-10）者 {len(S['G93'] & exp)}；其中含 G93.4x {len(g934 & exp)}、"
    f"仅含非 G93.4x 的 G93 码 {len(g93other & exp)}")
# 「保留」= 除 G93 非 .4 之外的任何窄定义码
retained_other = set()
for k in ["F05", "R41", "F06", "F01_03", "G30", "G31", "c293", "c294", "c3483", "c310", "c7800"]:
    retained_other |= S[k]
retained_other |= g934
non4 = g93other & exp
only_non4 = non4 - retained_other
both34 = non4 & g934
other_ok = (non4 & retained_other) - g934
log("")
log("  | 组别 | 入住数 |")
log("  |---|---|")
log(f"  | 含非 G93.4x 的 G93 代码（合计） | {len(non4)} |")
log(f"  | 　同时含 G93.4x（限制后仍暴露） | {len(both34)} |")
log(f"  | 　无 G93.4x 但有其他保留的符合条件代码（限制后仍暴露） | {len(other_ok)} |")
log(f"  | 　除此之外无任何保留代码（限制后失去暴露资格） | {len(only_non4)} |")
exp_r = exp - only_non4
log(f"  | 限制后暴露数 | {len(exp_r)} |")
log(f"  → 92 → {len(exp_r)}，减少 {len(exp) - len(exp_r)} 例")
if len(exp_r) != 79:
    log("  ⚠ 与 v24 报告的 79 不同：v24 的实现先剔除所有 G93 阳性者、再补回 G93.4x，")
    log("     因此连「有其他保留代码」的那 1 例也一并剔除，属实现偏差。")
m["np_g934"] = m.stay_id.isin(exp_r).astype(int)
m["np_g934_v24"] = m.stay_id.isin((exp - S["G93"]) | (g934 & exp)).astype(int)
a = pd.crosstab(m.np_g934, m.aki)
log(f"  限制后暴露组 AKI：{a.loc[1, 1]}/{a.loc[1].sum()} = "
    f"{100*a.loc[1,1]/a.loc[1].sum():.1f}%")
f6 = "aki ~ np_g934 + " + " + ".join(COV)
fit = smf.logit(f6, data=m).fit(disp=0)
ci = fit.conf_int().loc["np_g934"]
log(f"  六协变量 aOR = {np.exp(fit.params['np_g934']):.2f} "
    f"({np.exp(ci[0]):.2f}–{np.exp(ci[1]):.2f}), p = {fit.pvalues['np_g934']:.4f}, n = {int(fit.nobs)}")
# 与 v24 版本对比
bv = m.assign(aki=m.aki).pipe(lambda d: smf.logit("aki ~ np_g934_v24 + " + " + ".join(COV), data=d).fit(disp=0))
log(f"  （v24 实现口径下：暴露 {int(m.np_g934_v24.sum())}，"
    f"aOR {np.exp(bv.params['np_g934_v24']):.2f}，n = {int(bv.nobs)}）")

# CAM 一致性（限制后）
cam["v"] = cam.value.astype(str).str.strip().str.lower()
campos = cam[cam.v == "positive"].stay_id.unique()
camass = cam[cam.v.isin(["positive", "negative"])].stay_id.unique()
ass = m[m.stay_id.isin(camass)]
for tag, col in [("原暴露", "npsle_narrow"), ("限制后", "np_g934")]:
    tp = ((ass[col] == 1) & (ass.stay_id.isin(campos))).sum()
    fp = ((ass[col] == 1) & (~ass.stay_id.isin(campos))).sum()
    fn = ((ass[col] == 0) & (ass.stay_id.isin(campos))).sum()
    tn = ((ass[col] == 0) & (~ass.stay_id.isin(campos))).sum()
    se, sp = tp / (tp + fn), tn / (tn + fp)
    ppv, npv = tp / (tp + fp), tn / (tn + fn)
    po = (tp + tn) / (tp + tn + fp + fn)
    pe = ((tp + fp) * (tp + fn) + (fn + tn) * (fp + tn)) / (tp + tn + fp + fn) ** 2
    log(f"  {tag} CAM 2×2: TP {tp} FP {fp} FN {fn} TN {tn} | "
        f"敏感 {100*se:.1f}% 特异 {100*sp:.1f}% PPV {100*ppv:.1f}% NPV {100*npv:.1f}% "
        f"符合率 {100*po:.1f}% κ {(po-pe)/(1-pe):.3f}")

log("")
log("=" * 80)
log("C. G93 子码完整枚举（暴露入住内）")
log("=" * 80)
LAB = {"G931": "G93.1 Anoxic brain damage", "G9340": "G93.40 Encephalopathy, unspecified",
       "G9341": "G93.41 Metabolic encephalopathy", "G9349": "G93.49 Other encephalopathy",
       "G935": "G93.5 Compression of brain", "G936": "G93.6 Cerebral oedema",
       "G9381": "G93.81 Temporal sclerosis", "G9382": "G93.82 Brain death",
       "G9389": "G93.89 Other specified disorders of brain"}
g = dx[(dx.v == 10) & dx.code.str.startswith("G93") & dx.stay_id.isin(exp)]
sub = g.groupby("code").stay_id.nunique().sort_values(ascending=False)
for c, n in sub.items():
    log(f"  {c:7s} {LAB.get(c, '(not in label map)'):40s} {n:3d}")
log(f"  G93.x distinct {g.stay_id.nunique()}；子码计数和 {int(sub.sum())}（子码可重叠）")

log("")
log("=" * 80)
log("D. AKI 时间窗与首次达标标准")
log("=" * 80)
kd["h"] = (kd.charttime - kd.stay_id.map(imap)).dt.total_seconds() / 3600.0
pos = kd[kd.aki_stage >= 1]
any_aki = set(pos.stay_id)
during = set(pos[pos.h >= 0].stay_id)
pre = set(pos[pos.h < 0].stay_id)
log(f"  任意窗 AKI {len(any_aki)}；ICU 内曾达标 {len(during)}；入 ICU 前曾达标 {len(pre)}")
log(f"  两者重叠 {len(during & pre)}；仅入 ICU 前 {len(pre - during)}；ICU 内且入 ICU 前无 {len(during - pre)}")
log(f"  校验：{len(pre)} + ({len(during)} − {len(during & pre)}) = "
    f"{len(pre) + len(during) - len(during & pre)}（应等于任意窗 {len(any_aki)}）")


def crit_first(sub_df, tag):
    out = {}
    for sid, g in sub_df.groupby("stay_id"):
        g = g.sort_values("charttime")
        r = g.iloc[0]
        c = (r.aki_stage_creat >= 1, r.aki_stage_uo >= 1, r.aki_stage_crrt >= 1)
        key = ("creat" if c[0] else "") + ("uo" if c[1] else "") + ("crrt" if c[2] else "")
        out[sid] = key or "none"
    s = pd.Series(out)
    log(f"  {tag}（n={len(s)}）：")
    for k, v in s.value_counts().items():
        log(f"    {k:16s} {v:4d}")
    n1 = int((s.isin(["creat", "uo", "crrt"])).sum())
    n2 = int((s.isin(["creatuo", "creatcrrt", "uocrrt"])).sum())
    n3 = int((s == "creatuocrrt").sum())
    log(f"    单一标准 {n1} / 两个标准 {n2} / 三个标准 {n3}")
    return s


c_icu = crit_first(pos[pos.h >= 0], "首次 ICU 内达标记录")
c_any = crit_first(pos, "首次任意窗达标记录")

log("")
log("=" * 80)
log("E. ICU 窗 logistic 的实际样本量")
log("=" * 80)
m["aki_icu"] = m.stay_id.isin(during).astype(int)
no_icu = (~m.stay_id.isin(kd.stay_id))
log(f"  完全无 kdigo_stages 记录（任何窗）：{int(no_icu.sum())} 例")
log(f"  无 ICU 内记录（h≥0）：{int((~m.stay_id.isin(during) & ~m.stay_id.isin(pre)).sum())} 例"
    "（= 无 ICU 内阳性记录且无入 ICU 前记录）")
log(f"  无任何阳性记录者 {int((~m.stay_id.isin(any_aki)).sum())} 例")
for lab, dd in [("全 550（无记录按无 AKI）", m),
                ("仅限有 ICU 内 KDIGO 记录者", m[m.stay_id.isin(kd[kd.h >= 0].stay_id)])]:
    f = "aki_icu ~ npsle_narrow + " + " + ".join(COV)
    fi = smf.logit(f, data=dd).fit(disp=0)
    ci = fi.conf_int().loc["npsle_narrow"]
    log(f"  {lab}: n = {int(fi.nobs)}, aOR {np.exp(fi.params['npsle_narrow']):.2f} "
        f"({np.exp(ci[0]):.2f}–{np.exp(ci[1]):.2f}), p = {fi.pvalues['npsle_narrow']:.4f}")

log("")
log("=" * 80)
log("F. 120 例排序队列 与 Cox 人群的关系")
log("=" * 80)
cam["h"] = (cam.charttime - cam.stay_id.map(imap)).dt.total_seconds() / 3600.0
campos_h = cam[cam.v == "positive"].groupby("stay_id")["h"].min()
camass_n = cam[cam.v.isin(["positive", "negative"])].groupby("stay_id").size()
m["t_pos"] = m.stay_id.map(campos_h)
m["n_assess"] = m.stay_id.map(camass_n).fillna(0).astype(int)
m["assessable"] = (m.n_assess > 0).astype(int)
pos_icu = pos[pos.h >= 0].groupby("stay_id")["h"].min()
pos_any = pos.groupby("stay_id")["h"].min()
m["t_aki"] = m.stay_id.map(pos_icu)
m["t_aki_all"] = m.stay_id.map(pos_any)
m["pre_aki"] = m.stay_id.isin(pre).astype(int)
both120 = m[m.t_pos.notna() & m.t_aki.notna()]
log(f"  Panel A 队列（CAM+ 且 ICU 内 AKI）：{len(both120)}")
log(f"    其中 assessable = 0：{int((both120.assessable == 0).sum())}（不可能，应为 0）")
log(f"    其中 pre_aki = 1（入 ICU 前已达标）：{int((both120.pre_aki == 1).sum())}")
log(f"    其中首次 CAM 阳性在 t=0：{int((both120.t_pos <= 0).sum())}")
log(f"    进入前向队列（assessable 且 pre_aki=0 且 t_aki>0 或缺失）："
    f"{int(((both120.assessable == 1) & (both120.pre_aki == 0)).sum())}")
log(f"    进入反向队列（另加 t_pos>0）："
    f"{int(((both120.assessable == 1) & (both120.pre_aki == 0) & (both120.t_pos > 0)).sum())}")
log("")
log("  | 人群 | n | 事件 |")
log("  |---|---|---|")
log(f"  | 可判读（CAM 有 Positive/Negative） | {int((m.assessable==1).sum())} | CAM 阳性 {int(((m.assessable==1)&m.t_pos.notna()).sum())} |")
log(f"  | 且 入 ICU 前无 AKI | {int(((m.assessable==1)&(m.pre_aki==0)).sum())} | 后续 AKI {int(((m.assessable==1)&(m.pre_aki==0)&m.t_aki.notna()).sum())} |")
log(f"  | 且 首次 CAM 阳性>0（反向队列） | {int(((m.assessable==1)&(m.pre_aki==0)&((m.t_pos.isna())|(m.t_pos>0))).sum())} | 谵妄 {int(((m.assessable==1)&(m.pre_aki==0)&((m.t_pos.isna())|(m.t_pos>0))&m.t_pos.notna()).sum())} |")
log(f"  | 两事件均发生（Panel A） | {len(both120)} | — |")
in_f = ((both120.assessable == 1) & (both120.pre_aki == 0)).sum()
log(f"  → Panel A 的 {len(both120)} 例中，{len(both120)-in_f} 例因入 ICU 前已达标而不进入时变模型；"
    f"{in_f} 例进入前向模型")
excl_pre = both120[both120.pre_aki == 1]
if len(excl_pre):
    log(f"     被排除者：ICU 前首次达标距入 ICU 中位 "
        f"{-(excl_pre.stay_id.map(pos[lambda d: d.h < 0].groupby('stay_id').h.min())).median():.1f} h")

log("")
log("=" * 80)
log("G. ESRD / 透析依赖在时序模型人群中的分布")
log("=" * 80)
esrd = hit(["N186"], 10) | hit(["5856"], 9) | hit(["Z992"], 10) | hit(["V4511"], 9)
m["esrd"] = m.stay_id.isin(esrd).astype(int)
log(f"  全体 550 中 coded ESRD/透析依赖 {int(m.esrd.sum())}（{100*m.esrd.mean():.1f}%）")
fwd = m[(m.assessable == 1) & (m.pre_aki == 0)]
log(f"  前向队列 n={len(fwd)}，其中 ESRD {int(fwd.esrd.sum())}；"
    f"ESRD 者后续 AKI {int(fwd[fwd.esrd==1].t_aki.notna().sum())}/{int(fwd.esrd.sum())}")
revq = fwd[(fwd.t_pos.isna()) | (fwd.t_pos > 0)]
log(f"  反向队列 n={len(revq)}，其中 ESRD {int(revq.esrd.sum())}；"
    f"ESRD 者谵妄 {int(revq[revq.esrd==1].t_pos.notna().sum())}/{int(revq.esrd.sum())}")
log(f"  ESRD 与暴露交叉（全体）：")
log(pd.crosstab(m.esrd, m.npsle_narrow).to_string())

log("")
log("=" * 80)
log("H. 匹配后 LOS 的配对检验（89 对）")
log("=" * 80)
pp = pd.read_csv(os.path.join(BASE, "analysis", "psm_pairs_v23.csv"))
log(f"  配对文件列：{list(pp.columns)}；行数 {len(pp)}")
c1, c0 = pp.columns[0], pp.columns[1]
los = m.set_index("stay_id")["los_hospital"]
a_, b_ = pp[c1].map(los), pp[c0].map(los)
w = stats.wilcoxon(a_, b_)
log(f"  暴露组 LOS 中位 {a_.median():.1f}（IQR {a_.quantile(.25):.1f}–{a_.quantile(.75):.1f}）")
log(f"  对照组 LOS 中位 {b_.median():.1f}（IQR {b_.quantile(.25):.1f}–{b_.quantile(.75):.1f}）")
log(f"  Wilcoxon 符号秩检验 p = {w.pvalue:.4g}（配对）；中位差值 "
    f"{(a_-b_).median():.1f} d")

with io.open(OUT, "w", encoding="utf-8") as f:
    f.write("\n".join(L))
print("\n[写出]", OUT)
