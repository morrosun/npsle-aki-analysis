# -*- coding: utf-8 -*-
"""v27 审计：第五轮评审两项硬核查。

A. CAM-ICU 的时间窗口
   chartevents itemid 228332 是按 stay_id 全量取用、未限制 intime–outtime。
   本脚本量化：记录在窗外的规模；限制到 intime–outtime 后
   376 / 141 / 326 / 325 / 115 各变成多少；逐例检查唯一入 ICU 前阳性者
   （stay 33815705）在 ICU 内有无任何 Positive / Negative 记录。

B. AKI 综合起点 vs 各分项起点
   对 ICU 内有 KDIGO>=1 的每例，输出
   first_composite / first_creat / first_uo / first_crrt 以及
   component values at first composite，检验
   first_composite == min(first_creat, first_uo, first_crrt)。
   并给出 217 个前向 AKI 事件的分项分布（而非 374 例的口径）。

输出同时写 stdout 与 analysis/v27_audit_cam_aki_report.txt
"""
import io
import os

import numpy as np
import pandas as pd
import psycopg2

BASE = os.environ.get("SLECI_BASE", os.environ.get("SLECI_BASE", "."))
PG = dict(host="localhost", port=5432, user="postgres",
          password=os.environ.get("SLECI_PGPASSWORD", ""), dbname="mimiciv")
OUT = io.StringIO()


def P(*a):
    line = " ".join(str(x) for x in a)
    print(line)
    OUT.write(line + "\n")


m = pd.read_csv(f"{BASE}/data/mimiciv_sle.csv")
ids = m["stay_id"].astype("int64").tolist()
assert len(m) == 550

con = psycopg2.connect(**PG)
icu = pd.read_sql("select stay_id, hadm_id, intime, outtime from mimiciv_icu.icustays "
                  "where stay_id = any(%s)", con, params=(ids,))
kd = pd.read_sql("""select stay_id, charttime, aki_stage,
                           aki_stage_creat, aki_stage_uo, aki_stage_crrt,
                           creat, creat_low_past_7day, creat_low_past_48hr
                    from mimiciv_derived.kdigo_stages
                    where stay_id = any(%s) order by stay_id, charttime""",
                 con, params=(ids,))
cam = pd.read_sql("""select stay_id, charttime, value
                     from mimiciv_icu.chartevents
                     where itemid = 228332 and stay_id = any(%s)
                     order by stay_id, charttime""", con, params=(ids,))
con.close()

for d in (kd, cam):
    d["charttime"] = pd.to_datetime(d.charttime)
icu["intime"] = pd.to_datetime(icu.intime)
icu["outtime"] = pd.to_datetime(icu.outtime)
icu["t_end"] = (icu.outtime - icu.intime).dt.total_seconds() / 3600.0
end = icu.set_index("stay_id")["t_end"]
imap = icu.set_index("stay_id")["intime"]

kd["h"] = (kd.charttime - kd.stay_id.map(imap)).dt.total_seconds() / 3600.0
cam["h"] = (cam.charttime - cam.stay_id.map(imap)).dt.total_seconds() / 3600.0
cam["t_end"] = cam.stay_id.map(end)
cam["v"] = cam.value.astype(str).str.strip().str.lower()

# ============================================================ A. CAM 窗口
P("=" * 84)
P("A. CAM-ICU（itemid 228332）的时间窗口")
P("=" * 84)
P(f"总记录 {len(cam)} 条 / {cam.stay_id.nunique()} 例入住；取值分布：")
P(cam.v.value_counts().to_string().replace("\n", " | "))
judge = cam[cam.v.isin(["positive", "negative", "uta"])]
P(f"\n可判读类（positive/negative/uta）{len(judge)} 条")
P(f"  charttime < intime        : {int((cam.h < 0).sum())} 条 / "
  f"{cam.loc[cam.h < 0, 'stay_id'].nunique()} 例")
P(f"  charttime > outtime       : {int((cam.h > cam.t_end).sum())} 条 / "
  f"{cam.loc[cam.h > cam.t_end, 'stay_id'].nunique()} 例")
inw = cam[(cam.h >= 0) & (cam.h <= cam.t_end)]
P(f"  在 intime–outtime 内       : {len(inw)} 条 / {inw.stay_id.nunique()} 例")
P(f"  h 范围（全量）: {cam.h.min():.2f} – {cam.h.max():.2f} h；"
  f"住院窗 [0, {(cam.t_end.max()):.1f}]")

camj = cam[cam.v.isin(["positive", "negative"])]
P("\nA1. 判读记录（positive/negative）逐条窗口归属")
P(f"  全量（不看窗口）: {len(camj)} 条 / {camj.stay_id.nunique()} 例")
for lab, sel in [("仅窗内", camj[(camj.h >= 0) & (camj.h <= camj.t_end)]),
                 ("窗内+前", camj[camj.h <= camj.t_end])]:
    P(f"  {lab}: {len(sel)} 条 / {sel.stay_id.nunique()} 例")


def build(cam_use, kd_use, label):
    """按给定 CAM 与 KDIGO 记录集合重建关键计数。"""
    pos = kd_use[kd_use.aki_stage >= 1]
    pre = set(pos.loc[pos.h < 0, "stay_id"])
    t_aki = pos[pos.h >= 0].groupby("stay_id")["h"].min()
    t_aki2 = pos[pos.aki_stage >= 2].groupby("stay_id")["h"].min()
    cpos = cam_use[cam_use.v == "positive"].groupby("stay_id")["h"].min()
    cass = cam_use[cam_use.v.isin(["positive", "negative"])]
    n_ass = cass.groupby("stay_id").size()
    t_last = cass.groupby("stay_id")["h"].max()
    base = m[m.stay_id.isin(n_ass.index)].copy()
    base["pre_aki"] = base.stay_id.isin(pre)
    base["t_aki"] = base.stay_id.map(t_aki)
    base["t_pos"] = base.stay_id.map(cpos)
    base["t_last"] = base.stay_id.map(t_last)
    # 前向：assessable 且无 pre-AKI
    fwd = base[~base.pre_aki]
    # 反向：同上，排除首次阳性 <=0（无正向风险时间）
    rev = fwd[~(fwd.t_pos.fillna(1e9) <= 0)]
    n_ev = int((rev.t_pos > 0).sum())
    P(f"\n  [{label}]")
    P(f"    assessable                    = {len(base)}")
    P(f"    measured delirium（窗内阳性）  = {int(fwd.t_pos.notna().sum())}")
    P(f"    KDIGO>=2 阳性                 = {int(fwd.t_aki.isin(t_aki2.values).sum())}")
    P(f"    前向队列（assessable 且无 pre）= {len(fwd)}  "
      f"（其中 AKI 事件 {int(fwd.t_aki.notna().sum())}）")
    P(f"    反向队列（t_pos>0 或缺失）     = {len(rev)}  反向事件 {n_ev}")
    # Table 5 四格（assessable 全体，与 pre_aki 无关）
    a = base.copy()
    a["cam_pos"] = a.t_pos.notna()
    a["icd"] = m.set_index("stay_id").loc[a.stay_id, "npsle_narrow"].values.astype(bool)
    tp = int((a.cam_pos & a.icd).sum())
    fn = int((~a.cam_pos & a.icd).sum())
    fp = int((a.cam_pos & ~a.icd).sum())
    tn = int((~a.cam_pos & ~a.icd).sum())
    P(f"    Table 5 四格 TP/FN/FP/TN      = {tp}/{fn}/{fp}/{tn}  n={tp+fn+fp+tn}")
    return base, fwd, rev


kd_all = kd.copy()
base0, fwd0, rev0 = build(camj, kd_all, "全量 CAM，全窗 KDIGO（现行口径）")
base1, fwd1, rev1 = build(camj[(camj.h >= 0) & (camj.h <= camj.t_end)], kd_all,
                          "严格 intime–outtime CAM")
base2, fwd2, rev2 = build(camj[(camj.h >= 0) & (camj.h <= camj.t_end)],
                          kd[(kd.h >= 0) & (kd.h <= kd.stay_id.map(end))],
                          "严格 ICU 窗 CAM + KDIGO")

P("\nA2. 唯一入 ICU 前 CAM 阳性的入住（33815705）逐条记录")
one = cam[cam.stay_id == 33815705].sort_values("charttime")
P(f"  该 stay 的 CAM 记录数 = {len(one)}；t_end = {end.get(33815705):.2f} h")
if len(one):
    P(one[["charttime", "h", "v"]].to_string(index=False))
P(f"  ICU 窗内 positive: {int(((one.h >= 0) & (one.v == 'positive')).sum())}；"
  f"ICU 窗内 negative: {int(((one.h >= 0) & (one.v == 'negative')).sum())}")
P(f"  全量 positive: {int((one.v == 'positive').sum())}；"
  f"全量 negative: {int((one.v == 'negative').sum())}")
# 该例在严格窗口下是否仍留在反向队列
for lab, b in [("全量", base0), ("严格", base1)]:
    r = b[b.stay_id == 33815705]
    if len(r):
        P(f"  [{lab}] assessable={True}, t_pos={r.t_pos.iloc[0]}, "
          f"在反向队列={33815705 in set(rev0.stay_id if lab == '全量' else rev1.stay_id)}")
    else:
        P(f"  [{lab}] 不在 assessable")

P("\nA3. 首次阳性时间与末次可判读时间的分布（全量口径 / 严格口径）")
for lab, b in [("全量", base0), ("严格", base1)]:
    p = b.loc[b.t_pos.notna(), "t_pos"]
    l = b.loc[b.t_pos.notna(), "t_last"]
    P(f"  {lab}: 首次阳性 t_pos 中位 {p.median():.2f} h（IQR {p.quantile(.25):.2f}–"
      f"{p.quantile(.75):.2f}）；<0 者 {int((p < 0).sum())}，=0 者 {int((p == 0).sum())}")
    P(f"        末次可判读中位 {l.median():.2f} h；t_pos > t_last 者 "
      f"{int((p > l).sum())}")

P("\nA4. 受影响入住的窗外记录占比")
out_rec = camj[(camj.h < 0) | (camj.h > camj.t_end)]
P(f"  窗外判读记录 {len(out_rec)} 条 / {out_rec.stay_id.nunique()} 例；"
  f"其中前 {int((out_rec.h < 0).sum())} 条、后 {int((out_rec.h > out_rec.t_end).sum())} 条")
chg = set(base0.stay_id) ^ set(base1.stay_id)
P(f"  限制窗口后 assessable 集合变化: {len(chg)} 例 → {sorted(chg)}")
lost_pos = set(base0.loc[base0.t_pos.notna(), "stay_id"]) - \
    set(base1.loc[base1.t_pos.notna(), "stay_id"]) if len(base1) else set()
P(f"  限制窗口后失去阳性资格: {len(lost_pos)} 例 → {sorted(lost_pos)}")

# ============================================================ B. AKI 分项
P("")
P("=" * 84)
P("B. AKI 综合起点 vs 各分项起点")
P("=" * 84)
kdi = kd[(kd.h >= 0) & (kd.h <= kd.stay_id.map(end))].copy()
P(f"ICU 窗内 KDIGO 记录 {len(kdi)} 条 / {kdi.stay_id.nunique()} 例")
comp = kdi[kdi.aki_stage >= 1].groupby("stay_id")["h"].min().rename("t_comp")
tc = kdi[kdi.aki_stage_creat >= 1].groupby("stay_id")["h"].min().rename("t_cr")
tu = kdi[kdi.aki_stage_uo >= 1].groupby("stay_id")["h"].min().rename("t_uo")
tr = kdi[kdi.aki_stage_crrt >= 1].groupby("stay_id")["h"].min().rename("t_crrt")
B = pd.concat([comp, tc, tu, tr], axis=1)
P(f"  综合达标 {len(B)} 例；creat 达标 {int(B.t_cr.notna().sum())}；"
  f"uo 达标 {int(B.t_uo.notna().sum())}；crrt 达标 {int(B.t_crrt.notna().sum())}")

B["t_min"] = B[["t_cr", "t_uo", "t_crrt"]].min(axis=1)
B["diff"] = (B.t_comp - B.t_min).round(4)
P(f"\nB1. t_comp == min(分项首达) ?  相等 {int((B['diff'] == 0).sum())}，"
  f"不等 {int((B['diff'] != 0).sum())}")
if (B["diff"] != 0).any():
    P(B.loc[B["diff"] != 0, ["t_comp", "t_cr", "t_uo", "t_crrt", "t_min", "diff"]]
      .head(20).to_string())
    P(f"  diff 分布：{B.loc[B['diff'] != 0, 'diff'].describe().to_dict()}")

P("\nB2. 首个综合阳性记录上的分项取值（分项口径 1）")
first = kdi[kdi.aki_stage >= 1].sort_values("h").groupby("stay_id").head(1).set_index("stay_id")
P(f"  例数 {len(first)}")
c1 = (first.aki_stage_creat >= 1)
u1 = (first.aki_stage_uo >= 1)
r1 = (first.aki_stage_crrt >= 1)
P(f"  creat>=1 且 uo<1        : {int((c1 & ~u1 & ~r1).sum())}")
P(f"  uo>=1   且 creat<1      : {int((u1 & ~c1 & ~r1).sum())}")
P(f"  creat>=1 且 uo>=1       : {int((c1 & u1).sum())}")
P(f"  crrt>=1                 : {int(r1.sum())}")
P(f"  三者全 <1（异常）       : {int(((~c1) & (~u1) & (~r1)).sum())}")

P("\nB3. 为什么 both 与『无同时首次跨阈』可以并存：107 例的具体时间")
both_ids = first.index[(first.aki_stage_creat >= 1) & (first.aki_stage_uo >= 1)]
P(f"  both 例数 = {len(both_ids)}")
sub = B.loc[B.index.intersection(both_ids)]
P(f"  其中 t_cr == t_uo（真同时）        : {int((sub.t_cr == sub.t_uo).sum())}")
P(f"  其中 t_cr <  t_uo（creat 更早）    : {int((sub.t_cr < sub.t_uo).sum())}")
P(f"  其中 t_uo <  t_cr（uo 更早）       : {int((sub.t_uo < sub.t_cr).sum())}")
P(f"  both 组 t_comp 与 t_min 一致       : {int((sub.t_comp == sub.t_min).sum())}/{len(sub)}")
if (sub.t_cr < sub.t_uo).any() or (sub.t_uo < sub.t_cr).any():
    ex = sub[(sub.t_cr != sub.t_uo)].head(8)
    P("  示例（t_comp, t_cr, t_uo）：")
    P(ex[["t_comp", "t_cr", "t_uo"]].to_string())
    # 更早者对应该时点该分项的分期
    P("\n  核查：更早分项首达时，该时点 kdigo 记录上的分期值")
    rows = []
    for sid, r in ex.iterrows():
        tt = r.t_min
        snap = kdi[(kdi.stay_id == sid) & (kdi.h == tt)]
        if len(snap):
            s = snap.iloc[0]
            rows.append(dict(stay_id=sid, h=tt, aki_stage=s.aki_stage,
                             creat=s.aki_stage_creat, uo=s.aki_stage_uo,
                             crrt=s.aki_stage_crrt, creat_val=s.creat,
                             low48=s.creat_low_past_48hr, low7=s.creat_low_past_7day))
    P(pd.DataFrame(rows).to_string(index=False))

P("\nB4. 两种分项计数（374 例 vs 前向 217 事件）")
P("  374 例（ICU 内首次达标，含 intercurrent/前向混算）")
P(f"    仅 creat {int((c1 & ~u1 & ~r1).sum())} / 仅 uo {int((u1 & ~c1 & ~r1).sum())} / "
  f"both {int((c1 & u1).sum())} / crrt {int(r1.sum())}")
P(f"    各分项各自首达（不要求同一条记录）：creat {int(B.t_cr.notna().sum())} / "
  f"uo {int(B.t_uo.notna().sum())} / crrt {int(B.t_crrt.notna().sum())}")
P(f"    同时首达（t_cr == t_uo == t_comp 且两者均非空）: "
  f"{int(((B.t_cr == B.t_uo) & B.t_cr.notna() & B.t_uo.notna()).sum())}")

# 前向 217 事件：assessable 且无 pre-AKI 者的 AKI 事件
base0_ = base0.set_index("stay_id")
ev = base0_[(~base0_.pre_aki) & base0_.t_aki.notna()]
ev_ids = ev.index
fc = first.loc[first.index.intersection(ev_ids)]
c1e, u1e, r1e = (fc.aki_stage_creat >= 1), (fc.aki_stage_uo >= 1), (fc.aki_stage_crrt >= 1)
P(f"\n  前向模型事件（assessable、无 pre-AKI、ICU 内 AKI）n = {len(fc)}")
P(f"    仅 creat {int((c1e & ~u1e & ~r1e).sum())} / 仅 uo {int((u1e & ~c1e & ~r1e).sum())} / "
  f"both {int((c1e & u1e).sum())} / crrt {int(r1e.sum())}")
Be = B.loc[B.index.intersection(ev_ids)]
P(f"    各分项首达：creat {int(Be.t_cr.notna().sum())} / uo {int(Be.t_uo.notna().sum())} / "
  f"crrt {int(Be.t_crrt.notna().sum())}")

P("\nB5. kdigo_stages 是否平滑（同一 h 是否存在多条）")
dup = kdi.groupby(["stay_id", "h"]).size()
P(f"  同一 stay 同一 charttime 多条的组数: {int((dup > 1).sum())}")
P(f"  aki_stage 与 max(分项) 不符的条数: "
  f"{int((kdi.aki_stage != kdi[['aki_stage_creat','aki_stage_uo','aki_stage_crrt']].max(axis=1)).sum())}")

with open(f"{BASE}/analysis/v27_audit_cam_aki_report.txt", "w", encoding="utf-8") as f:
    f.write(OUT.getvalue())
print("\n[saved] analysis/v27_audit_cam_aki_report.txt")
