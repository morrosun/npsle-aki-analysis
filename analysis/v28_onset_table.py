# -*- coding: utf-8 -*-
"""v28：为 374 例 ICU 窗内 AKI 生成可核对的起点表，并保存 374/374 一致性检查。

输出：
  analysis/v28_aki_onset_table.csv   每例：复合首达时间、三分项首达时间、首个达标分项
  analysis/v28_aki_onset_report.txt  一致性检查与计数
"""
import io, os, warnings
import numpy as np, pandas as pd, psycopg2

warnings.filterwarnings("ignore")
BASE = os.environ.get("SLECI_BASE", os.environ.get("SLECI_BASE", "."))
DATA, ANA = os.path.join(BASE, "data"), os.path.join(BASE, "analysis")
PG = dict(host="localhost", port=5432, user="postgres",
          password=os.environ.get("SLECI_PGPASSWORD", ""), dbname="mimiciv")
L = []


def log(*a):
    t = " ".join(str(x) for x in a); print(t); L.append(t)


m = pd.read_csv(os.path.join(DATA, "mimiciv_sle.csv"))
ids = m.stay_id.astype("int64").tolist()
con = psycopg2.connect(**PG)
kd = pd.read_sql("SELECT stay_id,charttime,aki_stage,aki_stage_creat,aki_stage_uo,aki_stage_crrt "
                 "FROM mimiciv_derived.kdigo_stages WHERE stay_id=ANY(%s)", con, params=(ids,))
icu = pd.read_sql("SELECT stay_id,intime,outtime FROM mimiciv_icu.icustays "
                  "WHERE stay_id=ANY(%s)", con, params=(ids,))
con.close()
kd["charttime"] = pd.to_datetime(kd.charttime)
icu["intime"] = pd.to_datetime(icu.intime); icu["outtime"] = pd.to_datetime(icu.outtime)
kd["h"] = (kd.charttime - kd.stay_id.map(icu.set_index("stay_id").intime)).dt.total_seconds() / 3600.0
TEND = ((icu.outtime - icu.intime).dt.total_seconds() / 3600.0)
TEND.index = icu.stay_id.values
kd["t_end"] = kd.stay_id.map(TEND)
w = kd[(kd.h >= 0) & (kd.h <= kd.t_end)].copy()

for c in ["aki_stage", "aki_stage_creat", "aki_stage_uo", "aki_stage_crrt"]:
    w[c] = w[c].fillna(0)

first = lambda mask, col="h": (w[mask].sort_values("h").groupby("stay_id")[col].first())
comp = first(w.aki_stage >= 1)
cr = first(w.aki_stage_creat >= 1)
uo = first(w.aki_stage_uo >= 1)
rr = first(w.aki_stage_crrt >= 1)

tab = pd.DataFrame({"first_composite_positive_time": comp, "first_creatinine_positive_time": cr,
                    "first_urine_output_positive_time": uo, "first_crrt_positive_time": rr})
mn = tab.min(axis=1)
tab["min_component_time"] = mn
tab["consistent"] = (tab.first_composite_positive_time - mn).abs() < 1e-9
tab["first_criterion"] = np.select(
    [tab.first_creatinine_positive_time.eq(mn),
     tab.first_urine_output_positive_time.eq(mn),
     tab.first_crrt_positive_time.eq(mn)],
    ["creatinine", "urine_output", "crrt"], default="none")
tab.index.name = "stay_id"
tab = tab.reset_index()
tab.to_csv(os.path.join(ANA, "v28_aki_onset_table.csv"), index=False)

log("=" * 70)
log("ICU 窗内达到 KDIGO≥1 的入住数:", len(tab))
log("复合首达时间 == 最早分项首达时间:", int(tab.consistent.sum()), "/", len(tab))
log("不一致者:", int((~tab.consistent).sum()))
log("首个达标分项:", tab.first_criterion.value_counts().to_dict())
log("分项计数之和:", int((tab.first_criterion != "none").sum()))

# 前向模型 217 个 AKI 事件的分项构成
pos_pre = kd[(kd.h < 0) & (kd.aki_stage >= 1)].stay_id.unique()
cam = pd.read_sql("SELECT stay_id,charttime,value FROM mimiciv_icu.chartevents "
                  "WHERE stay_id=ANY(%s) AND itemid=228332", con, params=(ids,)) if False else None
con = psycopg2.connect(**PG)
cam = pd.read_sql("SELECT stay_id,charttime,value FROM mimiciv_icu.chartevents "
                  "WHERE stay_id=ANY(%s) AND itemid=228332", con, params=(ids,))
con.close()
cam["charttime"] = pd.to_datetime(cam.charttime)
cam["h"] = (cam.charttime - cam.stay_id.map(icu.set_index("stay_id").intime)).dt.total_seconds() / 3600.0
cam["v"] = cam.value.astype(str).str.strip().str.lower()
cam["t_end"] = cam.stay_id.map(TEND)
cam = cam[(cam.h >= 0) & (cam.h <= cam.t_end)]
assessable = set(cam[cam.v.isin(["positive", "negative"])].stay_id)
fwd = tab[tab.stay_id.isin(assessable) & ~tab.stay_id.isin(pos_pre)]
log("前向模型事件数:", len(fwd), "分项:", fwd.first_criterion.value_counts().to_dict())
io.open(os.path.join(ANA, "v28_aki_onset_report.txt"), "w", encoding="utf-8").write("\n".join(L))
print("\nwritten")
