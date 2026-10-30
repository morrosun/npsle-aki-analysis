# -*- coding: utf-8 -*-
"""Build a descriptive baseline table for the temporal-analysis cohort (n = 326).

Scope
-----
This script RE-RUNS the derived-table queries that define the temporal cohort
(KDIGO staging from mimiciv_derived.kdigo_stages and CAM-ICU from chartevents,
itemid 228332) and then produces a purely DESCRIPTIVE table grouped by whether
AKI was recorded during the ICU stay.

No hypothesis tests are computed and no model is fitted: grouping is
descriptive, and the table footnote says so. The intent is to describe the
population actually analysed in the time-varying models, which is a subset of
the 550-stay cohort used for the concurrent analysis.

Run:  python analysis/make_table1b_tvc.py
Out:  output_nephro/table1b_tvc.md, output_nephro/table1b_tvc.csv
"""
import io
import os
import sys

import numpy as np
import pandas as pd
import psycopg2

BASE = os.environ.get("SLECI_BASE", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA = os.path.join(BASE, "data")
OUTDIR = os.path.join(BASE, "output_nephro")
PG = dict(host="localhost", port=5432, user="postgres",
          password=os.environ.get("SLECI_PGPASSWORD", ""), dbname="mimiciv")

log_lines = []


def log(s=""):
    print(s)
    log_lines.append(str(s))


# ---------------------------------------------------------------- 1 cohort
df = pd.read_csv(os.path.join(DATA, "mimiciv_sle.csv"))
ids = df["stay_id"].astype("int64").tolist()
assert len(df) == 550, len(df)

con = psycopg2.connect(**PG)
icu = pd.read_sql("SELECT stay_id, intime, outtime FROM mimiciv_icu.icustays "
                  "WHERE stay_id = ANY(%s)", con, params=(ids,))
kd = pd.read_sql("SELECT stay_id, charttime, aki_stage FROM mimiciv_derived.kdigo_stages "
                 "WHERE stay_id = ANY(%s) ORDER BY stay_id, charttime", con, params=(ids,))
cam = pd.read_sql("SELECT stay_id, charttime, value FROM mimiciv_icu.chartevents "
                  "WHERE stay_id = ANY(%s) AND itemid = 228332", con, params=(ids,))
con.close()

for d in (kd, cam):
    d["charttime"] = pd.to_datetime(d["charttime"])
icu["intime"] = pd.to_datetime(icu["intime"])
icu["outtime"] = pd.to_datetime(icu["outtime"])

m = df.merge(icu, on="stay_id", how="left")
m["t_end"] = (m["outtime"] - m["intime"]).dt.total_seconds() / 3600.0
m["t_end"] = m["t_end"].fillna(m["los_icu"] * 24.0)

# first KDIGO >=1 relative to intime
kd = kd.merge(icu[["stay_id", "intime"]], on="stay_id", how="left")
kd["h"] = (kd["charttime"] - kd["intime"]).dt.total_seconds() / 3600.0
pos = kd[kd["aki_stage"] >= 1]
t_aki_all = pos.groupby("stay_id")["h"].min().rename("t_aki_all")
t_aki = pos[pos["h"] >= 0].groupby("stay_id")["h"].min().rename("t_aki")
t_aki2 = pos[pos["aki_stage"] >= 2].groupby("stay_id")["h"].min().rename("t_aki2")
n_pre = pos[pos["h"] < 0].groupby("stay_id").size().rename("n_pre")
m = (m.merge(t_aki_all, on="stay_id", how="left")
      .merge(t_aki, on="stay_id", how="left")
      .merge(t_aki2, on="stay_id", how="left")
      .merge(n_pre, on="stay_id", how="left"))
m["pre_aki"] = m["n_pre"].notna().astype(int)

# CAM-ICU timing
cam["h"] = (cam["charttime"] - cam["stay_id"].map(icu.set_index("stay_id")["intime"])
            ).dt.total_seconds() / 3600.0
cam["v"] = cam["value"].astype(str).str.strip().str.lower()
assess = cam[cam["v"].isin(["positive", "negative"])]
m = m.merge(assess.groupby("stay_id").size().rename("n_assess"), on="stay_id", how="left")
m = m.merge(assess.groupby("stay_id")["h"].min().rename("t_first_assess"), on="stay_id", how="left")
m["n_assess"] = m["n_assess"].fillna(0).astype(int)
m["assessable"] = (m["n_assess"] > 0).astype(int)

# Measured delirium: at least one POSITIVE CAM-ICU charted inside the ICU window
# (intime <= charttime <= outtime), the definition used in the manuscript.
t_end_map = m.set_index("stay_id")["t_end"]
cam["in_icu"] = (cam["h"] >= 0) & (cam["h"] <= cam["stay_id"].map(t_end_map))
pos_in = cam[(cam["v"] == "positive") & (cam["in_icu"] == True)]  # noqa: E712
m = m.merge(pos_in.groupby("stay_id").size().rename("n_pos").to_frame(), on="stay_id", how="left")
m["n_pos"] = m["n_pos"].fillna(0).astype(int)
m["delirium"] = (m["n_pos"] > 0).astype(int)

# ---------------------------------------------------------------- 2 the 326
coh = m[(m["assessable"] == 1) & (m["pre_aki"] == 0)].copy()
coh["aki_icu"] = coh["t_aki"].notna().astype(int)
log("=" * 78)
log("Temporal-analysis cohort (re-derived)")
log("=" * 78)
log(f"550 总队列 -> 可判读 CAM {int(m['assessable'].sum())} -> "
    f"排除入 ICU 前已 AKI 后 n = {len(coh)}")
log(f"ICU 内 AKI：{int(coh['aki_icu'].sum())}；无 AKI：{int((1-coh['aki_icu']).sum())}")
log(f"AKI 分期 >=2：{int(coh['t_aki2'].notna().sum())}")
log(f"Measured delirium (>=1 positive CAM-ICU in ICU window)：{int(coh['delirium'].sum())}")
log(f"  no-AKI 组 {int(coh.loc[coh['aki_icu'] == 0, 'delirium'].sum())} / "
    f"AKI 组 {int(coh.loc[coh['aki_icu'] == 1, 'delirium'].sum())}")
assert len(coh) == 326, len(coh)
assert int(coh["aki_icu"].sum()) == 217, int(coh["aki_icu"].sum())
assert int(coh["delirium"].sum()) == 115, int(coh["delirium"].sum())
assert int(coh.loc[coh["aki_icu"] == 0, "delirium"].sum()) == 20
assert int(coh.loc[coh["aki_icu"] == 1, "delirium"].sum()) == 95


# ---------------------------------------------------------------- 3 table
def ms(series, dp=1):
    s = pd.to_numeric(series, errors="coerce").dropna()
    return f"{s.mean():.{dp}f} \u00b1 {s.std():.{dp}f}"


def miqr(series, dp=1):
    s = pd.to_numeric(series, errors="coerce").dropna()
    return f"{s.median():.{dp}f} [{s.quantile(.25):.{dp}f}\u2013{s.quantile(.75):.{dp}f}]"


def n_pct(mask, d):
    n = int(mask.sum())
    return f"{n} ({100 * n / len(d):.1f})"


groups = [("All", coh), ("No AKI in ICU", coh[coh["aki_icu"] == 0]),
          ("AKI in ICU", coh[coh["aki_icu"] == 1])]
head = "| Characteristic | " + " | ".join(f"{g} (n = {len(d)})" for g, d in groups) + " |"
sep = "|" + "---|" * (len(groups) + 1)

rows = []


def cont(label, fn, series_name):
    rows.append("| " + label + " | " + " | ".join(fn(d[series_name]) for _, d in groups) + " |")


def cat(label, mask_fn):
    rows.append("| " + label + " | " + " | ".join(n_pct(mask_fn(d), d) for _, d in groups) + " |")


cont("Age, years", ms, "age")
cat("Female sex", lambda d: d["female"] == 1)
cont("Charlson index", ms, "charlson_norheum")
cat("Lupus nephritis", lambda d: d["lupus_nephritis"] == 1)
cat("Coded renal disease", lambda d: d["renal_disease"] == 1)
cat("Diabetes", lambda d: d["diabetes"] == 1)
cat("Cerebrovascular disease", lambda d: d["cerebrovascular_disease"] == 1)
cat("Coded neuropsychiatric involvement (narrow)",
    lambda d: d["npsle_narrow"] == 1)
cont("OASIS", ms, "oasis")
cat("First CAM-ICU assessment within 24 h", lambda d: d["t_first_assess"].le(24))
cont("CAM-ICU assessments per stay", miqr, "n_assess")
cat("Measured delirium (\u22651 positive CAM-ICU)", lambda d: d["delirium"] == 1)
cat("KDIGO stage \u22652 during ICU", lambda d: d["t_aki2"].notna())
cat("Invasive mechanical ventilation", lambda d: d["vent"] == 1)
cat("Vasopressor use", lambda d: d["vaso"] == 1)
cat("Renal replacement therapy", lambda d: d["rrt"] == 1)
cont("ICU length of stay, hours", miqr, "t_end")
cat("In-hospital mortality", lambda d: d["mortality"] == 1)

table = "\n".join([head, sep] + rows)
note = ("*Descriptive only. Values are mean \u00b1 SD, median [IQR] or n (%), and no hypothesis "
        "test was performed for this table. The cohort is the population entering the time-varying "
        "models: stays with at least one assessable CAM-ICU record and no qualifying AKI record "
        "before ICU admission. Measured delirium is at least one positive CAM-ICU result charted "
        "between ICU admission and ICU discharge. Grouping by any recorded ICU-window AKI is "
        "descriptive only; the Cox models treated AKI as a time-varying exposure. Organ support, "
        "assessments and length of stay are stay-level characteristics and are not necessarily "
        "baseline measurements. Recorded RRT is descriptive and may include maintenance dialysis.*")

md = ("### Table 2. Characteristics of the temporal-analysis cohort, by whether AKI was recorded "
      "during the ICU stay (MIMIC-IV, n = 326)\n\n" + table + "\n\n" + note + "\n")
io.open(os.path.join(OUTDIR, "table1b_tvc.md"), "w", encoding="utf-8").write(md)
coh.to_csv(os.path.join(OUTDIR, "table1b_tvc.csv"), index=False)

log("")
log(table)
log("")
log("写出: output_nephro/table1b_tvc.md , table1b_tvc.csv")
io.open(os.path.join(OUTDIR, "table1b_tvc.log"), "w", encoding="utf-8").write("\n".join(log_lines) + "\n")
