# -*- coding: utf-8 -*-
"""v25 审计：导出时变模型数据集（含 esrd 标志），供 ESRD 排除与 PH 专项复核。"""
import os
import io, os, json, warnings
import numpy as np, pandas as pd, psycopg2

warnings.filterwarnings("ignore")
BASE = os.environ.get("SLECI_BASE", os.environ.get("SLECI_BASE", "."))
DATA, ANA = os.path.join(BASE, "data"), os.path.join(BASE, "analysis")
EX = os.path.join(ANA, "_cpexp2")
os.makedirs(EX, exist_ok=True)
PG = dict(host="localhost", port=5432, user="postgres",
          password=os.environ.get("SLECI_PGPASSWORD", ""), dbname="mimiciv")
COV = ["age", "female", "charlson_norheum", "lupus_nephritis", "renal_disease", "diabetes"]

m = pd.read_csv(os.path.join(DATA, "mimiciv_sle.csv"))
ids = m.stay_id.astype("int64").tolist()
con = psycopg2.connect(**PG)
icu = pd.read_sql("SELECT stay_id,intime,outtime FROM mimiciv_icu.icustays "
                  "WHERE stay_id=ANY(%s)", con, params=(ids,))
kd = pd.read_sql("SELECT stay_id,charttime,aki_stage FROM mimiciv_derived.kdigo_stages "
                 "WHERE stay_id=ANY(%s)", con, params=(ids,))
cam = pd.read_sql("SELECT stay_id,charttime,value FROM mimiciv_icu.chartevents "
                  "WHERE stay_id=ANY(%s) AND itemid=228332", con, params=(ids,))
sep = pd.read_sql("SELECT stay_id,suspected_infection_time FROM mimiciv_derived.sepsis3 "
                  "WHERE stay_id=ANY(%s)", con, params=(ids,))
vaso = pd.read_sql("SELECT stay_id,starttime FROM mimiciv_derived.vasoactive_agent "
                   "WHERE stay_id=ANY(%s)", con, params=(ids,))
vent = pd.read_sql("SELECT stay_id,starttime,ventilation_status FROM mimiciv_derived.ventilation "
                   "WHERE stay_id=ANY(%s)", con, params=(ids,))
con.close()
for d in (kd, cam):
    d["charttime"] = pd.to_datetime(d["charttime"])
icu["intime"] = pd.to_datetime(icu.intime); icu["outtime"] = pd.to_datetime(icu.outtime)
sep["suspected_infection_time"] = pd.to_datetime(sep.suspected_infection_time)
vaso["starttime"] = pd.to_datetime(vaso.starttime)
vent["starttime"] = pd.to_datetime(vent.starttime)
imap = icu.set_index("stay_id")["intime"]

m = m.merge(icu[["stay_id", "intime", "outtime"]], on="stay_id", how="left")
m["t_end"] = (m.outtime - m.intime).dt.total_seconds() / 3600.0
m["t_end"] = m.t_end.fillna(m.los_icu * 24.0)
kd["h"] = (kd.charttime - kd.stay_id.map(imap)).dt.total_seconds() / 3600.0
pos = kd[kd.aki_stage >= 1]
m = m.merge(pos[pos.h >= 0].groupby("stay_id")["h"].min().rename("t_aki"), on="stay_id", how="left")
m = m.merge(pos[pos.aki_stage >= 2].groupby("stay_id")["h"].min().rename("t_aki2"), on="stay_id", how="left")
m = m.merge(pos[pos.h < 0].groupby("stay_id").size().rename("n_pre"), on="stay_id", how="left")
m["pre_aki"] = m.n_pre.notna().astype(int)

cam["h"] = (cam.charttime - cam.stay_id.map(imap)).dt.total_seconds() / 3600.0
cam["v"] = cam.value.astype(str).str.strip().str.lower()
ass = cam[cam.v.isin(["positive", "negative"])]
m = m.merge(cam[cam.v == "positive"].groupby("stay_id")["h"].min().rename("t_pos"), on="stay_id", how="left")
m = m.merge(cam[cam.v == "uta"].groupby("stay_id")["h"].min().rename("t_uta"), on="stay_id", how="left")
m = m.merge(ass.groupby("stay_id").size().rename("n_assess"), on="stay_id", how="left")
m = m.merge(ass.groupby("stay_id")["h"].min().rename("t_first"), on="stay_id", how="left")
m = m.merge(ass.groupby("stay_id")["h"].max().rename("t_last"), on="stay_id", how="left")
m["n_assess"] = m.n_assess.fillna(0).astype(int)
m["assessable"] = (m.n_assess > 0).astype(int)
m["t_pos_uta"] = m[["t_pos", "t_uta"]].min(axis=1)

sep["t_sep"] = (sep.suspected_infection_time - sep.stay_id.map(imap)).dt.total_seconds() / 3600.0
m = m.merge(sep[["stay_id", "t_sep"]], on="stay_id", how="left")
vaso["h"] = (vaso.starttime - vaso.stay_id.map(imap)).dt.total_seconds() / 3600.0
m = m.merge(vaso.groupby("stay_id")["h"].min().rename("t_vaso"), on="stay_id", how="left")
vent = vent[vent.ventilation_status.isin(["InvasiveVent", "Tracheostomy"])].copy()
vent["h"] = (vent.starttime - vent.stay_id.map(imap)).dt.total_seconds() / 3600.0
m = m.merge(vent.groupby("stay_id")["h"].min().rename("t_vent"), on="stay_id", how="left")

# ---- coded ESRD / dialysis dependence（评审第 4 项：时序模型是否纳入这类患者）----
_hadm = pd.read_sql("SELECT stay_id,hadm_id FROM mimiciv_icu.icustays WHERE stay_id=ANY(%s)",
                    psycopg2.connect(**PG), params=(ids,))
_dx = pd.read_sql("SELECT hadm_id,icd_code,icd_version FROM mimiciv_hosp.diagnoses_icd "
                  "WHERE hadm_id=ANY(%s)",
                  psycopg2.connect(**PG), params=(_hadm.hadm_id.dropna().astype('int64').unique().tolist(),))
_dx["code"] = _dx.icd_code.astype(str).str.strip().str.upper()
_dx["v"] = _dx.icd_version.astype(int)
_dx = _dx.merge(_hadm, on="hadm_id")
_esrd = set(_dx[((_dx.v == 10) & (_dx.code.str.startswith("N186") | _dx.code.str.startswith("Z992")))
                | ((_dx.v == 9) & (_dx.code.str.startswith("5856") | _dx.code.str.startswith("V4511")))].stay_id)
m["esrd"] = m.stay_id.isin(_esrd).astype(int)
print("[esrd]", int(m.esrd.sum()))

base = m[(m.assessable == 1) & (m.pre_aki == 0)].copy()
base = base[(base.t_aki.isna()) | (base.t_aki > 0)].reset_index(drop=True)
rev = m[(m.assessable == 1) & (m.pre_aki == 0) & m.t_first.notna()].copy()
rev["t_cam_end"] = rev[["t_last", "t_end"]].min(axis=1)
rev = rev[(rev.t_pos.isna()) | (rev.t_pos > 0)].reset_index(drop=True)
for dd in (base, rev):
    f = dd.sort_values(["subject_id", "intime"]).drop_duplicates("subject_id", keep="first")
    dd["first_stay"] = dd.stay_id.isin(f.stay_id).astype(int)

TVCOLS = {"tv2": [("vaso_tv", "t_vaso"), ("vent_tv", "t_vent")],
          "tv3": [("vaso_tv", "t_vaso"), ("vent_tv", "t_vent"), ("sep_tv", "t_sep")]}


def build(d, t_out, t_cen, expo, landmark=0.0, tv=(), extra_cuts=()):
    """tv 非空时，每一行按该时段起点赋值各时变状态。"""
    rows = []
    for _, r in d.iterrows():
        cand = [x for x in (r[t_out], r[t_cen]) if pd.notna(x) and np.isfinite(x)]
        if not cand:
            continue
        end = min(cand) - landmark
        if end <= 0:
            continue
        ev = (r[t_out] - landmark) if pd.notna(r[t_out]) else np.inf
        cuts = set(x - landmark for x in extra_cuts if landmark < x < min(cand))
        for col in [expo] + [c for _, c in tv]:
            t = r[col]
            if pd.notna(t) and np.isfinite(t) and landmark < t < min(cand):
                cuts.add(float(t) - landmark)
        bnd = [0.0] + sorted(cuts) + [end]
        for a, b in zip(bnd[:-1], bnd[1:]):
            if b <= a:
                continue
            seg = {"entry": a, "stop": b, "event": 1 if (ev > a and ev <= b) else 0,
                   "stay_id": int(r.stay_id), "subject_id": int(r.subject_id),
                   "esrd": int(r["esrd"]) if "esrd" in r.index else 0,
                   **{c: float(r[c]) for c in COV}}
            t = r[expo]
            seg["expo"] = 1 if (pd.notna(t) and np.isfinite(t) and (t - landmark) <= a) else 0
            for nm, col in tv:
                t2 = r[col]
                seg[nm] = 1 if (pd.notna(t2) and np.isfinite(t2) and (t2 - landmark) <= a) else 0
            rows.append(seg)
    return pd.DataFrame(rows)


b = base.copy(); b["t_cen72"] = np.minimum(b.t_end, 72.0)
r = rev.copy(); r["t_cen72"] = np.minimum(r.t_cam_end, 72.0)

JOBS = [
    ("f_main", b, "t_aki", "t_end", "t_pos", 0.0, "", ""),
    ("r_main", r, "t_pos", "t_cam_end", "t_aki", 0.0, "", ""),
    ("f_noesrd", b[b.esrd == 0], "t_aki", "t_end", "t_pos", 0.0, "", ""),
    ("r_noesrd", r[r.esrd == 0], "t_pos", "t_cam_end", "t_aki", 0.0, "", ""),
]

meta = {}
for name, d, t_out, t_cen, expo, lm, tvkey, extra in JOBS:
    e = build(d, t_out, t_cen, expo, landmark=lm, tv=TVCOLS.get(tvkey, ()),
              extra_cuts=(48.0,) if extra == "period" else ())
    if extra == "nass":
        e["n_assess"] = d.set_index("stay_id").loc[e.stay_id, "n_assess"].values
    if extra == "period":
        e["period"] = np.where(e.entry >= 48 - 1e-9, 2, 1)
        assert (e.period == 2).sum() > 0, name
    e.to_csv(os.path.join(EX, name + ".csv"), index=False)
    meta[name] = {"stays": int(len(d)), "rows": int(len(e)), "events": int(e.event.sum())}

json.dump(meta, io.open(os.path.join(EX, "meta.json"), "w", encoding="utf-8"), indent=1)
for k, v in meta.items():
    print(f"{k:10s} stays {v['stays']:4d} rows {v['rows']:5d} events {v['events']:4d}")
