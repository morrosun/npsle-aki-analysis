# -*- coding: utf-8 -*-
"""Add illness-severity scores to the SLE ICU cohort CSVs (psql COPY path, avoids psycopg2 encode issue).
- MIMIC-IV: OASIS (mimiciv_derived.oasis) -- single admission severity score, NO serum-creatinine component (non-renal).
- eICU:     APACHE IV total score (eicu_crd.apachepatientresult.apachescore).
NWICU is descriptive-only and has no derived severity table; left as-is.
Idiom: CSV stay_id == MIMIC/NWICU icustay_id == eICU patientunitstayid.
"""
import os, subprocess
import pandas as pd

DATA = os.environ.get("SLECI_DATA", "data")
PSQL = "D:/Program Files/PostgreSQL/18/bin/psql.exe"
ENV = dict(os.environ, PGPASSWORD=os.environ.get("SLECI_PGPASSWORD", ""), PGCLIENTENCODING="UTF8")

def psql_copy(db, sql, out_csv):
    with open(out_csv, "w") as f:
        subprocess.run([PSQL, "-h", "localhost", "-U", "postgres", "-d", db,
                        "-c", f"COPY ({sql}) TO STDOUT WITH CSV HEADER"],
                       env=ENV, stdout=f, check=True)

# ---------- MIMIC-IV: OASIS ----------
psql_copy("mimiciv", "SELECT stay_id, oasis FROM mimiciv_derived.oasis", f"{DATA}/_oasis_tmp.csv")
oasis = pd.read_csv(f"{DATA}/_oasis_tmp.csv").dropna(subset=["oasis"])
mim = pd.read_csv(f"{DATA}/mimiciv_sle.csv").drop(columns=[c for c in ["oasis","apache_iv"] if c in pd.read_csv(f"{DATA}/mimiciv_sle.csv").columns])
mim = mim.merge(oasis, on="stay_id", how="left")
print(f"MIMIC-IV: {len(mim)} rows; OASIS available for {mim['oasis'].notna().sum()} ({100*mim['oasis'].notna().mean():.1f}%)")
print("  OASIS median [IQR]:", round(mim['oasis'].median(),1), [round(mim['oasis'].quantile(q),1) for q in (0.25,0.75)])
mim.to_csv(f"{DATA}/mimiciv_sle.csv", index=False)

# ---------- eICU: APACHE IV ----------
psql_copy("eicu", "SELECT patientunitstayid AS stay_id, apachescore AS apache_iv FROM eicu_crd.apachepatientresult", f"{DATA}/_apache_tmp.csv")
ap = pd.read_csv(f"{DATA}/_apache_tmp.csv").dropna(subset=["apache_iv"]).drop_duplicates(subset=["stay_id"], keep="first")
eic = pd.read_csv(f"{DATA}/eicu_sle.csv").drop(columns=[c for c in ["oasis","apache_iv"] if c in pd.read_csv(f"{DATA}/eicu_sle.csv").columns])
eic = eic.merge(ap, on="stay_id", how="left").drop_duplicates(subset=["stay_id"], keep="first")
print(f"eICU: {len(eic)} rows; APACHE IV available for {eic['apache_iv'].notna().sum()} ({100*eic['apache_iv'].notna().mean():.1f}%)")
print("  APACHE IV median [IQR]:", round(eic['apache_iv'].median(),1), [round(eic['apache_iv'].quantile(q),1) for q in (0.25,0.75)])
for df, col in [(mim, "oasis"), (eic, "apache_iv")]:
    g0 = df[df.npsle_narrow==0][col].dropna(); g1 = df[df.npsle_narrow==1][col].dropna()
    print(f"  {col}: NPSLE- median {g0.median():.1f} (n={len(g0)}) vs NPSLE+ median {g1.median():.1f} (n={len(g1)})")
eic.to_csv(f"{DATA}/eicu_sle.csv", index=False)
os.remove(f"{DATA}/_oasis_tmp.csv"); os.remove(f"{DATA}/_apache_tmp.csv")
print("Severity extraction complete.")
