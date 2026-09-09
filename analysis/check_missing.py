import os
import pandas as pd, numpy as np
DATA = os.environ.get("SLECI_DATA", "data")
df = pd.read_csv(f"{DATA}/mimiciv_sle.csv")
mim = df[df.db == "MIMIC-IV"].copy()
print("MIMIC-IV rows:", len(mim))

cols = ["age","female","charlson_norheum","lupus_nephritis","renal_disease",
        "npsle_narrow","diabetes","cerebrovascular_disease","chf","copd",
        "aki","rrt","vent","vaso","mortality","los_hospital"]
print("\nPer-column missing % (MIMIC-IV):")
for c in cols:
    miss = mim[c].isna().mean()*100
    print(f"  {c:24s} {miss:6.2f}%  (n_missing={int(mim[c].isna().sum())})")

# Primary adjusted model covariates (age, sex, Charlson, lupus nephritis, renal disease)
prim = ["age","female","charlson_norheum","lupus_nephritis","renal_disease","npsle_narrow"]
print("\nPrimary-model covariate set max missing %:",
      round(mim[prim].isna().mean().max()*100, 2))

# Complete-case counts for each outcome model (outcome + exposure + 5 primary covariates)
for out in ["aki","rrt","vent","vaso","mortality","los_hospital"]:
    sub = mim.dropna(subset=[out,"npsle_narrow"]+["age","female","charlson_norheum","lupus_nephritis","renal_disease"])
    print(f"  complete-case {out:14s}: n={len(sub)} (of {len(mim)})")

# PSM model complete-case (9 covariates + outcomes)
psm_cov = ["age","female","charlson_norheum","lupus_nephritis","renal_disease",
           "diabetes","cerebrovascular_disease","chf","copd"]
for out in ["aki","rrt","vent","vaso","mortality","los_hospital"]:
    sub = mim.dropna(subset=[out,"npsle_narrow"]+psm_cov)
    print(f"  PSM complete-case {out:14s}: n={len(sub)} (of {len(mim)})")
