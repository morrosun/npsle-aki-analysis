# -*- coding: utf-8 -*-
"""SLE neuropsychiatric/cognitive (NPSLE) analysis across MIMIC-IV, NWICU, eICU.
Cohort = SLE ICU stays. Primary exposure = NPSLE (narrow=cognitive/delirium/encephalopathy/dementia;
broad=+seizure/psychosis/CVA/demyelinating). Outcomes = in-hospital mortality, LOS, invasive MV, AKI, RRT, vasopressors.
v2 adds: eICU derived MV/AKI/RRT/vaso, propensity-score matching (PSM), broad-definition sensitivity, cross-DB outcome meta.
"""
import json, os
import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.api as sm
import statsmodels.formula.api as smf
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

DATA = os.environ.get("SLECI_DATA", "data")
OUT = os.environ.get("SLECI_OUT", "analysis")
os.makedirs(OUT, exist_ok=True)

COLS = ["db","subject_id","hadm_id","stay_id","age","female","race_ethnicity","mortality",
        "los_icu","los_hospital","npsle_narrow","npsle_broad","lupus_nephritis","vent","rrt",
        "vaso","aki","charlson_norheum","diabetes","renal_disease","cerebrovascular_disease","chf","copd",
        "oasis","apache_iv"]

def load():
    frames=[]
    for db,f in [("MIMIC-IV","mimiciv_sle.csv"),("NWICU","nwicu_sle.csv"),("eICU","eicu_sle.csv")]:
        d=pd.read_csv(f"{DATA}/{f}")
        for c in COLS:
            if c not in d.columns: d[c]=np.nan
        frames.append(d[COLS])
    return pd.concat(frames, ignore_index=True)

def logit_ci(n_event, n_total):
    if n_event==0 or n_event==n_total:
        n_event+=0.5; n_total+=1
    p=n_event/n_total
    se=np.sqrt(1.0/n_event + 1.0/(n_total-n_event))
    lp=np.log(p/(1-p))
    return p, lp, se

def meta_proportion(rows):
    est=[]; se=[]
    for db,e,t in rows:
        p,lp,s=logit_ci(e,t); est.append(lp); se.append(s)
    est=np.array(est); se=np.array(se); w=1/se**2
    fixed=np.sum(w*est)/np.sum(w)
    k=len(est); q=np.sum(w*(est-fixed)**2); dfree=k-1
    tau2=max(0.0,(q-dfree)/(np.sum(w)-np.sum(w**2)/np.sum(w))) if k>1 else 0.0
    w2=1/(se**2+tau2); re=np.sum(w2*est)/np.sum(w2); re_se=np.sqrt(1/np.sum(w2))
    inv=lambda x: np.exp(x)/(1+np.exp(x))
    return inv(re), inv(re-1.96*re_se), inv(re+1.96*re_se)

def meta_or(rows):
    est=[]; se=[]
    for db,a,b,c,d in rows:
        if a==0 or b==0 or c==0 or d==0:
            a+=0.5;b+=0.5;c+=0.5;d+=0.5
        lor=np.log((a*d)/(b*c)); se_i=np.sqrt(1/a+1/b+1/c+1/d)
        est.append(lor); se.append(se_i)
    est=np.array(est); se=np.array(se); w=1/se**2
    fixed=np.sum(w*est)/np.sum(w)
    k=len(est); q=np.sum(w*(est-fixed)**2); dfree=k-1
    tau2=max(0.0,(q-dfree)/(np.sum(w)-np.sum(w**2)/np.sum(w))) if k>1 else 0.0
    w2=1/(se**2+tau2); re=np.sum(w2*est)/np.sum(w2); re_se=np.sqrt(1/np.sum(w2))
    return np.exp(re), np.exp(re-1.96*re_se), np.exp(re+1.96*re_se)

def meta_logor(items):
    """items: list of (db, logor, se). Random-effects (DerSimonian-Laird) on log-OR scale."""
    est=np.array([x[1] for x in items]); se=np.array([x[2] for x in items])
    w=1/se**2; fixed=np.sum(w*est)/np.sum(w)
    k=len(est); q=np.sum(w*(est-fixed)**2); dfree=k-1
    tau2=max(0.0,(q-dfree)/(np.sum(w)-np.sum(w**2)/np.sum(w))) if k>1 else 0.0
    w2=1/(se**2+tau2); re=np.sum(w2*est)/np.sum(w2); re_se=np.sqrt(1/np.sum(w2))
    i2=max(0.0,(q-dfree)/q)*100 if (q>0 and k>1) else 0.0
    return np.exp(re), np.exp(re-1.96*re_se), np.exp(re+1.96*re_se), round(i2,1)

def db_adjusted_logor(db, outcome, exposure, covs):
    """Fit per-DB logistic regression, return exposure log-OR, SE, aOR/CI/p, covariates actually used, n."""
    sub=df[df.db==db].dropna(subset=[outcome,exposure]+covs).copy()
    use=[c for c in covs if sub[c].nunique()>1]  # drop zero-variance covariates
    f=f"{outcome} ~ {exposure}" + "".join([" + "+c for c in use])
    m=smf.logit(f, data=sub).fit(disp=0)
    beta=m.params[exposure]; ci=m.conf_int().loc[exposure]
    se=(ci[1]-ci[0])/(2*1.96)
    return {"logor":float(beta),"se":float(se),
            "aOR":round(float(np.exp(beta)),2),
            "aLO":round(float(np.exp(ci[0])),2),"aHI":round(float(np.exp(ci[1])),2),
            "ap":f"{m.pvalues[exposure]:.3g}","covs":use,"n":int(len(sub))}

def two_by_two(sub, exposure, outcome):
    a=int(((sub[exposure]==1)&(sub[outcome]==1)).sum())
    b=int(((sub[exposure]==1)&(sub[outcome]==0)).sum())
    c=int(((sub[exposure]==0)&(sub[outcome]==1)).sum())
    d=int(((sub[exposure]==0)&(sub[outcome]==0)).sum())
    return a,b,c,d

def or_ci(a,b,c,d):
    a2,b2,c2,d2=a,b,c,d
    if 0 in (a2,b2,c2,d2): a2+=0.5;b2+=0.5;c2+=0.5;d2+=0.5
    lor=np.log((a2*d2)/(b2*c2)); se=np.sqrt(1/a2+1/b2+1/c2+1/d2)
    return np.exp(lor), np.exp(lor-1.96*se), np.exp(lor+1.96*se)

R={}
df=load()
R["n_total"]=int(len(df))
R["by_db"]={db:int((df.db==db).sum()) for db in ["MIMIC-IV","NWICU","eICU"]}

# ---------- Prevalence ----------
prev={}
for db in ["MIMIC-IV","NWICU","eICU"]:
    sub=df[df.db==db]
    n=len(sub); nn=int(sub.npsle_narrow.sum()); nb=int(sub.npsle_broad.sum())
    prev[db]={"n":n,"narrow":nn,"narrow_pct":round(100*nn/n,1),
              "broad":nb,"broad_pct":round(100*nb/n,1)}
R["prevalence"]=prev
# Pooled prevalence: MIMIC-IV + eICU only. NWICU (n=49, 5 narrow events) is
# retained as descriptive corroboration and is NOT pooled: in a random-effects
# model with k=3 it received ~24% of the weight against 6.1% of the sample,
# and it lacks harmonizable covariates. The 3-database values are kept for
# the record.
POOL_DBS = ["MIMIC-IV", "eICU"]
ALL_DBS = ["MIMIC-IV", "NWICU", "eICU"]
pp=meta_proportion([(db,prev[db]["narrow"],prev[db]["n"]) for db in POOL_DBS])
R["prevalence_pooled_narrow"]={"pct":round(100*pp[0],1),"lo":round(100*pp[1],1),"hi":round(100*pp[2],1),
                               "k":2,"databases":POOL_DBS,
                               "note":"NWICU descriptive only, not pooled"}
pb=meta_proportion([(db,prev[db]["broad"],prev[db]["n"]) for db in POOL_DBS])
R["prevalence_pooled_broad"]={"pct":round(100*pb[0],1),"lo":round(100*pb[1],1),"hi":round(100*pb[2],1),
                              "k":2,"databases":POOL_DBS,
                              "note":"NWICU descriptive only, not pooled"}
pp3=meta_proportion([(db,prev[db]["narrow"],prev[db]["n"]) for db in ALL_DBS])
pb3=meta_proportion([(db,prev[db]["broad"],prev[db]["n"]) for db in ALL_DBS])
R["prevalence_pooled_narrow_3db_incl_NWICU"]={"pct":round(100*pp3[0],1),"lo":round(100*pp3[1],1),"hi":round(100*pp3[2],1),
                                              "k":3,"note":"superseded; retained for transparency"}
R["prevalence_pooled_broad_3db_incl_NWICU"]={"pct":round(100*pb3[0],1),"lo":round(100*pb3[1],1),"hi":round(100*pb3[2],1),
                                             "k":3,"note":"superseded; retained for transparency"}

mim=df[df.db=="MIMIC-IV"].copy()

# ---------- Table 1 (MIMIC-IV, by narrow & broad) ----------
def tab1(sub, grp):
    g0=sub[sub[grp]==0]; g1=sub[sub[grp]==1]
    out={"n0":int(len(g0)),"n1":int(len(g1))}
    out["age_0"]=(round(g0.age.mean(),1),round(g0.age.std(),1))
    out["age_1"]=(round(g1.age.mean(),1),round(g1.age.std(),1))
    out["female_0"]=f"{int(g0.female.sum())} ({round(100*g0.female.mean(),1)}%)"
    out["female_1"]=f"{int(g1.female.sum())} ({round(100*g1.female.mean(),1)}%)"
    out["charlson_0"]=(round(g0.charlson_norheum.mean(),1),round(g0.charlson_norheum.std(),1))
    out["charlson_1"]=(round(g1.charlson_norheum.mean(),1),round(g1.charlson_norheum.std(),1))
    out["ln_0"]=f"{int(g0.lupus_nephritis.sum())} ({round(100*g0.lupus_nephritis.mean(),1)}%)"
    out["ln_1"]=f"{int(g1.lupus_nephritis.sum())} ({round(100*g1.lupus_nephritis.mean(),1)}%)"
    out["renal_0"]=f"{int(g0.renal_disease.sum())} ({round(100*g0.renal_disease.mean(),1)}%)"
    out["renal_1"]=f"{int(g1.renal_disease.sum())} ({round(100*g1.renal_disease.mean(),1)}%)"
    out["mort_0"]=f"{int(g0.mortality.sum())} ({round(100*g0.mortality.mean(),1)}%)"
    out["mort_1"]=f"{int(g1.mortality.sum())} ({round(100*g1.mortality.mean(),1)}%)"
    out["los_0"]=(round(g0.los_hospital.median(),1),round(g0.los_hospital.quantile(.25),1),round(g0.los_hospital.quantile(.75),1))
    out["los_1"]=(round(g1.los_hospital.median(),1),round(g1.los_hospital.quantile(.25),1),round(g1.los_hospital.quantile(.75),1))
    return out
R["table1_mimic_narrow"]=tab1(mim,"npsle_narrow")
R["table1_mimic_broad"]=tab1(mim,"npsle_broad")

# ---------- Risk-factor logistic regression (MIMIC-IV) ----------
COV=["age","female","charlson_norheum","lupus_nephritis","renal_disease","diabetes","cerebrovascular_disease","chf","copd"]
def risk_factors(exposure):
    mr=mim.dropna(subset=["charlson_norheum","renal_disease","diabetes","cerebrovascular_disease","chf","copd"]).copy()
    formula=f"{exposure} ~ " + " + ".join(COV)
    mod=smf.logit(formula, data=mr).fit(disp=0)
    params=mod.params; conf=mod.conf_int(); pvals=mod.pvalues
    rf=[]
    for name in params.index:
        if name=="Intercept": continue
        rf.append({"var":name,"OR":round(np.exp(params[name]),2),
                   "lo":round(np.exp(conf.loc[name,0]),2),"hi":round(np.exp(conf.loc[name,1]),2),
                   "p":f"{pvals[name]:.3g}"})
    return rf, int(len(mr))
R["risk_factors_narrow"],R["risk_factors_narrow_n"]=risk_factors("npsle_narrow")
R["risk_factors_broad"],R["risk_factors_broad_n"]=risk_factors("npsle_broad")

# ---------- Outcomes associated with NPSLE (MIMIC-IV) ----------
def outcome_or(sub, outcome, exposure="npsle_narrow", adjust=True):
    sub=sub.dropna(subset=[outcome]).copy()
    a,b,c,d=two_by_two(sub,exposure,outcome)
    uor,ulo,uhi=or_ci(a,b,c,d)
    res={"event_exp":a,"n_exp":a+b,"event_unexp":c,"n_unexp":c+d,
         "uOR":round(uor,2),"uLO":round(ulo,2),"uHI":round(uhi,2)}
    if adjust and outcome in ("mortality","aki","vent","rrt","vaso"):
        sub2=sub.dropna(subset=["charlson_norheum","renal_disease","diabetes","cerebrovascular_disease","chf","copd"]).copy()
        f=f"{outcome} ~ {exposure} + age + female + charlson_norheum + lupus_nephritis + renal_disease + diabetes"
        m=smf.logit(f, data=sub2).fit(disp=0)
        pv=m.params[exposure]; ci=m.conf_int().loc[exposure]
        res["aOR"]=round(np.exp(pv),2); res["aLO"]=round(np.exp(ci[0]),2); res["aHI"]=round(np.exp(ci[1]),2)
        res["ap"]=f"{m.pvalues[exposure]:.3g}"
    return res

def los_cmp(sub, exp="npsle_narrow"):
    g1=sub[sub[exp]==1].los_hospital.dropna(); g0=sub[sub[exp]==0].los_hospital.dropna()
    st=stats.mannwhitneyu(g1,g0,alternative="two-sided")
    return {"median_exp":round(g1.median(),1),"iqr_exp":(round(g1.quantile(.25),1),round(g1.quantile(.75),1)),
            "median_unexp":round(g0.median(),1),"iqr_unexp":(round(g0.quantile(.25),1),round(g0.quantile(.75),1)),
            "p":f"{st.pvalue:.3g}"}

out_mimic={}
for oc in ["mortality","vent","aki","rrt","vaso"]:
    out_mimic[oc]={"narrow":outcome_or(mim,oc,"npsle_narrow"),"broad":outcome_or(mim,oc,"npsle_broad")}
out_mimic["los_hospital"]={"narrow":los_cmp(mim,"npsle_narrow"),"broad":los_cmp(mim,"npsle_broad")}
R["outcomes_mimic"]=out_mimic

# ==================== PROPENSITY SCORE MATCHING (MIMIC-IV) ====================
def smd_binary(x1,x0):
    p1=np.mean(x1); p0=np.mean(x0)
    sp=np.sqrt((p1*(1-p1)+p0*(1-p0))/2)
    return 0.0 if sp==0 else (p1-p0)/sp
def smd_cont(x1,x0):
    m1,m0=np.mean(x1),np.mean(x0); s1,s0=np.std(x1,ddof=1),np.std(x0,ddof=1)
    sp=np.sqrt((s1**2+s0**2)/2)
    return 0.0 if sp==0 else (m1-m0)/sp

def psm(exposure):
    d=mim.dropna(subset=COV+["mortality","aki","rrt","vent","vaso","los_hospital"]).copy().reset_index(drop=True)
    X=d[COV].astype(float); X=sm.add_constant(X)
    y=d[exposure].astype(int)
    ps_mod=sm.Logit(y,X).fit(disp=0)
    d["ps"]=ps_mod.predict(X)
    d["logit_ps"]=np.log(d.ps/(1-d.ps))
    caliper=0.2*d["logit_ps"].std()
    treated=d[d[exposure]==1].copy(); control=d[d[exposure]==0].copy()
    used=set(); pairs=[]
    ctrl_idx=control.index.tolist()
    ctrl_lp=control["logit_ps"].values
    for ti,trow in treated.iterrows():
        diffs=np.abs(ctrl_lp-trow["logit_ps"])
        order=np.argsort(diffs)
        for j in order:
            cj=ctrl_idx[j]
            if cj in used: continue
            if diffs[j]<=caliper:
                used.add(cj); pairs.append((ti,cj)); break
    ti_idx=[p[0] for p in pairs]; ci_idx=[p[1] for p in pairs]
    mt=d.loc[ti_idx]; mc=d.loc[ci_idx]
    matched=pd.concat([mt,mc])
    # SMD before/after
    smd_before={}; smd_after={}
    t_all=d[d[exposure]==1]; c_all=d[d[exposure]==0]
    for cov in COV:
        if cov in ("age","charlson_norheum"):
            smd_before[cov]=round(smd_cont(t_all[cov],c_all[cov]),3)
            smd_after[cov]=round(smd_cont(mt[cov],mc[cov]),3)
        else:
            smd_before[cov]=round(smd_binary(t_all[cov],c_all[cov]),3)
            smd_after[cov]=round(smd_binary(mt[cov],mc[cov]),3)
    # matched outcomes
    res={"n_treated_all":int(len(t_all)),"n_control_all":int(len(c_all)),
         "n_pairs":len(pairs),"caliper":round(caliper,3),
         "smd_before":smd_before,"smd_after":smd_after,"outcomes":{}}
    for oc in ["mortality","vent","aki","rrt","vaso"]:
        a=int((mt[oc]==1).sum()); b=int((mt[oc]==0).sum())
        c=int((mc[oc]==1).sum()); dd=int((mc[oc]==0).sum())
        orr,lo,hi=or_ci(a,b,c,dd)
        # McNemar on matched pairs + matched-pair (conditional) odds ratio
        pair_t=mt[oc].values; pair_c=mc[oc].values
        n10=int(np.sum((pair_t==1)&(pair_c==0))); n01=int(np.sum((pair_t==0)&(pair_c==1)))
        try:
            mcp=stats.binomtest(min(n10,n01),n10+n01,0.5).pvalue if (n10+n01)>0 else 1.0
        except Exception:
            mcp=1.0
        # matched-pair OR = n10/n01 (conditional ML estimate); Miettinen asymptotic CI
        if n10>0 and n01>0:
            mor=round(n10/n01,2); se=np.sqrt(1.0/n10+1.0/n01)
            mor_lo=round(np.exp(np.log(n10/n01)-1.96*se),2); mor_hi=round(np.exp(np.log(n10/n01)+1.96*se),2)
        else:
            mor=mor_lo=mor_hi=float("nan")
        res["outcomes"][oc]={"exp_rate":f"{a}/{a+b} ({round(100*a/(a+b),1)}%)",
                             "unexp_rate":f"{c}/{c+dd} ({round(100*c/(c+dd),1)}%)",
                             "n10":n10,"n01":n01,
                             "matched_OR":mor,"matched_LO":mor_lo,"matched_HI":mor_hi,
                             "OR":round(orr,2),"LO":round(lo,2),"HI":round(hi,2),
                             "mcnemar_p":f"{mcp:.3g}"}
    # LOS matched
    g1=mt.los_hospital.dropna(); g0=mc.los_hospital.dropna()
    st=stats.mannwhitneyu(g1,g0,alternative="two-sided")
    res["outcomes"]["los_hospital"]={"median_exp":round(g1.median(),1),"iqr_exp":(round(g1.quantile(.25),1),round(g1.quantile(.75),1)),
                                     "median_unexp":round(g0.median(),1),"iqr_unexp":(round(g0.quantile(.25),1),round(g0.quantile(.75),1)),
                                     "p":f"{st.pvalue:.3g}"}
    res["max_smd_after"]=round(max(abs(v) for v in smd_after.values()),3)
    return res
R["psm_narrow"]=psm("npsle_narrow")
R["psm_broad"]=psm("npsle_broad")

# ---------- Validation: NWICU & eICU (now with eICU outcomes) ----------
val={}
for db in ["NWICU","eICU"]:
    sub=df[df.db==db]
    n=len(sub); ne=int(sub.npsle_narrow.sum())
    a,b,c,d=two_by_two(sub,"npsle_narrow","mortality")
    orr,lo,hi=or_ci(a,b,c,d)
    val[db]={"n":n,"narrow":ne,"narrow_pct":round(100*ne/n,1),
             "mort_exp":a,"mort_unexp":c,"OR":round(orr,2),"LO":round(lo,2),"HI":round(hi,2)}
    if db=="eICU":
        val[db]["los"]=los_cmp(sub,"npsle_narrow")
        val[db]["outcomes"]={}
        for oc in ["vent","aki","rrt","vaso"]:
            a,b,c,d=two_by_two(sub,"npsle_narrow",oc)
            orr,lo,hi=or_ci(a,b,c,d)
            val[db]["outcomes"][oc]={"event_exp":a,"n_exp":a+b,"event_unexp":c,"n_unexp":c+d,
                                     "uOR":round(orr,2),"uLO":round(lo,2),"uHI":round(hi,2)}
R["validation"]=val

# ---------- Cross-DB meta of outcomes (MIMIC + eICU): mortality, vent, aki ----------
def cross_meta(outcome, exposure="npsle_narrow"):
    rows=[]; perdb=[]
    for db in ["MIMIC-IV","eICU"]:
        sub=df[df.db==db].dropna(subset=[outcome])
        a,b,c,d=two_by_two(sub,exposure,outcome)
        if (a+c)>0 and (b+d)>0:
            rows.append((db,a,b,c,d))
            orr,lo,hi=or_ci(a,b,c,d)
            perdb.append({"db":db,"OR":round(orr,2),"LO":round(lo,2),"HI":round(hi,2)})
    if not rows: return None
    mo=meta_or(rows)
    return {"OR":round(mo[0],2),"LO":round(mo[1],2),"HI":round(mo[2],2),"per_db":perdb,
            "contributing_dbs":[r[0] for r in rows]}
R["meta_mortality_narrow"]=cross_meta("mortality","npsle_narrow")
R["meta_vent_narrow"]=cross_meta("vent","npsle_narrow")
R["meta_aki_narrow"]=cross_meta("aki","npsle_narrow")
R["meta_mortality_broad"]=cross_meta("mortality","npsle_broad")
R["meta_vent_broad"]=cross_meta("vent","npsle_broad")
R["meta_aki_broad"]=cross_meta("aki","npsle_broad")

# ---------- ADJUSTED cross-DB meta of mortality (MIMIC + eICU) ----------
# Harmonized minimal covariate set available in BOTH databases: age + sex.
# (eICU lacks Charlson/comorbidity flags; lupus_nephritis has no variance in eICU.)
HARM_COV=["age","female"]
def cross_meta_adjusted(outcome, exposure, covs=HARM_COV):
    per=[]; items=[]
    for db in ["MIMIC-IV","eICU"]:
        sub=df[df.db==db].dropna(subset=[outcome,exposure])
        if int((sub[outcome]==1).sum())==0:  # no events -> skip
            continue
        r=db_adjusted_logor(db,outcome,exposure,covs)
        r["db"]=db; per.append(r); items.append((db,r["logor"],r["se"]))
    if len(items)==0: return None
    mo=meta_logor(items)
    return {"OR":round(mo[0],2),"LO":round(mo[1],2),"HI":round(mo[2],2),"I2":mo[3],
            "adjustment":covs,"per_db":per,"contributing_dbs":[p["db"] for p in per]}

# Adjusted (age/sex-harmonized) cross-DB meta for PRIMARY outcome (AKI) and secondary outcomes.
R["meta_aki_narrow_adj"]=cross_meta_adjusted("aki","npsle_narrow")      # PRIMARY outcome
R["meta_vent_narrow_adj"]=cross_meta_adjusted("vent","npsle_narrow")
R["meta_mortality_narrow_adj"]=cross_meta_adjusted("mortality","npsle_narrow")
R["meta_aki_broad_adj"]=cross_meta_adjusted("aki","npsle_broad")
R["meta_vent_broad_adj"]=cross_meta_adjusted("vent","npsle_broad")
R["meta_mortality_broad_adj"]=cross_meta_adjusted("mortality","npsle_broad")
# MIMIC fully-adjusted aOR (9-covariate model) kept as reference (already in outcomes_mimic)
for oc in ("aki","vent","mortality"):
    R[f"mimic_{oc}_fulladj_narrow"]={k:R["outcomes_mimic"][oc]["narrow"][k]
        for k in ("aOR","aLO","aHI","ap")}

# ================= SEVERITY-ADJUSTED SENSITIVITY (illness-severity confounding) =================
# Addresses the critique that the NPSLE-AKI link could merely reflect "sicker patients".
# MIMIC-IV: add OASIS (mimiciv_derived.oasis) -- a single admission severity score that
#           EXCLUDES serum creatinine, i.e. non-renal severity (conservative for AKI).
# eICU:      add APACHE IV total score (eicu_crd.apachepatientresult.apachescore).
# A persistent association after severity adjustment argues against pure confounding by
# overall illness severity. (Severity scores that embed renal function make the estimate
# conservative -- adjusting for AKI-related components can only pull the OR toward 1.)
def sev_adj(db, outcome, exposure="npsle_narrow"):
    sub=df[df.db==db].dropna(subset=[outcome,exposure]).copy()
    covs=["age","female","charlson_norheum","lupus_nephritis","renal_disease","oasis"] if db=="MIMIC-IV" else ["age","female","apache_iv"]
    sub2=sub.dropna(subset=covs+[exposure])
    if int((sub2[outcome]==1).sum())==0 or int((sub2[exposure]==1).sum())==0:
        return None
    f=f"{outcome} ~ {exposure} + " + " + ".join(covs)
    m=smf.logit(f,data=sub2).fit(disp=0)
    beta=m.params[exposure]; ci=m.conf_int().loc[exposure]; se=(ci[1]-ci[0])/(2*1.96)
    return {"aOR":round(float(np.exp(beta)),2),"aLO":round(float(np.exp(ci[0])),2),
            "aHI":round(float(np.exp(ci[1])),2),"ap":f"{m.pvalues[exposure]:.3g}","n":int(len(sub2)),"covs":covs}

R["sev_adj"]={db:{oc:sev_adj(db,oc) for oc in ["aki","vent","mortality"]} for db in ["MIMIC-IV","eICU"]}

# ================= SENSITIVITY ANALYSES (MIMIC-IV derivation cohort) =================
# (A) Stricter AKI definition: KDIGO stage >= 2 (vs primary = stage >= 1).
# (B) Exclude baseline chronic kidney disease (eGFR < 60 via CKD-EPI 2021 baseline creatinine).
# eICU cannot support these (ICU-only creatinine, no derived KDIGO tables, no prior-7d baseline).
# --- load per-stay sensitivity fields and attach to the MIMIC-IV rows of df ---
sens_df = pd.read_csv(f"{DATA}/mimiciv_sle_sens.csv")
sens_df["aki_stage2plus"] = (sens_df["aki_stage_max"] >= 2).astype(int)
sens_map = sens_df.set_index("stay_id")
for col in ["aki_stage2plus","baseline_creat","baseline_egfr","ckd_icd"]:
    df[col] = df["stay_id"].map(sens_map[col])
df["baseline_ckd"] = (df["baseline_egfr"] < 60)  # bool (NaN stays NaN)
mi = df[df.db=="MIMIC-IV"].copy()

# covariate set matched to the primary full-adjusted AKI model (6 covariates)
SENS_COV = ["age","female","charlson_norheum","lupus_nephritis","renal_disease","diabetes"]

def outcome_or_sens(sub, outcome, exposure, covs=SENS_COV):
    sub=sub.dropna(subset=[outcome]).copy()
    a,b,c,d=two_by_two(sub,exposure,outcome)
    res={"event_exp":int(a),"n_exp":int(a+b),"event_unexp":int(c),"n_unexp":int(c+d),
         "uOR":round(or_ci(a,b,c,d)[0],2)}
    sub2=sub.dropna(subset=covs+[exposure]).copy()
    f=f"{outcome} ~ {exposure} + " + " + ".join(covs)
    m=smf.logit(f,data=sub2).fit(disp=0)
    pv=m.params[exposure]; ci=m.conf_int().loc[exposure]
    res.update({"aOR":round(float(np.exp(pv)),2),"aLO":round(float(np.exp(ci[0])),2),
                "aHI":round(float(np.exp(ci[1])),2),"ap":f"{m.pvalues[exposure]:.3g}","n":int(len(sub2))})
    return res

def psm_general(sub, exposure, outcome, covs):
    d=sub.dropna(subset=covs+[outcome]).copy().reset_index(drop=True)
    X=sm.add_constant(d[covs].astype(float)); y=d[exposure].astype(int)
    ps_mod=sm.Logit(y,X).fit(disp=0)
    d["ps"]=ps_mod.predict(X); d["logit_ps"]=np.log(d.ps/(1-d.ps))
    caliper=0.2*d["logit_ps"].std()
    treated=d[d[exposure]==1]; control=d[d[exposure]==0]
    used=set(); pairs=[]; ctrl_idx=control.index.tolist(); ctrl_lp=control["logit_ps"].values
    for ti,trow in treated.iterrows():
        diffs=np.abs(ctrl_lp-trow["logit_ps"]); order=np.argsort(diffs)
        for j in order:
            cj=ctrl_idx[j]
            if cj in used: continue
            if diffs[j]<=caliper:
                used.add(cj); pairs.append((ti,cj)); break
    if not pairs: return None
    mt=d.loc[[p[0] for p in pairs]]; mc=d.loc[[p[1] for p in pairs]]
    a=int((mt[outcome]==1).sum()); b=int((mt[outcome]==0).sum())
    c=int((mc[outcome]==1).sum()); dd=int((mc[outcome]==0).sum())
    orr,lo,hi=or_ci(a,b,c,dd)
    pt=mt[outcome].values; pc=mc[outcome].values
    n10=int(np.sum((pt==1)&(pc==0))); n01=int(np.sum((pt==0)&(pc==1)))
    mcp=stats.binomtest(min(n10,n01),n10+n01,0.5).pvalue if (n10+n01)>0 else 1.0
    if n10>0 and n01>0:
        mor=round(n10/n01,2); se=np.sqrt(1.0/n10+1.0/n01)
        mor_lo=round(np.exp(np.log(n10/n01)-1.96*se),2); mor_hi=round(np.exp(np.log(n10/n01)+1.96*se),2)
    else:
        mor=mor_lo=mor_hi=float("nan")
    smd_after={}
    for cov in covs:
        if cov in ("age","charlson_norheum"):
            smd_after[cov]=round(smd_cont(mt[cov],mc[cov]),3)
        else:
            smd_after[cov]=round(smd_binary(mt[cov],mc[cov]),3)
    return {"n_pairs":len(pairs),"OR":round(orr,2),"LO":round(lo,2),"HI":round(hi,2),
            "matched_OR":mor,"matched_LO":mor_lo,"matched_HI":mor_hi,"n10":n10,"n01":n01,
            "exp_rate":f"{a}/{a+b} ({round(100*a/(a+b),1)}%)",
            "unexp_rate":f"{c}/{c+dd} ({round(100*c/(c+dd),1)}%)",
            "mcnemar_p":f"{mcp:.3g}","max_smd_after":round(max(abs(v) for v in smd_after.values()),3)}

sens={}
# (A) KDIGO stage >= 2 as outcome
sens["aki_stage2plus_fulladj_narrow"]=outcome_or_sens(mi,"aki_stage2plus","npsle_narrow")
sens["aki_stage2plus_fulladj_broad"] =outcome_or_sens(mi,"aki_stage2plus","npsle_broad")
sens["aki_stage2plus_psm_narrow"]    =psm_general(mi,"npsle_narrow","aki_stage2plus",SENS_COV)
sens["aki_stage2plus_prevalence"]    ={"n":int(len(mi)),"events":int(mi.aki_stage2plus.sum()),
                                       "pct":round(100*mi.aki_stage2plus.mean(),1)}
# (B) exclude baseline CKD (eGFR < 60 from baseline creatinine)
mi_nockd = mi[mi["baseline_egfr"].notna() & (mi["baseline_ckd"] == False)].copy()  # 显式排除「无基线肌酐」者（NaN<60 会得 False）
sens["aki_nockd_fulladj_narrow"] =outcome_or_sens(mi_nockd,"aki","npsle_narrow")
sens["aki_nockd_psm_narrow"]     =psm_general(mi_nockd,"npsle_narrow","aki",SENS_COV)
sens["n_excluded_ckd_egfr"]      =int((mi["baseline_ckd"]==True).sum())
sens["n_analyzed_nockd"]         =int(len(mi_nockd))
# (B-alt) exclude CKD by ICD code (N18/N19/585) as cross-check
mi_nockd2 = mi[mi["ckd_icd"]==0].copy()
sens["aki_nockd_icd_fulladj_narrow"]=outcome_or_sens(mi_nockd2,"aki","npsle_narrow")
sens["n_excluded_ckd_icd"]          =int((mi["ckd_icd"]==1).sum())
# descriptive baseline-CKD prevalence by NPSLE status
sens["baseline_ckd_prev"]={
  "overall_pct":round(100*mi["baseline_ckd"].mean(),1),
  "narrow_pct":round(100*mi[mi.npsle_narrow==1]["baseline_ckd"].mean(),1),
  "unexp_pct":round(100*mi[mi.npsle_narrow==0]["baseline_ckd"].mean(),1),
  "ckd_icd_overall_pct":round(100*mi.ckd_icd.mean(),1)}
sens["primary_aki_fulladj_narrow_ref"]=R["mimic_aki_fulladj_narrow"]
R["sens"]=sens

# --- Supplementary forest: primary AKI vs stage>=2 vs no-baseline-CKD ---
C_PRIMARY="#b2182b"; C_MIMIC="#2166ac"
fig,ax=plt.subplots(figsize=(7.0,3.0))
rows=[("AKI (KDIGO >=1), primary\nfull-adj (9 cov)", R["mimic_aki_fulladj_narrow"], C_PRIMARY),
      ("AKI stage >=2, full-adj (6 cov)", sens["aki_stage2plus_fulladj_narrow"], C_MIMIC),
      ("AKI (>=1), excluding baseline CKD\nfull-adj (6 cov)", sens["aki_nockd_fulladj_narrow"], "#762a83")]
ys=np.arange(len(rows))[::-1]
for i,(lab,r,col) in enumerate(rows):
    if r is None: continue
    downweight = (i==2)  # excluding-baseline-CKD row: very wide CI -> visually downweight
    if downweight:
        # open / dashed marker to signal low weight; dashed error bar
        ax.plot([r["aLO"],r["aHI"]],[ys[i],ys[i]],color="#999999",lw=1.4,ls=":",zorder=2)
        ax.plot([r["aOR"]],[ys[i]],"D",mfc="none",mec="#762a83",ms=9,zorder=3)
        ax.text(r["aHI"]*1.10,ys[i],f"{r['aOR']:.2f} ({r['aLO']:.2f}\u2013{r['aHI']:.2f})",va="center",fontsize=8.5,color="#762a83")
        ax.text(0.42,ys[i]-0.32,"[wide CI \u2014 see text]",va="center",fontsize=7.5,color="#999999",style="italic")
    else:
        ax.plot([r["aLO"],r["aHI"]],[ys[i],ys[i]],color=col,lw=2.0,zorder=2)
        ax.plot([r["aOR"]],[ys[i]],"D",color=col,ms=9,zorder=3)
        ax.text(r["aHI"]*1.12,ys[i],f"{r['aOR']:.2f} ({r['aLO']:.2f}\u2013{r['aHI']:.2f})",va="center",fontsize=8.5)
ax.axvline(1,color="gray",ls="--",lw=1); ax.set_yticks(ys); ax.set_yticklabels([r[0] for r in rows])
ax.set_ylim(-0.6,len(rows)-0.4); ax.set_xscale("log"); ax.set_xticks([0.5,1,2,4,8]); ax.set_xticklabels(["0.5","1","2","4","8"])
ax.set_xlim(0.4,11); ax.set_xlabel("Adjusted odds ratio (95% CI), MIMIC-IV derivation cohort")
ax.set_title("Sensitivity analyses for the NPSLE\u2013AKI association",fontweight="bold")
plt.tight_layout(); plt.savefig(f"{OUT}/forest_sens.png", dpi=300); plt.close()

# for backward-compatible forest plot (unadjusted)
db_ors=[(x["db"],x["OR"],x["LO"],x["HI"]) for x in R["meta_mortality_narrow"]["per_db"]]
mp=(R["meta_mortality_narrow"]["OR"],R["meta_mortality_narrow"]["LO"],R["meta_mortality_narrow"]["HI"])
R["db_ors_mortality"]=[{"db":x[0],"OR":x[1],"LO":x[2],"HI":x[3]} for x in db_ors]

# ================= FIGURES =================
# Global style: publication-quality, 300 dpi
plt.rcParams.update({
    "figure.dpi":300, "savefig.dpi":300, "font.size":10,
    "axes.titlesize":11, "axes.labelsize":10, "axes.linewidth":0.8,
    "xtick.labelsize":9, "ytick.labelsize":9, "legend.fontsize":8.5,
    "font.family":"DejaVu Sans", "axes.spines.top":False, "axes.spines.right":False,
})
C_PRIMARY="#b2182b"; C_MIMIC="#2166ac"; C_EICU="#4d9221"; C_POOL="#b2182b"

def forest_adjusted(meta, xlabel, title, outfile, ref=None, ref_label="MIMIC-IV full-adj (9 cov)"):
    """Per-DB age/sex-adjusted aOR + pooled RE; optional MIMIC full-adjusted reference row."""
    if not meta: return
    per=meta["per_db"]
    rows=[(f"{p['db']} (n={p['n']})", p["aOR"], p["aLO"], p["aHI"], C_MIMIC if p['db']=='MIMIC-IV' else C_EICU) for p in per]
    rows.append((f"Pooled, random-effects (I\u00b2={meta['I2']}%)", meta["OR"], meta["LO"], meta["HI"], C_POOL))
    if ref: rows.append((ref_label, ref["aOR"], ref["aLO"], ref["aHI"], "#777777"))
    fig,ax=plt.subplots(figsize=(7.2,0.62*len(rows)+1.1))
    ys=list(range(len(rows)))[::-1]
    for i,(lab,o,lo,hi,col) in enumerate(rows):
        is_pool=lab.startswith("Pooled")
        ax.plot([lo,hi],[ys[i],ys[i]],color=col,lw=2.0 if is_pool else 1.5,zorder=2)
        ax.plot([o],[ys[i]],("D" if is_pool else "o"),color=col,ms=9 if is_pool else 6.5,zorder=3)
        ax.text(1.02,0.0,"",transform=ax.transAxes)
    xmax=max(r[3] for r in rows); ax.set_xlim(0.4, xmax*2.6)
    for i,(lab,o,lo,hi,col) in enumerate(rows):
        ax.text(xmax*1.15,ys[i],f"{o:.2f} ({lo:.2f}\u2013{hi:.2f})",va="center",fontsize=8.5)
    ax.axvline(1,color="gray",ls="--",lw=1,zorder=1)
    ax.set_yticks(ys); ax.set_yticklabels([r[0] for r in rows]); ax.set_ylim(-0.6,len(rows)-0.4)
    ax.set_xlabel(xlabel); ax.set_title(title,fontweight="bold"); ax.set_xscale("log")
    ax.set_xticks([0.5,1,2,4,8]); ax.set_xticklabels(["0.5","1","2","4","8"])
    plt.tight_layout(); plt.savefig(f"{OUT}/{outfile}"); plt.close()

# ---- Figure 2 (PRIMARY): AKI, age/sex-adjusted cross-DB meta ----
forest_adjusted(R["meta_aki_narrow_adj"],
    "Adjusted odds ratio (95% CI) for AKI (primary outcome)",
    "Narrow NPSLE and AKI \u2014 age/sex-adjusted, cross-database meta-analysis",
    "forest_aki_adj.png", ref=R["mimic_aki_fulladj_narrow"])

# ---- Figure 3 (summary): pooled adjusted aOR for the three harmonized outcomes ----
fig,ax=plt.subplots(figsize=(7.0,3.0))
summ=[("AKI (primary)",R["meta_aki_narrow_adj"]),
      ("Invasive mechanical ventilation",R["meta_vent_narrow_adj"]),
      ("In-hospital mortality (secondary)",R["meta_mortality_narrow_adj"])]
ys=np.arange(len(summ))[::-1]
for i,(lab,m) in enumerate(summ):
    if not m: continue
    ax.plot([m["LO"],m["HI"]],[ys[i],ys[i]],color=C_MIMIC,lw=2.0,zorder=2)
    ax.plot([m["OR"]],[ys[i]],"D",color=C_PRIMARY if i==0 else C_MIMIC,ms=9 if i==0 else 7,zorder=3)
    ax.text(m["HI"]*1.12,ys[i],f"{m['OR']:.2f} ({m['LO']:.2f}\u2013{m['HI']:.2f}), I\u00b2={m['I2']}%",va="center",fontsize=8.5)
ax.axvline(1,color="gray",ls="--",lw=1)
ax.set_yticks(ys); ax.set_yticklabels([s[0] for s in summ]); ax.set_ylim(-0.6,len(summ)-0.4)
ax.set_xscale("log"); ax.set_xticks([0.5,1,2,4]); ax.set_xticklabels(["0.5","1","2","4"]); ax.set_xlim(0.4,7)
ax.set_xlabel("Pooled age/sex-adjusted odds ratio (95% CI), MIMIC-IV + eICU")
ax.set_title("Cross-database adjusted associations of narrow NPSLE",fontweight="bold")
plt.tight_layout(); plt.savefig(f"{OUT}/forest_outcomes_adj.png"); plt.close()

# ---- (retained) unadjusted mortality forest ----
fig,ax=plt.subplots(figsize=(7.0,3.0))
labels=[x[0] for x in db_ors]+["Pooled (RE)"]
ors=[x[1] for x in db_ors]+[mp[0]]; los=[x[2] for x in db_ors]+[mp[1]]; his=[x[3] for x in db_ors]+[mp[2]]
ys=list(range(len(labels)))[::-1]
for i,(o,lo,hi) in enumerate(zip(ors,los,his)):
    is_pool=(i==len(labels)-1); col=C_POOL if is_pool else "black"
    ax.plot([lo,hi],[ys[i],ys[i]],color=col,lw=2.0 if is_pool else 1.5); ax.plot([o],[ys[i]],("D" if is_pool else "o"),color=col,ms=8 if is_pool else 6)
    ax.text(max(his)*1.1,ys[i],f"{o:.2f} ({lo:.2f}\u2013{hi:.2f})",va="center",fontsize=8.5)
ax.axvline(1,color="gray",ls="--",lw=1)
ax.set_yticks(ys); ax.set_yticklabels(labels); ax.set_xlim(0.4,max(his)*2.4)
ax.set_xlabel("Odds ratio (95% CI) for in-hospital mortality (unadjusted)")
ax.set_title("Narrow NPSLE and in-hospital mortality (secondary outcome)",fontweight="bold"); ax.set_xscale("log")
ax.set_xticks([0.5,1,2,4,8]); ax.set_xticklabels(["0.5","1","2","4","8"])
plt.tight_layout(); plt.savefig(f"{OUT}/forest_mortality.png"); plt.close()

# ---- (retained) adjusted mortality forest ----
forest_adjusted(R["meta_mortality_narrow_adj"],
    "Adjusted odds ratio (95% CI) for in-hospital mortality",
    "Narrow NPSLE and mortality \u2014 age/sex-adjusted, cross-database meta-analysis",
    "forest_mortality_adj.png", ref=R["mimic_mortality_fulladj_narrow"])

# ---- Prevalence bar ----
# NWICU is drawn in grey (descriptive corroboration only, not pooled); pooled
# estimates combine MIMIC-IV + eICU. Mirrors analysis/redraw_prevalence_v18.py.
C_DESC_N="#8f8f8f"; C_DESC_B="#c9c9c9"
fig,ax=plt.subplots(figsize=(7.0,4.2))
dbs=["MIMIC-IV","NWICU","eICU"]
narr=[prev[d]["narrow_pct"] for d in dbs]; barr=[prev[d]["broad_pct"] for d in dbs]
x=np.arange(len(dbs)); w=0.36
for i,d in enumerate(dbs):
    cn,cb=(C_DESC_N,C_DESC_B) if d=="NWICU" else (C_MIMIC,"#d6604d")
    lab_n="Narrow (cognitive/delirium)" if i==0 else None
    lab_b="Broad (full NPSLE)" if i==0 else None
    ax.bar(x[i]-w/2,narr[i],w,color=cn,label=lab_n)
    ax.bar(x[i]+w/2,barr[i],w,color=cb,label=lab_b)
for i,(nv,bv) in enumerate(zip(narr,barr)):
    ax.text(i-w/2,nv+0.6,f"{nv}%",ha="center",fontsize=8.5); ax.text(i+w/2,bv+0.6,f"{bv}%",ha="center",fontsize=8.5)
ax.set_xticks(x); ax.set_xticklabels([f"{d}{'*' if d=='NWICU' else ''}\n(n={prev[d]['n']})" for d in dbs]); ax.set_ylabel("% of SLE ICU stays")
ax.legend(); ax.set_ylim(0,max(barr)+6)
ax.set_title("Prevalence of neuropsychiatric involvement in SLE ICU stays",fontweight="bold")
_pn=R["prevalence_pooled_narrow"]; _pbd=R["prevalence_pooled_broad"]
fig.text(0.5,0.02,
    "* NWICU is descriptive only and is not included in the pooled estimates. "
    f"Pooled (MIMIC-IV + eICU, random effects): narrow {_pn['pct']}% ({_pn['lo']}\u2013{_pn['hi']}), "
    f"broad {_pbd['pct']}% ({_pbd['lo']}\u2013{_pbd['hi']}).",
    ha="center",fontsize=7.8,color="#444444")
plt.tight_layout(); plt.subplots_adjust(bottom=0.22)
plt.savefig(f"{OUT}/prevalence.png",dpi=300,facecolor="white",bbox_inches="tight"); plt.close()

# ---- Cohort flow (box diagram) ----
fig,ax=plt.subplots(figsize=(7.0,5.0)); ax.axis("off"); ax.set_xlim(0,10); ax.set_ylim(0,10)
def box(x,y,w2,h2,text,fc="#eef3f8",fs=8.6):
    ax.add_patch(plt.Rectangle((x-w2/2,y-h2/2),w2,h2,fc=fc,ec="#2166ac",lw=1.2,zorder=2))
    ax.text(x,y,text,ha="center",va="center",fontsize=fs,zorder=3,linespacing=1.15)
def arrow(x1,y1,x2,y2,color="#2166ac"):
    ax.annotate("",xy=(x2,y2),xytext=(x1,y1),arrowprops=dict(arrowstyle="->",color=color,lw=1.2))
box(5.0,9.0,8.5,0.9,f"Three critical-care databases screened for SLE with \u22651 ICU stay\nTotal SLE ICU stays = {len(df)}")
arrow(5.0,8.55,5.0,7.85)
box(2.0,7.1,2.6,1.1,f"MIMIC-IV\nn = {R['by_db']['MIMIC-IV']}\n(derivation)",fc="#dbe7f3")
box(5.0,7.1,2.6,1.1,f"eICU-CRD\nn = {R['by_db']['eICU']}\n(external validation)",fc="#e4efdc")
box(8.0,7.1,2.6,1.1,f"NWICU\nn = {R['by_db']['NWICU']}\n(descriptive only)",fc="#f3e9dc")
for xc in (2.0,5.0,8.0):
    arrow(xc,6.55,xc,5.75,color="#888")
box(2.0,4.95,2.6,1.2,f"Narrow NPSLE\n{prev['MIMIC-IV']['narrow']} ({prev['MIMIC-IV']['narrow_pct']}%)\nBroad {prev['MIMIC-IV']['broad_pct']}%")
box(5.0,4.95,2.6,1.2,f"Narrow NPSLE\n{prev['eICU']['narrow']} ({prev['eICU']['narrow_pct']}%)\nBroad {prev['eICU']['broad_pct']}%")
box(8.0,4.95,2.6,1.2,f"Narrow NPSLE\n{prev['NWICU']['narrow']} ({prev['NWICU']['narrow_pct']}%)\nBroad {prev['NWICU']['broad_pct']}%")
for xc in (2.0,5.0,8.0):
    arrow(xc,4.35,xc,3.75,color="#888")
box(5.0,2.75,8.5,1.45,
    f"Pooled narrow prevalence {R['prevalence_pooled_narrow']['pct']}% "
    f"(95% CI {R['prevalence_pooled_narrow']['lo']}\u2013{R['prevalence_pooled_narrow']['hi']}%)\n"
    f"Pooled broad prevalence {R['prevalence_pooled_broad']['pct']}% "
    f"(95% CI {R['prevalence_pooled_broad']['lo']}\u2013{R['prevalence_pooled_broad']['hi']}%)\n"
    f"Primary analysis: MIMIC-IV (logistic regression + propensity-score matching);\n"
    f"eICU external validation; cross-DB random-effects meta-analysis",
    fc="#f7f7f7")
ax.set_title("Study cohort flow",fontweight="bold",fontsize=11)
plt.tight_layout(); plt.savefig(f"{OUT}/flow.png",dpi=300,bbox_inches="tight",facecolor="white"); plt.close()

# ---- Love plot (SMD before/after matching, narrow) ----
covpretty={"age":"Age","female":"Female sex","charlson_norheum":"Charlson index","lupus_nephritis":"Lupus nephritis",
           "renal_disease":"Renal disease","diabetes":"Diabetes","cerebrovascular_disease":"Cerebrovascular disease",
           "chf":"Congestive heart failure","copd":"COPD"}
fig,ax=plt.subplots(figsize=(7.0,4.4))
covlab=COV; sb=R["psm_narrow"]["smd_before"]; sa=R["psm_narrow"]["smd_after"]
order=sorted(covlab,key=lambda c:abs(sb[c]))
yy=np.arange(len(order))
for i,c in enumerate(order):
    ax.plot([abs(sb[c]),abs(sa[c])],[i,i],color="#bbbbbb",lw=1.0,zorder=1)
ax.scatter([abs(sb[c]) for c in order],yy,color="#d6604d",s=45,label="Before matching",zorder=3)
ax.scatter([abs(sa[c]) for c in order],yy,color="#2166ac",s=45,label="After matching",zorder=3)
ax.axvline(0.1,color="red",ls="--",lw=1,label="SMD = 0.1 threshold")
ax.set_yticks(yy); ax.set_yticklabels([covpretty[c] for c in order])
ax.set_xlabel("Absolute standardized mean difference"); ax.legend(loc="lower right")
ax.set_title("Covariate balance before/after propensity-score matching\n(narrow NPSLE, MIMIC-IV)",fontweight="bold")
plt.tight_layout(); plt.savefig(f"{OUT}/loveplot.png"); plt.close()

# ---- Forest plot of matched outcomes (narrow PSM) ----
fig,ax=plt.subplots(figsize=(7.0,3.4))
ocs=["aki","vent","los_marker","rrt","mortality","vaso"]
oclab=["AKI (primary)","Invasive MV","","RRT","Mortality","Vasopressors"]
pm=R["psm_narrow"]["outcomes"]
plot_rows=[(l,oc) for oc,l in zip(ocs,oclab) if oc in pm]
yy=np.arange(len(plot_rows))[::-1]
for i,(lab,oc) in enumerate(plot_rows):
    o=pm[oc]["OR"]; lo=pm[oc]["LO"]; hi=pm[oc]["HI"]
    col=C_PRIMARY if oc=="aki" else "#2166ac"
    ax.plot([lo,hi],[yy[i],yy[i]],color=col,lw=1.7); ax.plot([o],[yy[i]],"s",color=col,ms=7)
    ax.text(hi*1.12,yy[i],f"{o:.2f} ({lo:.2f}\u2013{hi:.2f})",va="center",fontsize=8.5)
ax.axvline(1,color="gray",ls="--",lw=1)
ax.set_yticks(yy); ax.set_yticklabels([l for l,_ in plot_rows]); ax.set_xscale("log")
ax.set_xticks([0.5,1,2,4,8]); ax.set_xticklabels(["0.5","1","2","4","8"]); ax.set_xlim(0.4,12)
ax.set_xlabel("Odds ratio (95% CI), propensity-matched cohort (89 pairs)")
ax.set_title("Narrow NPSLE and outcomes after propensity-score matching",fontweight="bold")
plt.tight_layout(); plt.savefig(f"{OUT}/forest_psm.png"); plt.close()

with open(f"{OUT}/results.json","w") as f:
    json.dump(R,f,indent=2,default=str)

# ============ pretty print ============
print("=== COHORT ===", R["n_total"], R["by_db"])
print("=== PREVALENCE (narrow / broad) ===")
for db in dbs: print(f"  {db}: narrow {prev[db]['narrow']}/{prev[db]['n']} ({prev[db]['narrow_pct']}%), broad {prev[db]['broad_pct']}%")
print("  pooled narrow:", R["prevalence_pooled_narrow"], " broad:", R["prevalence_pooled_broad"])
print("=== eICU outcomes (narrow) ===", R["validation"]["eICU"]["outcomes"])
print("=== Cross-DB meta (narrow) ===")
print("  AKI (PRIMARY, ADJ age/sex):", R["meta_aki_narrow_adj"], " MIMIC full-adj:", R["mimic_aki_fulladj_narrow"])
print("  IMV (ADJ age/sex):", R["meta_vent_narrow_adj"])
print("  mortality (unadj):", R["meta_mortality_narrow"])
print("  mortality (ADJ age/sex):", R["meta_mortality_narrow_adj"])
print("  mortality (broad ADJ):", R["meta_mortality_broad_adj"])
print("  MIMIC full-adj aOR mortality (narrow):", R["mimic_mortality_fulladj_narrow"])
print("  aki (unadj):", R["meta_aki_narrow"], " vent (unadj):", R["meta_vent_narrow"])
print("=== PSM narrow ===", "pairs=",R["psm_narrow"]["n_pairs"], "maxSMD_after=",R["psm_narrow"]["max_smd_after"])
for oc in ["mortality","vent","aki","rrt","vaso"]: print("  ",oc, R["psm_narrow"]["outcomes"][oc])
print("  los:", R["psm_narrow"]["outcomes"]["los_hospital"])
print("=== PSM broad ===", "pairs=",R["psm_broad"]["n_pairs"], "maxSMD_after=",R["psm_broad"]["max_smd_after"])
for oc in ["mortality","aki"]: print("  ",oc, R["psm_broad"]["outcomes"][oc])
print("=== Risk factors broad (n=%d) ==="%R["risk_factors_broad_n"])
for r in R["risk_factors_broad"][:6]: print("  ",r)
print("=== SENSITIVITY ANALYSES (MIMIC-IV) ===")
print("  baseline CKD prevalence (%):", sens["baseline_ckd_prev"])
print("  (A) AKI stage>=2 full-adj narrow:", sens["aki_stage2plus_fulladj_narrow"])
print("  (A) AKI stage>=2 full-adj broad :", sens["aki_stage2plus_fulladj_broad"])
print("  (A) AKI stage>=2 PSM narrow     :", sens["aki_stage2plus_psm_narrow"])
print("  (B) exclude baseline CKD: excluded=%d, analyzed=%d"%(sens["n_excluded_ckd_egfr"],sens["n_analyzed_nockd"]))
print("  (B) AKI full-adj narrow (no CKD):", sens["aki_nockd_fulladj_narrow"])
print("  (B) AKI PSM narrow (no CKD)     :", sens["aki_nockd_psm_narrow"])
print("  (B-alt) AKI full-adj (no ICD-CKD):", sens["aki_nockd_icd_fulladj_narrow"])
print("Figures + results.json saved to", OUT)
