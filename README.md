# NPSLE-AKI Analysis

**Reproducible analysis code** for the study:
*Cognitive/Neuropsychiatric Involvement and Acute Kidney Injury in Critically Ill Patients With Systemic Lupus Erythematosus: A Derivation and External Validation Study.*

This repository contains **only the analysis code**. No patient-level data are
included; all cohorts are derived from PhysioNet-credentialed databases
(MIMIC-IV, eICU-CRD, NWICU) under their respective Data Use Agreements (DUA).
Researchers with PhysioNet credentials can reconstruct the cohorts by running
the SQL extraction scripts against their local database copies.

## Pipeline

1. **Cohort extraction** — `sql/extract_*.sql` build the SLE ICU cohorts
   (exposure = coded cognitive/neuropsychiatric involvement; outcomes = AKI,
   invasive mechanical ventilation, RRT, in-hospital mortality, LOS,
   vasopressors) in MIMIC-IV (derivation), eICU-CRD (external validation),
   and NWICU (descriptive corroboration).
2. **Severity scores** — `analysis/extract_severity.py` adds OASIS (MIMIC-IV)
   and APACHE IV (eICU) illness-severity scores to the cohort CSVs.
3. **Main analysis** — `analysis/analyze.py` runs multivariable logistic
   regression, 1:1 propensity-score matching, severity-adjusted sensitivity,
   and the exploratory cross-database random-effects pooled estimate, writing
   `analysis/results.json`.
4. **QA / figures (primary)** — `analysis/check_missing.py` (missingness check),
   `analysis/redraw_flow.py` (Figure 1, Panel A), `analysis/rebuild_figure1_v31.py`
   (Figure 1, Panel B, and the composite), `analysis/make_figures_v10.py` and
   `analysis/make_figure3_temporal.py` (Figures 2 and 3), and
   `analysis/redraw_forest_psm_v33.py`, `analysis/redraw_forest_sens_v33.py`,
   `analysis/redraw_forest_mortality_v33.py` (Figure 4 and Supplementary
   Figures S2–S3).
5. **Measured delirium and outcome windows** — `analysis/delirium_vs_icd.py`
   (CAM-ICU agreement with the coded phenotype), `analysis/v28_onset_table.py`
   (recorded sequence of delirium and AKI), and
   `analysis/eicu_kdigo_windows.py` (time-windowed KDIGO rule in eICU).
6. **Time-varying Cox models** — `analysis/v27_export_cox.py` exports the
   counting-process (start, stop] datasets, `analysis/v27_tvc.py` fits them in
   Python, and `analysis/v27_r_cox.R`, `analysis/v28_r_cox.R` and
   `analysis/v29_r_check.R` refit them in R (`survival::coxph`, Efron ties,
   patient-clustered variance), including the `cox.zph` checks and the log-time
   interaction specifications. The R scripts read the exported tables from
   `SLECI_EXPORT_DIR` (default `_cpexp3`).
7. **Phenotype and consistency checks** — `analysis/aki_timing.py`,
   `analysis/aki_confounders.py`, `analysis/v30_eicu_patterns.py` (which
   free-text patterns the eICU definition actually matched),
   `analysis/v31_encephalitis_only.py` (reclassification of the two
   encephalitis-only stays) and `analysis/v31_pooled.py`
   (DerSimonian–Laird pooling).

## Environment & paths

- Python 3.13 with: `pandas`, `numpy`, `scipy`, `statsmodels`, `matplotlib`,
  `psycopg2`; plus R 4.6 with `survival` 3.8 for stage 6.
- A local PostgreSQL 18 instance with the PhysioNet databases loaded is
  required for `extract_severity.py` (uses `psql` via subprocess) and for the
  DB-based scripts in stages 5–7.
- The scripts read paths from environment variables:

  | Variable           | Purpose                          |
  |--------------------|----------------------------------|
  | `SLECI_DATA`       | cohort CSV directory             |
  | `SLECI_OUT`        | analysis output directory        |
  | `SLECI_BASE`       | project root (figures)           |
  | `SLECI_EXPORT_DIR` | counting-process exports (R)     |
  | `SLECI_PGPASSWORD` | PostgreSQL password              |

  Set these before running, e.g.:
  `export SLECI_DATA=/path/to/data SLECI_PGPASSWORD=****`

  If unset, the scripts fall back to paths relative to the working directory
  (`data`, `analysis`, and `.` respectively), so a clone with cohort CSVs
  placed in `./data` runs as-is.

## Role of NWICU in pooled estimates

NWICU (n = 49, including only 5 stays with narrow involvement) is retained as
**descriptive corroboration only** and is **not entered into any pooled
estimate** — for prevalence or for odds ratios. Two reasons: it lacks
harmonizable covariates, and in a random-effects model with few studies it
would have received roughly 24% of the pooling weight against 6% of the
sample. All pooled estimates in the manuscript therefore combine
**MIMIC-IV + eICU** (k = 2).

`analyze.py` records this explicitly in `results.json`:
`prevalence_pooled_narrow` / `prevalence_pooled_broad` are the k = 2 estimates
used in the paper, while `prevalence_pooled_narrow_3db_incl_NWICU` /
`prevalence_pooled_broad_3db_incl_NWICU` keep the earlier three-database
values for transparency.

## Data availability

Cohorts are reconstructed from MIMIC-IV, eICU-CRD, and NWICU, accessible to
certified researchers through PhysioNet (https://physionet.org). Analysis code
and extracted cohort schemas are available in this repository; individual
patient data are not published, per the PhysioNet DUA.

## License

MIT — see [LICENSE](LICENSE).

## Citation

Wang K, et al. *Cognitive/Neuropsychiatric Involvement and Acute Kidney Injury
in Critically Ill Patients With Systemic Lupus Erythematosus: A Derivation and
External Validation Study.* (submitted). Code: https://github.com/morrosun/npsle-aki-analysis
(archived at Zenodo: https://doi.org/10.5281/zenodo.22066849).
