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
4. **QA / figures** — `analysis/check_missing.py` (missingness check),
   `analysis/redraw_flow.py` (cohort-flow figure), and
   `analysis/redraw_prevalence_v18.py` (prevalence panel, can be regenerated
   on its own from an existing `results.json`).

## Environment & paths

- Python 3.13 with: `pandas`, `numpy`, `scipy`, `statsmodels`, `matplotlib`.
- A local PostgreSQL 18 instance with the PhysioNet databases loaded is
  required for `extract_severity.py` (uses `psql` via subprocess).
- The scripts read paths from environment variables:

  | Variable           | Purpose                        |
  |--------------------|--------------------------------|
  | `SLECI_DATA`       | cohort CSV directory           |
  | `SLECI_OUT`        | analysis output directory      |
  | `SLECI_BASE`       | project root (figures)         |
  | `SLECI_PGPASSWORD` | PostgreSQL password for `psql` |

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
