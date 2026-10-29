-- Sensitivity-analysis supplement for the MIMIC-IV SLE ICU cohort.
-- Adds per-stay: (1) max KDIGO stage during ICU stay (for stage>=2 outcome),
--                (2) baseline creatinine (KDIGO baseline = lowest in prior 7 days)
--                    -> baseline eGFR via CKD-EPI 2021 (no race coefficient)
--                    -> baseline CKD = eGFR < 60 mL/min/1.73m^2
--                (3) chronic kidney disease ICD flag (N18/N19, 585) as cross-check.
DROP TABLE IF EXISTS sle_sens;
CREATE TEMP TABLE sle_sens AS
WITH sleadm AS (
  SELECT DISTINCT subject_id, hadm_id
  FROM mimiciv_hosp.diagnoses_icd
  WHERE (icd_version=9 AND icd_code='7100') OR (icd_version=10 AND icd_code LIKE 'M32%')
),
stays AS (
  SELECT d.subject_id, d.hadm_id, d.stay_id, d.admission_age AS age,
         CASE WHEN d.gender='F' THEN 1 ELSE 0 END AS female
  FROM mimiciv_derived.icustay_detail d
  JOIN sleadm s ON s.subject_id=d.subject_id AND s.hadm_id=d.hadm_id
  WHERE d.first_icu_stay = true
),
akistage AS (
  SELECT k.stay_id,
         MAX(k.aki_stage) AS aki_stage_max,
         -- true baseline = lowest ACTUAL creatinine within first 48h of ICU
         -- (pre-progression baseline; avoids capturing a later AKI peak)
         MIN(CASE WHEN k.charttime <= (d.icu_intime + interval '48 hours')
                  THEN k.creat END)                      AS base_creat_48h,
         MIN(k.creat_low_past_7day)  AS base_creat_7,
         MIN(k.creat_low_past_48hr)  AS base_creat_48,
         MIN(k.creat)                AS creat_min
  FROM mimiciv_derived.kdigo_stages k
  JOIN mimiciv_derived.icustay_detail d ON d.stay_id = k.stay_id
  WHERE k.stay_id IN (SELECT stay_id FROM stays)
  GROUP BY k.stay_id
),
ckd_icd AS (
  SELECT DISTINCT subject_id, hadm_id
  FROM mimiciv_hosp.diagnoses_icd
  WHERE (icd_version=10 AND (icd_code LIKE 'N18%' OR icd_code LIKE 'N19%'))
     OR (icd_version=9  AND icd_code LIKE '585%')
)
SELECT s.stay_id,
       COALESCE(a.aki_stage_max,0)                                  AS aki_stage_max,
       COALESCE(a.base_creat_48h, a.base_creat_7, a.base_creat_48, a.creat_min) AS baseline_creat,
       -- eGFR uses the SAME baseline creatinine as baseline_creat (48-h window,
       -- no lower bound). Earlier versions used MIN(creat_low_past_7day) here,
       -- which is the lowest creatinine anywhere in the stay, not the early value.
       CASE WHEN COALESCE(a.base_creat_48h, a.base_creat_7, a.base_creat_48, a.creat_min) IS NULL THEN NULL
            ELSE
              142.0
              * POWER( LEAST( COALESCE(a.base_creat_48h, a.base_creat_7, a.base_creat_48, a.creat_min)
                              / (CASE WHEN s.female=1 THEN 0.7 ELSE 0.9 END), 1.0),
                       CASE WHEN s.female=1 THEN -0.241 ELSE -0.302 END )
              * POWER( GREATEST( COALESCE(a.base_creat_48h, a.base_creat_7, a.base_creat_48, a.creat_min)
                              / (CASE WHEN s.female=1 THEN 0.7 ELSE 0.9 END), 1.0),
                       -1.200 )
              * POWER( 0.9938, s.age )
              * (CASE WHEN s.female=1 THEN 1.012 ELSE 1.0 END)
       END                                                          AS baseline_egfr,
       CASE WHEN ci.subject_id IS NOT NULL THEN 1 ELSE 0 END        AS ckd_icd
FROM stays s
LEFT JOIN akistage a ON a.stay_id = s.stay_id
LEFT JOIN ckd_icd ci  ON ci.subject_id = s.subject_id AND ci.hadm_id = s.hadm_id;
\copy sle_sens TO 'mimiciv_sle_sens.csv' WITH CSV HEADER;
