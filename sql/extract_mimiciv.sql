DROP TABLE IF EXISTS sle_out;
CREATE TEMP TABLE sle_out AS
WITH sleadm AS (
  SELECT DISTINCT subject_id, hadm_id
  FROM mimiciv_hosp.diagnoses_icd
  WHERE (icd_version=9 AND icd_code='7100') OR (icd_version=10 AND icd_code LIKE 'M32%')
),
dx AS (
  SELECT subject_id, hadm_id,
    BOOL_OR(
      (icd_version=10 AND (icd_code LIKE 'F05%' OR icd_code LIKE 'R41%' OR icd_code LIKE 'F06%' OR icd_code LIKE 'G93%' OR icd_code LIKE 'F01%' OR icd_code LIKE 'F02%' OR icd_code LIKE 'F03%' OR icd_code LIKE 'G30%' OR icd_code LIKE 'G31%'))
      OR (icd_version=9 AND (icd_code LIKE '293%' OR icd_code LIKE '294%' OR icd_code='3483' OR icd_code LIKE '310%' OR icd_code LIKE '7800%'))
    ) AS np_narrow,
    BOOL_OR(
      (icd_version=10 AND (icd_code LIKE 'F05%' OR icd_code LIKE 'R41%' OR icd_code LIKE 'F06%' OR icd_code LIKE 'G93%' OR icd_code LIKE 'F01%' OR icd_code LIKE 'F02%' OR icd_code LIKE 'F03%' OR icd_code LIKE 'G30%' OR icd_code LIKE 'G31%' OR icd_code LIKE 'G40%' OR icd_code LIKE 'R56%' OR icd_code LIKE 'F23%' OR icd_code LIKE 'F29%' OR icd_code LIKE 'I6%' OR icd_code LIKE 'G45%' OR icd_code LIKE 'G35%' OR icd_code LIKE 'G36%' OR icd_code LIKE 'G37%'))
      OR (icd_version=9 AND (icd_code LIKE '293%' OR icd_code LIKE '294%' OR icd_code='3483' OR icd_code LIKE '310%' OR icd_code LIKE '7800%' OR icd_code LIKE '345%' OR icd_code LIKE '7803%' OR icd_code LIKE '298%' OR icd_code LIKE '437%' OR icd_code LIKE '3623' OR icd_code LIKE '340%'))
    ) AS np_broad,
    BOOL_OR( (icd_version=10 AND icd_code='M3214') OR (icd_version=9 AND icd_code='5838') ) AS lupus_nephritis
  FROM mimiciv_hosp.diagnoses_icd
  WHERE (subject_id, hadm_id) IN (SELECT subject_id, hadm_id FROM sleadm)
  GROUP BY subject_id, hadm_id
),
vent AS ( SELECT stay_id, BOOL_OR(ventilation_status IN ('InvasiveVent','Tracheostomy')) AS vent FROM mimiciv_derived.ventilation GROUP BY stay_id ),
rrt AS ( SELECT stay_id, BOOL_OR(dialysis_present=1) AS rrt FROM mimiciv_derived.rrt GROUP BY stay_id ),
vaso AS ( SELECT stay_id, BOOL_OR(COALESCE(dopamine,0)>0 OR COALESCE(epinephrine,0)>0 OR COALESCE(norepinephrine,0)>0 OR COALESCE(phenylephrine,0)>0 OR COALESCE(vasopressin,0)>0 OR COALESCE(dobutamine,0)>0 OR COALESCE(milrinone,0)>0) AS vaso FROM mimiciv_derived.vasoactive_agent GROUP BY stay_id ),
aki AS ( SELECT stay_id, MAX(aki_stage) AS aki_stage_max FROM mimiciv_derived.kdigo_stages GROUP BY stay_id )
SELECT
  'MIMIC-IV' AS db,
  d.subject_id, d.hadm_id, d.stay_id,
  d.admission_age AS age,
  CASE WHEN d.gender='F' THEN 1 ELSE 0 END AS female,
  d.race AS race_ethnicity,
  d.hospital_expire_flag AS mortality,
  d.los_icu, d.los_hospital,
  CASE WHEN dx.np_narrow THEN 1 ELSE 0 END AS npsle_narrow,
  CASE WHEN dx.np_broad THEN 1 ELSE 0 END AS npsle_broad,
  CASE WHEN dx.lupus_nephritis THEN 1 ELSE 0 END AS lupus_nephritis,
  CASE WHEN vent.vent THEN 1 ELSE 0 END AS vent,
  CASE WHEN rrt.rrt THEN 1 ELSE 0 END AS rrt,
  CASE WHEN vaso.vaso THEN 1 ELSE 0 END AS vaso,
  CASE WHEN aki.aki_stage_max >= 1 THEN 1 ELSE 0 END AS aki,
  (c.charlson_comorbidity_index - COALESCE(c.rheumatic_disease,0)) AS charlson_norheum,
  CASE WHEN (c.diabetes_without_cc=1 OR c.diabetes_with_cc=1) THEN 1 ELSE 0 END AS diabetes,
  c.renal_disease, c.cerebrovascular_disease, c.congestive_heart_failure AS chf, c.chronic_pulmonary_disease AS copd
FROM mimiciv_derived.icustay_detail d
JOIN sleadm s ON s.subject_id=d.subject_id AND s.hadm_id=d.hadm_id
LEFT JOIN dx ON dx.subject_id=d.subject_id AND dx.hadm_id=d.hadm_id
LEFT JOIN mimiciv_derived.charlson c ON c.hadm_id=d.hadm_id
LEFT JOIN vent ON vent.stay_id=d.stay_id
LEFT JOIN rrt ON rrt.stay_id=d.stay_id
LEFT JOIN vaso ON vaso.stay_id=d.stay_id
LEFT JOIN aki ON aki.stay_id=d.stay_id
WHERE d.first_icu_stay = true;
\copy sle_out TO 'mimiciv_sle.csv' WITH CSV HEADER;
