DROP TABLE IF EXISTS sle_out;
CREATE TEMP TABLE sle_out AS
WITH sleadm AS (
  SELECT DISTINCT subject_id, hadm_id FROM hosp.diagnoses_icd
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
  FROM hosp.diagnoses_icd
  WHERE (subject_id, hadm_id) IN (SELECT subject_id, hadm_id FROM sleadm)
  GROUP BY subject_id, hadm_id
),
comorb AS (
  SELECT subject_id, hadm_id,
    BOOL_OR(icd_code LIKE '250%') AS diabetes,
    BOOL_OR(icd_code LIKE '585%' OR icd_code LIKE '586%' OR icd_code='5839') AS renal_disease,
    BOOL_OR(icd_code LIKE '430%' OR icd_code LIKE '431%' OR icd_code LIKE '432%' OR icd_code LIKE '433%' OR icd_code LIKE '434%' OR icd_code LIKE '435%' OR icd_code LIKE '436%' OR icd_code LIKE '437%' OR icd_code LIKE '438%' OR icd_code LIKE '3623' OR icd_code LIKE 'G45%' OR icd_code LIKE 'I6%') AS cerebrovascular_disease,
    BOOL_OR(icd_code LIKE '428%' OR icd_code='39891' OR icd_code LIKE '402%1' OR icd_code LIKE '404%1' OR icd_code LIKE '425%') AS chf,
    BOOL_OR(icd_code LIKE '490%' OR icd_code LIKE '491%' OR icd_code LIKE '492%' OR icd_code LIKE '493%' OR icd_code LIKE '494%' OR icd_code LIKE '495%' OR icd_code LIKE '496%' OR icd_code LIKE 'J4%') AS copd
  FROM hosp.diagnoses_icd
  WHERE (subject_id, hadm_id) IN (SELECT subject_id, hadm_id FROM sleadm)
  GROUP BY subject_id, hadm_id
)
SELECT
  'NWICU' AS db,
  a.subject_id, a.hadm_id, s.stay_id,
  p.anchor_age AS age,
  CASE WHEN p.gender='F' THEN 1 ELSE 0 END AS female,
  a.race AS race_ethnicity,
  a.hospital_expire_flag AS mortality,
  s.los AS los_icu,
  EXTRACT(EPOCH FROM (a.dischtime - a.admittime))/86400.0 AS los_hospital,
  CASE WHEN dx.np_narrow THEN 1 ELSE 0 END AS npsle_narrow,
  CASE WHEN dx.np_broad THEN 1 ELSE 0 END AS npsle_broad,
  CASE WHEN dx.lupus_nephritis THEN 1 ELSE 0 END AS lupus_nephritis,
  NULL AS vent, NULL AS rrt, NULL AS vaso, NULL AS aki,
  NULL AS charlson_norheum,
  CASE WHEN comorb.diabetes THEN 1 ELSE 0 END AS diabetes,
  CASE WHEN comorb.renal_disease THEN 1 ELSE 0 END AS renal_disease,
  CASE WHEN comorb.cerebrovascular_disease THEN 1 ELSE 0 END AS cerebrovascular_disease,
  CASE WHEN comorb.chf THEN 1 ELSE 0 END AS chf,
  CASE WHEN comorb.copd THEN 1 ELSE 0 END AS copd
FROM hosp.admissions a
JOIN sleadm ON sleadm.subject_id=a.subject_id AND sleadm.hadm_id=a.hadm_id
JOIN hosp.patients p ON p.subject_id=a.subject_id
JOIN icu.icustays s ON s.hadm_id=a.hadm_id
LEFT JOIN dx ON dx.subject_id=a.subject_id AND dx.hadm_id=a.hadm_id
LEFT JOIN comorb ON comorb.subject_id=a.subject_id AND comorb.hadm_id=a.hadm_id
WHERE s.stay_id = (SELECT MIN(stay_id) FROM icu.icustays s2 WHERE s2.hadm_id=a.hadm_id);
\copy sle_out TO 'nwicu_sle.csv' WITH CSV HEADER;
