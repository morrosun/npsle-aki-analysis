-- eICU SLE cohort extraction with derived outcomes (MV / AKI / RRT / vasopressors)
DROP TABLE IF EXISTS sle_out;
CREATE TEMP TABLE sle_out AS
WITH sle AS (
  SELECT DISTINCT patientunitstayid FROM eicu_crd.diagnosis
  WHERE diagnosisstring='general|immunological diseases|systemic lupus erythematosus'
),
dx AS (
  SELECT patientunitstayid,
    BOOL_OR(diagnosisstring ILIKE ANY(ARRAY['%delirium%','%encephal%','%dementia%','%cognitive%','%amnesia%','%psychosis%','%hallucin%'])) AS np_narrow,
    BOOL_OR(diagnosisstring ILIKE ANY(ARRAY['%delirium%','%encephal%','%dementia%','%cognitive%','%amnesia%','%psychosis%','%hallucin%','%seizure%','%convuls%','%stroke%','%cerebrovascular%','%cerebr%','%mening%','%demyelinating%','%multiple sclerosis%'])) AS np_broad
  FROM eicu_crd.diagnosis
  WHERE patientunitstayid IN (SELECT patientunitstayid FROM sle)
  GROUP BY patientunitstayid
),
-- Invasive mechanical ventilation: treatment (mechanical ventilation) OR respiratorycare vent start OR apache intubated/vent
mv_treat AS (
  SELECT DISTINCT patientunitstayid FROM eicu_crd.treatment
  WHERE patientunitstayid IN (SELECT patientunitstayid FROM sle)
    AND treatmentstring LIKE 'pulmonary|ventilation and oxygenation|mechanical ventilation%'
),
mv_resp AS (
  SELECT DISTINCT patientunitstayid FROM eicu_crd.respiratorycare
  WHERE patientunitstayid IN (SELECT patientunitstayid FROM sle)
    AND ventstartoffset IS NOT NULL AND ventstartoffset <> 0
),
mv_apache AS (
  SELECT DISTINCT patientunitstayid FROM eicu_crd.apacheapsvar
  WHERE patientunitstayid IN (SELECT patientunitstayid FROM sle)
    AND (intubated=1 OR vent=1)
),
-- Renal replacement therapy: treatment dialysis OR apache dialysis
rrt_treat AS (
  SELECT DISTINCT patientunitstayid FROM eicu_crd.treatment
  WHERE patientunitstayid IN (SELECT patientunitstayid FROM sle)
    AND treatmentstring LIKE 'renal|dialysis%'
),
rrt_apache AS (
  SELECT DISTINCT patientunitstayid FROM eicu_crd.apacheapsvar
  WHERE patientunitstayid IN (SELECT patientunitstayid FROM sle) AND dialysis=1
),
-- Vasopressors: treatment vasopressors OR infusiondrug vasoactive agents
vaso_treat AS (
  SELECT DISTINCT patientunitstayid FROM eicu_crd.treatment
  WHERE patientunitstayid IN (SELECT patientunitstayid FROM sle)
    AND treatmentstring LIKE 'cardiovascular|shock|vasopressors%'
),
vaso_inf AS (
  SELECT DISTINCT patientunitstayid FROM eicu_crd.infusiondrug
  WHERE patientunitstayid IN (SELECT patientunitstayid FROM sle)
    AND (drugname ILIKE '%norepinephrine%' OR drugname ILIKE '%epinephrine%'
         OR drugname ILIKE '%vasopressin%' OR drugname ILIKE '%phenylephrine%'
         OR drugname ILIKE '%dopamine%' OR drugname ILIKE '%dobutamine%'
         OR drugname ILIKE '%milrinone%' OR drugname ILIKE '%neosynephrine%')
),
-- AKI: KDIGO creatinine criterion (peak >= 1.5x baseline OR rise >= 0.3 mg/dL) OR dialysis
creat AS (
  SELECT patientunitstayid,
    MIN(labresult) AS creat_min,
    MAX(labresult) AS creat_max
  FROM eicu_crd.lab
  WHERE labname='creatinine' AND labresult IS NOT NULL AND labresult > 0 AND labresult < 30
    AND patientunitstayid IN (SELECT patientunitstayid FROM sle)
  GROUP BY patientunitstayid
),
aki AS (
  SELECT c.patientunitstayid,
    CASE WHEN c.creat_max >= 1.5*c.creat_min OR (c.creat_max - c.creat_min) >= 0.3 THEN 1 ELSE 0 END AS aki_creat
  FROM creat c
)
SELECT
  'eICU' AS db,
  p.uniquepid AS subject_id, NULL AS hadm_id, p.patientunitstayid AS stay_id,
  CASE WHEN p.age='> 89' THEN 90 ELSE p.age::numeric END AS age,
  CASE WHEN p.gender='Female' THEN 1 ELSE 0 END AS female,
  p.ethnicity AS race_ethnicity,
  CASE WHEN p.hospitaldischargestatus='Expired' THEN 1 ELSE 0 END AS mortality,
  NULL AS los_icu,
  (p.hospitaldischargeoffset - p.hospitaladmitoffset)/1440.0 AS los_hospital,
  CASE WHEN dx.np_narrow THEN 1 ELSE 0 END AS npsle_narrow,
  CASE WHEN dx.np_broad THEN 1 ELSE 0 END AS npsle_broad,
  0 AS lupus_nephritis,
  CASE WHEN mt.patientunitstayid IS NOT NULL OR mr.patientunitstayid IS NOT NULL OR ma.patientunitstayid IS NOT NULL THEN 1 ELSE 0 END AS vent,
  CASE WHEN rt.patientunitstayid IS NOT NULL OR ra.patientunitstayid IS NOT NULL THEN 1 ELSE 0 END AS rrt,
  CASE WHEN vt.patientunitstayid IS NOT NULL OR vi.patientunitstayid IS NOT NULL THEN 1 ELSE 0 END AS vaso,
  ak.aki_creat AS aki,
  NULL AS charlson_norheum,
  NULL AS diabetes, NULL AS renal_disease, NULL AS cerebrovascular_disease, NULL AS chf, NULL AS copd
FROM eicu_crd.patient p
JOIN sle ON sle.patientunitstayid=p.patientunitstayid
LEFT JOIN dx ON dx.patientunitstayid=p.patientunitstayid
LEFT JOIN mv_treat mt ON mt.patientunitstayid=p.patientunitstayid
LEFT JOIN mv_resp mr ON mr.patientunitstayid=p.patientunitstayid
LEFT JOIN mv_apache ma ON ma.patientunitstayid=p.patientunitstayid
LEFT JOIN rrt_treat rt ON rt.patientunitstayid=p.patientunitstayid
LEFT JOIN rrt_apache ra ON ra.patientunitstayid=p.patientunitstayid
LEFT JOIN vaso_treat vt ON vt.patientunitstayid=p.patientunitstayid
LEFT JOIN vaso_inf vi ON vi.patientunitstayid=p.patientunitstayid
LEFT JOIN aki ak ON ak.patientunitstayid=p.patientunitstayid;
\copy sle_out TO 'eicu_sle.csv' WITH CSV HEADER;
