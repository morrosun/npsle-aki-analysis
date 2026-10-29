# -*- coding: utf-8 -*-
"""V30 audit: what do the eICU LIKE patterns actually capture?

Writes analysis/v30_eicu_patterns_report.txt (incremental append).
"""
import os
import psycopg2

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                   "v30_eicu_patterns_report.txt")
if os.path.exists(OUT):
    os.remove(OUT)
f = open(OUT, "w", encoding="utf-8")


def log(*a):
    s = " ".join(str(x) for x in a)
    f.write(s + "\n")
    f.flush()
    print(s)


def q(cur, sql, args=None):
    cur.execute(sql, args or ())
    return cur.fetchall()


cn = psycopg2.connect(host="localhost", port=5432, user="postgres",
                      password=os.environ.get("SLECI_PGPASSWORD", ""),
                      dbname="eicu")
cur = cn.cursor()

NAR = ['%delirium%', '%encephal%', '%dementia%', '%cognitive%',
       '%amnesia%', '%psychosis%', '%hallucin%']
EXTRA = ['%seizure%', '%convuls%', '%stroke%', '%cerebrovascular%',
         '%cerebr%', '%mening%', '%demyelinating%', '%multiple sclerosis%']

SLE = "SELECT DISTINCT patientunitstayid FROM eicu_crd.diagnosis " \
      "WHERE diagnosisstring='general|immunological diseases|systemic lupus erythematosus'"

q(cur, "SELECT count(*) FROM (%s) t" % SLE)
n_sle = q(cur, "SELECT count(*) FROM (%s) t" % SLE)[0][0]
log("== eICU SLE stays:", n_sle)

# exposure recomputed exactly as in extract_eicu.sql
cur.execute("""
CREATE TEMP TABLE dx AS
SELECT patientunitstayid, diagnosisstring FROM eicu_crd.diagnosis
WHERE patientunitstayid IN (%s)
""" % SLE)
q(cur, "SELECT count(*) FROM dx")
log("diagnosis rows:", q(cur, "SELECT count(*) FROM dx")[0][0])

cur.execute("""
CREATE TEMP TABLE ex AS
SELECT patientunitstayid,
  bool_or(diagnosisstring ILIKE ANY(%s)) AS nar,
  bool_or(diagnosisstring ILIKE ANY(%s)) AS brd
FROM dx GROUP BY patientunitstayid
""", (NAR, NAR + EXTRA))

n_nar, n_brd, n_both = q(cur, """
SELECT count(*) FILTER (WHERE nar), count(*) FILTER (WHERE brd),
       count(*) FILTER (WHERE nar AND brd) FROM ex""")[0]
log("narrow =", n_nar, " broad =", n_brd, " narrow&broad =", n_both)
log("broad-only (broad but not narrow) =", n_brd - n_both)

# ---- which pattern matched, per narrow stay -------------------------------
log("\n== narrow stays: matched patterns and raw strings ==")
rows = q(cur, """
SELECT patientunitstayid,
       array_agg(DISTINCT diagnosisstring) FILTER (
         WHERE diagnosisstring ILIKE ANY(%s)) AS hit
FROM dx
WHERE patientunitstayid IN (SELECT patientunitstayid FROM ex WHERE nar)
GROUP BY patientunitstayid
ORDER BY patientunitstayid
""", (NAR,))
for sid, hit in rows:
    pats = []
    for p in NAR:
        stem = p.strip("%")
        if any(stem in (h or "").lower() for h in hit):
            pats.append(stem)
    log(sid, "| patterns:", ",".join(pats), "| strings:", " ;; ".join(sorted(hit)))

# ---- how many narrow stays are matched ONLY by %encephal% ----------------
log("\n== narrow stays whose only narrow pattern is 'encephal' ==")
cur.execute("""
CREATE TEMP TABLE pat_per_stay AS
SELECT patientunitstayid,
  max(CASE WHEN diagnosisstring ILIKE '%%delirium%%' THEN 1 ELSE 0 END) d_del,
  max(CASE WHEN diagnosisstring ILIKE '%%encephal%%' THEN 1 ELSE 0 END) d_enc,
  max(CASE WHEN diagnosisstring ILIKE '%%dementia%%' THEN 1 ELSE 0 END) d_dem,
  max(CASE WHEN diagnosisstring ILIKE '%%cognitive%%' THEN 1 ELSE 0 END) d_cog,
  max(CASE WHEN diagnosisstring ILIKE '%%amnesia%%' THEN 1 ELSE 0 END) d_amn,
  max(CASE WHEN diagnosisstring ILIKE '%%psychosis%%' THEN 1 ELSE 0 END) d_psy,
  max(CASE WHEN diagnosisstring ILIKE '%%hallucin%%' THEN 1 ELSE 0 END) d_hal
FROM dx GROUP BY patientunitstayid
""")
rows = q(cur, """
SELECT patientunitstayid, d_del, d_enc, d_dem, d_cog, d_amn, d_psy, d_hal
FROM pat_per_stay WHERE patientunitstayid IN
  (SELECT patientunitstayid FROM ex WHERE nar) ORDER BY patientunitstayid""")
only_enc = [r[0] for r in rows if r[2] == 1 and sum(r[1:]) == 1]
only_dem = [r[0] for r in rows if r[3] == 1 and sum(r[1:]) == 1]
log("only-encephal stays:", len(only_enc), only_enc)
log("only-dementia stays:", len(only_dem), only_dem)
log("exclusive pattern counts:",
    "del", sum(1 for r in rows if r[1] and sum(r[1:]) == 1),
    "enc", len(only_enc),
    "dem", len(only_dem),
    "cog", sum(1 for r in rows if r[4] and sum(r[1:]) == 1),
    "amn", sum(1 for r in rows if r[5] and sum(r[1:]) == 1),
    "psy", sum(1 for r in rows if r[6] and sum(r[1:]) == 1),
    "hal", sum(1 for r in rows if r[7] and sum(r[1:]) == 1))

log("\n== every matched string among narrow stays (exact text) ==")
rows = q(cur, """
SELECT diagnosisstring, count(DISTINCT patientunitstayid)
FROM dx WHERE patientunitstayid IN (SELECT patientunitstayid FROM ex WHERE nar)
  AND diagnosisstring ILIKE ANY(%s)
GROUP BY diagnosisstring ORDER BY 2 DESC, 1
""", (NAR,))
for s, n in rows:
    log("  %3d  %s" % (n, s))

log("\n== does any matched string contain letters outside 'encephalopathy'? ==")
rows = q(cur, """
SELECT DISTINCT diagnosisstring FROM dx
WHERE patientunitstayid IN (SELECT patientunitstayid FROM ex WHERE nar)
  AND diagnosisstring ILIKE '%%encephal%%'""")
for (s,) in rows:
    log("  ", s)

# ---- broad increment from %cerebr% / %mening% ----------------------------
log("\n== broad stays added ONLY by 'cerebr'/'mening' ==")
cur.execute("""
CREATE TEMP TABLE inc AS
SELECT patientunitstayid, diagnosisstring FROM dx
WHERE patientunitstayid IN (SELECT patientunitstayid FROM ex WHERE brd AND NOT nar)
""")
rows = q(cur, """
SELECT diagnosisstring, count(DISTINCT patientunitstayid)
FROM inc GROUP BY diagnosisstring ORDER BY 2 DESC, 1""")
for s, n in rows:
    log("  %3d  %s" % (n, s))

log("\n== which extra stem accounts for each broad-only stay ==")
cur.execute("""
CREATE TEMP TABLE po AS
SELECT patientunitstayid,
  max(CASE WHEN diagnosisstring ILIKE '%%seizure%%' THEN 1 ELSE 0 END) e_sei,
  max(CASE WHEN diagnosisstring ILIKE '%%convuls%%' THEN 1 ELSE 0 END) e_con,
  max(CASE WHEN diagnosisstring ILIKE '%%stroke%%' THEN 1 ELSE 0 END) e_str,
  max(CASE WHEN diagnosisstring ILIKE '%%cerebrovascular%%' THEN 1 ELSE 0 END) e_cv,
  max(CASE WHEN diagnosisstring ILIKE '%%cerebr%%' THEN 1 ELSE 0 END) e_cer,
  max(CASE WHEN diagnosisstring ILIKE '%%mening%%' THEN 1 ELSE 0 END) e_men,
  max(CASE WHEN diagnosisstring ILIKE '%%demyelinating%%' THEN 1 ELSE 0 END) e_dem,
  max(CASE WHEN diagnosisstring ILIKE '%%multiple sclerosis%%' THEN 1 ELSE 0 END) e_ms
FROM inc GROUP BY patientunitstayid
""")
rows = q(cur, "SELECT * FROM po ORDER BY patientunitstayid")
log("broad-only stays:", len(rows))
for r in rows:
    flags = [k for k, v in zip(["seizure", "convuls", "stroke", "cerebrovascular",
                                "cerebr", "mening", "demyelinating", "MS"], r[1:]) if v]
    log("  ", r[0], flags)

log("\n== exclusive contributor among broad-only stays (only one stem) ==")
cnt = {}
for r in rows:
    flags = tuple(k for k, v in zip(["seizure", "convuls", "stroke", "cerebrovascular",
                                     "cerebr", "mening", "demyelinating", "MS"], r[1:]) if v)
    cnt[flags] = cnt.get(flags, 0) + 1
for k, v in sorted(cnt.items(), key=lambda x: -x[1]):
    log("  ", v, k)

log("\n== broad-only stays matched ONLY by cerebr and/or mening ==")
solo = [r for r in rows if r[1] == 0 and r[2] == 0 and r[3] == 0 and r[4] == 0
        and (r[5] == 1 or r[6] == 1)]
log("count:", len(solo), [r[0] for r in solo])
for r in solo:
    ids = r[0]
    ss = q(cur, "SELECT DISTINCT diagnosisstring FROM inc WHERE patientunitstayid=%s", (ids,))
    log("   ", ids, "->", " ;; ".join(x[0] for x in ss))

f.close()
cn.close()
print("done")
