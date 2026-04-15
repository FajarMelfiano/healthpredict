-- ==========================================================
-- athena_queries.sql
-- HealthPredict AI — Amazon Athena Exploration Queries
-- Workgroup: healthpredict-workgroup
-- Database:  healthpredict_health_db
-- ==========================================================
-- Run these queries in the AWS Athena Console Query Editor.
-- Select workgroup: healthpredict-workgroup
-- Select database:  healthpredict_health_db
-- ==========================================================


-- ── QUERY 1: Record count and column completeness ─────────
-- Run this first to confirm Glue ETL completed successfully.
-- Expected: ~768 rows, zero NULLs in key columns.
SELECT
    COUNT(*)                   AS total_records,
    COUNT(glucose)             AS glucose_not_null,
    COUNT(bmi)                 AS bmi_not_null,
    COUNT(blood_pressure)      AS bp_not_null,
    COUNT(outcome)             AS outcome_not_null,
    SUM(CASE WHEN outcome = 1 THEN 1 ELSE 0 END) AS positive_cases,
    SUM(CASE WHEN outcome = 0 THEN 1 ELSE 0 END) AS negative_cases
FROM healthpredict_health_db.processed_patient_data;


-- ── QUERY 2: Class distribution ──────────────────────────
-- Verify class imbalance — expected ~65% negative, ~35% positive.
SELECT
    outcome,
    COUNT(*) AS count,
    ROUND(COUNT(*) * 100.0 /
          SUM(COUNT(*)) OVER(), 1) AS percentage
FROM healthpredict_health_db.processed_patient_data
GROUP BY outcome
ORDER BY outcome;


-- ── QUERY 3: Feature statistics by class ─────────────────
-- Compare average feature values between diabetic and non-diabetic groups.
-- Useful for understanding which features differ most between classes.
SELECT
    outcome,
    ROUND(AVG(glucose), 2)           AS avg_glucose,
    ROUND(AVG(bmi), 2)               AS avg_bmi,
    ROUND(AVG(age), 1)               AS avg_age,
    ROUND(AVG(insulin), 2)           AS avg_insulin,
    ROUND(AVG(blood_pressure), 1)    AS avg_blood_pressure,
    ROUND(AVG(diabetes_pedigree), 3) AS avg_pedigree,
    ROUND(AVG(pregnancies), 1)       AS avg_pregnancies
FROM healthpredict_health_db.processed_patient_data
GROUP BY outcome
ORDER BY outcome;


-- ── QUERY 4: BMI category distribution ───────────────────
-- Verify that the Glue ETL engineered features were created correctly.
-- Expected categories: Underweight, Normal, Overweight, Obese
SELECT
    bmi_category,
    COUNT(*)              AS count,
    ROUND(AVG(bmi), 2)    AS avg_bmi,
    ROUND(AVG(CAST(outcome AS DOUBLE)), 3) AS diabetes_rate
FROM healthpredict_health_db.processed_patient_data
WHERE bmi_category IS NOT NULL
GROUP BY bmi_category
ORDER BY count DESC;


-- ── QUERY 5: Age group vs diabetes rate ──────────────────
SELECT
    age_group,
    COUNT(*) AS count,
    SUM(CASE WHEN outcome = 1 THEN 1 ELSE 0 END) AS diabetic,
    ROUND(AVG(CAST(outcome AS DOUBLE)) * 100, 1)  AS diabetes_pct,
    ROUND(AVG(glucose), 1) AS avg_glucose
FROM healthpredict_health_db.processed_patient_data
WHERE age_group IS NOT NULL
GROUP BY age_group
ORDER BY diabetes_pct DESC;


-- ── QUERY 6: Glucose risk category breakdown ─────────────
SELECT
    glucose_risk,
    COUNT(*)                                       AS count,
    ROUND(AVG(CAST(outcome AS DOUBLE)) * 100, 1)   AS diabetes_pct,
    ROUND(AVG(glucose), 1)                         AS avg_glucose,
    ROUND(AVG(insulin), 1)                         AS avg_insulin
FROM healthpredict_health_db.processed_patient_data
WHERE glucose_risk IS NOT NULL
GROUP BY glucose_risk
ORDER BY avg_glucose;


-- ── QUERY 7: Correlation proxy — glucose × bmi interaction ─
-- High interaction value should correlate with higher diabetes rate.
SELECT
    CASE
        WHEN glucose_bmi_interaction < 3.0  THEN 'Low   (<3.0)'
        WHEN glucose_bmi_interaction < 4.5  THEN 'Med   (3.0-4.5)'
        ELSE                                     'High  (>4.5)'
    END AS interaction_tier,
    COUNT(*) AS count,
    ROUND(AVG(CAST(outcome AS DOUBLE)) * 100, 1) AS diabetes_pct
FROM healthpredict_health_db.processed_patient_data
GROUP BY 1
ORDER BY diabetes_pct;


-- ── QUERY 8: Raw data vs processed data comparison ────────
-- Compare row counts between raw and processed tables.
-- Use UNION ALL to run in one query.
SELECT 'raw'       AS source, COUNT(*) AS records
FROM healthpredict_health_db.raw_patient_data
UNION ALL
SELECT 'processed' AS source, COUNT(*) AS records
FROM healthpredict_health_db.processed_patient_data;


-- ── QUERY 9: Post-prediction analytics ────────────────────
-- Run AFTER making at least one POST /predict API call.
-- Queries the prediction_results table populated by Lambda.
SELECT
    risk_level,
    COUNT(*)                              AS total,
    ROUND(AVG(CAST(risk_score AS DOUBLE)), 4) AS avg_score,
    MIN(prediction_timestamp)             AS first_prediction,
    MAX(prediction_timestamp)             AS latest_prediction
FROM healthpredict_health_db.prediction_results
GROUP BY risk_level
ORDER BY risk_level;


-- ── QUERY 10: High-risk patient list ──────────────────────
-- Retrieve all HIGH-risk predictions sorted by score descending.
SELECT
    patient_id,
    prediction_timestamp,
    CAST(risk_score AS DOUBLE) AS score,
    model_version
FROM healthpredict_health_db.prediction_results
WHERE risk_level = 'HIGH'
ORDER BY CAST(risk_score AS DOUBLE) DESC
LIMIT 20;
