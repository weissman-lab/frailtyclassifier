


SELECT e.PAT_ENC_CSN_ID
     , mrn.IDENTITY_ID MRN
     , mrn.PAT_ID
     , e.EFFECTIVE_DATE_DTTM
     , e.ENTRY_TIME
     , datediff(day, pat.BIRTH_DATE, e.ENTRY_TIME) / 365.0      AGE
     , ZHAT.NAME AS ADMSN_TYPE
     , zpc.NAME as PATIENT_CLASS
     , dep.DEPT_ABBREVIATION AS UNIT
     , dep.DEPARTMENT_NAME
     , dep.EXTERNAL_NAME
     , dep.SPECIALTY
     , CL.LOCATION_ABBR
     , zs.NAME AS SEX
     , zm.NAME AS MARITAL_STATUS
     , zr.NAME AS RELIGION
     , zes.NAME AS EMPY_STAT
     , zpr.NAME AS RACE
     , ze.NAME AS ETHNICITY
     , zl.NAME AS LANGUAGE
     , zc.NAME AS COUNTY
     , pat.ZIP
     , ZDET.NAME    AS  ENCOUNTER_TYPE
FROM PAT_ENC AS e
  left join PAT_ENC_2 AS e2 on e2.PAT_ENC_CSN_ID = e.PAT_ENC_CSN_ID
       left join PAT_ENC_HSP AS hsp ON hsp.PAT_ENC_CSN_ID = e.PAT_ENC_CSN_ID
       INNER JOIN PATIENT AS pat ON pat.PAT_ID = e.PAT_ID
       LEFT JOIN ZC_PAT_CLASS AS zpc ON e2.ADT_PAT_CLASS_C = zpc.ADT_PAT_CLASS_C
       LEFT JOIN ZC_HOSP_ADMSN_TYPE AS ZHAT ON e.HOSP_ADMSN_TYPE_C = ZHAT.HOSP_ADMSN_TYPE_C
       LEFT JOIN ZC_PAT_STATUS AS zps ON hsp.ADT_PATIENT_STAT_C = zps.ADT_PATIENT_STAT_C
       LEFT JOIN CLARITY_DEP AS dep ON dep.DEPARTMENT_ID = e.DEPARTMENT_ID
       LEFT JOIN CLARITY_LOC AS CL ON dep.REV_LOC_ID = CL.LOC_ID
       LEFT JOIN ZC_SEX AS zs on pat.SEX_C = zs.RCPT_MEM_SEX_C
       LEFT JOIN ZC_ETHNIC_GROUP AS ze on pat.ETHNIC_GROUP_C = ze.ETHNIC_GROUP_C
       LEFT JOIN ZC_MARITAL_STATUS AS zm on pat.MARITAL_STATUS_C = zm.MARITAL_STATUS_C
       LEFT JOIN ZC_RELIGION AS zr on pat.RELIGION_C = zr.RELIGION_C
       LEFT JOIN ZC_LANGUAGE AS zl on pat.LANGUAGE_C = zl.LANGUAGE_C
       LEFT JOIN ZC_FIN_CLASS AS zf on pat.DEF_FIN_CLASS_C = zf.FIN_CLASS_C
       LEFT JOIN ZC_VETERAN_STAT AS zv on pat.VETERAN_STATUS_C = zv.VETERAN_STATUS_C
       lEFT JOIN ZC_EMPY_STAT AS zes on pat.EMPY_STATUS_C = zes.EMPY_STAT_C
       LEFT JOIN ZC_COUNTY AS zc on pat.COUNTY_C = zc.COUNTY_C
       LEFT JOIN PATIENT_RACE AS pr on pr.PAT_ID = e.PAT_ID and pr.LINE = 1
       LEFT JOIN ZC_PATIENT_RACE AS zpr on pr.PATIENT_RACE_C = zpr.PATIENT_RACE_C
       inner join IDENTITY_ID AS mrn ON mrn.PAT_ID = e.PAT_ID AND mrn.IDENTITY_TYPE_ID = 100
       left join ZC_DISP_ENC_TYPE AS ZDET ON e.ENC_TYPE_C = ZDET.DISP_ENC_TYPE_C
WHERE e.ENTRY_TIME >= '2017-01-01'
-- AND zpc.NAME = 'Outpatient'
AND dep.SPECIALTY != 'Laboratory'
AND ZDET.NAME IN ('Appointment', 'Office Visit')
AND e.ENTRY_TIME >= '2017-01-01'
AND mrn.PAT_ID in :ids