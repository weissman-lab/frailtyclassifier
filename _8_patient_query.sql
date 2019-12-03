


select e.PAT_ENC_CSN_ID
     , mrn.IDENTITY_ID MRN
     , mrn.PAT_ID
     , e.EFFECTIVE_DATE_DTTM
     , e.ENTRY_TIME
     , datediff(day, pat.BIRTH_DATE, e.ENTRY_TIME) / 365.0      AGE
     , ZHAT.NAME as ADMSN_TYPE
     , zpc.NAME as PATIENT_CLASS
     , dep.DEPT_ABBREVIATION as UNIT
     , dep.DEPARTMENT_NAME
     , dep.EXTERNAL_NAME
     , dep.SPECIALTY
     , CL.LOCATION_ABBR
     , zs.NAME as SEX
     , zm.NAME as MARITAL_STATUS
     , zr.NAME as RELIGION
     , zes.NAME as EMPY_STAT
     , zpr.NAME as RACE
     , ze.NAME as ETHNICITY
     , zl.NAME as LANGUAGE
     , zc.NAME as COUNTY
     , pat.ZIP
     , ZDET.NAME    as  ENCOUNTER_TYPE
FROM PAT_ENC as e
  left join PAT_ENC_2 as e2 on e2.PAT_ENC_CSN_ID = e.PAT_ENC_CSN_ID
       left join PAT_ENC_HSP as hsp on hsp.PAT_ENC_CSN_ID = e.PAT_ENC_CSN_ID
       inner join PATIENT as pat on pat.PAT_ID = e.PAT_ID
       left join ZC_PAT_CLASS as zpc on e2.ADT_PAT_CLASS_C = zpc.ADT_PAT_CLASS_C
       left join ZC_HOSP_ADMSN_TYPE as ZHAT on e.HOSP_ADMSN_TYPE_C = ZHAT.HOSP_ADMSN_TYPE_C
       left join ZC_PAT_STATUS as zps on hsp.ADT_PATIENT_STAT_C = zps.ADT_PATIENT_STAT_C
       left join CLARITY_DEP as dep on dep.DEPARTMENT_ID = e.DEPARTMENT_ID
       left join CLARITY_LOC as CL on dep.REV_LOC_ID = CL.LOC_ID
       left join ZC_SEX as zs on pat.SEX_C = zs.RCPT_MEM_SEX_C
       left join ZC_ETHNIC_GROUP as ze on pat.ETHNIC_GROUP_C = ze.ETHNIC_GROUP_C
       left join ZC_MARITAL_STATUS as zm on pat.MARITAL_STATUS_C = zm.MARITAL_STATUS_C
       left join ZC_RELIGION as zr on pat.RELIGION_C = zr.RELIGION_C
       left join ZC_LANGUAGE as zl on pat.LANGUAGE_C = zl.LANGUAGE_C
       left join ZC_FIN_CLASS as zf on pat.DEF_FIN_CLASS_C = zf.FIN_CLASS_C
       left join ZC_VETERAN_STAT as zv on pat.VETERAN_STATUS_C = zv.VETERAN_STATUS_C
       left join ZC_EMPY_STAT as zes on pat.EMPY_STATUS_C = zes.EMPY_STAT_C
       left join ZC_COUNTY as zc on pat.COUNTY_C = zc.COUNTY_C
       left join PATIENT_RACE as pr on pr.PAT_ID = e.PAT_ID and pr.LINE = 1
       left join ZC_PATIENT_RACE as zpr on pr.PATIENT_RACE_C = zpr.PATIENT_RACE_C
       inner join IDENTITY_ID as mrn on mrn.PAT_ID = e.PAT_ID and mrn.IDENTITY_TYPE_ID = 100
       left join ZC_DISP_ENC_TYPE as ZDET on e.ENC_TYPE_C = ZDET.DISP_ENC_TYPE_C
where e.ENTRY_TIME >= '2017-01-01'
and dep.SPECIALTY in ('INTERNAL MEDICINE', 'PULMONARY', 'FAMILY PRACTICE', 'GERONTOLOGY')
and ZDET.NAME in ('Appointment', 'Office Visit')
and e.ENTRY_TIME >= '2017-01-01'
and mrn.PAT_ID in :ids