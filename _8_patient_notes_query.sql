


select
        pe.PAT_ENC_CSN_ID                                           as CSN
    ,   pe.PAT_ID
    ,   hi.NOTE_ID
    ,   pe.CONTACT_DATE                                             as ENC_DATE
    ,   datediff(day, p.BIRTH_DATE, pe.CONTACT_DATE) / 365.0        as AGE
    ,   zpc.NAME                                                    as PATIENT_CLASS
    ,   cd.DEPARTMENT_NAME
    ,   cd.SPECIALTY
    ,   znti.NAME                                                   as NOTE_TYPE
    ,   nei.note_status_c                                           as NOTE_STATUS
    ,   hnt.LINE                                                    as NOTE_LINE
    ,   nei.CONTACT_NUM
    ,   zdet.NAME                                                   as ENCOUNTER_TYPE
    ,   hnt.NOTE_TEXT
    ,   COALESCE(nei.ENTRY_INSTANT_DTTM, nei.NOTE_FILE_TIME_DTTM)   as NOTE_ENTRY_TIME
    ,   cl.LOCATION_ABBR
    ,   zs.NAME                                                     as SEX
    ,   zms.NAME                                                    as MARITAL_STATUS
    ,   zr.NAME                                                     as RELIGION
    ,   zes.NAME                                                    as EMPY_STAT
    ,   zpr.NAME                                                    as RACE
    ,   zeg.NAME                                                    as ETHNICITY
    ,   zl.NAME                                                     as LANGUAGE
    ,   zc.NAME                                                     as COUNTY
    ,   zv.ABBR                                                     as VETERAN_STATUS
    ,   p.ZIP
FROM PAT_ENC as pe
join PAT_ENC_2 as pe2 on pe2.PAT_ENC_CSN_ID = pe.PAT_ENC_CSN_ID
left join ZC_PAT_CLASS as zpc on pe2.ADT_PAT_CLASS_C = zpc.ADT_PAT_CLASS_C
join PATIENT as p on p.PAT_ID = pe.PAT_ID
left join CLARITY_DEP as cd on cd.DEPARTMENT_ID = pe.DEPARTMENT_ID
left join CLARITY_LOC as cl on cd.REV_LOC_ID = CL.LOC_ID
left join ZC_SEX as zs on p.SEX_C = zs.RCPT_MEM_SEX_C
left join ZC_ETHNIC_GROUP as zeg on p.ETHNIC_GROUP_C = zeg.ETHNIC_GROUP_C
left join ZC_MARITAL_STATUS as zms on p.MARITAL_STATUS_C = zms.MARITAL_STATUS_C
left join ZC_RELIGION as zr on p.RELIGION_C = zr.RELIGION_C
left join ZC_LANGUAGE as zl on p.LANGUAGE_C = zl.LANGUAGE_C
left join ZC_FIN_CLASS as zf on p.DEF_FIN_CLASS_C = zf.FIN_CLASS_C
left join ZC_VETERAN_STAT as zv on p.VETERAN_STATUS_C = zv.VETERAN_STATUS_C
left join ZC_EMPY_STAT as zes on p.EMPY_STATUS_C = zes.EMPY_STAT_C
left join ZC_COUNTY as zc on p.COUNTY_C = zc.COUNTY_C
left join PATIENT_RACE as pr on pr.PAT_ID = pe.PAT_ID and pr.LINE = 1
left join ZC_PATIENT_RACE as zpr on pr.PATIENT_RACE_C = zpr.PATIENT_RACE_C
left join ZC_DISP_ENC_TYPE as zdet on pe.ENC_TYPE_C = zdet.DISP_ENC_TYPE_C
join HNO_INFO as hi on  hi.PAT_ENC_CSN_ID = pe.PAT_ENC_CSN_ID
join ZC_NOTE_TYPE_IP as znti on hi.IP_NOTE_TYPE_C = znti.TYPE_IP_C
join NOTE_ENC_INFO as nei on nei.NOTE_ID = hi.NOTE_ID
join HNO_NOTE_TEXT as hnt on nei.CONTACT_SERIAL_NUM = hnt.NOTE_CSN_ID and hnt.NOTE_TEXT <> ''
