select e.PAT_ENC_CSN_ID,
  mrn.IDENTITY_ID,
  mrn.PAT_ID,
  icd10.CODE,
  CE.DX_NAME,
  e.ENTRY_TIME,
  e.EFFECTIVE_DATE_DTTM,
  'PAT_ENC_DX' as SOURCE
from PAT_ENC_DX as PED
  inner join PAT_ENC as e on e.PAT_ENC_CSN_ID = PED.PAT_ENC_CSN_ID
  inner join CLARITY_EDG as CE on PED.DX_ID = CE.DX_ID
  inner join EDG_CURRENT_ICD10 as icd10 on icd10.DX_ID = CE.DX_ID
  inner join IDENTITY_ID as mrn on mrn.PAT_ID = e.PAT_ID and mrn.IDENTITY_TYPE_ID = 100
where icd10.CODE is not null and e.ENTRY_TIME > '2017-01-01' and
  (icd10.CODE like '%J44%' or icd10.CODE like '%J43%' or icd10.CODE like '%J41%' or icd10.CODE like '%J42%' or icd10.CODE like '%J84%' or icd10.CODE like '%D86%' or icd10.CODE like '%J84%' or icd10.CODE like '%M34%' or icd10.CODE like '%J99%')

union

select
  e.PAT_ENC_CSN_ID,
  mrn.IDENTITY_ID,
  mrn.PAT_ID,
  icd10.CODE,
  CE.DX_NAME,
  e.ENTRY_TIME,
  e.EFFECTIVE_DATE_DTTM,
  'HSP_ACCT_DX_LIST' as SOURCE
from HSP_ACCT_DX_LIST as PED
  LEFT join CLARITY_EDG as CE on PED.DX_ID = CE.DX_ID
  LEFT join ZC_DX_CC_HA as z1 on z1.DX_CC_HA_C = ped.DX_COMORBIDITY_C
  LEFT join ZC_DX_POA as z2 on z2.DX_POA_C = ped.FINAL_DX_POA_C
  inner join EDG_CURRENT_ICD10 icd10 on icd10.DX_ID = CE.DX_ID
  LEFT join PAT_ENC as e on e.HSP_ACCOUNT_ID = ped.HSP_ACCOUNT_ID
  left join IDENTITY_ID as mrn on mrn.PAT_ID = e.PAT_ID and mrn.IDENTITY_TYPE_ID = 100
where
  icd10.CODE is not null and e.ENTRY_TIME > '2017-01-01' and
  (icd10.CODE like '%J44%' or icd10.CODE like '%J43%' or icd10.CODE like '%J41%' or icd10.CODE like '%J42%' or icd10.CODE like '%J84%' or icd10.CODE like '%D86%' or icd10.CODE like '%J84%' or icd10.CODE like '%M34%' or icd10.CODE like '%J99%')

union

select
  e.PAT_ENC_CSN_ID,
  mrn.IDENTITY_ID,
  mrn.PAT_ID,
  icd10.CODE,
  edg.DX_NAME,
  e.ENTRY_TIME,
  e.EFFECTIVE_DATE_DTTM,
  'PROBLEM_LIST' as SOURCE
from
  PAT_ENC as e
inner join PROBLEM_LIST_HX as ph on ph.HX_PROBLEM_EPT_CSN = e.PAT_ENC_CSN_ID
inner join CLARITY_EDG as edg on edg.DX_ID = ph.HX_PROBLEM_ID
inner join EDG_CURRENT_ICD10 as icd10 on icd10.DX_ID = edg.DX_ID
left join ZC_DX_POA as z1 on z1.DX_POA_C = ph.HX_PROBLEM_POA_C
left join IDENTITY_ID as mrn on mrn.PAT_ID = e.PAT_ID and mrn.IDENTITY_TYPE_ID = 100

where
  icd10.CODE is not null and e.ENTRY_TIME > '2017-01-01' and
  (icd10.CODE like '%J44%' or icd10.CODE like '%J43%' or icd10.CODE like '%J41%' or icd10.CODE like '%J42%' or icd10.CODE like '%J84%' or icd10.CODE like '%D86%' or icd10.CODE like '%J84%' or icd10.CODE like '%M34%' or icd10.CODE like '%J99%')