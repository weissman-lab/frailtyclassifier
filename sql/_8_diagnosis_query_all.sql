with dx as (
    select
            ped.PAT_ENC_CSN_ID  as CSN
        ,   ped.PAT_ID
        ,   icd10.CODE          as icd10
    --     ,   'PAT_ENC_DX'        as source
    from PAT_ENC_DX as ped
    join EDG_CURRENT_ICD10 as icd10 on icd10.DX_ID = ped.DX_ID
    where icd10.CODE is not null
    and ped.CONTACT_DATE > '2017-01-01'

    union

    select
            pe.PAT_ENC_CSN_ID   as CSN
        ,   pe.PAT_ID
        ,   icd10.CODE          as icd10
    --     ,   'HSP_ACCT_DX_LIST'  as source
    from HSP_ACCT_DX_LIST as hadl
    join PAT_ENC as pe on pe.HSP_ACCOUNT_ID = hadl.HSP_ACCOUNT_ID
    join EDG_CURRENT_ICD10 as icd10 on icd10.DX_ID = hadl.DX_ID
    where icd10.CODE is not null
    and pe.CONTACT_DATE > '2017-01-01'

    union

    select
            pe.PAT_ENC_CSN_ID   as CSN
        ,   pe.PAT_ID
        ,   icd10.CODE          as icd10
    --     ,   'PROBLEM_LIST'  as source
    from PAT_ENC as pe
    join PROBLEM_LIST_HX as plh on plh.HX_PROBLEM_EPT_CSN = pe.PAT_ENC_CSN_ID
    join CLARITY_EDG as ce on ce.DX_ID = plh.HX_PROBLEM_ID
    join EDG_CURRENT_ICD10 as icd10 on icd10.DX_ID = ce.DX_ID
    where icd10.CODE is not null
    and pe.CONTACT_DATE > '2017-01-01'
) select distinct * from dx