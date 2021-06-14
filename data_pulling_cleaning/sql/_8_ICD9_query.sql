
select
        ped.PAT_ID
    ,   eci.CODE as ICD9
    ,   ped.CONTACT_DATE
from PAT_ENC_DX as ped
inner join EDG_CURRENT_ICD9 as eci on ped.DX_ID = eci.DX_ID
where ped.CONTACT_DATE > '2017-01-01'
and ped.PAT_ID in :ids
