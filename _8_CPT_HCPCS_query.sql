
select
        op.PAT_ENC_CSN_ID as CSN
    ,   ceo.CPT_CODE as CPT
    ,   op.PAT_ID
    ,   ceo.CODE_TYPE_C as TYPE
    , 	op.ORDERING_DATE
    , 	op.PROC_ID
    ,	op.DESCRIPTION
    ,   maxdate = Max(ceo.CONTACT_DATE_REAL)
from ORDER_PROC as op
inner join CLARITY_EAP_OT as ceo on ceo.PROC_ID = op.PROC_ID
where ceo.CODE_TYPE_C in (1,2)
and ceo.CPT_CODE is not null
and op.PAT_ID in :ids
and op.ORDERING_DATE > '2017-01-01'
group by op.PAT_ENC_CSN_ID, op.PAT_ID, ceo.CPT_CODE, ceo.CODE_TYPE_C, op.PROC_ID, op.ORDERING_DATE, op.DESCRIPTION, op.PROC_ID