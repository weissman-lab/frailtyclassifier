
select
        ti.PAT_ID
    ,   ti.TX_SURG_DT
from TRANSPLANT_INFO as ti
inner join TRANSPLANT_CLASS as tc on ti.SUMMARY_BLOCK_ID = tc.SUMMARY_BLOCK_ID
-- where tc.TX_CLASS_C = 2 and ti.TX_SURG_DT is not NULL -- lungs
-- and ti.PAT_ID in :ids --this will get subbed out for the IDs that we've already got
