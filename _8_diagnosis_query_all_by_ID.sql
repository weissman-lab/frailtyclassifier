
-------------------------------------------------
WITH dxq AS (
    SELECT e.PAT_ENC_CSN_ID,
      mrn.IDENTITY_ID,
      mrn.PAT_ID,
      icd10.CODE,
      CE.DX_NAME,
      e.ENTRY_TIME,
      e.EFFECTIVE_DATE_DTTM,
      'PAT_ENC_DX' as SOURCE
    FROM PAT_ENC_DX as PED
      INNER JOIN PAT_ENC as e ON e.PAT_ENC_CSN_ID = PED.PAT_ENC_CSN_ID
      INNER JOIN CLARITY_EDG as CE ON PED.DX_ID = CE.DX_ID
      INNER JOIN EDG_CURRENT_ICD10 as icd10 on icd10.DX_ID = CE.DX_ID
      INNER JOIN IDENTITY_ID as mrn ON mrn.PAT_ID = e.PAT_ID AND mrn.IDENTITY_TYPE_ID = 100
    WHERE icd10.CODE is not null and e.ENTRY_TIME > '2017-01-01'

    UNION

    SELECT
      e.PAT_ENC_CSN_ID,
      mrn.IDENTITY_ID,
      mrn.PAT_ID,
      icd10.CODE,
      CE.DX_NAME,
      e.ENTRY_TIME,
      e.EFFECTIVE_DATE_DTTM,
      'HSP_ACCT_DX_LIST' AS SOURCE
    FROM HSP_ACCT_DX_LIST AS PED
      LEFT JOIN CLARITY_EDG AS CE ON PED.DX_ID = CE.DX_ID
      LEFT JOIN ZC_DX_CC_HA AS z1 on z1.DX_CC_HA_C = ped.DX_COMORBIDITY_C
      LEFT JOIN ZC_DX_POA AS z2 ON z2.DX_POA_C = ped.FINAL_DX_POA_C
      INNER JOIN EDG_CURRENT_ICD10 icd10 on icd10.DX_ID = CE.DX_ID
      LEFT JOIN PAT_ENC AS e on e.HSP_ACCOUNT_ID = ped.HSP_ACCOUNT_ID
      left join IDENTITY_ID AS mrn ON mrn.PAT_ID = e.PAT_ID AND mrn.IDENTITY_TYPE_ID = 100
    WHERE
      icd10.CODE is not null and e.ENTRY_TIME > '2017-01-01'

    UNION

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
      PAT_ENC AS e
    INNER JOIN PROBLEM_LIST_HX AS ph on ph.HX_PROBLEM_EPT_CSN = e.PAT_ENC_CSN_ID
    -- inner join PROBLEM_LIST AS l on l.PROBLEM_LIST_ID = ph.PROBLEM_LIST_ID
    inner join CLARITY_EDG AS edg on edg.DX_ID = ph.HX_PROBLEM_ID
    inner join EDG_CURRENT_ICD10 AS icd10 on icd10.DX_ID = edg.DX_ID
    left join ZC_DX_POA AS z1 on z1.DX_POA_C = ph.HX_PROBLEM_POA_C
    left join IDENTITY_ID AS mrn ON mrn.PAT_ID = e.PAT_ID AND mrn.IDENTITY_TYPE_ID = 100

    WHERE
      icd10.CODE is not null and e.ENTRY_TIME > '2017-01-01'
) SELECT dxq.PAT_ENC_CSN_ID,
         dxq.IDENTITY_ID,
         dxq.PAT_ID,
         dxq.CODE,
         dxq.ENTRY_TIME
	FROM dxq
  	  WHERE dxq.PAT_ID in :ids
