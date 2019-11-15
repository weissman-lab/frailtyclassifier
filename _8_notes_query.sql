
--Query to get notes.
select hi.PAT_ENC_CSN_ID,
       nei.CONTACT_SERIAL_NUM                                    NOTE_SERIAL_NUM,
       ZC_NOTE_TYPE_IP.NAME                                      IP_NOTE_TYPE, --ERIK ADDED THIS 11/15/16
       COALESCE(nei.ENTRY_INSTANT_DTTM, nei.NOTE_FILE_TIME_DTTM) NOTE_ENTRY_TIME,
       hi.NOTE_ID,
       t.LINE                                                    NOTE_LINE,
       nei.CONTACT_NUM,
       t.NOTE_TEXT
from NOTE_ENC_INFO AS nei
  INNER JOIN HNO_INFO AS hi
    ON nei.NOTE_ID = hi.NOTE_ID -- Stores overtime information
    and 19 = hi.note_type_noadd_c
    AND hi.PAT_ENC_CSN_ID in :ids
  INNER JOIN ZC_NOTE_TYPE_IP
    ON hi.IP_NOTE_TYPE_C = ZC_NOTE_TYPE_IP.TYPE_IP_C --IP Note type --ERIK ADDED THIS 11/15/16
  INNER JOIN HNO_NOTE_TEXT AS t
    ON nei.CONTACT_SERIAL_NUM = t.NOTE_CSN_ID
    and t.NOTE_TEXT <> ''
WHERE 2 = nei.note_status_c

