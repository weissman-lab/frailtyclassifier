
--Query to get notes.
select hi.PAT_ENC_CSN_ID,
       nei.CONTACT_SERIAL_NUM                                    NOTE_SERIAL_NUM,
       ZC_NOTE_TYPE_IP.NAME                                      IP_NOTE_TYPE, 
       COALESCE(nei.ENTRY_INSTANT_DTTM, nei.NOTE_FILE_TIME_DTTM) NOTE_ENTRY_TIME,
       hi.NOTE_ID,
       t.LINE                                                    NOTE_LINE,
       nei.CONTACT_NUM,
       t.NOTE_TEXT
from NOTE_ENC_INFO as nei
  inner join HNO_INFO as hi
    on nei.NOTE_ID = hi.NOTE_ID 
    and 19 = hi.note_type_noadd_c
    and hi.PAT_ENC_CSN_ID in :ids
  inner join ZC_NOTE_TYPE_IP
    on hi.IP_NOTE_TYPE_C = ZC_NOTE_TYPE_IP.TYPE_IP_C 
  inner join HNO_NOTE_TEXT as t
    on nei.CONTACT_SERIAL_NUM = t.NOTE_CSN_ID
    and t.NOTE_TEXT <> ''
where 2 = nei.note_status_c

