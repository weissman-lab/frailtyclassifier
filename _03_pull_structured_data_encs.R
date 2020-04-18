library(skimr)
library(dplyr)
library(magrittr)
library(tidyr)
library(stringr)
`%ni%` <- Negate(`%in%`)

cwd <- getwd()
setwd(paste0(cwd, "/output"))
outdir = getwd()

# read csv
encdf <- read.csv('encs_r_start.csv', stringsAsFactors = FALSE)

office_visit <- encdf %>%
  filter(ENCOUNTER_TYPE == 'Office Visit' | ENCOUNTER_TYPE == 'Post Hospitalization' | ENCOUNTER_TYPE == 'Post Emergency')%>%
  filter(PATIENT_CLASS == 'MAPS' | PATIENT_CLASS == 'Outpatient' | PATIENT_CLASS == "") %>%
  mutate(office_visit = 1) %>%
  select(-'PATIENT_CLASS', -'ENCOUNTER_TYPE')

ED_visit <- encdf %>%
  filter(ENCOUNTER_TYPE == 'Hospital Encounter' | ENCOUNTER_TYPE == 'Emergency Department') %>%
  filter(PATIENT_CLASS == 'Emergency' | PATIENT_CLASS == 'Observation') %>%
  mutate(ED_visit = 1) %>%
  select(-'PATIENT_CLASS', -'ENCOUNTER_TYPE')

admission <- encdf %>%
  filter(ENCOUNTER_TYPE == 'Hospital Encounter' | ENCOUNTER_TYPE == 'Emergency Department') %>%
  filter(PATIENT_CLASS == 'Inpatient'| PATIENT_CLASS == 'ICU' | PATIENT_CLASS == 	'Semi-Private/Med-Surg'| PATIENT_CLASS == "") %>%
  mutate(admission = 1) %>%
  select(-'PATIENT_CLASS', -'ENCOUNTER_TYPE')

encdf2 <- bind_rows(office_visit, ED_visit, admission)

#send back to python script
write.csv(encdf2, paste0(outdir, "/encs_r_finish.csv"))

encdf[encdf$ENCOUNTER_TYPE == "Emergency Department" & encdf$PATIENT_CLASS == "",] %>% dim
encdf[encdf$ENCOUNTER_TYPE == "Hospital Encounter" & encdf$PATIENT_CLASS == "",] %>% dim
