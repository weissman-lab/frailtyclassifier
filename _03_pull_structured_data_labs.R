library(skimr)
library(dplyr)
library(magrittr)
library(tidyr)

cwd <- getwd()
setwd(paste0(cwd, "/output"))
outdir = getwd()
print(getwd())
labdf3 <- read.csv('labs_r_start.csv', stringsAsFactors = FALSE)
print('loaded file from R')
#clean labs
labdf4 <- 
  labdf3 %>%
  # convert albumin mg/dl to g/dl & drop values > 10
  mutate(VAL_NUM = ifelse(COMMON_NAME == 'ALBUMIN' & REFERENCE_UNIT == 'mg/dL', VAL_NUM/1000, VAL_NUM)) %>%
  # drop albumin extreme outliers
  filter(!(COMMON_NAME == 'ALBUMIN' & VAL_NUM > 10)) %>%
  # drop alk phos with units '%'
  filter(!(COMMON_NAME == 'ALKALINE_PHOSPHATASE' & REFERENCE_UNIT == '%')) %>%
  # drop tbili & dbili (keep total bili)
  filter(!(COMMON_NAME == 'BILIRUBIN DIRECT' | COMMON_NAME == 'BILIRUBIN INDIRECT')) %>%
  # drop calcium with units 'q'
  filter(!(COMMON_NAME == 'CALCIUM' & REFERENCE_UNIT == 'q')) %>%
  # drop calcium extreme outliers
  filter(!(COMMON_NAME == 'CALCIUM' & VAL_NUM > 25)) %>%
  # drop CO2 with units '31'
  filter(!(COMMON_NAME == 'CO2' & REFERENCE_UNIT == '31')) %>%
  # drop CO2 extreme outliers
  filter(!(COMMON_NAME == 'CO2' & VAL_NUM > 200)) %>%
  # drop Cr extreme outliers
  filter(!(COMMON_NAME == 'CREATININE' & VAL_NUM > 50)) %>%
  # recode Cr 30 - 50 as Cr 30
  mutate(VAL_NUM = ifelse(COMMON_NAME == 'CREATININE' & VAL_NUM > 30, 30, VAL_NUM)) %>%
  #will cap ferritin at 10000 and recode outliers to 10000
  mutate(VAL_NUM = ifelse(COMMON_NAME == 'FERRITIN' & VAL_NUM > 10000, 10000, VAL_NUM)) %>%
  #drop hbg with units '15.5' and '%'
  filter(!(COMMON_NAME == 'HEMOGLOBIN' & REFERENCE_UNIT == '%')) %>%
  filter(!(COMMON_NAME == 'HEMOGLOBIN' & REFERENCE_UNIT == '15.5')) %>%
  #convert LDL nmol/L to mmol/dL
  mutate(VAL_NUM = ifelse(COMMON_NAME == 'LDL' & REFERENCE_UNIT == 'nmol/L', (VAL_NUM/1e6)*38.61, VAL_NUM))%>%
  #remove LDL zeros and negatives
  filter(!(COMMON_NAME == 'LDL' & VAL_NUM < 1)) %>%
  #drop Mg extreme outliers
  filter(!(COMMON_NAME == 'MAGNESIUM' & VAL_NUM > 20)) %>%
  #drop MCV > 200 or < 30
  filter(!(COMMON_NAME == 'MCV' & VAL_NUM < 30)) %>%
  filter(!(COMMON_NAME == 'MCV' & VAL_NUM > 200)) %>%
  #drop phos without units (many outliers)
  filter(!(COMMON_NAME == 'PHOSPHATE' & REFERENCE_UNIT == '')) %>%
  #divide plt >10000 by 1000 to change units
  mutate(VAL_NUM = ifelse(COMMON_NAME == 'PLATELETS' & VAL_NUM > 10000, VAL_NUM/1000, VAL_NUM)) %>%
  #drop K > 9 or < 1 (likely spurious)
  filter(!(COMMON_NAME == 'POTASSIUM' & VAL_NUM > 9)) %>%
  filter(!(COMMON_NAME == 'POTASSIUM' & VAL_NUM < 1)) %>%
  #drop protein extreme outliers
  filter(!(COMMON_NAME == 'PROTEIN' & VAL_NUM > 50)) %>%
  #drop PT <5
  filter(!(COMMON_NAME == 'PT' & VAL_NUM < 5)) %>%
  #recode PTT > 150 to 150 (upper limit of assay)
  mutate(VAL_NUM = ifelse(COMMON_NAME == 'PTT' & VAL_NUM > 150, 150, VAL_NUM)) %>%
  #drop ptt < 10
  filter(!(COMMON_NAME == 'PTT' & VAL_NUM < 10)) %>%
  #drop ptt without units because they are distributed differently
  filter(!(COMMON_NAME == 'PTT' & REFERENCE_UNIT == '')) %>%
  #drop RDW ref units '15.4'
  filter(!(COMMON_NAME == 'RDW' & REFERENCE_UNIT == '15.4')) %>%
  #drop RDW > 100 and < 9
  filter(!(COMMON_NAME == 'RDW' & VAL_NUM > 100)) %>%
  filter(!(COMMON_NAME == 'RDW' & VAL_NUM < 9)) %>%
  #drop Na > 200 & < 100
  filter(!(COMMON_NAME == 'SODIUM' & VAL_NUM < 100)) %>%
  filter(!(COMMON_NAME == 'SODIUM' & VAL_NUM > 200)) %>%
  #drop tsat > 99
  filter(!(COMMON_NAME == 'TRANSFERRIN_SAT' & VAL_NUM > 99)) %>%
  #drop wbc > 300
  filter(!(COMMON_NAME == 'WBC' & VAL_NUM > 300)) %>%
  #drop REFERENCE_UNIT because no longer needed
  select(-REFERENCE_UNIT)

print(table(labdf4$COMMON_NAME))

#send back to python script
write.csv(labdf4, paste0(outdir, "/labs_r_finish.csv"))
