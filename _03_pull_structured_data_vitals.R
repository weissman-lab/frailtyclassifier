library(skimr)
library(dplyr)
library(magrittr)
library(tidyr)
library(stringr)

cwd <- getwd()
setwd(paste0(cwd, "/output"))
outdir = getwd()

vitdf <- read.csv('vitals_r_start.csv', stringsAsFactors = FALSE)

#clean vitals
vitdf2 <- vitdf %>%
  # ensure missing are idenfitied properly
  mutate(HEIGHT = ifelse(HEIGHT == '', NA, HEIGHT)) %>%
  mutate(WEIGHT = ifelse(WEIGHT == 'NA', NA, WEIGHT)) %>%
  # set as missing if no feet in height
  mutate(HEIGHT = ifelse(!str_detect(HEIGHT, "\'"), NA, HEIGHT))

# convert height into cm
vitdf3 <- vitdf2 %>% 
  separate(HEIGHT, sep = "'", c('ft', 'inch')) %>%
  mutate(inch = str_replace(inch, "\"", "")) %>%
  mutate(inch = ifelse(inch == '', NA, inch)) %>%
  mutate(HEIGHT_CM = ifelse(!is.na(inch), (as.numeric(ft)*30.48 + as.numeric(inch)*2.54), as.numeric(ft)*30.48)) %>%
  select(-'ft', -'inch') %>%
  #drop height > 245 cm
  mutate(HEIGHT_CM = ifelse(HEIGHT_CM > 245, NA, HEIGHT_CM)) %>%
  #drop height < 110
  mutate(HEIGHT_CM = ifelse(HEIGHT_CM < 110, NA, HEIGHT_CM))

#calculate difference in height measurements
heightdiff <- vitdf3 %>%
  group_by(PAT_ID) %>%
  summarise(max(HEIGHT_CM)-min(HEIGHT_CM)) %>%
  ungroup() %>%
  arrange(PAT_ID) %>%
  filter(`max(HEIGHT_CM) - min(HEIGHT_CM)` > 15)
heightdiff <- heightdiff$PAT_ID
#drop height difference >15 cm (approx 95th percentile)
vitdf5 <- vitdf3 %>%
  mutate(HEIGHT_CM = ifelse(PAT_ID %in% heightdiff, NA, HEIGHT_CM))

vitdf6 <- vitdf5 %>%
  #convert weight from ounces to kg
  mutate(WEIGHT = WEIGHT*0.028) %>%
  #drop weight > 300 kg (note: this will effect # missing height)
  mutate(WEIGHT = ifelse(WEIGHT > 300, NA, WEIGHT)) %>%
  #drop weight < 30 kg
  mutate(WEIGHT = ifelse(WEIGHT < 30, NA, WEIGHT)) %>%
  #calculate BMI
  mutate(BMI = WEIGHT/((HEIGHT_CM/100)^2)) %>%
  #drop systolic bp > 250 or < 70
  mutate(BP_SYSTOLIC = ifelse(BP_SYSTOLIC > 250 | BP_SYSTOLIC < 70, NA, BP_SYSTOLIC)) %>%
  #drop diastolic bp < 25 or > 170
  mutate(BP_DIASTOLIC = ifelse(BP_DIASTOLIC < 25 | BP_DIASTOLIC > 170, NA, BP_DIASTOLIC)) %>%
  #drop age > 110 OR < 16
  mutate(AGE = ifelse(AGE > 110 | AGE < 16, NA, AGE))

#send back to python script
write.csv(vitdf6, paste0(outdir, "/vitals_r_finish.csv"))
