

## This script combines and imputes the structured data

library(ranger)
library(missRanger)
library(doParallel)
library(foreach)
library(dplyr)
library(readr)
registerDoParallel(detectCores())
`%ni%` <- Negate(`%in%`)
if (grepl("crandrew", getwd())){
  datadir <- "~/projects/GW_PAIR_frailty_classifier/data/"
  outdir <- "~/projects/GW_PAIR_frailty_classifier/output/"
  figdir <- "~/projects/GW_PAIR_frailty_classifier/figures/"
} else {
  datadir <- "/media/drv2/andrewcd2/frailty/data/"
  outdir <- "/media/drv2/andrewcd2/frailty/output/"
  figdir <- "/media/drv2/andrewcd2/frailty/figures/"
}

#
df <- read.csv(paste0(outdir, "structured_data_merged_partially_cleaned.csv"))
# clean it some more
df$X <- df$RELIGION <- NULL
df$RACE[df$RACE %in% c('Unknown', '')] <- NA
df$RACE[df$RACE %ni% c('White', "Black") & !is.na(df$RACE)] <- "Other"
df$RACE <- factor(df$RACE)
df$AGE[df$AGE>100 | df$AGE < 18] <- NA
df$SEX[df$SEX == ''] <- NA
df$SEX <- factor(df$SEX)
df$EMPY_STAT[df$EMPY_STAT %in% c('', 'Unknown')] <- NA
df$EMPY_STAT <- as.character(df$EMPY_STAT)
df$EMPY_STAT[df$EMPY_STAT %in% c('Homemaker', 'Self Employed', 
                                 'On Active Military Duty', 'Per Diem', 
                                 'Student - Part Time', 'Student - Full Time')] <- "Other"
df$EMPY_STAT[df$EMPY_STAT == 'Retired Military'] <- "Retired"
df$EMPY_STAT <- factor(df$EMPY_STAT)
df$MARITAL_STATUS[df$MARITAL_STATUS == ''] <- NA
df$MARITAL_STATUS <- as.character(df$MARITAL_STATUS)
df$MARITAL_STATUS[df$MARITAL_STATUS == "Separated"] <- 'Divorced'
df$MARITAL_STATUS[df$MARITAL_STATUS == "Partner"] <- 'Other'
df$MARITAL_STATUS <- factor(df$MARITAL_STATUS)

df$LANGUAGE[df$LANGUAGE == ''] <- NA
df$LANGUAGE <- factor(df$LANGUAGE)
df$COUNTY[df$COUNTY == ''] <- NA
df$COUNTY <- factor(df$COUNTY)
#missranger
impdat <- missRanger(df[,2:ncol(df)], returnOOB = T, num.threads = detectCores(), seed = 8675309)
impdat <- data.frame("PAT_ID" = df$PAT_ID, impdat)
write.csv(impdat, file = paste0(outdir, 'impdat.csv'))

impdat_dums <- model.matrix(~.-PAT_ID-1, data = impdat)
impdat_dums <- data.frame("PAT_ID" = df$PAT_ID, impdat_dums)
write.csv(impdat_dums, file = paste0(outdir, 'impdat_dums.csv'))

