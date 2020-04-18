

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
# df$COUNTY[df$COUNTY == ''] <- NA
# df$COUNTY <- factor(df$COUNTY)
options(na.action='na.pass')
missdat <- model.matrix(~.-PAT_ID-1, data = df, na.rm=F)
options(na.action='na.omit')

# make data frame of missing values
mdums <- df
for (i in colnames(mdums)){
  mdums[[i]] <- is.na(mdums[[i]])
  if (all(mdums[[i]] == FALSE)){
    mdums[[i]] <- NULL
  }
}
print(dim(mdums))
colnames(mdums) <- paste0("MV_", colnames(mdums))

#missranger
impdat <- missRanger(df[,2:ncol(df)], returnOOB = T, num.threads = detectCores(), seed = 8675309)
impdat <- data.frame("PAT_ID" = df$PAT_ID, impdat)
write.csv(impdat, file = paste0(outdir, 'impdat.csv'))

impdat_dums <- model.matrix(~.-PAT_ID-1, data = impdat)
impdat_dums <- data.frame("PAT_ID" = df$PAT_ID, impdat_dums)
impdat_dums <- cbind(impdat_dums, mdums)
write.csv(impdat_dums, file = paste0(outdir, 'impdat_dums.csv'))

# table
impdat <- read.csv(paste0(outdir, 'impdat_dums.csv'))
colnames(impdat) <- gsub(" |[.]", "_", colnames(impdat)) %>% tolower
colnames(missdat) <- gsub(" |[.]", "_", colnames(missdat)) %>% tolower

colnames(impdat) %in% colnames(missdat)
colnames(impdat)[colnames(impdat) %ni% colnames(missdat)]
missdat <- as.data.frame(missdat)

head(impdat)
sumstats <- foreach(i = colnames(impdat)[4:ncol(impdat)], .combine = 'rbind') %do% {
  m <- data.frame(term = i, 
                  nmiss = sum(is.na(missdat[[i]])),
                  mean = mean(missdat[[i]], na.rm=T),
                  impmean = mean(impdat[[i]]))
  m
}

sumstats$impmean <- prettyNum(signif(sumstats$impmean,2))
sumstats$mean <- prettyNum(signif(sumstats$mean,2))
write.csv(sumstats, paste0(figdir, "sumstats_imp.csv"))

# what is up with all of the missing encounters?
df[is.na(df$n_encs),] %>% head

df[df$PAT_ID == "000059949",] %>% dim
