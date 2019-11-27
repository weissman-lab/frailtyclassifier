
# this script does the parallelized identification of last-12-month diagnoses

library(jsonlite)
library(dplyr)
library(foreach)
library(doParallel)
library(readr)
registerDoParallel(detectCores())
datadir <- "/Users/crandrew/projects/GW_PAIR_frailty_classifier/data/"

znotes <- read_csv(paste0(datadir, "znotes.csv"))
znotes$X1 <- NULL
znotes <- znotes[as.Date(znotes$ENTRY_TIME) >= as.Date("2018-01-01"),]
znotes$ENTRY_TIME <- as.Date(znotes$ENTRY_TIME)

dx_df <- read_csv(paste0(datadir, "dx_df.csv"))
dx_df$X1 <- NULL
dx_df <- dx_df[!duplicated(dx_df),]
dx_df$ENTRY_TIME <- as.Date(dx_df$ENTRY_TIME)

PT <- proc.time()
dx_windowed_by_csn <- foreach(i = 1:nrow(dx_df[1:1000,]), .combine = rbind, .errorhandling = "remove") %do% {
  print(i)
  pidi <- znotes$PAT_ID[i]
  timei <- znotes$ENTRY_TIME[i]
  dxi <- dx_df[dx_df$PAT_ID == pidi &
                 timei - dx_df$ENTRY_TIME < 365 &
                 timei - dx_df$ENTRY_TIME >= 0,]
  out = data.frame(
    PAT_ENC_CSN_ID = znotes$PAT_ENC_CSN_ID[i],
    dx_list = paste(dxi$CODE, collapse = ",")
  )
  return(out)  
}
print(proc.time() - PT)

out = list()
for(i in 1:nrow(dx_df[1:1000,])) {
  print(i)
  pidi <- znotes$PAT_ID[i]
  timei <- znotes$ENTRY_TIME[i]
  dxi <- dx_df[dx_df$PAT_ID == pidi &
                 timei - dx_df$ENTRY_TIME < 365 &
                 timei - dx_df$ENTRY_TIME >= 0,]
  out[[i]] = data.frame(
    PAT_ENC_CSN_ID = znotes$PAT_ENC_CSN_ID[i],
    dx_list = paste(dxi$CODE, collapse = ",")
  )

}
