

library(comorbidity)
library(reticulate)
library(tidyverse)
library(foreach)
library(doParallel)
registerDoParallel(detectCores())
outdir <- "/Users/crandrew/projects/GW_PAIR_frailty_classifier/output/"

# set the path the the conda python so that I can use pandas under reticulate
use_condaenv("GW_PAIR_frailty_classifier", conda = "/opt/anaconda3/bin/conda")
pd <- import('pandas')

df <- pd$read_pickle(paste0(outdir, "conc_notes_df.pkl"))
df$combined_notes <- NULL
# loop through the rows of the DF and compute the scores, rather than futzing with reshaping 
e_score <- foreach(i = 1:nrow(df), .errorhandling = "pass") %dopar% {
  foo <- data.frame(id =i, dx = df$dxs[[i]], stringsAsFactors = F)
  comorbidity(x = foo, id = "id", code = "dx", score = "elixhauser", assign0 = FALSE)$score  
}

xx <- sapply(e_score, class)
table(xx) #they're all numeric
plot(df$n_comorb, df$elixhauser) # and they correlate with the number of comorbidities

df$elixhauser <- sapply(e_score, function(x){x})
out <- df[,c("PAT_ID", "LATEST_TIME", "CSNS", "elixhauser")]

write.csv(out, paste0(outdir, "elixhauser_scores.csv"))

