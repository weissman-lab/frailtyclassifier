
library(ranger)
library(doParallel)
library(foreach)
library(dplyr)
library(readr)
registerDoParallel(detectCores())
`%ni%` <- Negate(`%in%`)

outdir <- "~/projects/GW_PAIR_frailty_classifier/output/"
figdir <- "~/projects/GW_PAIR_frailty_classifier/figures/"
if (Sys.info()['nodename'] == 'grace') {
  outdir <- "/media/drv2/andrewcd2/frailty/output/"
  figdir <- "/media/drv2/andrewcd2/frailty/figures/"
}
# load all the datasets that we'll be testing
ds <- list.files(outdir, pattern = "test_data_")
yvars <- c("Msk_prob","Nutrition","Resp_imp", "Fall_risk")
nboot = 100
####################################
# loop through training and test set definitions
bootreps <- foreach(b = 1:nboot, .errorhandling = 'remove') %do%{
  set.seed(b, kind = "L'Ecuyer-CMRG")
  ##################################
  # loop through datasets
  ds_loop <- foreach(d = ds, .combine = cbind, .errorhandling = 'remove') %do% {
    df <- read_csv(paste0(outdir, d))
    df <- as.data.frame(df) # get rid of that tidyverse nonsense that makes my code break
    tr <- sample(unique(df$note), 20, replace = F)
    te <- sample(unique(df$note)[unique(df$note) %ni% tr], 5, replace = F)
    
    Xtr = df[df$note %in% tr, grepl("wmean|max_|min_|identity|lag", colnames(df))]
    Xte = df[df$note %in% te, grepl("wmean|max_|min_|identity|lag", colnames(df))]
    
    tridx <- which(df$note %in% tr)
    teidx <- which(df$note %in% te)
    
    # loop through variables
    stats_by_mod <- foreach(y = yvars, .errorhandling = 'remove', .combine = cbind) %do% {
      print(paste("Doing boot", b, "model", y, "dataset", d))
      m = ranger(y = factor(df[tridx, y]), x = Xtr,
                 num.threads = detectCores(),
                 probability = TRUE,
                 verbose = F)
      pred <- predict(m, Xte)$predictions
      ymat = model.matrix(~.-1, data = data.frame(y = factor(df[teidx, y])))
      colnames(ymat) <- gsub("y", "", colnames(ymat))
      ymat <- ymat[,colnames(pred)]
      rmse_all <- (sum((pred-ymat)^2)/3/nrow(ymat))^.5
      yrare <- ymat[ymat[,"0"]==0,]
      prare <- pred[ymat[,"0"]==0,]
      rmse_rare <- (sum((prare-yrare)^2)/3/nrow(yrare))^.5
      out = data.frame(b = b, ds = d, y = y, rmse_rare = rmse_rare, rmse_all = rmse_all)
      print(out)
      write.table(out, file = paste0(outdir, "incremental_output.csv"), sep = ",", 
                  col.names = !file.exists(paste0(outdir, "incremental_output.csv")), 
                  append = T)
      return(out)
    }  
    return(stats_by_mod)
  }
  print(d)
  print(summary(bootreps))
  return(bootreps)
}
write_csv(ds_loop, paste0(outdir, "embedding_selection_output.csv"))

# df <- read.csv(paste0(outdir, "incremental_output.csv"))
# df$bw <- ifelse(grepl("bw5", df$ds), "5", "30")
# df$corp <- ifelse(grepl("_oa_", df$ds, ignore.case = T), "OA", "UP")
# df$type = ifelse(grepl("w2v", df$ds), "w2v", "ft")
# df$dim <- strsplit(as.character(df$ds), "_") %>% 
#   sapply(function(x){x[length(x)-1]}) %>% 
#   gsub(pattern = "d", replacement = "") %>% 
#   as.numeric()
# 
# 
# m <- lm(rmse_rare~bw+corp+type+factor(dim)+y, data = df)
# summary(m)
# m <- lm(rmse_all~bw+corp+type+factor(dim)+y, data = df)
# summary(m)
# 
# 
# 
