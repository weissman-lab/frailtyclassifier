
library(ranger)
library(doParallel)
library(foreach)
library(dplyr)
library(glmnet)
library(readr)
registerDoParallel(detectCores())
`%ni%` <- Negate(`%in%`)
library(reticulate)
if (grepl("crandrew", getwd())){
  use_python("/Users/crandrew/.conda/envs/GW_PAIR_frailty_classifier/bin/python")  
  datadir <- "~/projects/GW_PAIR_frailty_classifier/data/"
  outdir <- "~/projects/GW_PAIR_frailty_classifier/output/"
  figdir <- "~/projects/GW_PAIR_frailty_classifier/figures/"
} else {
  use_python("/home/andrewcd/anaconda3/envs/frailty/bin/python")  
  datadir <- "/media/drv2/andrewcd2/frailty/data/"
  outdir <- "/media/drv2/andrewcd2/frailty/output/"
  figdir <- "/media/drv2/andrewcd2/frailty/figures/"
}

pd <- import("pandas")

# load all the datasets that we'll be testing
ds <- list.files(outdir, pattern = "batch")
yvars <- c("Msk_prob","Nutrition","Resp_imp", "Fall_risk", "Frailty_nos")

df <- read_csv(paste0(outdir, ds[grepl("batch3", ds)]))
df <- as.data.frame(df)

# load the structured data
# first parse the note names to get patient ID's and months
notes <- unique(df$note)
mm <- data.frame(PAT_ID = strsplit(notes, "_") %>% 
                   sapply(function(x){x[4]}),
                 month = strsplit(notes, "_") %>% 
                   sapply(function(x){x[3]}) %>% 
                   gsub(pattern = 'm', replacement = '') %>% 
                   as.numeric,
                 note = notes
)


strdat = list.files(outdir, pattern = "_6m")
for (i in 1:length(strdat)){
  strd <- pd$read_pickle(paste0(outdir, strdat[i]))
  strd$PAT_ENC_CSN_ID <- NULL
  mm <- mm %>% left_join(strd)
  print(dim(mm))
}

# deal with missing data
for (i in 1:ncol(mm)){
  if (any(is.na(mm[,i]))){
    mm[[paste0(colnames(mm)[i], "_missing")]] <- ifelse(is.na(mm[,i]), 1, 0)
    mm[is.na(mm[,i]), i] <- 0
  }
}

df <- inner_join(df, mm)

# identify the feature columns
features <- c(colnames(df)[grepl("wmean|max_|min_|identity|lag", colnames(df))],
              colnames(mm)[4:ncol(mm)]
)

for (y in yvars){
  for (mod in c('rf', 'ridge', 'lasso')){
    toadd <- matrix(rep(NA, nrow(df)*2), ncol = 2)
    colnames(toadd) <- paste0(y, "_", c("neg", "pos"), '_', mod)
    df <- cbind(df, toadd)
  }
}

Xval <- df[, features]

# random forest
rflist <- readRDS(paste0(outdir, "RF_model_list_march_3.rds"))
for (y in yvars){
  pred <- predict(rflist[[y]], Xval)$predictions
  df[, paste0(y,"_pos_rf")] <- pred[,3]
  df[, paste0(y,"_neg_rf")] <- pred[,1]    
}

# ridge
ridgelist <- readRDS(paste0(outdir, "logitlist_ridge.rds"))
for (y in yvars){
  pred <- predict(ridgelist[[y]], as.matrix(Xval), type = 'response')
  df[, paste0(y,"_pos_ridge")] <- pred[,3,1]
  df[, paste0(y,"_neg_ridge")] <- pred[,1,1]    
}

# ridge
lassolist <- readRDS(paste0(outdir, "logitlist_lasso.rds"))
for (y in yvars){
  pred <- predict(lassolist[[y]], as.matrix(Xval), type = 'response')
  df[, paste0(y,"_pos_lasso")] <- pred[,3,1]
  df[, paste0(y,"_neg_lasso")] <- pred[,1,1]    
}

preddf <- df[,colnames(df) %ni% features]
colnames(preddf)
write.csv(preddf, file = paste0(outdir, "batch3_OOSpreds.csv"))

# RMSE

rmselist <- foreach(mod = c("rf", 'ridge', 'lasso'), .combine = rbind) %do% {
  yhat <- foreach(y = yvars, .combine = cbind) %do% {
    yhat <- df[,paste0(y, c("_neg", "_pos"), "_", mod)]
    yhat[[paste0(y, "_neutral_",mod)]] <- 1 - rowSums(yhat)
    return(yhat)
  }
  ymat_all <- foreach(y = yvars, .combine = cbind) %do% {
    ymat <- model.matrix(as.formula(paste0("~as.factor(", y, ")-1")), data = df)
    ymat <- ymat[,c(1,3,2)]
    return(ymat)
  }
  ymat_rare <- foreach(y = yvars, .combine = cbind) %do% {
    ymat <- model.matrix(as.formula(paste0("~as.factor(", y, ")-1")), data = df)
    ymat <- ymat[,c(1,3,2)]
    ymat[which(ymat[,3]==1),] = NA
    return(ymat)
  }
  rmse_all <- colMeans((yhat - ymat_all)^2)
  rmse_rare <- colMeans((yhat - ymat_rare)^2, na.rm=T)
  disag_rmse_rare <- data.frame(mod = mod,
                           aspect = strsplit(names(rmse_rare), "_") %>% sapply(function(x){paste0(x[1:2], collapse = "_")}),
                           type = "rare",
                           class = strsplit(names(rmse_rare), "_") %>% sapply(function(x){paste0(x[3], collapse = "_")}),
                           value = rmse_rare,
                           stringsAsFactors = F
                           )
  disag_rmse_all <- data.frame(mod = mod,
                                aspect = strsplit(names(rmse_all), "_") %>% sapply(function(x){paste0(x[1], collapse = "_")}),
                                type = "rare",
                                class = strsplit(names(rmse_all), "_") %>% sapply(function(x){paste0(x[length(x)-1], collapse = "_")}),
                                value = rmse_all,
                               stringsAsFactors = F
  )
  comb_rmse <- foreach(y = yvars, .combine = rbind) %dopar%{
    yh <- yhat[,grep(y, colnames(yhat))]
    ym <- model.matrix(as.formula(paste0("~as.factor(", y, ")-1")), data = df)
    ym <- ym[,c(1,3,2)]
    rmse_all <- (sum((yh-ym)^2)/3/nrow(ym))^.5
    yrare <- ym[ym[,3]==0,]
    prare <- yh[ym[,3]==0,]
    rmse_rare <- (sum((prare-yrare)^2)/3/nrow(yrare))^.5
    return(data.frame(mod = mod, aspect = y, type = c("all", "rare"), class = 'combined', value = c(rmse_all, rmse_rare), stringsAsFactors = F))
  }
  # all df

  return(bind_rows(disag_rmse_all, disag_rmse_rare, comb_rmse))
}
write.csv(rmselist, file = paste0(outdir, "batch3_rmse_stats.csv"))

m = lm(value~mod, data = subset(rmselist, class = 'combined'))
summary(m)

outrmse = data.frame(model = NA, aspect = NA, type = NA)

names(rmselist) <- c("rf", 'ridge', 'lasso')
rmselist$rf$comb_rmse %>% unlist


sapply(rmselist[['rf']], function(x){x$comb_rmse})
- sapply(ridgelist, function(x){x$rmse_all})




