library(ranger)
library(Matrix)
workdir <- paste0(getwd(), '/')
outdir <- paste0(getwd(), '/')

tr_tfidf <- as(Matrix::readMM(paste0(workdir,'tr_df_tfidf.mtx')), "dgCMatrix")
tr_tfidf <- provideDimnames(tr_tfidf)
tr_df <- data.table::fread(paste0(workdir,'tr_df.csv'))

te_tfidf <- as(Matrix::readMM(paste0(workdir,'te_df_tfidf.mtx')), "dgCMatrix")
te_tfidf <- provideDimnames(te_tfidf)
te_df <- data.table::fread(paste0(workdir,'te_df.csv'))

#fit rf
tr_tfidf_rg <- ranger(y = as.factor(tr_df$Resp_imp_1),
                      x= tr_tfidf,
                      probability = TRUE,
                      num.trees = 10)

#Macro? Brier score
print(tr_tfidf_rg)

#make predictions
te_tfidf_pred <- predict(tr_tfidf_rg, data=te_tfidf)$predictions

dim(te_tfidf)
dim(te_tfidf_pred)

#Function for calculating Brier scores
brier_score <- function(obs, pred) {
  mean((obs - pred)^2)}

#compute microaveraged brier score






#Andrew's code:

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
df1 <- read_csv(paste0(outdir, ds[grepl("batch1", ds)]))
df2 <- read_csv(paste0(outdir, ds[grepl("batch2", ds)]))
df <- rbind(df1, df2)
df <- as.data.frame(df)

set.seed(8675309)
notes = df$note %>% unique %>% sample(replace = F)

# load the structured data
# first parse the note names to get patient ID's and months
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
# augment the data frame with predicted probability columns
for (y in yvars){
  toadd <- matrix(rep(NA, nrow(df)*2), ncol = 2)
  colnames(toadd) <- paste0(y, "_", c("neg", "pos"))
  df <- cbind(df, toadd)
}

# using all the data
modlist <- foreach(y = yvars) %do% {
  m = ranger(y = factor(df[[y]]), x = df[,features],
             num.threads = detectCores(),
             probability = TRUE,
             verbose = F)
  return(m)
}
names(modlist) <- yvars
saveRDS(modlist, paste0(outdir, "RF_model_list_march_3.rds"))
# for OOS comparisons
for (i in 1:10){
  print(((i-1)*5+1):(i*5))
  te <- notes[((i-1)*5+1):(i*5+1)]
  tr <- unique(notes[notes %ni% te])
  
  Xtr = df[df$note %in% tr, features]
  Xte = df[df$note %in% te, features]
  
  tridx <- which(df$note %in% tr)
  teidx <- which(df$note %in% te)
  
  for (y in yvars){
    m = ranger(y = factor(df[tridx, y]), x = Xtr,
               num.threads = detectCores(),
               probability = TRUE,
               verbose = F)
    pred <- predict(m, Xte)$predictions
    df[teidx, paste0(y,"_pos")] <- pred[,3]
    df[teidx, paste0(y,"_neg")] <- pred[,1]    
  }
}
write.csv(df[,!grepl("wmean|max_|min_|identity|lag", colnames(df))], file = paste0(outdir, "batch2_OOSpreds.csv"))
############################
# logit -- ridge
# first specify the folds by note, just as was done for the RFs
foldvec <- rep(NA, nrow(df))
for (i in 1:10){
  te <- notes[((i-1)*5+1):(i*5+1)]
  foldvec[df$note %in% te] <- i
}
modlist <- foreach(y = yvars, .errorhandling = 'pass') %do% {
  m <- cv.glmnet(x = as.matrix(df[,features]), y = df[, y], 
                 family = 'multinomial', foldid = foldvec,
                 parallel = TRUE, alpha = 0)
  plot(m)
  pred = predict(m, newx = as.matrix(df[,features]), 
                 type = "response")[,c("-1", "0", "1"), 1]
  ymat = model.matrix(~.-1, data = data.frame(y = factor(df[, y])))
  colnames(ymat) <- gsub("y", "", colnames(ymat))
  ymat <- ymat[,colnames(pred)]
  m$rmse_all <- (sum((pred-ymat)^2)/3/nrow(ymat))^.5
  yrare <- ymat[ymat[,"0"]==0,]
  prare <- pred[ymat[,"0"]==0,]
  m$rmse_rare <- (sum((prare-yrare)^2)/3/nrow(yrare))^.5
  return(m)
}
names(modlist) <- yvars
saveRDS(modlist, paste0(outdir, "logitlist_ridge.rds"))
# same thing for lasso
modlist <- foreach(y = yvars, .errorhandling = 'pass') %do% {
  m <- cv.glmnet(x = as.matrix(df[,features]), y = df[, y], 
                 family = 'multinomial', foldid = foldvec,
                 parallel = TRUE, alpha = 1)
  plot(m)
  pred = predict(m, newx = as.matrix(df[,features]), 
                 type = "response")[,c("-1", "0", "1"), 1]
  ymat = model.matrix(~.-1, data = data.frame(y = factor(df[, y])))
  colnames(ymat) <- gsub("y", "", colnames(ymat))
  ymat <- ymat[,colnames(pred)]
  
  m$rmse_all <- (sum((pred-ymat)^2)/3/nrow(ymat))^.5
  yrare <- ymat[ymat[,"0"]==0,]
  prare <- pred[ymat[,"0"]==0,]
  m$rmse_rare <- (sum((prare-yrare)^2)/3/nrow(yrare))^.5
  return(m)
}
names(modlist) <- yvars
saveRDS(modlist, paste0(outdir, "logitlist_lasso.rds"))
