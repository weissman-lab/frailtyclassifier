
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


######################
# GOF plots

preddf <- read_csv(paste0(outdir, "batch2_OOSpreds.csv"))

emat = matrix(rep(NA, 5*50), nrow = 50)
colnames(emat) <- yvars
for (y in yvars){
  # show goodness of fit
  yhat <- preddf[,paste0(y, c("_neg", "_pos"))]
  yhat[[paste0(y, "_neutral")]] <- 1 - rowSums(yhat)
  yhat <- yhat + cbind(rep(.001, nrow(yhat)),
                       rep(.001, nrow(yhat)),
                       rep(-.001, nrow(yhat)))

  entropy <- apply(yhat, 1, function(x){-sum(x*log(x))})
  par(mar = c(5.1, 4.1, 2.1, 4.1))
  plot(x = which(preddf[,y]==1),
       y = rep(1.02, length(which(df[,y]==1))),
       ylim = c(0,2), xlim = c(0, nrow(preddf)),
       pch = 15, col = "red",
       xlab = "token position in concatenated corpus",
       ylab = "",
       yaxt = 'n',
       main = gsub("_", " ", y))
  points(x = which(preddf[,y]==-1),
         y = rep(.98, length(which(preddf[,y]==-1))),
         pch = 15, col = "darkgreen")
  lines(yhat[[paste0(y,"_pos")]], col = "red")
  lines(yhat[[paste0(y,"_neg")]], col = "darkgreen")
  ll =0
  k = 1
  for (i in c(which(preddf[(1:nrow(preddf)), "note"] != preddf[(1:nrow(preddf))+1, "note"]))){
    abline(v = i, lty = 2)
    ee <- mean(entropy[ll:i], na.rm = T)
    print(ee)
    emat[k, y] <- ee
    text(x = mean(c(ll, i)), y = 1.7, labels = round(mean(entropy[ll:i]),3), col = 'blue')
    ll = i
    k = k+1
  } # do the last one
  text(x = mean(c(ll, nrow(preddf))), y = 1.7, labels = round(mean(entropy[ll:nrow(preddf)]),3), col = 'blue')
  
  lines(entropy+1, col = "blue")
  abline(h=1)
  axis(side = 2, at = c(0, .5, 1))
  axis(side = 4, at = c(0, .5, 1)+1, labels = c(0, .5, 1), col = 'blue', col.axis = 'blue')
  mtext("Estimated probability", side = 2, line = 2, at = .5)
  mtext("Entropy", side = 4, line = 2, at = 1.5, srt = 90)
  dev.copy2pdf(file = paste0(figdir, "batch2_RF_CV_GOF_", y, '.pdf'), height = 6, width = 8*10)
}

# compute RMSE
yhat <- foreach(y = yvars, .combine = cbind) %do% {
  yhat <- preddf[,paste0(y, c("_neg", "_pos"))]
  yhat[[paste0(y, "_neutral")]] <- 1 - rowSums(yhat)
  return(yhat)
}
ymat_all <- foreach(y = yvars, .combine = cbind) %do% {
  ymat <- model.matrix(as.formula(paste0("~as.factor(", y, ")-1")), data = preddf)
  ymat <- ymat[,c(1,3,2)]
  return(ymat)
}
ymat_rare <- foreach(y = yvars, .combine = cbind) %do% {
  ymat <- model.matrix(as.formula(paste0("~as.factor(", y, ")-1")), data = preddf)
  ymat <- ymat[,c(1,3,2)]
  ymat[which(ymat[,3]==1),] = NA
  return(ymat)
}
rmse_all <- colMeans((yhat - ymat_all)^2)
rmse_rare <- colMeans((yhat - ymat_rare)^2, na.rm=T)
comb_rmse <- foreach(y = yvars) %do%{
  yh <- yhat[,grep(y, colnames(yhat))]
  ym <- model.matrix(as.formula(paste0("~as.factor(", y, ")-1")), data = preddf)
  ym <- ym[,c(1,3,2)]
  rmse_all <- (sum((yh-ym)^2)/3/nrow(ym))^.5
  yrare <- ym[ym[,3]==0,]
  prare <- yh[ym[,3]==0,]
  rmse_rare <- (sum((prare-yrare)^2)/3/nrow(yrare))^.5
  return(list(rmse_all = rmse_all, rmse_rare = rmse_rare))
}
names(comb_rmse) = yvars
comb_rmse



# ridge
ridgelist <- readRDS(paste0(outdir, "logitlist_ridge.rds"))
sapply(ridgelist, function(x){x$rmse_all})
sapply(ridgelist, function(x){x$rmse_rare})

# lasso
lassolist <- readRDS(paste0(outdir, "logitlist_lasso.rds"))
sapply(lassolist, function(x){x$rmse_all})
sapply(lassolist, function(x){x$rmse_rare})

#ridge minus lasso
sapply(ridgelist, function(x){x$rmse_all}) - sapply(lassolist, function(x){x$rmse_all})
sapply(ridgelist, function(x){x$rmse_rare}) - sapply(lassolist, function(x){x$rmse_rare})


# RF minus Ridge
sapply(comb_rmse, function(x){x$rmse_all}) - sapply(ridgelist, function(x){x$rmse_all})
sapply(comb_rmse, function(x){x$rmse_rare}) - sapply(ridgelist, function(x){x$rmse_rare})

# RF minus lasso
sapply(comb_rmse, function(x){x$rmse_all}) - sapply(lassolist, function(x){x$rmse_all})
sapply(comb_rmse, function(x){x$rmse_rare}) - sapply(lassolist, function(x){x$rmse_rare})



# ##################################
# # loop through datasets
# ds_loop <- foreach(d = ds, .combine = cbind, .errorhandling = 'remove') %do% {
#   df <- read_csv(paste0(outdir, d))
#   ####################################
#   # loop through training and test set definitions
#   bootreps <- foreach(b = 1:nboot, .errorhandling = 'remove') %dopar%{
#     set.seed(b, kind = "L'Ecuyer-CMRG")
#     tr <- sample(unique(df$note), 20, replace = F)
#     te <- sample(unique(df$note)[unique(df$note) %ni% tr], 5, replace = F)
#     
#     Xtr = df[df$note %in% tr, grepl("wmean|max_|min_|identity|lag", colnames(df))]
#     Xte = df[df$note %in% te, grepl("wmean|max_|min_|identity|lag", colnames(df))]
#     
#     tridx <- which(df$note %in% tr)
#     teidx <- which(df$note %in% te)
#     
#     # loop through variables
#     stats_by_mod <- foreach(y = yvars, .errorhandling = 'remove', .combine = cbind) %do% {
#       m = ranger(y = factor(df[tridx, y]), x = Xtr,
#                  num.threads = 1,
#                  probability = TRUE,
#                  verbose = F)
#       pred <- predict(m, Xte)$predictions
#       ymat = model.matrix(~.-1, data = data.frame(y = factor(df[teidx, y])))
#       colnames(ymat) <- gsub("y", "", colnames(ymat))
#       ymat <- ymat[,colnames(pred)]
#       rmse_all <- (sum((pred-ymat)^2)/3/nrow(ymat))^.5
#       yrare <- ymat[ymat[,"0"]==0,]
#       prare <- pred[ymat[,"0"]==0,]
#       rmse_rare <- (sum((prare-yrare)^2)/3/nrow(yrare))^.5
#       print(y)
#       print(m$rmse_rare)
#       print(m$rmse_all)
#       out = data.frame(b = b, ds = ds[i], y = y, rmse_rare = rmse_rare, rmse_all = rmse_all)
#       return(out)
#     }  
#     return(stats_by_mod)
#   }
#   print(d)
#   print(summary(bootreps))
#   return(bootreps)
# }
# write_csv(ds_loop, paste0(outdir, "embedding_selection_output.csv"))
# 
# names(modlist) <- yvars
# saveRDS(modlist, paste0(outdir, "rflist.rds"))
# sapply(modlist, function(x){x$rmse_rare})
# 
# # importance
# 
# 
# emat = matrix(rep(NA, 5*5), 5, 5)
# colnames(emat) <- yvars
# for (y in yvars){
#   # show goodness of fit
#   yhat <- predict(modlist[[y]], Xte)
#   ymat <- yhat$predictions
#   if (ncol(ymat) == 2){
#     ymat <- cbind(ymat, 0)
#     colnames(ymat)[3] = "1"
#   }
#   # post-hoc fix
#   ymat <- ymat + cbind(rep(.001, nrow(ymat)),
#                        rep(-.001, nrow(ymat)),
#                        rep(.001, nrow(ymat)))
#   
#   entropy <- apply(ymat, 1, function(x){-sum(x*log(x))})
#   par(mar = c(5.1, 4.1, 2.1, 4.1))
#   plot(x = which(df[teidx,y]==1),
#        y = rep(1.02, length(which(df[teidx,y]==1))),
#        ylim = c(0,2), xlim = c(0, length(teidx)),
#        pch = 15, col = "red",
#        xlab = "token position in concatenated corpus",
#        ylab = "",
#        yaxt = 'n',
#        main = gsub("_", " ", y))
#   points(x = which(df[teidx,y]==-1),
#          y = rep(.98, length(which(df[teidx,y]==-1))),
#          pch = 15, col = "darkgreen")
#   if ("1" %in% colnames(yhat$predictions)){
#     lines(yhat$predictions[,"1"], col = "red")    
#   }
#   lines(yhat$predictions[,"-1"], col = "darkgreen")
#   ll =0
#   k = 1
#   for (i in c(which(df[teidx, "note"] != df[teidx+1, "note"]))){
#     abline(v = i, lty = 2)
#     ee <- mean(entropy[ll:i], na.rm = T)
#     print(ee)
#     emat[k, y] <- ee
#     text(x = mean(c(ll, i)), y = 1.7, labels = round(mean(entropy[ll:i]),3), col = 'blue')
#     ll = i
#     k = k+1
#   }
#   lines(entropy+1, col = "blue")
#   abline(h=1)
#   axis(side = 2, at = c(0, .5, 1))
#   axis(side = 4, at = c(0, .5, 1)+1, labels = c(0, .5, 1), col = 'blue', col.axis = 'blue')
#   mtext("Estimated probability", side = 2, line = 2, at = .5)
#   mtext("Entropy", side = 4, line = 2, at = 1.5, srt = 90)
#   dev.copy2pdf(file = paste0(figdir, "exploratory_", y, '.pdf'), height = 6, width = 8)
# }
# 
# # grouped importance
# y = yvars[2]
# cols <- colnames(Xte)[grepl("max_", colnames(Xte))]
# getimps <- function(y, cols, N, threads_per_imp = 1){
#   m = modlist[[y]]
#   pred <- predict(m, Xte)$predictions
#   ymat = model.matrix(~.-1, data = data.frame(y = factor(df[teidx, y])))
#   colnames(ymat) <- gsub("y", "", colnames(ymat))
#   ymat <- ymat[,colnames(pred)]
#   rmse_all <- (sum((pred-ymat)^2)/3/nrow(ymat))^.5
#   yrare <- ymat[ymat[,"0"]==0,]
#   prare <- pred[ymat[,"0"]==0,]
#   rmse_rare <- (sum((prare-yrare)^2)/3/nrow(yrare))^.5
#   
#   cols_to_perm <- Xte[,cols]
#   imps <- foreach(i = 1:N, .combine = c, .errorhandling = "remove") %dopar% {
#     Xperm <- Xte
#     Xperm[,cols] <- cols_to_perm[sample(1:nrow(cols_to_perm), replace = FALSE),]
#     predperm <- predict(m, Xperm, num.threads = threads_per_imp)$predictions
#     # rmse_all_perm <- (sum((predperm-ymat)^2)/3/nrow(ymat))^.5
#     ppermrare <- predperm[ymat[,"0"]==0,]
#     rmse_rare_perm <- (sum((ppermrare-yrare)^2)/3/nrow(yrare))^.5    
#     return(rmse_rare_perm)
#   }
#   mu <- mean(imps) - rmse_rare
#   sig <- sd(imps - rmse_rare)
#   return(list(mu = mu, sig = sig, len = length(imps)))
# }
# 
# implist <- foreach(y = yvars) %do% {
#   out = list(
#     id_imp = getimps(y, colnames(Xte)[grepl("idenity", colnames(Xte))], 100, 2),
#     lag_imp = getimps(y, colnames(Xte)[grepl("lag", colnames(Xte))], 100, 2),
#     minmax_imp = getimps(y, colnames(Xte)[grepl("min_|max_", colnames(Xte))], 100, 2),
#     wmean_imp = getimps(y, colnames(Xte)[grepl("wmean_", colnames(Xte))], 100, 2)
#   )
#   print(out)
#   return(out)  
# }
# 
# 
# sapply(implist, function(x){x$id_imp$mu})
# sapply(implist, function(x){x$wmean_imp$mu})
# 
# 
# print(y)
# print(m$rmse_rare)
# print(m$rmse_all)
# 
# 
# 
# 
# 
# emat
# # picking new examples based on average entropy
# # weighting by rmse of rare class
# rmse_rare <- sapply(modlist, function(x){x$rmse_rare})
# apply(emat, 1, weighted.mean, weights = rmse_rare)
# # picking new models based on inconsistency
# ydf <- df[,yvars]
# ydf <- data.frame(sapply(ydf, as.factor))
# ytr <- model.matrix(~.-1, data = ydf)
# pca <- prcomp(ytr, scale. = TRUE, center = TRUE)
# rot <- ytr %*% pca$rotation[,1]
# plot(density(rot))
# 
# yhat <- foreach(y = yvars, .combine = cbind) %do% {
#   yhat <- predict(modlist[[y]], Xte)
#   ymat <- yhat$predictions
#   if (ncol(ymat) == 2){
#     ymat <- cbind(ymat, 0)
#     colnames(ymat)[3] = "1"
#   }
#   # post-hoc fix
#   ymat <- ymat + cbind(rep(.001, nrow(ymat)),
#                        rep(-.001, nrow(ymat)),
#                        rep(.001, nrow(ymat)))
#   return(ymat)
# }
# dim(ytr)
# 
# 
# x = c(0,1)
# y = c(1,0)
# 
# p = c(.01, .98, .01)
# -sum(p*log(p))
# 
# e <- function(p){
#   -sum(p*log(p))
# }
# e(c(.01, .98, .01))
# e(c(.33, .33, .33))
# e(c(.1, .88, .01))
# 
# crossprod(x, y)/(crossprod(x,x))
# 
# 
# -sum(c(.33, .33, .33)*log(c(.33, .33, .33)))
# -sum(c(.8, .1, .1)*log(c(.8, .1, .1)))
# 
# # lines(entropy/5+.4, col = "blue")
# -.33*log(.33)
# sum(-rep(.33, 3)*log(rep(.33, 3)))
# 
# hist(entropy)
# 
# 
# ifelse(df[teidx,"Fall_risk"]==0, 1, 0), col = 'blue')
# 
# plot(ifelse(df[teidx,"Fall_risk"]==-1, 1, 0), col = 'red')
# lines(yhat$predictions[,"-1"], col = "red")
# #
# points(ifelse(df[teidx,"Fall_risk"]==0, 1, 0), col = 'blue')
# lines(yhat$predictions[,"0"], col = "blue")
# #
# points(ifelse(df[teidx,"Fall_risk"]==1, 1, 0), col = 'darkgreen')
# lines(yhat$predictions[,"1"], col = "darkgreen")
# 
# head(yhat)
# modlist[['Fall_risk']]$predictions
# 
# 
# # compute average entropy of new notes
# 
# # weight it by
# 
# # logit
# y = "Resp_imp"
# modlist <- foreach(y = yvars, .errorhandling = 'pass') %do% {
#   unique(as.numeric((df[tridx, "note"])))
#   sort(unique(as.numeric(factor(df[tridx, "note"]))))
#   
#   m <- cv.glmnet(x = as.matrix(Xtr), y = df[tridx, y], family = 'multinomial', foldid = as.numeric(factor(df[tridx, "note"])),
#                  parallel = TRUE, alpha = 0)
#   plot(m)
#   pred = predict(m, newx = as.matrix(Xte), type = "response")[,c("-1", "0", "1"), 1]
#   ymat = model.matrix(~.-1, data = data.frame(y = factor(df[teidx, y])))
#   colnames(ymat) <- gsub("y", "", colnames(ymat))
#   ymat <- ymat[,colnames(pred)]
#   dim(ymat)
#   
#   m$rmse_all <- (sum((pred-ymat)^2)/3/nrow(ymat))^.5
#   yrare <- ymat[ymat[,"0"]==0,]
#   prare <- pred[ymat[,"0"]==0,]
#   m$rmse_rare <- (sum((prare-yrare)^2)/3/nrow(yrare))^.5
#   print(y)
#   print(m$rmse_rare)
#   print(m$rmse_all)
#   return(m)
# }
# names(modlist) <- yvars
# saveRDS(modlist, paste0(outdir, "logitlist.rds"))



