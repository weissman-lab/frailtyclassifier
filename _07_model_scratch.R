
library(ranger)
library(doParallel)
library(foreach)
library(dplyr)
library(glmnet)
registerDoParallel(detectCores())
`%ni%` <- Negate(`%in%`)

outdir <- "~/projects/GW_PAIR_frailty_classifier/output/"
figdir <- "~/projects/GW_PAIR_frailty_classifier/figures/"

df <- read.csv(paste0(outdir, "foo.csv"))

colnames(df)
set.seed(8675309)
tr <- sample(unique(df$note), 20, replace = F)
te <- sample(unique(df$note)[unique(df$note) %ni% tr], 5, replace = F)

Xtr = df[df$note %in% tr, grepl("wmean|max_|min_|idenity|lag", colnames(df))]
Xte = df[df$note %in% te, grepl("wmean|max_|min_|idenity|lag", colnames(df))]

tridx <- which(df$note %in% tr)
teidx <- which(df$note %in% te)

yvars <- c("Frailty_nos", "Msk_prob","Nutrition","Resp_imp", "Fall_risk")

  
# rf
modlist <- foreach(y = yvars) %do% {
  m = ranger(y = factor(df[tridx, y]), x = Xtr,
         num.threads = detectCores(),
         probability = TRUE,
         verbose = T)
  pred <- predict(m, Xte)$predictions
  ymat = model.matrix(~.-1, data = data.frame(y = factor(df[teidx, y])))
  colnames(ymat) <- gsub("y", "", colnames(ymat))
  ymat <- ymat[,colnames(pred)]
  m$rmse_all <- (sum((pred-ymat)^2)/3/nrow(ymat))^.5
  yrare <- ymat[ymat[,"0"]==0,]
  prare <- pred[ymat[,"0"]==0,]
  m$rmse_rare <- (sum((prare-yrare)^2)/3/nrow(yrare))^.5
  print(y)
  print(m$rmse_rare)
  print(m$rmse_all)
  return(m)
}
names(modlist) <- yvars
saveRDS(modlist, paste0(outdir, "rflist.rds"))
sapply(modlist, function(x){x$rmse_rare})

# importance


emat = matrix(rep(NA, 5*5), 5, 5)
colnames(emat) <- yvars
for (y in yvars){
  # show goodness of fit
  yhat <- predict(modlist[[y]], Xte)
  ymat <- yhat$predictions
  if (ncol(ymat) == 2){
    ymat <- cbind(ymat, 0)
    colnames(ymat)[3] = "1"
  }
  # post-hoc fix
  ymat <- ymat + cbind(rep(.001, nrow(ymat)),
                       rep(-.001, nrow(ymat)),
                       rep(.001, nrow(ymat)))
  
  entropy <- apply(ymat, 1, function(x){-sum(x*log(x))})
  par(mar = c(5.1, 4.1, 2.1, 4.1))
  plot(x = which(df[teidx,y]==1),
       y = rep(1.02, length(which(df[teidx,y]==1))),
       ylim = c(0,2), xlim = c(0, length(teidx)),
       pch = 15, col = "red",
       xlab = "token position in concatenated corpus",
       ylab = "",
       yaxt = 'n',
       main = gsub("_", " ", y))
  points(x = which(df[teidx,y]==-1),
         y = rep(.98, length(which(df[teidx,y]==-1))),
         pch = 15, col = "darkgreen")
  if ("1" %in% colnames(yhat$predictions)){
    lines(yhat$predictions[,"1"], col = "red")    
  }
  lines(yhat$predictions[,"-1"], col = "darkgreen")
  ll =0
  k = 1
  for (i in c(which(df[teidx, "note"] != df[teidx+1, "note"]))){
    abline(v = i, lty = 2)
    ee <- mean(entropy[ll:i], na.rm = T)
    print(ee)
    emat[k, y] <- ee
    text(x = mean(c(ll, i)), y = 1.7, labels = round(mean(entropy[ll:i]),3), col = 'blue')
    ll = i
    k = k+1
  }
  lines(entropy+1, col = "blue")
  abline(h=1)
  axis(side = 2, at = c(0, .5, 1))
  axis(side = 4, at = c(0, .5, 1)+1, labels = c(0, .5, 1), col = 'blue', col.axis = 'blue')
  mtext("Estimated probability", side = 2, line = 2, at = .5)
  mtext("Entropy", side = 4, line = 2, at = 1.5, srt = 90)
  dev.copy2pdf(file = paste0(figdir, "exploratory_", y, '.pdf'), height = 6, width = 8)
}

# grouped importance
y = yvars[2]
cols <- colnames(Xte)[grepl("max_", colnames(Xte))]
getimps <- function(y, cols, N, threads_per_imp = 1){
  m = modlist[[y]]
  pred <- predict(m, Xte)$predictions
  ymat = model.matrix(~.-1, data = data.frame(y = factor(df[teidx, y])))
  colnames(ymat) <- gsub("y", "", colnames(ymat))
  ymat <- ymat[,colnames(pred)]
  rmse_all <- (sum((pred-ymat)^2)/3/nrow(ymat))^.5
  yrare <- ymat[ymat[,"0"]==0,]
  prare <- pred[ymat[,"0"]==0,]
  rmse_rare <- (sum((prare-yrare)^2)/3/nrow(yrare))^.5
  
  cols_to_perm <- Xte[,cols]
  imps <- foreach(i = 1:N, .combine = c, .errorhandling = "remove") %dopar% {
    Xperm <- Xte
    Xperm[,cols] <- cols_to_perm[sample(1:nrow(cols_to_perm), replace = FALSE),]
    predperm <- predict(m, Xperm, num.threads = threads_per_imp)$predictions
    # rmse_all_perm <- (sum((predperm-ymat)^2)/3/nrow(ymat))^.5
    ppermrare <- predperm[ymat[,"0"]==0,]
    rmse_rare_perm <- (sum((ppermrare-yrare)^2)/3/nrow(yrare))^.5    
    return(rmse_rare_perm)
  }
  mu <- mean(imps) - rmse_rare
  sig <- sd(imps - rmse_rare)
  return(list(mu = mu, sig = sig, len = length(imps)))
}

implist <- foreach(y = yvars) %do% {
  out = list(
    id_imp = getimps(y, colnames(Xte)[grepl("idenity", colnames(Xte))], 100, 2),
    lag_imp = getimps(y, colnames(Xte)[grepl("lag", colnames(Xte))], 100, 2),
    minmax_imp = getimps(y, colnames(Xte)[grepl("min_|max_", colnames(Xte))], 100, 2),
    wmean_imp = getimps(y, colnames(Xte)[grepl("wmean_", colnames(Xte))], 100, 2)
  )
  print(out)
  return(out)  
}


sapply(implist, function(x){x$id_imp$mu})
sapply(implist, function(x){x$wmean_imp$mu})


print(y)
print(m$rmse_rare)
print(m$rmse_all)





emat
# picking new examples based on average entropy
# weighting by rmse of rare class
rmse_rare <- sapply(modlist, function(x){x$rmse_rare})
apply(emat, 1, weighted.mean, weights = rmse_rare)
# picking new models based on inconsistency
ydf <- df[,yvars]
ydf <- data.frame(sapply(ydf, as.factor))
ytr <- model.matrix(~.-1, data = ydf)
pca <- prcomp(ytr, scale. = TRUE, center = TRUE)
rot <- ytr %*% pca$rotation[,1]
plot(density(rot))

yhat <- foreach(y = yvars, .combine = cbind) %do% {
  yhat <- predict(modlist[[y]], Xte)
  ymat <- yhat$predictions
  if (ncol(ymat) == 2){
    ymat <- cbind(ymat, 0)
    colnames(ymat)[3] = "1"
  }
  # post-hoc fix
  ymat <- ymat + cbind(rep(.001, nrow(ymat)),
                       rep(-.001, nrow(ymat)),
                       rep(.001, nrow(ymat)))
  return(ymat)
}
dim(ytr)


x = c(0,1)
y = c(1,0)

p = c(.01, .98, .01)
-sum(p*log(p))

e <- function(p){
  -sum(p*log(p))
}
e(c(.01, .98, .01))
e(c(.33, .33, .33))
e(c(.1, .88, .01))

crossprod(x, y)/(crossprod(x,x))


-sum(c(.33, .33, .33)*log(c(.33, .33, .33)))
-sum(c(.8, .1, .1)*log(c(.8, .1, .1)))

# lines(entropy/5+.4, col = "blue")
-.33*log(.33)
sum(-rep(.33, 3)*log(rep(.33, 3)))

hist(entropy)


ifelse(df[teidx,"Fall_risk"]==0, 1, 0), col = 'blue')

plot(ifelse(df[teidx,"Fall_risk"]==-1, 1, 0), col = 'red')
lines(yhat$predictions[,"-1"], col = "red")
#
points(ifelse(df[teidx,"Fall_risk"]==0, 1, 0), col = 'blue')
lines(yhat$predictions[,"0"], col = "blue")
#
points(ifelse(df[teidx,"Fall_risk"]==1, 1, 0), col = 'darkgreen')
lines(yhat$predictions[,"1"], col = "darkgreen")

head(yhat)
modlist[['Fall_risk']]$predictions


# compute average entropy of new notes

# weight it by

# logit
y = "Resp_imp"
modlist <- foreach(y = yvars, .errorhandling = 'pass') %do% {
  unique(as.numeric((df[tridx, "note"])))
  sort(unique(as.numeric(factor(df[tridx, "note"]))))
  
  m <- cv.glmnet(x = as.matrix(Xtr), y = df[tridx, y], family = 'multinomial', foldid = as.numeric(factor(df[tridx, "note"])),
                 parallel = TRUE, alpha = 0)
  plot(m)
  pred = predict(m, newx = as.matrix(Xte), type = "response")[,c("-1", "0", "1"), 1]
  ymat = model.matrix(~.-1, data = data.frame(y = factor(df[teidx, y])))
  colnames(ymat) <- gsub("y", "", colnames(ymat))
  ymat <- ymat[,colnames(pred)]
  dim(ymat)
  
  m$rmse_all <- (sum((pred-ymat)^2)/3/nrow(ymat))^.5
  yrare <- ymat[ymat[,"0"]==0,]
  prare <- pred[ymat[,"0"]==0,]
  m$rmse_rare <- (sum((prare-yrare)^2)/3/nrow(yrare))^.5
  print(y)
  print(m$rmse_rare)
  print(m$rmse_all)
  return(m)
}
names(modlist) <- yvars
saveRDS(modlist, paste0(outdir, "logitlist.rds"))