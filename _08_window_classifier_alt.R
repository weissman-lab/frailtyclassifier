library(data.table)
library(ranger)
library(dplyr)
library(tidyr)
library(ggplot2)
library(gmish)
library(doParallel)
registerDoParallel(detectCores())

knitr::opts_chunk$set(echo = FALSE)

workdir <- paste0(getwd(), '/')
outdir <- paste0(getwd(), '/')


#brier score function
brier_score <- function(obs, pred) {
  mean((obs - pred)^2)
}

seed = 92120
folds <- seq(1, 10)
svd <- c(50, 300, 1000)
frail_lab <- c('Resp_imp', 'Msk_prob', 'Fall_risk', 'Nutrition')

#load data (structured data & text with windowing)
assign(paste0('f', folds[1], '_tr'), fread(paste0(outdir, 'f', folds[1], '_tr_df.csv')))
assign(paste0('f', folds[1], '_te'), fread(paste0(outdir, 'f', folds[1], '_te_df.csv')))

#load caseweights (weight non-neutral tokens by the inverse of their prevalence)
#e.g. 1.3% of fall_risk tokens are non-neutral. Therefore, non-neutral tokens are weighted * (1/0.013)
assign(paste0('f', folds[1], '_tr_cw'), fread(paste0(outdir, 'f', folds[1], '_tr_cw.csv')))


#full hyper grid
# hyper_grid <- expand.grid(
#   ntree      = signif(seq(10, 500, length.out = 5), 0),
#   mtry       = seq(2, 50, length.out = 5),
#   node_size  = signif(seq(1, 15, length.out = 5), 1)
# )

#small hyper grid
hyper_grid <- expand.grid(
  ntree      = signif(seq(2, 5, length.out = 2), 0),
  mtry       = seq(2, 5, length.out = 2),
  node_size  = signif(seq(1, 2, length.out = 2), 1)
)

#med hyper grid
# hyper_grid <- expand.grid(
#   ntree      = signif(seq(10, 500, length.out = 5), 0),
#   mtry       = seq(2, 50, length.out = 3),
#   node_size  = signif(seq(1, 15, length.out = 3), 1)
# )


hyper_grid5 = data.frame()

for (f in 1:length(frail_lab)) {
  
  for(s in 1:length(svd)) {
    
    #load truncated SVD of tf-idf of text & remove first row (error)
    assign(paste0('f', folds[1], '_tr_svd', svd[s]), fread(paste0(outdir, 'f', folds[1], '_tr_svd', svd[s], '.csv'), skip = 1))
    assign(paste0('f', folds[1], '_te_svd', svd[s]), fread(paste0(outdir, 'f', folds[1], '_te_svd', svd[s], '.csv'), skip = 1))
    
    for(i in 1:nrow(hyper_grid)) {
      
      #get matching training and test data
      x_train <- get(paste0('f', folds[1], '_tr_svd', svd[s]))
      x_test <- get(paste0('f', folds[1], '_te_svd', svd[s]))
      y_train <- f1_tr[[paste0(frail_lab[f])]]
      y_test_neut <- f1_te[[paste0(frail_lab[f], '_0')]]
      y_test_pos <- f1_te[[paste0(frail_lab[f], '_1')]]
      y_test_neg <- f1_te[[paste0(frail_lab[f], '_-1')]]
      #get matching caseweights
      cw <- get(paste0('f', folds[1], '_tr_cw'))[[paste0(frail_lab[f], '_cw')]]
      
      frail_rf <- ranger(y = factor(y_train, levels = c(0, 1, -1)), #relevel factors to match class.weights
                         x = select(x_train, -1),
                         num.threads = detectCores(),
                         probability = TRUE,
                         num.trees = hyper_grid$ntree[i],
                         mtry = hyper_grid$mtry[i],
                         min.node.size = hyper_grid$node_size[i],
                         #implemented class.weights rather than case.weights
                         #in order of outcome factor levels
                         class.weights =
                           as.integer(c(levels(factor(cw))[1], levels(factor(cw))[2], levels(factor(cw))[2])),
                         #later, can set oob.error=FALSE to save time/memory (using CV error)
                         oob.error = TRUE,
                         seed = seed)
      
      #oob brier score for all classes
      hyper_grid$oob_brier[i] <- frail_rf$prediction.error
      
      #make predictions on test fold
      preds <- predict(frail_rf, data=x_test)$predictions
      
      #cv brier score for all classes (currently fold 1)
      hyper_grid$cv_brier_all[i] <- brier_score(
        c(y_test_neut, y_test_pos, y_test_neg), 
        c(preds[,'0'], preds[,'1'], preds[,'-1']))
      
      #cv brier score for each class (currently fold 1)
      hyper_grid$cv_brier_neut[i] <- brier_score(y_test_neut, preds[,'0'])
      
      hyper_grid$cv_brier_pos[i] <- brier_score(y_test_pos, preds[,'1'])
      
      hyper_grid$cv_brier_neg[i] <- brier_score(y_test_neg, preds[,'-1'])
    }
    
    hyper_grid2 <- hyper_grid
    hyper_grid2$SVD <- svd[s]
    
    #start building the hyper_grid for the current loop
    if (exists(paste0('hyper_grid_', f)) == FALSE) {
      assign(paste0('hyper_grid_', f), hyper_grid2)
    } else {
      #add new results from each svd loop
      assign(paste0('hyper_grid_', f), rbind(get(paste0('hyper_grid_', f)), hyper_grid2))
    }
    
  }
  
  #add frail aspect label
  hyper_grid4 <- get(paste0('hyper_grid_', f))
  hyper_grid4$frail_lab <- frail_lab[f]
  
  #probably need to do same as above (if else statement) but for each fold
  hyper_grid5 <- rbind(hyper_grid5, hyper_grid4)
  
  #save
  #saveRDS(hyper_grid5, paste0(outdir, 'hyper_grid_f', folds[1], '.rda'))
}
