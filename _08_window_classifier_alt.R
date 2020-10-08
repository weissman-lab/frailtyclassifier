library(data.table)
library(ranger)
library(dplyr)
library(tidyr)
library(doParallel)
registerDoParallel(detectCores())

outdir <- paste0(getwd(), '/output/_08_window_classifier_alt/')



#Experiment number (based on date):
exp <- '100820'


#brier score function
brier_score <- function(obs, pred) {
  mean((obs - pred)^2)
}

#scaled brier score function
scaled_brier_score <- function(obs, pred) {
  1 - (brier_score(obs, pred) / brier_score(obs, mean(obs)))
}

#cross-entropy function
cross_entropy_1 <- function(obs, pred){
  plus <- obs * log(pred)
  minus <- (1-obs) * (log (1-pred))
  mat <- cbind(plus, minus)
  mean(rowSums(mat))
}

#re-writing as -ylog(yhat) - (1-y)log(1-yhat)
cross_entropy_2 <- function(obs, pred){
  plus <- -1 * (obs * log(pred))
  minus <- -1 * ((1-obs) * (log (1-pred)))
  mat <- cbind(plus, minus)
  mean(rowSums(mat))
}


seed = 92120
folds <- seq(1, 10)
svd <- c(50, 300, 1000)
frail_lab <- c('Msk_prob', 'Fall_risk', 'Nutrition', 'Resp_imp')


start_time <- Sys.time()

for (d in 1:length(folds)) {
  
  #load data (structured data & text with windowing)
  assign(paste0('f', folds[d], '_tr'), fread(paste0(outdir, 'f_', folds[d], '_tr_df.csv')))
  assign(paste0('f', folds[d], '_te'), fread(paste0(outdir, 'f_', folds[d], '_te_df.csv')))
  
  #load caseweights (weight non-neutral tokens by the inverse of their prevalence)
  #e.g. 1.3% of fall_risk tokens are non-neutral. Therefore, non-neutral tokens are weighted * (1/0.013)
  assign(paste0('f', folds[d], '_tr_cw'), fread(paste0(outdir, 'f_', folds[d], '_tr_cw.csv')))
  
  # #hyper grid
  # hyper_grid <- expand.grid(
  #   ntree           = 300,
  #   mtry            = signif(seq(7, 45, length.out = 4), 2),
  #   sample_frac = signif(seq(0.6, 1, length.out = 3), 1)
  # )
  
  #tree grid
  # hyper_grid <- expand.grid(
  #   ntree      = signif(seq(100, 700, length.out = 4), 0),
  #   mtry       = 20,
  #   node_size  = 10
  # )  
  
  #very small grid
  hyper_grid <- expand.grid(
    ntree           = c(2, 3),
    mtry            = 1,
    sample_frac = .2
  )
  
  for (f in 1:length(frail_lab)) {
    
    for(s in 1:length(svd)) {
      
      #load truncated SVD of tf-idf of text & remove first row and first column (junk)
      assign(paste0('f', folds[d], '_tr_svd', svd[s]), fread(paste0(outdir, 'f_', folds[d], '_tr_svd', svd[s], '.csv'), skip = 1, drop = 1))
      assign(paste0('f', folds[d], '_te_svd', svd[s]), fread(paste0(outdir, 'f_', folds[d], '_te_svd', svd[s], '.csv'), skip = 1, drop = 1))
      
      for(i in 1:nrow(hyper_grid)) {
        #get matching training and test data
        x_train <- get(paste0('f', folds[d], '_tr_svd', svd[s]))
        x_test <- get(paste0('f', folds[d], '_te_svd', svd[s]))
        y_train <- get(paste0('f', folds[d], '_tr'))[[paste0(frail_lab[f])]]
        y_test_neut <- get(paste0('f', folds[d], '_te'))[[paste0(frail_lab[f], '_0')]]
        y_test_pos <- get(paste0('f', folds[d], '_te'))[[paste0(frail_lab[f], '_1')]]
        y_test_neg <- get(paste0('f', folds[d], '_te'))[[paste0(frail_lab[f], '_-1')]]
        y_test <- cbind(y_test_neut, y_test_pos, y_test_neg)
        #get matching caseweights
        cw <- get(paste0('f', folds[d], '_tr_cw'))[[paste0(frail_lab[f], '_cw')]]
        
        frail_rf <- ranger(y = factor(y_train, levels = c(0, 1, -1)), #relevel factors to match class.weights
                           x = select(x_train, -1),
                           num.threads = detectCores(),
                           probability = TRUE,
                           num.trees = hyper_grid$ntree[i],
                           mtry = hyper_grid$mtry[i],
                           sample.fraction = hyper_grid$sample_frac[i],
                           #leaving node size as default
                           #min.node.size = hyper_grid$node_size[i],
                           #class.weights in order of outcome factor levels
                           #class.weights = as.integer(c(levels(factor(cw))[1], levels(factor(cw))[2], levels(factor(cw))[2])),
                           #later, can set oob.error=FALSE to save time/memory (using CV error)
                           #case.weights
                           case.weights = cw,
                           oob.error = TRUE,
                           seed = seed)
        
        
        #oob brier score for all classes
        hyper_grid$oob_brier[i] <- frail_rf$prediction.error
        
        
        #make predictions on test fold
        preds <- predict(frail_rf, data=x_test)$predictions
        
        
        #cv brier score for each class
        hyper_grid$cv_brier_neut[i] <- brier_score(y_test_neut, preds[,'0'])
        hyper_grid$cv_brier_pos[i] <- brier_score(y_test_pos, preds[,'1'])
        hyper_grid$cv_brier_neg[i] <- brier_score(y_test_neg, preds[,'-1'])
        #cv brier score for all classes
        hyper_grid$cv_brier_all[i] <- brier_score(
          c(y_test_neut, y_test_pos, y_test_neg), 
          c(preds[,'0'], preds[,'1'], preds[,'-1']))
        
        
        #cv scaled brier score for each class
        hyper_grid$scaled_brier_neut[i] <- scaled_brier_score(y_test_neut, preds[,'0'])
        hyper_grid$scaled_brier_pos[i] <- scaled_brier_score(y_test_pos, preds[,'1'])
        hyper_grid$scaled_brier_neg[i] <- scaled_brier_score(y_test_neg, preds[,'-1'])
        #mean scaled Brier score for all classes
        hyper_grid$scaled_brier_all[i] <- mean(
          scaled_brier_score(y_test_neut, preds[,'0']),
          scaled_brier_score(y_test_pos, preds[,'1']),
          scaled_brier_score(y_test_neg, preds[,'-1']))
        
        
        #calculate cross-entropy
        preds_ce <- preds
        #set floor and ceiling for predictions (predictions of 1 or 0 create entropy of -inf)
        preds_ce[preds_ce==0] <- 1e-3
        preds_ce[preds_ce==1] <- 0.999
        #calculate
        hyper_grid$cross_entropy_2[i] <- cross_entropy_2(y_test, preds_ce)
      }
      
      hyper_grid2 <- hyper_grid
      hyper_grid2$SVD <- svd[s]
      
      #start building the hyper_grid for the current loop
      if (exists(paste0('hyper_grid_d', d, '_f', f)) == FALSE) {
        assign(paste0('hyper_grid_d', d, '_f', f), hyper_grid2)
      } else {
        #add new results from each svd loop
        assign(paste0('hyper_grid_d', d, '_f', f), rbind(get(paste0('hyper_grid_d', d, '_f', f)), hyper_grid2))
      }
    }
    
    #add frail aspect label
    hyper_grid4 <- get(paste0('hyper_grid_d', d, '_f', f))
    hyper_grid4$frail_lab <- frail_lab[f]
    
    if (exists(paste0('hyper_', frail_lab[f], '_fold_', d)) == FALSE) {
      assign(paste0('hyper_', frail_lab[f], '_fold_', d), hyper_grid4)
    } else{
      #add new results from each aspect loop
      assign(paste0('hyper_', frail_lab[f], '_fold_', d), rbind(get(paste0('hyper_', frail_lab[f], '_fold_', d)), hyper_grid4))
    }
    
    #save each fold for each aspect
    write.csv(get(paste0('hyper_', frail_lab[f], '_fold_', d)), paste0(outdir, 'exp', exp, '_hyper_', frail_lab[f], '_fold_', d, '.csv'))
    
    #calculate & save run time for each fold for each aspect
    end_time <- Sys.time()
    duration <- difftime(end_time, start_time, units = 'sec')
    run_time <- paste0('The start time is: ', start_time, '. The end time is: ', end_time, '. Time difference of: ', duration, ' seconds.')
    #save
    write(run_time, paste0(outdir, 'exp', exp, '_duration_hyper_', frail_lab[f], '_fold_', d, '.txt'))
  }
}

#calculate total run time
end_time <- Sys.time()
duration <- difftime(end_time, start_time, units = 'sec')
run_time <- paste0('The start time is: ', start_time, '. The end time is: ', end_time, '. Time difference of: ', duration, ' seconds.')
#save
write(run_time, paste0(outdir, 'exp', exp, '_duration_winclass_alt_r.txt'))




################
# Pick the best hyper-parameters. Then, train RF again and save the predictions (for use in calibration curves).


#function to get mean loss across 10 folds
mean_loss <- function(h) {
  hyper_wide <- pivot_wider(h, id_cols = c('frail_lab', 'mtry', 'sample_frac', 'SVD'), names_from = 'fold',  values_from = 'cross_entropy_2', names_prefix = 'fold')
  hyper_wide2 <- hyper_wide %>%
    mutate(mean_entropy = rowMeans(hyper_wide[,grep('fold', colnames(hyper_wide), value = TRUE)])) %>%
    select(-grep('fold', colnames(hyper_wide)))
  return(hyper_wide2)
}


for (f in 1:length(frail_lab)) {
  
  #get the best hyperparameters for each aspect (avg across 10 folds)
  for (d in 1:length(folds)) {
    #get performance from each fold for each aspect
    assign(paste0('hyper_', frail_lab[f], '_fold_', folds[d]), fread(paste0(outdir, 'exp', exp, '_hyper_', frail_lab[f], '_fold_', d, '.csv')))
    #add fold label
    hyper <- get(paste0('hyper_', frail_lab[f], '_fold_', folds[d]))
    hyper$fold <- folds[d]
    assign(paste0('hyper_', frail_lab[f], '_fold_', folds[d]), hyper)
  }
  #combine all folds for each aspect
  assign(paste0(frail_lab[f], '_hyper'), do.call(rbind, mget(objects(pattern = paste0(frail_lab[f])))))
  #calculate mean loss for each set of hyperparameters
  best_hyper <- mean_loss(get(paste0(frail_lab[f], '_hyper')))
  #pick the combo with lowest loss
  best_hyper <- best_hyper %>%
    arrange(mean_entropy) %>%
    slice(1)
  #insert into hyper grid
  aspect_grid <- expand.grid(
    ntree           = 300,
    mtry            = best_hyper$mtry,
    sample_frac = best_hyper$sample_frac
  )
  
  for (d in 1:length(folds)) {
    
    #load data (structured data & text with windowing)
    assign(paste0('f', folds[d], '_tr'), fread(paste0(outdir, 'f_', folds[d], '_tr_df.csv')))
    assign(paste0('f', folds[d], '_te'), fread(paste0(outdir, 'f_', folds[d], '_te_df.csv')))
    
    #load caseweights (weight non-neutral tokens by the inverse of their prevalence)
    #e.g. 1.3% of fall_risk tokens are non-neutral. Therefore, non-neutral tokens are weighted * (1/0.013)
    assign(paste0('f', folds[d], '_tr_cw'), fread(paste0(outdir, 'f_', folds[d], '_tr_cw.csv')))
    
    #Reload hyper grid for each fold
    hyper_grid <- aspect_grid
    
    for(s in 1:length(svd)) {
      
      #load truncated SVD of tf-idf of text & remove first row and first column (junk)
      assign(paste0('f', folds[d], '_tr_svd', svd[s]), fread(paste0(outdir, 'f_', folds[d], '_tr_svd', svd[s], '.csv'), skip = 1, drop = 1))
      assign(paste0('f', folds[d], '_te_svd', svd[s]), fread(paste0(outdir, 'f_', folds[d], '_te_svd', svd[s], '.csv'), skip = 1, drop = 1))
      
      for(i in 1:nrow(hyper_grid)) {
        
        #get matching training and test data
        x_train <- get(paste0('f', folds[d], '_tr_svd', svd[s]))
        x_test <- get(paste0('f', folds[d], '_te_svd', svd[s]))
        y_train <- get(paste0('f', folds[d], '_tr'))[[paste0(frail_lab[f])]]
        y_test_neut <- get(paste0('f', folds[d], '_te'))[[paste0(frail_lab[f], '_0')]]
        y_test_pos <- get(paste0('f', folds[d], '_te'))[[paste0(frail_lab[f], '_1')]]
        y_test_neg <- get(paste0('f', folds[d], '_te'))[[paste0(frail_lab[f], '_-1')]]
        y_test <- cbind(y_test_neut, y_test_pos, y_test_neg)
        #get matching caseweights
        cw <- get(paste0('f', folds[d], '_tr_cw'))[[paste0(frail_lab[f], '_cw')]]
        
        frail_rf <- ranger(y = factor(y_train, levels = c(0, 1, -1)), #relevel factors to match class.weights
                           x = select(x_train, -1),
                           num.threads = detectCores(),
                           probability = TRUE,
                           num.trees = hyper_grid$ntree[i],
                           mtry = hyper_grid$mtry[i],
                           sample.fraction = hyper_grid$sample_frac[i],
                           #leaving node size as default
                           #min.node.size = hyper_grid$node_size[i],
                           #class.weights in order of outcome factor levels
                           #class.weights = as.integer(c(levels(factor(cw))[1], levels(factor(cw))[2], levels(factor(cw))[2])),
                           #later, can set oob.error=FALSE to save time/memory (using CV error)
                           #case.weights
                           case.weights = cw,
                           oob.error = TRUE,
                           seed = seed)
        
        
        #make predictions on test fold
        preds <- predict(frail_rf, data=x_test)$predictions
        #save predictions for the best set of hyperparameters
        saveRDS(preds, paste0(outdir, 'preds/', 'exp', exp, '_BestPred_', frail_lab[f], '_fold_', folds[d], '.rda'))
        
        
        #save performance metrics to compare to previous iteration
        #cv brier score for each class
        hyper_grid$cv_brier_neut[i] <- brier_score(y_test_neut, preds[,'0'])
        hyper_grid$cv_brier_pos[i] <- brier_score(y_test_pos, preds[,'1'])
        hyper_grid$cv_brier_neg[i] <- brier_score(y_test_neg, preds[,'-1'])
        #cv brier score for all classes
        hyper_grid$cv_brier_all[i] <- brier_score(
          c(y_test_neut, y_test_pos, y_test_neg), 
          c(preds[,'0'], preds[,'1'], preds[,'-1']))
        #cv scaled brier score for each class
        hyper_grid$scaled_brier_neut[i] <- scaled_brier_score(y_test_neut, preds[,'0'])
        hyper_grid$scaled_brier_pos[i] <- scaled_brier_score(y_test_pos, preds[,'1'])
        hyper_grid$scaled_brier_neg[i] <- scaled_brier_score(y_test_neg, preds[,'-1'])
        #mean scaled Brier score for all classes
        hyper_grid$scaled_brier_all[i] <- mean(
          scaled_brier_score(y_test_neut, preds[,'0']),
          scaled_brier_score(y_test_pos, preds[,'1']),
          scaled_brier_score(y_test_neg, preds[,'-1']))
        #calculate cross-entropy
        preds_ce <- preds
        #set floor and ceiling for predictions (predictions of 1 or 0 create entropy of -inf)
        preds_ce[preds_ce==0] <- 1e-3
        preds_ce[preds_ce==1] <- 0.999
        #calculate
        hyper_grid$cross_entropy_2[i] <- cross_entropy_2(y_test, preds_ce)
      }
      
      hyper_grid2 <- hyper_grid
      hyper_grid2$SVD <- svd[s]
      
      #start building the hyper_grid for the current loop
      if (exists(paste0('hyper_grid_d', d, '_f', f)) == FALSE) {
        assign(paste0('hyper_grid_d', d, '_f', f), hyper_grid2)
      } else {
        #add new results from each svd loop
        assign(paste0('hyper_grid_d', d, '_f', f), rbind(get(paste0('hyper_grid_d', d, '_f', f)), hyper_grid2))
      }
    }
    
    #add frail aspect label
    hyper_grid4 <- get(paste0('hyper_grid_d', d, '_f', f))
    hyper_grid4$frail_lab <- frail_lab[f]
    
    if (exists(paste0('hyper_', frail_lab[f], '_fold_', d)) == FALSE) {
      assign(paste0('hyper_', frail_lab[f], '_fold_', d), hyper_grid4)
    } else{
      #add new results from each aspect loop
      assign(paste0('hyper_', frail_lab[f], '_fold_', d), rbind(get(paste0('hyper_', frail_lab[f], '_fold_', d)), hyper_grid4))
    }
    
    #save each fold for each aspect
    write.csv(get(paste0('hyper_', frail_lab[f], '_fold_', d)), paste0(outdir, 'exp', exp, '_best_hyper_', frail_lab[f], '_fold_', d, '.csv'))
  }
}
