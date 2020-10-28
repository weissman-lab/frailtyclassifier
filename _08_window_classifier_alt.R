library(data.table)
library(ranger)
library(dplyr)
library(tidyr)
library(doParallel)
registerDoParallel(detectCores())



#Experiment number (based on date):
exp <- '102820'
#Update exp numbrer to indicate rf with tf-idf
exp <- paste0(exp, '_rf_tfidf')

#Include structured data?
inc_struc = TRUE

#Update exp number to indicate unstructured/structured
if (inc_struc == FALSE) {
  exp <- paste0(exp, 'un')
} else {exp <- paste0(exp, 'str')}



seed = 92120
folds <- seq(1, 10)
svd <- c(300, 1000, 3000)
frail_lab <- c('Msk_prob', 'Fall_risk', 'Nutrition', 'Resp_imp')



datadir <- paste0(getwd(), '/output/_08_window_classifier_alt/')
#new directory for each experiment
outdir <- paste0(getwd(), '/output/_08_window_classifier_alt/exp', exp, '/')
dir.create(outdir)
#new directory for predictions
predsdir <- paste0(outdir,'preds/')
dir.create(predsdir)



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



start_time <- Sys.time()

for (d in 1:length(folds)) {
  
  #load labels and structured data
  assign(paste0('f', folds[d], '_tr'), fread(paste0(datadir, 'f_', folds[d], '_tr_df.csv')))
  assign(paste0('f', folds[d], '_te'), fread(paste0(datadir, 'f_', folds[d], '_te_df.csv')))
  
  #load caseweights (weight non-neutral tokens by the inverse of their prevalence)
  #e.g. 1.3% of fall_risk tokens are non-neutral. Therefore, non-neutral tokens are weighted * (1/0.013)
  assign(paste0('f', folds[d], '_tr_cw'), fread(paste0(datadir, 'f_', folds[d], '_tr_cw.csv')))
  
  for (f in 1:length(frail_lab)) {
    
    #get matching training and test data
    y_train <- get(paste0('f', folds[d], '_tr'))[[paste0(frail_lab[f])]]
    y_test_neut <- get(paste0('f', folds[d], '_te'))[[paste0(frail_lab[f], '_0')]]
    y_test_pos <- get(paste0('f', folds[d], '_te'))[[paste0(frail_lab[f], '_1')]]
    y_test_neg <- get(paste0('f', folds[d], '_te'))[[paste0(frail_lab[f], '_-1')]]
    y_test <- cbind(y_test_neut, y_test_pos, y_test_neg)
    
    #get matching caseweights
    cw <- get(paste0('f', folds[d], '_tr_cw'))[[paste0(frail_lab[f], '_cw')]]
    
    for(s in 1:length(svd)) {
      
      #load features with or without structured data
      if (inc_struc == FALSE) {
        #load only the SVD features & remove first row and first column (junk)
        x_train <- fread(paste0(datadir, 'f_', folds[d], '_tr_svd', svd[s], '.csv'), skip = 1, drop = 1)
        x_test <- fread(paste0(datadir, 'f_', folds[d], '_te_svd', svd[s], '.csv'), skip = 1, drop = 1)
      } else {
        #concatenate structured data with SVD
        x_train <- cbind(get(paste0('f', folds[d], '_tr_svd', svd[s])), get(paste0('f', folds[d], '_tr'))[,27:82])
        x_test <- cbind(get(paste0('f', folds[d], '_te_svd', svd[s])), get(paste0('f', folds[d], '_te'))[,27:82])
      }
      
      # hyper grid
      hyper_grid <- expand.grid(
        ntree           = 300,
        mtry            = signif(seq(7, 45, length.out = 4), 2),
        sample_frac = signif(seq(0.6, 1, length.out = 3), 1))
      #label sample fraction (for naming .csv files)
      hyper_grid <- mutate(hyper_grid, sample_frac_l = ifelse(sample_frac == 0.6, 6,
                                                              ifelse(sample_frac == 0.8, 8,
                                                                     ifelse(sample_frac == 1.0, 10, NA))))
      #tree grid
      # hyper_grid <- expand.grid(
      #   ntree      = signif(seq(100, 700, length.out = 4), 0),
      #   mtry       = 20,
      #   node_size  = 10)
      #very small grid
      # hyper_grid <- expand.grid(
      #   ntree           = c(1, 2),
      #   mtry            = 1,
      #   sample_frac = .1)
      
      for(i in 1:nrow(hyper_grid)) {
        
        frail_rf <- ranger(y = factor(y_train, levels = c(0, 1, -1)), #relevel factors to match class.weights
                           x = x_train,
                           num.threads = detectCores(),
                           probability = TRUE,
                           num.trees = hyper_grid$ntree[i],
                           mtry = hyper_grid$mtry[i],
                           sample.fraction = hyper_grid$sample_frac[i],
                           #leaving node size as default
                           #min.node.size = hyper_grid$node_size[i],
                           #class.weights in order of outcome factor levels
                           #class.weights = as.integer(c(levels(factor(cw))[1], levels(factor(cw))[2], levels(factor(cw))[2])),
                           #set oob.error=FALSE to save time/memory (using CV error)
                           #case.weights
                           case.weights = cw,
                           oob.error = FALSE,
                           seed = seed)
        
        #make predictions on test fold
        preds <- predict(frail_rf, data=x_test)$predictions
        
        #save predictions
        fwrite(as.data.table(preds), paste0(predsdir, 'exp', exp, '_preds_', frail_lab[f], '_fold_', folds[d], '_mtry_', hyper_grid$mtry[i], '_sfr_', hyper_grid$sample_frac_l[i], '.csv'))
        
        #cv brier score for each class
        hyper_grid$bscore_neut[i] <- brier_score(y_test_neut, preds[,'0'])
        hyper_grid$bscore_pos[i] <- brier_score(y_test_pos, preds[,'1'])
        hyper_grid$bscore_neg[i] <- brier_score(y_test_neg, preds[,'-1'])
        #cv brier score for all classes
        hyper_grid$bscore_all[i] <- brier_score(
          c(y_test_neut, y_test_pos, y_test_neg), 
          c(preds[,'0'], preds[,'1'], preds[,'-1']))
        
        #cv scaled brier score for each class
        hyper_grid$sbrier_neut[i] <- scaled_brier_score(y_test_neut, preds[,'0'])
        hyper_grid$sbrier_pos[i] <- scaled_brier_score(y_test_pos, preds[,'1'])
        hyper_grid$sbrier_neg[i] <- scaled_brier_score(y_test_neg, preds[,'-1'])
        #mean scaled Brier score for all classes
        hyper_grid$sbrier_mean[i] <- mean(c(
          scaled_brier_score(y_test_neut, preds[,'0']),
          scaled_brier_score(y_test_pos, preds[,'1']),
          scaled_brier_score(y_test_neg, preds[,'-1'])))
        
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
      if (exists(paste0('hyper_grid_d', folds[d], '_f', f)) == FALSE) {
        assign(paste0('hyper_grid_d', folds[d], '_f', f), hyper_grid2)
      } else {
        #add new results from each svd loop
        assign(paste0('hyper_grid_d', folds[d], '_f', f), rbind(get(paste0('hyper_grid_d', folds[d], '_f', f)), hyper_grid2))
      }
    }
    
    #add frail aspect label
    hyper_grid4 <- get(paste0('hyper_grid_d', folds[d], '_f', f))
    hyper_grid4$frail_lab <- frail_lab[f]
    
    #continue building grid with each loop
    if (exists(paste0('hyper_', frail_lab[f], '_fold_', folds[d])) == FALSE) {
      assign(paste0('hyper_', frail_lab[f], '_fold_', folds[d]), hyper_grid4)
    } else{
      #add new results from each aspect loop
      assign(paste0('hyper_', frail_lab[f], '_fold_', folds[d]), rbind(get(paste0('hyper_', frail_lab[f], '_fold_', folds[d])), hyper_grid4))
    }
    #save each fold for each aspect
    fwrite(get(paste0('hyper_', frail_lab[f], '_fold_', folds[d])), paste0(outdir, 'exp', exp, '_hyper_', frail_lab[f], '_fold_', folds[d], '.csv'))
    
    #calculate & save run time for each fold for each aspect
    end_time <- Sys.time()
    duration <- difftime(end_time, start_time, units = 'sec')
    run_time <- paste0('The start time is: ', start_time, '. The end time is: ', end_time, '. Time difference of: ', duration, ' seconds.')
    #save
    write(run_time, paste0(outdir, 'exp', exp, '_duration_hyper_', frail_lab[f], '_fold_', folds[d], '.txt'))
  }
}

#calculate total run time
end_time <- Sys.time()
duration <- difftime(end_time, start_time, units = 'sec')
run_time <- paste0('The start time is: ', start_time, '. The end time is: ', end_time, '. Time difference of: ', duration, ' seconds.')
#save
write(run_time, paste0(outdir, 'exp', exp, '_duration_winclass_alt_r.txt'))