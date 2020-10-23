library(data.table)
library(glmnet)
library(dplyr)
library(tidyr)
library(foreach)
library(doParallel)
registerDoParallel(detectCores())






#Experiment number (based on date):
exp <- '102320'
#Update exp numbrer to indicate penalized regression
exp <- paste0(exp, '_logit')

#Include structured data?
inc_struc = TRUE

#Update exp number to indicate unstructured/structured
if (inc_struc == FALSE) {
  exp <- paste0(exp, '_un')
} else {exp <- paste0(exp, '_str')}



#  setwd(dirname(rstudioapi::getSourceEditorContext()$path))
datadir <- paste0(getwd(), '/output/_08_window_classifier_alt/')
#new directory for each experiment
outdir <- paste0(getwd(), '/output/_08_window_classifier_logit/exp', exp, '/')
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


seed = 92120
folds <- 1
svd <- c(300, 1000, 3000)
frail_lab <- c('Msk_prob', 'Fall_risk', 'Nutrition', 'Resp_imp')


# alpha sequence
alpha_seq <- c(0.9, 0.5, 0.1)

#lambda seq
lambda_seq <- c(10^seq(2, -5, length.out = 25))

#start timer
start_time <- Sys.time()

foreach (d = 1:length(folds)) %dopar% {
  
  #load data (structured data & text with windowing)
  assign(paste0('f', folds[d], '_tr'), fread(paste0(datadir, 'f_', folds[d], '_tr_df.csv')))
  assign(paste0('f', folds[d], '_te'), fread(paste0(datadir, 'f_', folds[d], '_te_df.csv')))
  
  #load caseweights (weight non-neutral tokens by the inverse of their prevalence)
  #e.g. 1.3% of fall_risk tokens are non-neutral. Therefore, non-neutral tokens are weighted * (1/0.013)
  assign(paste0('f', folds[d], '_tr_cw'), fread(paste0(datadir, 'f_', folds[d], '_tr_cw.csv')))
  
  
  for (f in 1:length(frail_lab)) {
    #get matching training and test labels
    y_train_neut <- get(paste0('f', folds[d], '_tr'))[[paste0(frail_lab[f], '_0')]]
    y_train_pos <- get(paste0('f', folds[d], '_tr'))[[paste0(frail_lab[f], '_1')]]
    y_train_neg <- get(paste0('f', folds[d], '_tr'))[[paste0(frail_lab[f], '_-1')]]
    y_test_neut <- get(paste0('f', folds[d], '_te'))[[paste0(frail_lab[f], '_0')]]
    y_test_pos <- get(paste0('f', folds[d], '_te'))[[paste0(frail_lab[f], '_1')]]
    y_test_neg <- get(paste0('f', folds[d], '_te'))[[paste0(frail_lab[f], '_-1')]]
    y_test <- cbind(y_test_neut, y_test_pos, y_test_neg)
    
    for(s in 1:length(svd)) {
      
      #load truncated SVD of tf-idf of text & remove first row and first column (junk)
      assign(paste0('f', folds[d], '_tr_svd', svd[s]), fread(paste0(datadir, 'f_', folds[d], '_tr_svd', svd[s], '.csv'), skip = 1, drop = 1))
      assign(paste0('f', folds[d], '_te_svd', svd[s]), fread(paste0(datadir, 'f_', folds[d], '_te_svd', svd[s], '.csv'), skip = 1, drop = 1))
      
      #load features with or without structured data
      if (inc_struc == FALSE) {
        #load only the SVD features
        x_train <- as.matrix(get(paste0('f', folds[d], '_tr_svd', svd[s])))
        x_test <- as.matrix(get(paste0('f', folds[d], '_te_svd', svd[s])))
      } else {
        #concatenate structured data with SVD
        x_train <- as.matrix(cbind(get(paste0('f', folds[d], '_tr_svd', svd[s])), get(paste0('f', folds[d], '_tr'))[,27:82]))
        x_test <- as.matrix(cbind(get(paste0('f', folds[d], '_te_svd', svd[s])), get(paste0('f', folds[d], '_te'))[,27:82]))
      }
      
      #run elastic net across lambda grid for each class
      classes <- c('neut', 'pos', 'neg')
      
      for (c in 1:length(classes)) {
        for(a in 1:length(alpha_seq)) {
          #train model for each class
          frail_logit <- glmnet(x = x_train,
                                y = get(paste0('y_train_', classes[c])),
                                family = 'binomial',
                                alpha = alpha_seq[a],
                                lambda = lambda_seq,
                                trace.it = 1)
          
          #make predictions on test fold for each alpha
          alpha_preds <- predict(frail_logit, x_test, type = 'response')
          
          #save predictions
          fwrite(alpha_preds, paste0(predsdir, 'exp', exp, '_preds_f', folds[d], '_', frail_lab[f], '_', svd[s], '_', classes[c], '_a', a, '.csv'))
          
          #build hyperparameter grid
          hyper_grid <- expand.grid(
            frail_lab = NA,
            fold = NA,
            SVD = NA,
            class = NA,
            lambda = rep(NA, ncol(alpha_preds)),
            alpha = NA,
            df = NA,
            cross_entropy_2 = NA,
            brier = NA,
            scaled_brier = NA)
          
          for (l in 1:ncol(alpha_preds)) {
            #label each row
            hyper_grid$frail_lab[l] <- frail_lab[f]
            hyper_grid$fold[l] <- folds[d]
            hyper_grid$SVD[l] <- svd[s]
            hyper_grid$class[l] <- classes[c]
            hyper_grid$alpha[l] <- alpha_seq[a]
            
            #lambda
            hyper_grid$lambda[l] <- frail_logit$lambda[l]
            #number of nonzero coefficients (Df)
            hyper_grid$df[l] <- frail_logit$df[l]
            #preds for this lambda
            preds <- alpha_preds[,l]
            
            #brier score
            hyper_grid$brier[l] <- brier_score(get(paste0('y_test_', classes[c])), preds)
            #scaled brier score
            hyper_grid$scaled_brier[l] <- scaled_brier_score(get(paste0('y_test_', classes[c])), preds)
            
            #calculate cross-entropy
            preds_ce <- preds
            #set floor and ceiling for predictions (predictions of 1 or 0 create entropy of -inf)
            preds_ce[preds_ce==0] <- 1e-3
            preds_ce[preds_ce==1] <- 0.999
            #calculate
            hyper_grid$cross_entropy_2[l] <- cross_entropy_2(get(paste0('y_test_', classes[c])), preds_ce)
            
          }
          
          #save each alpha for each class
          fwrite(hyper_grid, paste0(outdir, 'exp', exp, '_hyper_f', folds[d], '_', frail_lab[f], '_', svd[s], '_', classes[c], '_a', a, '.csv'))
          
          #concatenate the alphas for each class
          if (exists(paste0('hyper_grid_d', d, '_f', f, '_s', s, '_c', c)) == FALSE) {
            assign(paste0('hyper_grid_d', d, '_f', f, '_s', s, '_c', c), hyper_grid)
          } else {
            #add new results from each loop
            assign(paste0('hyper_grid_d', d, '_f', f, '_s', s, '_c', c), rbind(get(paste0('hyper_grid_d', d, '_f', f, '_s', s, '_c', c)), hyper_grid))
          }
        }
        #concatenate the classes for each svd  
        if (exists(paste0('hyper_grid_d', d, '_f', f, '_s', s)) == FALSE) {
          assign(paste0('hyper_grid_d', d, '_f', f, '_s', s), get(paste0('hyper_grid_d', d, '_f', f, '_s', s, '_c', c)))
        } else {
          #add new results from each loop
          assign(paste0('hyper_grid_d', d, '_f', f, '_s', s), rbind(get(paste0('hyper_grid_d', d, '_f', f, '_s', s)), get(paste0('hyper_grid_d', d, '_f', f, '_s', s, '_c', c))))
        }
      }
      #concatenate svd for each frailty aspect
      if (exists(paste0('hyper_grid_d', d, '_f', f)) == FALSE) {
        assign(paste0('hyper_grid_d', d, '_f', f), get(paste0('hyper_grid_d', d, '_f', f, '_s', s)))
      } else {
        #add new results from each loop
        assign(paste0('hyper_grid_d', d, '_f', f), rbind(get(paste0('hyper_grid_d', d, '_f', f)), get(paste0('hyper_grid_d', d, '_f', f, '_s', s))))
      }
    }
    #save each fold for each aspect
    fwrite(get(paste0('hyper_grid_d', d, '_f', f)), paste0(outdir, 'exp', exp, '_hyper_', frail_lab[f], '_fold_', folds[d], '.csv'))
    
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

# 
# hyper_grid_d1_f1
# 
# 
# ################
# # Pick the best hyper-parameters and save predictions
# 
# 
# #function to get mean loss across 10 folds
# mean_loss <- function(h) {
#   hyper_wide <- pivot_wider(h, id_cols = c('SVD', 'class', 'lambda', 'alpha', 'frail_lab'), names_from = 'fold',  values_from = 'cross_entropy_2', names_prefix = 'fold')
#   hyper_wide2 <- hyper_wide %>%
#     mutate(mean_entropy = rowMeans(hyper_wide[,grep('fold', colnames(hyper_wide), value = TRUE)])) %>%
#     select(-grep('fold', colnames(hyper_wide)))
#   return(hyper_wide2)
# }
# 
# 
# for (f in 1:length(frail_lab)) {
#   
#   #get the best hyperparameters for each aspect (avg across 10 folds)
#   for (d in 1:length(folds)) {
#     #get performance from each fold for each aspect
#     assign(paste0('best_hyper_', frail_lab[f], '_fold_', folds[d]), fread(paste0(outdir, 'exp', exp, '_hyper_', frail_lab[f], '_fold_', folds[d], '.csv')))
#     #add fold label
#     hyper <- get(paste0('best_hyper_', frail_lab[f], '_fold_', folds[d]))
#     hyper$fold <- folds[d]
#     assign(paste0('best_hyper_', frail_lab[f], '_fold_', folds[d]), hyper)
#   }
#   #combine all folds for each aspect
#   assign(paste0(frail_lab[f], '_hyper'), do.call(rbind, mget(objects(pattern = paste0(frail_lab[f])))))
#   #calculate mean loss for each set of hyperparameters
#   best_hyper <- mean_loss(get(paste0(frail_lab[f], '_hyper')))
#   #pick the combo with lowest loss
#   best_hyper <- best_hyper %>%
#     arrange(mean_entropy) %>%
#     slice(1)
#   #insert into hyper grid
#   aspect_grid <- expand.grid(
#     frail_lab = NA,
#     fold = NA,
#     SVD = best_hyper$SVD,
#     class = NA,
#     lambda = best_hyper$lambda,
#     alpha = best_hyper$alpha,
#     df = NA,
#     cross_entropy_2 = NA,
#     brier = NA,
#     scaled_brier = NA)
# }
