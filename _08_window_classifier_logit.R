library(data.table)
library(glmnet)
library(dplyr)
library(tidyr)
library(foreach)
library(doParallel)
registerDoParallel(detectCores())



#Experiment number (based on date):
exp <- '102420'
#Update exp numbrer to indicate penalized regression
exp <- paste0(exp, '_logit_tfidf')

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



#set seed
seed = 92120

#model grid
mg <- expand_grid(
  fold = seq(1,10),
  svd = seq(50, 300, 1000),
  frail_lab = c('Msk_prob', 'Fall_risk', 'Nutrition', 'Resp_imp'),
  alpha = c(0.9, 0.5, 0.1),
  class = c('neut', 'pos', 'neg')
)
#label alpha (for naming .csv files)
mg <- mutate(mg, alpha_l = ifelse(alpha == 0.9, 9,
                                  ifelse(alpha == 0.5, 5,
                                         ifelse(alpha == 0.1, 1, NA))))

#lambda seq
lambda_seq <- c(10^seq(2, -5, length.out = 25))



#start timer
start_time <- Sys.time()

foreach (r = 1:nrow(mg)) %dopar% {
  
  #load labels and structured data
  assign(paste0('f', mg$fold[r], '_tr'), fread(paste0(datadir, 'f_', mg$fold[r], '_tr_df.csv')))
  assign(paste0('f', mg$fold[r], '_te'), fread(paste0(datadir, 'f_', mg$fold[r], '_te_df.csv')))
  
  #load SVD with or without structured data & remove first row and first column (junk)
  if (inc_struc == FALSE) {
    #load only the SVD
    x_train <- as.matrix(fread(paste0(datadir, 'f_', mg$fold[r], '_tr_svd', mg$svd[r], '.csv'), skip = 1, drop = 1))
    x_test <- as.matrix(fread(paste0(datadir, 'f_', mg$fold[r], '_te_svd', mg$svd[r], '.csv'), skip = 1, drop = 1))
  } else {
    #concatenate structured data with SVD
    x_train <- as.matrix(cbind(fread(paste0(datadir, 'f_', mg$fold[r], '_tr_svd', mg$svd[r], '.csv'), skip = 1, drop = 1), get(paste0('f', mg$fold[r], '_tr'))[,27:82]))
    x_test <- as.matrix(cbind(fread(paste0(datadir, 'f_', mg$fold[r], '_te_svd', mg$svd[r], '.csv'), skip = 1, drop = 1), get(paste0('f', mg$fold[r], '_te'))[,27:82]))
  }
  
  #get matching training and test labels
  y_train_neut <- get(paste0('f', mg$fold[r], '_tr'))[[paste0(mg$frail_lab[r], '_0')]]
  y_train_pos <- get(paste0('f', mg$fold[r], '_tr'))[[paste0(mg$frail_lab[r], '_1')]]
  y_train_neg <- get(paste0('f', mg$fold[r], '_tr'))[[paste0(mg$frail_lab[r], '_-1')]]
  y_test_neut <- get(paste0('f', mg$fold[r], '_te'))[[paste0(mg$frail_lab[r], '_0')]]
  y_test_pos <- get(paste0('f', mg$fold[r], '_te'))[[paste0(mg$frail_lab[r], '_1')]]
  y_test_neg <- get(paste0('f', mg$fold[r], '_te'))[[paste0(mg$frail_lab[r], '_-1')]]
  
  #load caseweights (weight non-neutral tokens by the inverse of their prevalence)
  #e.g. 1.3% of fall_risk tokens are non-neutral. Therefore, non-neutral tokens are weighted * (1/0.013)
  assign(paste0('f', mg$fold[r], '_tr_cw'), fread(paste0(datadir, 'f_', mg$fold[r], '_tr_cw.csv')))
  
  #run elastic net across lambda grid for each class
  classes <- c('neut', 'pos', 'neg')
  
  #train model for each class
  frail_logit <- glmnet(x = x_train,
                        y = get(paste0('y_train_', mg$class[r])),
                        family = 'binomial',
                        alpha = mg$alpha[r],
                        lambda = lambda_seq)
  
  #make predictions on test fold for each alpha
  alpha_preds <- predict(frail_logit, x_test, type = 'response')
  
  #save predictions
  fwrite(alpha_preds, paste0(predsdir, 'exp', exp, '_preds_f', mg$fold[r], '_', mg$frail_lab[r], '_', mg$class[r], '_svd_', mg$svd[r], '_alpha', mg$alpha_l[r], '.csv'))
  
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
    bscore = NA,
    sbrier = NA)
  
  for (l in 1:ncol(alpha_preds)) {
    #label each row
    hyper_grid$frail_lab[l] <- mg$frail_lab[r]
    hyper_grid$fold[l] <- mg$fold[r]
    hyper_grid$SVD[l] <- mg$svd[r]
    hyper_grid$class[l] <- mg$class[r]
    hyper_grid$alpha[l] <- mg$alpha[r]
    
    #lambda
    hyper_grid$lambda[l] <- frail_logit$lambda[l]
    #number of nonzero coefficients (Df)
    hyper_grid$df[l] <- frail_logit$df[l]
    #preds for this lambda
    preds <- alpha_preds[,l]
    
    #brier score
    hyper_grid$bscore[l] <- brier_score(get(paste0('y_test_', mg$class[r])), preds)
    #scaled brier score
    hyper_grid$sbrier[l] <- scaled_brier_score(get(paste0('y_test_', mg$class[r])), preds)
    
    #calculate cross-entropy
    preds_ce <- preds
    #set floor and ceiling for predictions (predictions of 1 or 0 create entropy of -inf)
    preds_ce[preds_ce==0] <- 1e-3
    preds_ce[preds_ce==1] <- 0.999
    #calculate
    hyper_grid$cross_entropy_2[l] <- cross_entropy_2(get(paste0('y_test_', mg$class[r])), preds_ce)
  }
  
  #save hyper_grid for each glmnet run
  fwrite(hyper_grid, paste0(outdir, 'exp', exp, '_hyper_f', mg$fold[r], '_', mg$frail_lab[r], '_', mg$svd[r], '_', mg$class[r], '_r', r, '.csv'))
  
  #calculate & save run time for each glmnet
  end_time <- Sys.time()
  duration <- difftime(end_time, start_time, units = 'sec')
  run_time <- paste0('The start time is: ', start_time, '. The end time is: ', end_time, '. Time difference of: ', duration, ' seconds.')
  #save
  write(run_time, paste0(outdir, 'exp', exp, '_duration_hyper_', mg$fold[r], '_', mg$frail_lab[r], '_', mg$svd[r], '_', mg$class[r], '_r', r, '.txt'))
}

#calculate total run time
end_time <- Sys.time()
duration <- difftime(end_time, start_time, units = 'sec')
run_time <- paste0('The start time is: ', start_time, '. The end time is: ', end_time, '. Time difference of: ', duration, ' seconds.')
#save
write(run_time, paste0(outdir, 'exp', exp, '_duration_logit.txt'))

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
#     assign(paste0('best_hyper_', mg$frail_lab[r], '_fold_', mg$fold[r]), fread(paste0(outdir, 'exp', exp, '_hyper_', mg$frail_lab[r], '_fold_', mg$fold[r], '.csv')))
#     #add fold label
#     hyper <- get(paste0('best_hyper_', mg$frail_lab[r], '_fold_', mg$fold[r]))
#     hyper$fold <- mg$fold[r]
#     assign(paste0('best_hyper_', mg$frail_lab[r], '_fold_', mg$fold[r]), hyper)
#   }
#   #combine all folds for each aspect
#   assign(paste0(mg$frail_lab[r], '_hyper'), do.call(rbind, mget(objects(pattern = paste0(mg$frail_lab[r])))))
#   #calculate mean loss for each set of hyperparameters
#   best_hyper <- mean_loss(get(paste0(mg$frail_lab[r], '_hyper')))
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
#     bscore = NA,
#     sbrier = NA)
# }
