library(data.table)
library(glmnet)
library(dplyr)
library(tidyr)
library(foreach)
library(doParallel)
registerDoParallel(detectCores())



#Experiment number from cmd line:
exp <- commandArgs(trailingOnly = TRUE)
#test if there is an exp number argument: if not, return an error
if (length(exp)==0) {
  stop("Exp number must be specified as an argument", call.=FALSE)
}
#Update exp number to indicate penalized regression with embeddings
exp <- paste0(exp, '_logit_embed')

#Include structured data?
inc_struc = TRUE

#Update exp number to indicate unstructured/structured
if (inc_struc == FALSE) {
  exp <- paste0(exp, 'un')
} else {exp <- paste0(exp, 'str')}



# training & testing data:
# mb
# setwd(dirname(rstudioapi::getSourceEditorContext()$path))
# datadir <- paste0(getwd(), '/output/lin_trees/')
# grace
# datadir <- paste0(getwd(), '/output/lin_trees/')
# azure
datadir <- '/share/gwlab/frailty/output/lin_trees/'
SVDdir <- paste0(datadir, 'svd/') 
embeddingsdir <- paste0(datadir, 'embeddings/')
trtedatadir <- paste0(datadir, 'trtedata/')

#new output directory for each experiment:
outdir <- paste0(datadir, 'exp', exp, '/')
dir.create(outdir)
#new directory for predictions:
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
  svd = 0, #included to match glmnet tf-idf output
  frail_lab = c('Msk_prob', 'Fall_risk', 'Nutrition', 'Resp_imp'),
  alpha = c(0.9, 0.5, 0.1),
  class = c('neut', 'pos', 'neg')
)
#label alpha (for naming .csv files)
mg <- mutate(mg, alpha_l = ifelse(alpha == 0.9, 9,
                                  ifelse(alpha == 0.5, 5,
                                         ifelse(alpha == 0.1, 1, NA))))

#check for models that have already been completed & remove them from the grid
mg <- mg %>%
  mutate(filename = paste0('exp', exp, '_hyper_f', fold, '_', frail_lab, '_', class, '_svd_', svd, '_alpha', alpha_l, '.csv')) %>%
  filter(!filename %in% list.files(outdir)) %>%
  select(-'filename')


#lambda seq
lambda_seq <- c(10^seq(2, -5, length.out = 25))


#load data for all folds prior to parallelizing
folds <- seq(1, 10)
for (f in 1:length(folds)) {
  #load labels and structured data
  assign(paste0('f', folds[f], '_tr'), fread(paste0(trtedatadir, 'f_', folds[f], '_tr_df.csv')))
  assign(paste0('f', folds[f], '_te'), fread(paste0(trtedatadir, 'f_', folds[f], '_te_df.csv')))
  
  #load embeddings for each fold (drop index)
  embeddings_tr <- fread(paste0(embeddingsdir, 'f_', folds[f], '_tr_embed_mean_cent_lag_lead.csv'), drop = 1)
  embeddings_te <- fread(paste0(embeddingsdir, 'f_', folds[f], '_te_embed_mean_cent_lag_lead.csv'), drop = 1)
  
  #test that embeddings notes match training/test notes before dropping the 'notes' column
  if (identical(distinct(get(paste0('f', folds[f], '_tr')), note)$note, distinct(embeddings_tr, note)$note) == FALSE) stop("embeddings do not match training data")
  if (identical(distinct(get(paste0('f', folds[f], '_te')), note)$note, distinct(embeddings_te, note)$note) == FALSE) stop("embeddings do not match test data")
  
  #load embeddings with or without structured data
  if (inc_struc == FALSE) {
    #drop 'note' column
    assign(paste0('f', folds[f], '_x_train'), as.matrix(embeddings_tr[-1]))
    assign(paste0('f', folds[f], '_x_test'), as.matrix(embeddings_te[-1]))
  } else {
    #drop 'note' column & concatenate embeddings with structured data
    assign(paste0('f', folds[f], '_x_train'), as.matrix(cbind(embeddings_tr, get(paste0('f', folds[f], '_tr'))[,27:82])))
    assign(paste0('f', folds[f], '_x_test'), as.matrix(cbind(embeddings_te, get(paste0('f', folds[f], '_te'))[,27:82])))
  }
}


#start timer
start_time <- Sys.time()

foreach (r = 1:nrow(mg)) %dopar% {
  
  #get matching training and test labels
  y_train_neut <- get(paste0('f', mg$fold[r], '_tr'))[[paste0(mg$frail_lab[r], '_0')]]
  y_train_pos <- get(paste0('f', mg$fold[r], '_tr'))[[paste0(mg$frail_lab[r], '_1')]]
  y_train_neg <- get(paste0('f', mg$fold[r], '_tr'))[[paste0(mg$frail_lab[r], '_-1')]]
  y_test_neut <- get(paste0('f', mg$fold[r], '_te'))[[paste0(mg$frail_lab[r], '_0')]]
  y_test_pos <- get(paste0('f', mg$fold[r], '_te'))[[paste0(mg$frail_lab[r], '_1')]]
  y_test_neg <- get(paste0('f', mg$fold[r], '_te'))[[paste0(mg$frail_lab[r], '_-1')]]
  
  #train model for each class
  frail_logit <- glmnet(x = get(paste0('f', mg$fold[r], '_x_train')),
                        y = get(paste0('y_train_', mg$class[r])),
                        family = 'binomial',
                        alpha = mg$alpha[r],
                        lambda = lambda_seq)
  
  #make predictions on test fold for each alpha
  alpha_preds <- predict(frail_logit, get(paste0('f', mg$fold[r], '_x_test')), type = 'response')
  
  #save predictions
  fwrite(as.data.table(alpha_preds), paste0(predsdir, 'exp', exp, '_preds_f', mg$fold[r], '_', mg$frail_lab[r], '_', mg$class[r], '_svd_', mg$svd[r], '_alpha', mg$alpha_l[r], '.csv'))
  
  #build hyperparameter grid
  hyper_grid <- expand.grid(
    frail_lab = NA,
    fold = NA,
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
  fwrite(hyper_grid, paste0(outdir, 'exp', exp, '_hyper_f', mg$fold[r], '_', mg$frail_lab[r], '_', mg$class[r], '_svd_', mg$svd[r], '_alpha', mg$alpha_l[r], '.csv'))

  #calculate & save run time for each glmnet
  end_time <- Sys.time()
  duration <- difftime(end_time, start_time, units = 'sec')
  run_time <- paste0('The start time is: ', start_time, '. The end time is: ', end_time, '. Time difference of: ', duration, ' seconds.')
  #save
  write(run_time, paste0(outdir, 'exp', exp, '_duration_hyper_', mg$fold[r], '_', mg$frail_lab[r], '_', mg$class[r], '_svd_', mg$svd[r], '_alpha', mg$alpha_l[r], '.txt'))
}

#calculate total run time
end_time <- Sys.time()
duration <- difftime(end_time, start_time, units = 'sec')
run_time <- paste0('The start time is: ', start_time, '. The end time is: ', end_time, '. Time difference of: ', duration, ' seconds.')
#save
write(run_time, paste0(outdir, 'exp', exp, '_duration_logit.txt'))