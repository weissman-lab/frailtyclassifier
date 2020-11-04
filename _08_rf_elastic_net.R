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
#Update exp number to indicate penalized regression
exp <- paste0(exp, '_enet')

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

#-ylog(yhat) - (1-y)log(1-yhat)
cross_entropy_2 <- function(obs, pred){
  plus <- -1 * (obs * log(pred))
  minus <- -1 * ((1-obs) * (log (1-pred)))
  mat <- cbind(plus, minus)
  mean(rowSums(mat))
}


#set seed
seed = 92120


#load data in parallel
folds <- seq(1, 10)
for (f in 1:length(folds)) {
  assign(paste0('f', folds[f], '_tr'), fread(paste0(trtedatadir, 'f_', folds[f], '_tr_df.csv')))
  assign(paste0('f', folds[f], '_te'), fread(paste0(trtedatadir, 'f_', folds[f], '_te_df.csv')))
}
svd <- c('embed', '300', '1000', '3000')
for (s in 1:length(svd)) {
  if (svd[s] == 'embed') {
    embed_train <- foreach (f = 1:length(folds)) %dopar% {
      #load embeddings for each fold (drop index)
      embeddings_tr <- fread(paste0(embeddingsdir, 'f_', folds[f], '_tr_embed_mean_cent_lag_lead.csv'), drop = 1)
      #test that embeddings notes match training/test notes before dropping the 'notes' column
      if (identical(distinct(get(paste0('f', folds[f], '_tr')), note)$note, distinct(embeddings_tr, note)$note) == FALSE) stop("embeddings do not match training data")
      #embeddings with or without structured data
      if (inc_struc == FALSE) {
        #drop 'note' column
        embeddings_tr <- as.matrix(embeddings_tr[, -1])
      } else {
        #drop 'note' column & concatenate embeddings with structured data
        embeddings_tr <- as.matrix(cbind(embeddings_tr[, -1], get(paste0('f', folds[f], '_tr'))[,27:82]))
      }
      return(embeddings_tr)
    }
    embed_test <- foreach (f = 1:length(folds)) %dopar% {
      #load embeddings for each fold (drop index)
      embeddings_te <- fread(paste0(embeddingsdir, 'f_', folds[f], '_te_embed_mean_cent_lag_lead.csv'), drop = 1)
      #test that embeddings notes match training/test notes before dropping the 'notes' column
      if (identical(distinct(get(paste0('f', folds[f], '_te')), note)$note, distinct(embeddings_te, note)$note) == FALSE) stop("embeddings do not match test data")
      #embeddings with or without structured data
      if (inc_struc == FALSE) {
        #drop 'note' column
        embeddings_te <- as.matrix(embeddings_te[, -1])
      } else {
        #drop 'note' column & concatenate embeddings with structured data
        embeddings_te <- as.matrix(cbind(embeddings_te[, -1], get(paste0('f', folds[f], '_te'))[,27:82]))
      }
      return(embeddings_te)
    }
  } else {
      train <- foreach (f = 1:length(folds)) %dopar% {
        #svd with or without structured data
        if (inc_struc == FALSE) {
          #load only the svd - drop first 2 columns (index and note label)
          x_train <- as.matrix(fread(paste0(SVDdir, 'f_', folds[f], '_tr_svd', svd[s], '.csv'), skip = 1, drop = 1))
        } else {
          #concatenate svd with structured data
          x_train <- as.matrix(cbind(fread(paste0(SVDdir, 'f_', folds[f], '_tr_svd', svd[s], '.csv'), skip = 1, drop = 1), get(paste0('f', folds[f], '_tr'))[,27:82]))
        }
        #test that svd row length matches training/test row length
        if ((nrow(x_train) == nrow(get(paste0('f', folds[f], '_tr')))) == FALSE) stop("svd do not match training data")
        return(x_train)
      }
      test <- foreach (f = 1:length(folds)) %dopar% {
        #load labels and structured data
        df_te <- fread(paste0(trtedatadir, 'f_', folds[f], '_te_df.csv'))
        #svd with or without structured data
        if (inc_struc == FALSE) {
          #load only the svd - drop first 2 columns (index and note label)
          x_test <- as.matrix(fread(paste0(SVDdir, 'f_', folds[f], '_te_svd', svd[s], '.csv'), skip = 1, drop = 1))
        } else {
          #concatenate svd with structured data
          x_test <- as.matrix(cbind(fread(paste0(SVDdir, 'f_', folds[f], '_te_svd', svd[s], '.csv'), skip = 1, drop = 1), get(paste0('f', folds[f], '_te'))[,27:82]))
        }
        #test that svd row length matches training/test row length
        if ((nrow(x_test) == nrow(get(paste0('f', folds[f], '_te')))) == FALSE) stop("svd do not match training data")
        return(x_test)
      }
    }
  assign(paste0('s_', svd[s], '_x_train'), train)
  assign(paste0('s_', svd[s], '_x_test'), test)
}
#foreach output is in list format
#rename each element of the list as a df
for (s in 1:length(svd)) {
  for (f in 1:length(folds)) {
    assign(paste0('f', folds[f], '_s_', svd[s], '_x_train'), get(paste0('s_', svd[s], '_x_train'))[[f]])
    assign(paste0('f', folds[f], '_s_', svd[s], '_x_test'), get(paste0('s_', svd[s], '_x_test'))[[f]])
  }
}

# old version (works)
# #load data for all folds prior to parallelizing
# folds <- seq(1, 10)
# svd <- c('300', '1000', '3000')
# for (f in 1:length(folds)) {
#   #load labels and structured data
#   assign(paste0('f', folds[f], '_tr'), fread(paste0(trtedatadir, 'f_', folds[f], '_tr_df.csv')))
#   assign(paste0('f', folds[f], '_te'), fread(paste0(trtedatadir, 'f_', folds[f], '_te_df.csv')))
#   #load embeddings for each fold (drop index)
#   embeddings_tr <- fread(paste0(embeddingsdir, 'f_', folds[f], '_tr_embed_mean_cent_lag_lead.csv'), drop = 1)
#   embeddings_te <- fread(paste0(embeddingsdir, 'f_', folds[f], '_te_embed_mean_cent_lag_lead.csv'), drop = 1)
#   #test that embeddings notes match training/test notes before dropping the 'notes' column
#   if (identical(distinct(get(paste0('f', folds[f], '_tr')), note)$note, distinct(embeddings_tr, note)$note) == FALSE) stop("embeddings do not match training data")
#   if (identical(distinct(get(paste0('f', folds[f], '_te')), note)$note, distinct(embeddings_te, note)$note) == FALSE) stop("embeddings do not match test data")
#   #embeddings with or without structured data
#   if (inc_struc == FALSE) {
#     #drop 'note' column
#     assign(paste0('f', folds[f], '_s_embed',  '_x_train'), as.matrix(embeddings_tr[, -1]))
#     assign(paste0('f', folds[f], '_s_embed', '_x_test'), as.matrix(embeddings_te[, -1]))
#   } else {
#     #drop 'note' column & concatenate embeddings with structured data
#     assign(paste0('f', folds[f], '_s_embed', '_x_train'), as.matrix(cbind(embeddings_tr[, -1], get(paste0('f', folds[f], '_tr'))[,27:82])))
#     assign(paste0('f', folds[f], '_s_embed', '_x_test'), as.matrix(cbind(embeddings_te[, -1], get(paste0('f', folds[f], '_te'))[,27:82])))
#   }
#   #load svd for each fold
#   for (s in 1:length(svd)) {
#     #svd with or without structured data
#     if (inc_struc == FALSE) {
#       #load only the svd - drop first 2 columns (index and note label)
#       assign(paste0('f', folds[f], '_s_', svd[s], '_x_train'), as.matrix(fread(paste0(SVDdir, 'f_', folds[f], '_tr_svd', svd[s], '.csv'), skip = 1, drop = 1)))
#       assign(paste0('f', folds[f], '_s_', svd[s], '_x_test'), as.matrix(fread(paste0(SVDdir, 'f_', folds[f], '_te_svd', svd[s], '.csv'), skip = 1, drop = 1)))
#     } else {
#       #concatenate svd with structured data
#       assign(paste0('f', folds[f], '_s_', svd[s], '_x_train'), as.matrix(cbind(fread(paste0(SVDdir, 'f_', folds[f], '_tr_svd', svd[s], '.csv'), skip = 1, drop = 1), get(paste0('f', folds[f], '_tr'))[,27:82])))
#       assign(paste0('f', folds[f], '_s_', svd[s], '_x_test'), as.matrix(cbind(fread(paste0(SVDdir, 'f_', folds[f], '_te_svd', svd[s], '.csv'), skip = 1, drop = 1), get(paste0('f', folds[f], '_te'))[,27:82])))
#     }
#     #test that svd row length matches training/test row length
#     if ((nrow(get(paste0('f', folds[f], '_s_', svd[s], '_x_train'))) == nrow(get(paste0('f', folds[f], '_tr')))) == FALSE) stop("svd do not match training data")
#     if ((nrow(get(paste0('f', folds[f], '_s_', svd[s], '_x_test'))) == nrow(get(paste0('f', folds[f], '_te')))) == FALSE) stop("svd do not match test data")
#   }
# }


#start glmnet timer
start_time <- Sys.time()


#set sequence of lambda values to test
lambda_seq <- c(10^seq(2, -5, length.out = 25))


#model grid 1
mg1 <- expand_grid(
  fold = seq(1,10),
  svd = c('embed', '1000'),
  frail_lab = c('Msk_prob', 'Fall_risk', 'Nutrition', 'Resp_imp'),
  alpha = c(0.9, 0.5, 0.1),
  class = c('neut', 'pos', 'neg')
)
#label alpha (for naming .csv files)
mg1 <- mutate(mg1, alpha_l = ifelse(alpha == 0.9, 9,
                                  ifelse(alpha == 0.5, 5,
                                         ifelse(alpha == 0.1, 1, NA))))
#check for models that have already been completed & remove them from the grid
mg1 <- mg1 %>%
  mutate(filename = paste0('exp', exp, '_hyper_f', fold, '_', frail_lab, '_', class, '_svd_', svd, '_alpha', alpha_l, '.csv')) %>%
  filter(!filename %in% list.files(outdir)) %>%
  select(-'filename')
#run for first model grid
mg <- mg1
foreach (r = 1:nrow(mg)) %dopar% {
  #get matching training and test labels
  y_train_neut <- get(paste0('f', mg$fold[r], '_tr'))[[paste0(mg$frail_lab[r], '_0')]]
  y_train_pos <- get(paste0('f', mg$fold[r], '_tr'))[[paste0(mg$frail_lab[r], '_1')]]
  y_train_neg <- get(paste0('f', mg$fold[r], '_tr'))[[paste0(mg$frail_lab[r], '_-1')]]
  y_test_neut <- get(paste0('f', mg$fold[r], '_te'))[[paste0(mg$frail_lab[r], '_0')]]
  y_test_pos <- get(paste0('f', mg$fold[r], '_te'))[[paste0(mg$frail_lab[r], '_1')]]
  y_test_neg <- get(paste0('f', mg$fold[r], '_te'))[[paste0(mg$frail_lab[r], '_-1')]]
  #train model for each class
  frail_logit <- glmnet(x = get(paste0('f', mg$fold[r], '_s_', mg$svd[r], '_x_train')),
                        y = get(paste0('y_train_', mg$class[r])),
                        family = 'binomial',
                        alpha = mg$alpha[r],
                        lambda = lambda_seq)
  #make predictions on test fold for each alpha
  alpha_preds <- predict(frail_logit, get(paste0('f', mg$fold[r], '_s_', mg$svd[r], '_x_test')), type = 'response')
  #save predictions
  fwrite(as.data.table(alpha_preds), paste0(predsdir, 'exp', exp, '_preds_f', mg$fold[r], '_', mg$frail_lab[r], '_', mg$class[r], '_svd_', mg$svd[r], '_alpha', mg$alpha_l[r], '.csv'))
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
  fwrite(hyper_grid, paste0(outdir, 'exp', exp, '_hyper_f', mg$fold[r], '_', mg$frail_lab[r], '_', mg$class[r], '_svd_', mg$svd[r], '_alpha', mg$alpha_l[r], '.csv'))
  #calculate & save run time for each glmnet
  end_time <- Sys.time()
  duration <- difftime(end_time, start_time, units = 'sec')
  run_time <- paste0('The start time is: ', start_time, '. The end time is: ', end_time, '. Time difference of: ', duration, ' seconds.')
  #save
  write(run_time, paste0(outdir, 'exp', exp, '_duration_hyper_', mg$fold[r], '_', mg$frail_lab[r], '_', mg$class[r], '_svd_', mg$svd[r], '_alpha', mg$alpha_l[r], '.txt'))
  
  #remove objects & garbage collection
  rm(frail_logit, alpha_preds, hyper_grid, y_train_neut, y_train_pos, y_train_neg, y_test_neut, y_test_pos, y_test_neg)
  gc()
}


#model grid 2
mg2 <- expand_grid(
  fold = seq(1,10),
  svd = c('300', '3000'),
  frail_lab = c('Msk_prob', 'Fall_risk', 'Nutrition', 'Resp_imp'),
  alpha = c(0.9, 0.5, 0.1),
  class = c('neut', 'pos', 'neg')
)
#label alpha (for naming .csv files)
mg2 <- mutate(mg2, alpha_l = ifelse(alpha == 0.9, 9,
                                  ifelse(alpha == 0.5, 5,
                                         ifelse(alpha == 0.1, 1, NA))))
#check for models that have already been completed & remove them from the grid
mg2 <- mg2 %>%
  mutate(filename = paste0('exp', exp, '_hyper_f', fold, '_', frail_lab, '_', class, '_svd_', svd, '_alpha', alpha_l, '.csv')) %>%
  filter(!filename %in% list.files(outdir)) %>%
  select(-'filename')
#run for second model grid
mg <- mg2
foreach (r = 1:nrow(mg)) %dopar% {
  #get matching training and test labels
  y_train_neut <- get(paste0('f', mg$fold[r], '_tr'))[[paste0(mg$frail_lab[r], '_0')]]
  y_train_pos <- get(paste0('f', mg$fold[r], '_tr'))[[paste0(mg$frail_lab[r], '_1')]]
  y_train_neg <- get(paste0('f', mg$fold[r], '_tr'))[[paste0(mg$frail_lab[r], '_-1')]]
  y_test_neut <- get(paste0('f', mg$fold[r], '_te'))[[paste0(mg$frail_lab[r], '_0')]]
  y_test_pos <- get(paste0('f', mg$fold[r], '_te'))[[paste0(mg$frail_lab[r], '_1')]]
  y_test_neg <- get(paste0('f', mg$fold[r], '_te'))[[paste0(mg$frail_lab[r], '_-1')]]
  #train model for each class
  frail_logit <- glmnet(x = get(paste0('f', mg$fold[r], '_s_', mg$svd[r], '_x_train')),
                        y = get(paste0('y_train_', mg$class[r])),
                        family = 'binomial',
                        alpha = mg$alpha[r],
                        lambda = lambda_seq)
  #make predictions on test fold for each alpha
  alpha_preds <- predict(frail_logit, get(paste0('f', mg$fold[r], '_s_', mg$svd[r], '_x_test')), type = 'response')
  #save predictions
  fwrite(as.data.table(alpha_preds), paste0(predsdir, 'exp', exp, '_preds_f', mg$fold[r], '_', mg$frail_lab[r], '_', mg$class[r], '_svd_', mg$svd[r], '_alpha', mg$alpha_l[r], '.csv'))
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
  fwrite(hyper_grid, paste0(outdir, 'exp', exp, '_hyper_f', mg$fold[r], '_', mg$frail_lab[r], '_', mg$class[r], '_svd_', mg$svd[r], '_alpha', mg$alpha_l[r], '.csv'))
  #calculate & save run time for each glmnet
  end_time <- Sys.time()
  duration <- difftime(end_time, start_time, units = 'sec')
  run_time <- paste0('The start time is: ', start_time, '. The end time is: ', end_time, '. Time difference of: ', duration, ' seconds.')
  #save
  write(run_time, paste0(outdir, 'exp', exp, '_duration_hyper_', mg$fold[r], '_', mg$frail_lab[r], '_', mg$class[r], '_svd_', mg$svd[r], '_alpha', mg$alpha_l[r], '.txt'))
  
  #remove objects & garbage collection
  rm(frail_logit, alpha_preds, hyper_grid, y_train_neut, y_train_pos, y_train_neg, y_test_neut, y_test_pos, y_test_neg)
  gc()
}


#calculate total run time
end_time <- Sys.time()
duration <- difftime(end_time, start_time, units = 'sec')
run_time <- paste0('The start time is: ', start_time, '. The end time is: ', end_time, '. Time difference of: ', duration, ' seconds.')
#save
write(run_time, paste0(outdir, 'exp', exp, '_duration_elastic_net.txt'))
gc()



#RANDOM FOREST

#Experiment number from cmd line:
exp <- commandArgs(trailingOnly = TRUE)
#Update exp numbrer to indicate rf with tf-idf
exp <- paste0(exp, '_rf_tfidf')

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


#set seed
seed = 92120
#set models to run
folds <- seq(1, 10)
svd <- c('embed', '300', '1000', '3000')
frail_lab <- c('Msk_prob', 'Fall_risk', 'Nutrition', 'Resp_imp')


#start timer
start_time <- Sys.time()

for (d in 1:length(folds)) {
  #load caseweights (weight non-neutral tokens by the inverse of their prevalence)
  #e.g. 1.3% of fall_risk tokens are non-neutral. Therefore, non-neutral tokens are weighted * (1/0.013)
  assign(paste0('f', folds[d], '_tr_cw'), fread(paste0(trtedatadir, 'f_', folds[d], '_tr_cw.csv')))
  
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
      #get training & test data
      x_train <- get(paste0('f', folds[f], '_s_', svd[s], '_x_train'))
      x_test <- get(paste0('f', folds[f], '_s_', svd[s], '_x_test'))
      # hyper grid
      hyper_grid <- expand.grid(
        ntree       = 400,
        mtry        = signif(seq(7, 45, length.out = 4), 2),
        sample_frac = signif(seq(0.6, 1, length.out = 3), 1))
      #label sample fraction (for naming .csv files)
      hyper_grid <- mutate(hyper_grid, sample_frac_l = ifelse(sample_frac == 0.6, 6,
                                                              ifelse(sample_frac == 0.8, 8,
                                                                     ifelse(sample_frac == 1.0, 10, NA))))
      
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
      #add SVD label
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
write(run_time, paste0(outdir, 'exp', exp, '_duration_randomforest.txt'))
