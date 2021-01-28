library(data.table)
library(PRROC)
library(glmnet)
library(dplyr)
library(tidyr)
library(foreach)
library(doParallel)
library(ranger)
library(rbenchmark)
registerDoParallel(detectCores())


#Experiment number from cmd line:
exp <- commandArgs(trailingOnly = TRUE)
#test if there is an exp number argument: if not, return an error
if (length(exp)==0) {
  stop("Exp number must be specified as an argument", call.=FALSE)
}

#repeats & folds
repeats <- 1
folds <- 1
#text features
svd <- c('embed', '300', '1000')
#include structured data?
inc_struc = TRUE
#set seed
seed = 92120

#set directories based on location
dirs = c('/Users/martijac/Documents/Frailty/frailty_classifier/output/',
         '/media/drv2/andrewcd2/frailty/output/',
         '/share/gwlab/frailty/output/')
for (d in 1:length(dirs)) {
  if (dir.exists(dirs[d])) {
    rootdir = dirs[d]
  }
}
datadir <- paste0(rootdir, 'saved_models/AL03/processed_data/')
SVDdir <- paste0(datadir, 'svd/') 
embeddingsdir <- paste0(datadir, 'embeddings/')
trvadatadir <- paste0(datadir, 'trvadata/')
cwdir <- paste0(datadir, 'caseweights/')
#new output directory for each experiment:
outdir <- paste0(rootdir, 'lin_trees_SENT/exp', exp, '/')
#directory for performance for each rf model:
rf_modeldir <- paste0(outdir,'rf_models/')
#directory for duration for each rf model:
rf_durationdir <- paste0(outdir,'rf_durations/')
#directory for variable importance:
rf_importancedir <- paste0(outdir,'rf_importance/')
#directory for predictions:
rf_predsdir <- paste0(outdir,'rf_preds/')
#directory for performance for each enet model:
enet_modeldir <- paste0(outdir,'enet_models/')
#directory for duration for each enet model:
enet_durationdir <- paste0(outdir,'enet_durations/')
#directory for coefficients:
enet_coefsdir <- paste0(outdir,'enet_coefs/')
#directory for predictions:
enet_predsdir <- paste0(outdir,'enet_preds/')
#make directories
dir.create(outdir)
dir.create(rf_durationdir)
dir.create(rf_modeldir)
dir.create(rf_importancedir)
dir.create(rf_predsdir)
dir.create(enet_modeldir)
dir.create(enet_durationdir)
dir.create(enet_coefsdir)
dir.create(enet_predsdir)


# Brier score
Brier <- function(predictions, observations, positive_class) {
  obs = as.numeric(observations == positive_class)
  mean((predictions - obs)^2)
}

# scaled Brier score
scaled_Brier <- function(predictions, observations, positive_class) {
  1 - (Brier(predictions, observations, positive_class) / 
         Brier(mean(observations), observations, positive_class))
}

# multiclass Brier score
multi_Brier <- function(predictions, observations) {
  mean(rowSums((data.matrix(predictions) - data.matrix(observations))^2))
}

# multiclass scaled Brier score
multi_scaled_Brier <- function(predictions, observations) {
  event_rate_matrix <- matrix(colMeans(observations), 
                              ncol = length(colMeans(observations)), 
                              nrow = nrow(observations),
                              byrow = TRUE)
  1 - (multi_Brier(predictions, observations) / 
         multi_Brier(event_rate_matrix, observations))
}

#repeated k-fold cross validation
for (p in 1:length(repeats)) {
  #load data in parallel
  for (d in 1:length(folds)) {
    # load labels and structured data
    assign(paste0('r', repeats[p], '_f', folds[d], '_tr'),
           fread(paste0(trvadatadir, 'r', repeats[p], '_f', folds[d], '_tr_df.csv')))
    assign(paste0('r', repeats[p], '_f', folds[d], '_va'),
           fread(paste0(trvadatadir, 'r', repeats[p], '_f', folds[d], '_va_df.csv')))
    # load case weights
    assign(paste0('r', repeats[p], '_f', folds[d], '_tr_cw'), fread(
      paste0(cwdir, 'r', repeats[p], '_f', folds[d], '_tr_caseweights.csv')))
    # check for matching
    if ((nrow(get(paste0('r', repeats[p], '_f', folds[d], '_tr_cw'))) == 
         nrow(get(paste0('r', repeats[p], '_f', folds[d], '_tr')))) == FALSE)
      stop("caseweights do not match training data")
  }
  for (s in 1:length(svd)) {
    if (svd[s] == 'embed') {
      train <- foreach (d = 1:length(folds)) %dopar% {
        # load embeddings for each fold (drop index)
        embeddings_tr <- fread(
          paste0(embeddingsdir, 'r', repeats[p], '_f', folds[d], '_tr_embed_min_max_mean_SENT.csv'),
            drop = 1)
        pca_tr <- get(paste0('r', repeats[p], '_f', folds[d], '_tr'))
        emb_cols <- grep('min_|max_|mean_', colnames(embeddings_tr), value = TRUE)
        pca_cols <- grep('pca', colnames(pca_tr), value = TRUE)
        # test that embeddings match structured data
        if (identical(pca_tr$sentence_id, embeddings_tr$sentence_id) == FALSE)
          stop("embeddings do not match training data")
        #embeddings without vs with structured data
        if (inc_struc == FALSE) {
          #get only embeddings columns
          embeddings_tr <- data.matrix(embeddings_tr[, ..emb_cols])
        } else {
          # concatenate embeddings with structured data
          embeddings_tr <- data.matrix(cbind(embeddings_tr[, ..emb_cols], pca_tr[, ..pca_cols]))
        }
        return(embeddings_tr)
      }
      validation <- foreach (d = 1:length(folds)) %dopar% {
        # load embeddings for each fold (drop index)
        embeddings_va <- fread(
          paste0(embeddingsdir, 'r', repeats[p], '_f', folds[d], '_va_embed_min_max_mean_SENT.csv'),
          drop = 1)
        pca_va <- get(paste0('r', repeats[p], '_f', folds[d], '_va'))
        emb_cols <- grep('min_|max_|mean_', colnames(embeddings_va), value = TRUE)
        pca_cols <- grep('pca', colnames(pca_va), value = TRUE)
        # test that embeddings match structured data
        if (identical(pca_va$sentence_id, embeddings_va$sentence_id) == FALSE)
          stop("embeddings do not match validation data")
        #embeddings without vs with structured data
        if (inc_struc == FALSE) {
          #get only embeddings columns
          embeddings_va <- data.matrix(embeddings_va[, ..emb_cols])
        } else {
          # concatenate embeddings with structured data
          embeddings_va <- data.matrix(cbind(embeddings_va[, ..emb_cols], pca_va[, ..pca_cols]))
        }
        return(embeddings_va)
      }
    } else {
      train <- foreach (d = 1:length(folds)) %dopar% {
        #get svd and structured data for each fold
        svd_tr <- fread(
          paste0(SVDdir, 'r', repeats[p], '_f', folds[d], '_tr_svd', svd[s], '.csv'),
          skip = 1,
          drop = 1)
        colnames(svd_tr) <- c('sentence_id', paste0('svd', seq(1:as.integer(svd[s]))))
        pca_tr <- get(paste0('r', repeats[p], '_f', folds[d], '_tr'))
        svd_cols <- grep('svd', colnames(svd_tr), value = TRUE)
        pca_cols <- grep('pca', colnames(pca_tr), value = TRUE)
        #test that svd match structured data
        if (identical(pca_tr$sentence_id, svd_tr$sentence_id) == FALSE)
          stop("embeddings do not match validation data")
        #svd without vs with structured data
        if (inc_struc == FALSE) {
          #load only the svd columns
          svd_tr <- data.matrix(svd_tr[, ..svd_cols])
        } else {
          #concatenate svd with structured data
          svd_tr <- data.matrix(cbind(svd_tr[, ..svd_cols], pca_tr[, ..pca_cols]))
        }
        return(svd_tr)
      }
      validation <- foreach (d = 1:length(folds)) %dopar% {
        #get svd and structured data for each fold
        svd_va <- fread(
          paste0(SVDdir, 'r', repeats[p], '_f', folds[d], '_va_svd', svd[s], '.csv'),
          skip = 1,
          drop = 1)
        colnames(svd_va) <- c('sentence_id', paste0('svd', seq(1:as.integer(svd[s]))))
        pca_va <- get(paste0('r', repeats[p], '_f', folds[d], '_va'))
        svd_cols <- grep('svd', colnames(svd_va), value = TRUE)
        pca_cols <- grep('pca', colnames(pca_va), value = TRUE)
        #test that svd match structured data
        if (identical(pca_va$sentence_id, svd_va$sentence_id) == FALSE)
          stop("embeddings do not match validation data")
        #svd without vs with structured data
        if (inc_struc == FALSE) {
          #load only the svd columns
          svd_va <- data.matrix(svd_va[, ..svd_cols])
        } else {
          #concatenate svd with structured data
          svd_va <- data.matrix(cbind(svd_va[, ..svd_cols], pca_va[, ..pca_cols]))
        }
        return(svd_va)
      }
    }
    assign(paste0('r', repeats[p], '_s_', svd[s], '_x_train'), train)
    assign(paste0('r', repeats[p], '_s_', svd[s], '_x_validation'),
           validation)
  }
  #foreach output is in list format
  #rename each element of the list as a df
  for (s in 1:length(svd)) {
    for (d in 1:length(folds)) {
      assign(paste0('r', repeats[p], '_f', folds[d], '_s_', svd[s], '_x_train'),
             get(paste0('r', repeats[p], '_s_', svd[s], '_x_train'))[[d]])
      assign(paste0('r', repeats[p], '_f', folds[d], '_s_', svd[s], '_x_validation'),
             get(paste0(
               'r', repeats[p], '_s_', svd[s], '_x_validation'
             ))[[d]])
    }
  }
  
  #clear lists of dfs
  for (s in 1:length(svd)) {
    rm(list = paste0('r', repeats[p], '_s_', svd[s], '_x_train'))
    rm(list = paste0('r', repeats[p], '_s_', svd[s], '_x_validation'))
  }
  invisible(gc(verbose = FALSE))
  
  ############################## RANDOM FOREST ##############################
  
  #tuning grid
  mg3 <- expand_grid(
    fold = folds,
    svd = svd,
    frail_lab = c('Msk_prob', 'Fall_risk', 'Nutrition', 'Resp_imp'),
    ntree       = 400,
    mtry        = signif(seq(7, 45, length.out = 3), 2),
    sample_frac = signif(seq(0.6, 1, length.out = 3), 1),
    case_weights = c(TRUE, FALSE)
  )
  
  #label sample fraction (for naming .csv files)
  mg3 <- mutate(mg3, sample_frac_l = ifelse(sample_frac == 0.6, 6,
                                            ifelse(sample_frac == 0.8, 8,
                                                   ifelse(sample_frac == 1.0, 10, NA))))
  #check for models that have already been completed & remove them from the grid
  mg3 <- mg3 %>%
    mutate(filename = 
             paste0('exp', exp, '_hypergrid_r', repeats[p], '_f', fold, '_',
                    frail_lab, '_svd_', svd, '_mtry', mtry, '_sfrac',
                    sample_frac_l, '_cw_',  as.integer(case_weights), '.csv')) %>%
    filter(!filename %in% list.files(rf_modeldir)) %>%
    select(-'filename')
  
  #run rf if incomplete
  if ((nrow(mg3) == 0) == FALSE) {
    for (r in 1:nrow(mg3)){
      #get matching training and validation labels
      x_train <- get(
        paste0('r', repeats[p], '_f', mg3$fold[r], '_s_', mg3$svd[r], '_x_train'))
      x_validation <- get(
        paste0('r', repeats[p], '_f', mg3$fold[r], '_s_', mg3$svd[r], '_x_validation'))
      y_cols <- c(paste0(mg3$frail_lab[r], '_neut'),
                  paste0(mg3$frail_lab[r], '_pos'),
                  paste0(mg3$frail_lab[r], '_neg'))
      y_train <- get(paste0('r', repeats[p], '_f', mg3$fold[r], '_tr'))[, ..y_cols] %>%
        as_tibble() %>%
        mutate(factr = ifelse(.[[1]] == 1, 1,
                              ifelse(.[[2]] == 1, 2,
                                     ifelse(.[[3]] == 1, 3, NA))))
      y_train_factor <- as.factor(y_train$factr)
      y_validation <- get(paste0('r', repeats[p], '_f', mg3$fold[r], '_va'))[, ..y_cols]
      #get matching case weights
      if (mg3$case_weights[r] == FALSE) {
        cw <- NULL
      } else {
        cw <- get(
          paste0('r', repeats[p], '_f', mg3$fold[r], '_tr_cw'))[[paste0(mg3$frail_lab[r], '_cw')]]
      }
      #measure CPU time for rf
      benchmark <- benchmark("rf" = {
        frail_rf <- ranger(y = y_train_factor,
                           x = x_train,
                           num.threads = detectCores(),
                           probability = TRUE,
                           num.trees = mg3$ntree[r],
                           mtry = mg3$mtry[r],
                           sample.fraction = mg3$sample_frac[r],
                           case.weights = cw,
                           oob.error = FALSE,
                           importance = 'impurity',
                           seed = seed)
      }, replications = 1
      )
      #save benchmarking
      fwrite(benchmark, 
             paste0(rf_durationdir, 'exp', exp, '_duration_hyper_r', repeats[p],
                    '_f', mg3$fold[r], '_', mg3$frail_lab[r], '_svd_', mg3$svd[r],
                    '_mtry_', mg3$mtry[r], '_sfrac_', mg3$sample_frac_l[r], '_cw_',
                    as.integer(mg3$case_weights[r]), '.csv'))
      #save variable importance
      importance <- importance(frail_rf)
      i_names <- names(importance)
      importance <- transpose(as.data.table(importance))
      colnames(importance) <- i_names
      importance$cv_repeat <- repeats[p]
      importance$fold <- mg3$fold[r]
      importance$SVD <- mg3$svd[r]
      importance$mtry <- mg3$mtry[r]
      importance$sample_frac <- mg3$sample_frac[r]
      importance$case_weights <- mg3$case_weights[r]
      fwrite(importance, 
             paste0(rf_importancedir, 'exp', exp, '_importance_r', repeats[p],
                    '_f', mg3$fold[r], '_', mg3$frail_lab[r], '_svd_', mg3$svd[r],
                    '_mtry_', mg3$mtry[r], '_sfrac_', mg3$sample_frac_l[r], 
                    '_cw_',  as.integer(mg3$case_weights[r]), '.csv'))
      #make predictions on validation fold
      preds <- predict(frail_rf, data=x_validation)$predictions
      colnames(preds) <- y_cols
      preds_save <- as.data.table(preds)
      preds_save$sentence_id <- get(
        paste0('r', repeats[p], '_f', mg3$fold[r], '_va'))$sentence_id
      #save predictions
      fwrite(as.data.table(preds_save), 
             paste0(rf_predsdir, 'exp', exp, '_preds_r', repeats[p], '_f',
                    mg3$fold[r], '_', mg3$frail_lab[r], '_svd_', mg3$svd[r],
                    '_mtry_', mg3$mtry[r], '_sfrac_', mg3$sample_frac_l[r],
                    '_cw_',  as.integer(mg3$case_weights[r]), '.csv'))
      #label each row
      hyper_grid <- data.frame(frail_lab = mg3$frail_lab[r])
      hyper_grid$cv_repeat <- repeats[p]
      hyper_grid$fold <- mg3$fold[r]
      hyper_grid$SVD <- mg3$svd[r]
      hyper_grid$mtry <- mg3$mtry[r]
      hyper_grid$sample_frac <- mg3$sample_frac[r]
      hyper_grid$case_weights <- mg3$case_weights[r]
      hyper_grid$bscore_neut <- NA
      hyper_grid$bscore_pos <- NA
      hyper_grid$bscore_neg <- NA
      hyper_grid$sbrier_neut <- NA
      hyper_grid$sbrier_pos <- NA
      hyper_grid$sbrier_neg <- NA
      hyper_grid$bscore_multi <- NA
      hyper_grid$sbrier_multi <- NA
      hyper_grid$PR_AUC_neut <- NA
      hyper_grid$PR_AUC_pos <- NA
      hyper_grid$PR_AUC_neg <- NA
      hyper_grid$ROC_AUC_neut <- NA
      hyper_grid$ROC_AUC_pos <- NA
      hyper_grid$ROC_AUC_neg <- NA
      #check for missing values in preds and relevant obs in validation set
      if (((sum(is.na(preds)) > 0) == FALSE) &
          ((sum(y_validation[[2]]) > 0) == TRUE) & 
          ((sum(y_validation[[3]]) > 0) == TRUE))  {
        #single class Brier scores
        hyper_grid$bscore_neut <- Brier(preds[, 1], y_validation[[1]], 1)
        hyper_grid$bscore_pos <- Brier(preds[, 2], y_validation[[2]], 1)
        hyper_grid$bscore_neg <- Brier(preds[, 3], y_validation[[3]], 1)
        #single class scaled Brier scores
        hyper_grid$sbrier_neut <- scaled_Brier(preds[, 1], y_validation[[1]], 1)
        hyper_grid$sbrier_pos <- scaled_Brier(preds[, 2], y_validation[[2]], 1)
        hyper_grid$sbrier_neg <- scaled_Brier(preds[, 3], y_validation[[3]], 1)
        #multiclass brier score
        hyper_grid$bscore_multi <- multi_Brier(preds, y_validation)
        #multiclass scaled brier score
        hyper_grid$sbrier_multi <- multi_scaled_Brier(preds, y_validation)
        #Precision-recall area under the curve
        hyper_grid$PR_AUC_neut <- pr.curve(scores.class0 = preds[, 1],
                                           weights.class0 = y_validation[[1]])$auc.integral
        hyper_grid$PR_AUC_pos <- pr.curve(scores.class0 = preds[, 2],
                                          weights.class0 = y_validation[[2]])$auc.integral
        hyper_grid$PR_AUC_neg <- pr.curve(scores.class0 = preds[, 3],
                                          weights.class0 = y_validation[[3]])$auc.integral
        #Receiver operating characteristic area under the curve
        hyper_grid$ROC_AUC_neut <- roc.curve(scores.class0 = preds[, 1],
                                             weights.class0 = y_validation[[1]])$auc
        hyper_grid$ROC_AUC_pos <- roc.curve(scores.class0 = preds[, 2],
                                            weights.class0 = y_validation[[2]])$auc
        hyper_grid$ROC_AUC_neg <- roc.curve(scores.class0 = preds[, 3],
                                            weights.class0 = y_validation[[3]])$auc
      }
      #save hyper_grid for each rf run
      fwrite(hyper_grid, 
             paste0(rf_modeldir, 'exp', exp, '_hypergrid_r', repeats[p], '_f',
                    mg3$fold[r], '_', mg3$frail_lab[r], '_svd_', mg3$svd[r],
                    '_mtry_', mg3$mtry[r], '_sfrac_', mg3$sample_frac_l[r],
                    '_cw_',  as.integer(mg3$case_weights[r]), '.csv'))
      invisible(invisible(gc(verbose = FALSE)))
    }
  }
  
  ############################## REGRESSION ##############################
  
  #set sequence of lambda values to test
  lambda_seq <- signif(c(10^seq(2, -5, length.out = 25)), 4)
  #tuning grid
  mg1 <- expand_grid(
    fold = folds,
    svd = svd,
    frail_lab = c('Msk_prob', 'Fall_risk', 'Nutrition', 'Resp_imp'),
    alpha = c(0.9, 0.5, 0.1),
    case_weights = c(TRUE, FALSE)
  )
  #label alpha (for naming .csv files)
  mg1 <- mutate(mg1, alpha_l = ifelse(alpha == 0.9, 9,
                                      ifelse(alpha == 0.5, 5,
                                             ifelse(alpha == 0.1, 1, NA))))
  
  #check for models that have already been completed & remove them from the grid
  mg1 <- mg1 %>%
    mutate(filename = 
             paste0('exp', exp, '_hypergrid_r', repeats[p], '_f', fold, '_',
                    frail_lab, '_svd_', svd, '_alpha', alpha_l, '_cw',
                    as.integer(case_weights), '.csv')) %>%
    filter(!filename %in% list.files(enet_modeldir)) %>%
    select(-'filename')
  
  #run glmnet if incomplete
  if ((nrow(mg1) == 0) == FALSE) {
    #run for first model grid
    enet_error = foreach (r = 1:nrow(mg1), .errorhandling = "pass") %dopar% {
      tc_error <- tryCatch(
        {
      #get matching training and validation labels
      x_train <- get(
        paste0('r', repeats[p], '_f', mg1$fold[r], '_s_', mg1$svd[r], '_x_train'))
      x_validation <- get(
        paste0('r', repeats[p], '_f', mg1$fold[r], '_s_', mg1$svd[r], '_x_validation'))
      y_cols <- c(paste0(mg1$frail_lab[r], '_neut'),
                  paste0(mg1$frail_lab[r], '_pos'),
                  paste0(mg1$frail_lab[r], '_neg'))
      y_train <- data.matrix(get(
        paste0('r', repeats[p], '_f', mg1$fold[r], '_tr'))[, ..y_cols])
      y_validation <- data.matrix(get(
        paste0('r', repeats[p], '_f', mg1$fold[r], '_va'))[, ..y_cols])
      #get matching case weights
      if (mg1$case_weights[r] == FALSE) {
        cw <- NULL
      } else {
        cw <- get(
          paste0('r', repeats[p], '_f', mg1$fold[r], '_tr_cw'))[[paste0(mg1$frail_lab[r], '_cw')]]
      }
      #measure CPU time for glmnet
      benchmark <- benchmark("glmnet" = {
        #train model
        frail_logit <- glmnet(x_train, 
                              y_train,
                              family = 'multinomial',
                              alpha = mg1$alpha[r],
                              lambda = lambda_seq,
                              weights = cw)
      }, replications = 1
      )
      #save benchmarking
      fwrite(benchmark, 
             paste0(enet_durationdir, 'exp', exp, '_duration_hyper_r',
                    repeats[p], '_f', mg1$fold[r], '_', mg1$frail_lab[r], '_svd_',
                    mg1$svd[r], '_alpha', mg1$alpha_l[r], '_cw',
                    as.integer(mg1$case_weights[r]), '.txt'))
      #save coefficients
      coefs <- predict(frail_logit, x_validation, type = 'coefficients')
      for (c in 1:length(coefs)){
        coefs_s <- coefs[[y_cols[c]]]
        coefs_save <- as.data.table(t(as.matrix(coefs_s)))
        colnames(coefs_save)[1] <- 'intercept'
        coefs_save$lambda <- as.character(lambda_seq)
        coefs_save$frail_lab <- y_cols[c]
        coefs_save$cv_repeat <- repeats[p]
        coefs_save$fold <- mg1$fold[r]
        coefs_save$SVD <- mg1$svd[r]
        coefs_save$alpha <- mg1$alpha[r]
        coefs_save$case_weights <- mg1$case_weights[r]
        fwrite(coefs_save, 
               paste0(enet_coefsdir, 'exp', exp, '_coefs_r', repeats[p], '_f',
                      mg1$fold[r], '_', y_cols[c], '_svd_', mg1$svd[r], '_alpha',
                      mg1$alpha_l[r], '_cw', as.integer(mg1$case_weights[r]), '.csv'))
      }
      #make predictions on validation fold for each alpha
      alpha_preds <- predict(frail_logit, x_validation, type = 'response')
      #set lambas as dimnames for 3rd dimension
      dimnames(alpha_preds)[[3]] <- lambda_seq
      #save predictions
      preds_s <- list()
      for (d in 1:dim(alpha_preds)[3]) {
        preds_save <- as.data.table(alpha_preds[, , d])
        preds_save$lambda <- dimnames(alpha_preds)[[3]][d]
        preds_save$sentence_id <- get(
          paste0('r', repeats[p], '_f', mg1$fold[r], '_va'))$sentence_id
        preds_s[[d]] <- preds_save
      }
      preds_save <- rbindlist(preds_s)
      fwrite(preds_save, 
             paste0(enet_predsdir, 'exp', exp, '_preds_r', repeats[p], '_f', 
                    mg1$fold[r], '_', mg1$frail_lab[r], '_svd_', mg1$svd[r], 
                    '_alpha', mg1$alpha_l[r], '_cw', 
                    as.integer(mg1$case_weights[r]), '.csv'))
      #build hyperparameter grid
      hyper_grid <- expand.grid(
        frail_lab = NA,
        cv_repeat = NA,
        fold = NA,
        SVD = NA,
        lambda = rep(NA, dim(alpha_preds)[3]), #lambdas are in the 3rd dimension of this array
        alpha = NA,
        case_weights = NA,
        bscore_multi = NA,
        bscore_neut = NA,
        bscore_pos = NA,
        bscore_neg = NA,
        sbrier_multi = NA,
        sbrier_neut = NA,
        sbrier_pos = NA,
        sbrier_neg = NA,
        PR_AUC_neut = NA,
        PR_AUC_pos = NA,
        PR_AUC_neg = NA,
        ROC_AUC_neut = NA,
        ROC_AUC_pos = NA,
        ROC_AUC_neg = NA
      )
      for (l in 1:dim(alpha_preds)[3]) {
        #label each row
        hyper_grid$frail_lab[l] <- mg1$frail_lab[r]
        hyper_grid$cv_repeat[l] <- repeats[p]
        hyper_grid$fold[l] <- mg1$fold[r]
        hyper_grid$SVD[l] <- mg1$svd[r]
        hyper_grid$alpha[l] <- mg1$alpha[r]
        hyper_grid$case_weights[l] <- mg1$case_weights[r]
        #lambda
        hyper_grid$lambda[l] <- frail_logit$lambda[l]
        #preds for this lambda
        preds <- alpha_preds[, , l]
        #check for missing values in preds and relevant obs in validation set
        if (((sum(is.na(preds)) > 0) == FALSE) &
            ((sum(y_validation[, 2]) > 0) == TRUE) & 
            ((sum(y_validation[, 3]) > 0) == TRUE))  {
          #single class Brier scores
          hyper_grid$bscore_neut[l] = Brier(preds[, 1], y_validation[, 1], 1)
          hyper_grid$bscore_pos[l] = Brier(preds[, 2], y_validation[, 2], 1)
          hyper_grid$bscore_neg[l] = Brier(preds[, 3], y_validation[, 3], 1)
          #single class scaled Brier scores
          hyper_grid$sbrier_neut[l] = scaled_Brier(preds[, 1], y_validation[, 1], 1)
          hyper_grid$sbrier_pos[l] = scaled_Brier(preds[, 2], y_validation[, 2], 1)
          hyper_grid$sbrier_neg[l] = scaled_Brier(preds[, 3], y_validation[, 3], 1)
          #multiclass brier score
          hyper_grid$bscore_multi[l] <- multi_Brier(preds, y_validation)
          #multiclass scaled brier score
          hyper_grid$sbrier_multi[l] <- multi_scaled_Brier(preds, y_validation)
          #Precision-recall area under the curve
          hyper_grid$PR_AUC_neut[l] = pr.curve(scores.class0 = preds[, 1],
                                               weights.class0 = y_validation[, 1])$auc.integral
          hyper_grid$PR_AUC_pos[l] = pr.curve(scores.class0 = preds[, 2],
                                              weights.class0 = y_validation[, 2])$auc.integral
          hyper_grid$PR_AUC_neg[l] = pr.curve(scores.class0 = preds[, 3],
                                              weights.class0 = y_validation[, 3])$auc.integral
          #Receiver operating characteristic area under the curve
          hyper_grid$ROC_AUC_neut[l] = roc.curve(scores.class0 = preds[, 1],
                                                 weights.class0 = y_validation[, 1])$auc
          hyper_grid$ROC_AUC_pos[l] = roc.curve(scores.class0 = preds[, 2],
                                                weights.class0 = y_validation[, 2])$auc
          hyper_grid$ROC_AUC_neg[l] = roc.curve(scores.class0 = preds[, 3],
                                                weights.class0 = y_validation[, 3])$auc
        }
      }
      
      #save hyper_grid for each glmnet run
      fwrite(hyper_grid, 
             paste0(enet_modeldir, 'exp', exp, '_hypergrid_r', repeats[p], '_f',
                    mg1$fold[r], '_', mg1$frail_lab[r], '_svd_', mg1$svd[r], '_alpha',
                    mg1$alpha_l[r], '_cw', as.integer(mg1$case_weights[r]), '.csv'))
      #remove objects & garbage collection
      rm(x_train, x_validation, y_train, y_validation, frail_logit, benchmark, 
         alpha_preds, preds_save, preds_s, preds, hyper_grid)
      },
      
      #writing generic error messages to trace back later. Was not able to get 
      #foreach to reliably output the real error message AND where it occurred
      error = function(cond) {
        return(
          paste0('error in: exp', exp, '_r', repeats[p], '_f', mg1$fold[r], '_', 
                 mg1$frail_lab[r], '_svd_', mg1$svd[r], '_alpha', mg1$alpha_l[r], 
                 '_cw', as.integer(mg1$case_weights[r])))
      },
      warning = function(cond) {
        return(
          paste0('warning in: exp', exp, '_r', repeats[p], '_f', mg1$fold[r], 
                 '_', mg1$frail_lab[r], '_svd_', mg1$svd[r], '_alpha', 
                 mg1$alpha_l[r], '_cw', as.integer(mg1$case_weights[r])))
      })
      return(tc_error)
    }
    
    fwrite(as.data.table(enet_error), 
           paste0(outdir, 'exp', exp, '_enet_error_r', repeats[p], '.txt'))
  }
  
  invisible(gc(verbose = FALSE))
  
  #remove all objects related to current CV repeat
  objects_cv_repeat <- grep(paste0('r', repeats[p], '_f'), names(.GlobalEnv), value=TRUE)
  rm(list = objects_cv_repeat)
  invisible(invisible(gc(verbose = FALSE)))
}

#RANDOM FOREST - summary
#Summarize performance for all completed RF models
rf_output <- grep('_hypergrid_', list.files(rf_modeldir), value = TRUE)
rf_output <- lapply(paste0(rf_modeldir, rf_output), fread)
rf_output <- rbindlist(rf_output)
fwrite(rf_output, paste0(outdir, 'exp', exp, '_rf_performance.csv'))
#Summarize benchmarking for all completed RF models
rf_bench <- grep('_duration_hyper_', list.files(rf_durationdir), value = TRUE)
rf_bench <- lapply(paste0(rf_durationdir, rf_bench), fread)
rf_bench <- rbindlist(rf_bench)
fwrite(rf_bench, paste0(outdir, 'exp', exp, '_rf_cpu_time.csv'))
#Summarize variable importance
rf_importance <- grep('_importance_r', list.files(rf_importancedir), value = TRUE)
rf_importance <- lapply(paste0(rf_importancedir, rf_importance), fread)
rf_importance <- rbindlist(rf_importance, fill = TRUE)
fwrite(rf_importance, paste0(outdir, 'exp', exp, '_rf_importance.csv'))

#ELASTIC NET - summary
#Summarize performance for all completed enet models
enet_output <- grep('_hypergrid_', list.files(enet_modeldir), value = TRUE)
enet_output <- lapply(paste0(enet_modeldir, enet_output), fread)
enet_output <- rbindlist(enet_output)
fwrite(enet_output, paste0(outdir, 'exp', exp, '_enet_performance.csv'))
#Summarize benchmarking for all completed enet models
enet_bench <- grep('_duration_hyper_', list.files(enet_durationdir), value = TRUE)
enet_bench <- lapply(paste0(enet_durationdir, enet_bench), fread)
enet_bench <- rbindlist(enet_bench)
fwrite(enet_bench, paste0(outdir, 'exp', exp, '_enet_cpu_time.csv'))
#Summarize coefficients
enet_coefs <- grep('_coefs_r', list.files(enet_coefsdir), value = TRUE)
enet_coefs <- lapply(paste0(enet_coefsdir, enet_coefs), fread)
enet_coefs <- rbindlist(enet_coefs, fill = TRUE)
fwrite(enet_coefs, paste0(outdir, 'exp', exp, '_enet_coefs.csv'))


