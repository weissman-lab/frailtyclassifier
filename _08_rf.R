
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
batchstring <- commandArgs(trailingOnly = TRUE)
#test if there is an exp number argument: if not, return an error
if (length(batchstring)==0) {
  stop("AL round must be specified as an argument", call.=FALSE)
}
#repeats & folds
REPEATS <- seq(1, 3)
FOLDS <- seq(0, 9)
#text features
SVD <- c('embed', '300', '1000')
#include structured data?
INC_STRUC = TRUE

#set directories based on location
dirs = c(paste0('./output/saved_models/', batchstring, '/'),
         paste0('/gwshare/frailty/output/saved_models/', batchstring, '/'),
         '/Users/martijac/Documents/Frailty/frailty_classifier/output/',
         paste0('/Users/crandrew/projects/GW_PAIR_frailty_classifier/output/saved_models/', batchstring, '/'),
         '/media/drv2/andrewcd2/frailty/output/')
for (d in 1:length(dirs)) {
  if (dir.exists(dirs[d])) {
    rootdir = dirs[d]
  }
}
datadir <- paste0(rootdir, 'processed_data/')
SVDdir <- paste0(datadir, 'svd/') 
embeddingsdir <- paste0(datadir, 'embeddings/')
trvadatadir <- paste0(datadir, 'trvadata/')
cwdir <- paste0(datadir, 'caseweights/')
#new output directory for each experiment:
outdir <- paste0(rootdir, 'lin_trees/')
#directory for performance for each rf model:
rf_modeldir <- paste0(outdir,'rf_models/')
#directory for duration for each rf model:
rf_durationdir <- paste0(outdir,'rf_durations/')
#directory for variable importance:
rf_importancedir <- paste0(outdir,'rf_importance/')
#directory for predictions:
rf_predsdir <- paste0(outdir,'rf_preds/')
#directory for clobber check:
rf_clobberdir <- paste0(outdir,'rf_clobber/')
#directory for performance for each enet model:
enet_modeldir <- paste0(outdir,'enet_models/')
#directory for duration for each enet model:
enet_durationdir <- paste0(outdir,'enet_durations/')
#directory for coefficients:
enet_coefsdir <- paste0(outdir,'enet_coefs/')
#directory for predictions:
enet_predsdir <- paste0(outdir,'enet_preds/')
#directory for clobber check:
enet_clobberdir <- paste0(outdir,'enet_clobber/')
#make directories
dir.create(outdir)
dir.create(rf_durationdir)
dir.create(rf_modeldir)
dir.create(rf_importancedir)
dir.create(rf_predsdir)
dir.create(rf_clobberdir)
dir.create(enet_modeldir)
dir.create(enet_durationdir)
dir.create(enet_coefsdir)
dir.create(enet_predsdir)
dir.create(enet_clobberdir)


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



load_labels_caseweights <- function(rep, fold){
  tr <- fread(paste0(trvadatadir, 'r', rep, '_f', fold, '_tr_df.csv'))
  va <- fread(paste0(trvadatadir, 'r', rep, '_f', fold, '_va_df.csv'))
  cw <- fread(paste0(cwdir, 'r', rep, '_f', fold, '_tr_caseweights.csv'))
  if (nrow(tr) != nrow(cw)){
    stop("caseweights do not match training data")    
  }
  return(list(tr = tr, va = va, cw = cw))
}




load_x_embeddings <- function(rep, fold, lab_cw){
  embeddings_tr <- fread(paste0(embeddingsdir, 'r', rep, '_f', fold, '_tr_embed_min_max_mean_SENT.csv'))
  pca_tr <- lab_cw$tr
  emb_cols <- grep('min_|max_|mean_', colnames(embeddings_tr), value = TRUE)
  pca_cols <- grep('pca', colnames(pca_tr), value = TRUE)
  # test that embeddings match structured data
  if (identical(pca_tr$sentence_id, embeddings_tr$sentence_id) == FALSE)
    stop("embeddings do not match training data")
  #embeddings without vs with structured data
  if (INC_STRUC == FALSE) {
    #get only embeddings columns
    embeddings_tr <- data.matrix(embeddings_tr[, ..emb_cols])
  } else {
    # concatenate embeddings with structured data
    embeddings_tr <- data.matrix(cbind(embeddings_tr[, ..emb_cols], pca_tr[, ..pca_cols]))
  }
  embeddings_va <- fread(paste0(embeddingsdir, 'r', rep, '_f', fold, '_va_embed_min_max_mean_SENT.csv'), drop = 1)
  pca_va <- lab_cw$va
  emb_cols <- grep('min_|max_|mean_', colnames(embeddings_va), value = TRUE)
  pca_cols <- grep('pca', colnames(pca_va), value = TRUE)
  # test that embeddings match structured data
  if (identical(pca_va$sentence_id, embeddings_va$sentence_id) == FALSE)
    stop("embeddings do not match validation data")
  #embeddings without vs with structured data
  if (INC_STRUC == FALSE) {
    #get only embeddings columns
    embeddings_va <- data.matrix(embeddings_va[, ..emb_cols])
  } else {
    # concatenate embeddings with structured data
    embeddings_va <- data.matrix(cbind(embeddings_va[, ..emb_cols], pca_va[, ..pca_cols]))
  }
  outlist <- list(tr = embeddings_tr,
                  va = embeddings_va)
  return(outlist)
}

load_x_svd <- function(rep, fold, lab_cw, svd_dim){
  svd_tr <- fread(paste0(SVDdir, 'r', rep, '_f', fold, '_tr_svd', svd_dim, '.csv'),
                  skip = 1,
                  drop = 1)
  colnames(svd_tr) <- c('sentence_id', paste0('svd', seq(1:as.integer(svd_dim))))
  pca_tr <- lab_cw$tr
  svd_cols <- grep('svd', colnames(svd_tr), value = TRUE)
  pca_cols <- grep('pca', colnames(pca_tr), value = TRUE)
  #test that svd match structured data
  if (identical(pca_tr$sentence_id, svd_tr$sentence_id) == FALSE)
    stop("embeddings do not match validation data")
  #svd without vs with structured data
  if (INC_STRUC == FALSE) {
    #load only the svd columns
    svd_tr <- data.matrix(svd_tr[, ..svd_cols])
  } else {
    #concatenate svd with structured data
    svd_tr <- data.matrix(cbind(svd_tr[, ..svd_cols], pca_tr[, ..pca_cols]))
  }
  svd_va <- fread(paste0(SVDdir, 'r', rep, '_f', fold, '_va_svd', svd_dim, '.csv'),
                  skip = 1,
                  drop = 1)
  colnames(svd_va) <- c('sentence_id', paste0('svd', seq(1:as.integer(svd_dim))))
  pca_va <- lab_cw$va
  svd_cols <- grep('svd', colnames(svd_va), value = TRUE)
  pca_cols <- grep('pca', colnames(pca_va), value = TRUE)
  #test that svd match structured data
  if (identical(pca_va$sentence_id, svd_va$sentence_id) == FALSE)
    stop("embeddings do not match validation data")
  #svd without vs with structured data
  if (INC_STRUC == FALSE) {
    #load only the svd columns
    svd_va <- data.matrix(svd_va[, ..svd_cols])
  } else {
    #concatenate svd with structured data
    svd_va <- data.matrix(cbind(svd_va[, ..svd_cols], pca_va[, ..pca_cols]))
  }
  outlist <- list(tr = embeddings_tr,
                  va = embeddings_va)
  return(outlist)
}

load_x <- function(rep, fold, lab_cw, svd_dim = NULL){
  if (is.null(svd_dim) | svd_dim == "embed"){
    return(load_x_embeddings(rep, fold, lab_cw))
  } else {
    return(load_x_svd(rep, fold, lab_cw, svd_dim))
  }
}

rf <- function(rep, fold, frail_lab, svd, ntree, mtry, sample_frac, case_weights, sample_frac_l){
  # fold = 2
  # rep = 2
  # frail_lab = 'Msk_prob'
  # svd = 'embed'
  # ntree = 400
  # mtry = 7
  # case_weights = FALSE
  # sample_frac = 1
  # sample_frac_l = 10
  print(paste0(c("starting",match.call()), collapse = " "))
  fname_completed <- paste0(rf_modeldir, 
                            'exp', 
                            batchstring, 
                            '_hypergrid_r', 
                            rep, 
                            '_f',
                            fold, 
                            '_', 
                            frail_lab, 
                            '_svd_', 
                            svd,
                            '_mtry_', 
                            mtry, 
                            '_sfrac_', 
                            sample_frac_l,
                            '_cw_',  
                            as.integer(case_weights), 
                            '.csv')  
  if (file.exists(fname_completed)){
    return(NULL)
  }
  #write to prevent clobber
  fname_clobber <- gsub(rf_modeldir, rf_clobberdir, fname_completed)
  fname_clobber <- gsub("_hypergrid_", "_clobber_", fname_clobber)
  fwrite(data.frame(clb = 1), fname_clobber)
  #get matching training and validation labels
  lab_cw <- load_labels_caseweights(fold = fold, rep = rep)
  y_cols <- c(paste0(frail_lab, '_neut'),
              paste0(frail_lab, '_pos'),
              paste0(frail_lab, '_neg'))
  y_train <- lab_cw$tr[, ..y_cols] %>%
    as_tibble() %>%
    mutate(factr = ifelse(.[[1]] == 1, 1,
                          ifelse(.[[2]] == 1, 2,
                                 ifelse(.[[3]] == 1, 3, NA))))
  y_train_factor <- as.factor(y_train$factr)
  y_validation <- lab_cw$va[, ..y_cols]
  # load X
  X <- load_x(rep, fold, lab_cw, svd)
  if (case_weights == TRUE){
    cw <- lab_cw$cw
  } else {
    cw <- NULL
  }
  benchmark <- benchmark("rf" = {
    frail_rf <- ranger(y = y_train_factor,
                       x = X$tr,
                       num.threads = detectCores(),
                       probability = TRUE,
                       num.trees = ntree,
                       mtry = mtry,
                       sample.fraction = sample_frac,
                       case.weights = cw,
                       oob.error = FALSE,
                       importance = 'impurity')
  }, replications = 1)
  #save benchmarking
  fname_duration <- gsub(rf_modeldir, rf_durationdir, fname_completed)
  fname_duration <- gsub("_hypergrid_", "_duration_hyper_", fname_duration)  
  fwrite(benchmark, fname_duration)
  
  #save variable importance
  importance <- importance(frail_rf)
  i_names <- names(importance)
  importance <- transpose(as.data.table(importance))
  colnames(importance) <- i_names
  importance$cv_repeat <- rep
  importance$fold <- fold
  importance$SVD <- svd
  importance$mtry <- mtry
  importance$sample_frac <- sample_frac
  importance$case_weights <- case_weights
  fname_importance <- gsub(rf_modeldir, rf_importancedir, fname_completed)
  fname_importance <- gsub("_hypergrid_", "_importance_", fname_importance)  
  fwrite(importance, fname_importance)
  
  #make predictions on validation fold
  preds <- predict(frail_rf, data=X$va)$predictions
  colnames(preds) <- y_cols
  preds_save <- as.data.table(preds)
  preds_save$sentence_id <- lab_cw$va$sentence_id
  #save predictions
  fname_preds <- gsub(rf_modeldir, rf_importancedir, fname_completed)
  fname_preds <- gsub("_hypergrid_", "_preds_", fname_preds)  
  fwrite(as.data.table(preds_save), fname_preds)
  #label each row
  hyper_grid <- data.frame(frail_lab = frail_lab)
  hyper_grid$cv_repeat <- rep
  hyper_grid$fold <- fold
  hyper_grid$SVD <- svd
  hyper_grid$mtry <- mtry
  hyper_grid$sample_frac <- sample_frac
  hyper_grid$case_weights <- case_weights
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
  fname_hyper <- gsub(rf_modeldir, rf_importancedir, fname_completed)
  fname_hyper <- gsub("_hypergrid_", "_hypergrid_", fname_hyper)  
  fwrite(as.data.table(preds_save), fname_preds)
  fwrite(hyper_grid, fname_completed)
  invisible(invisible(gc(verbose = FALSE)))
  
  #delete clobber
  file.remove(fname_clobber)
  print(paste0(c("finished",match.call()), collapse = " "))
}

############################## RANDOM FOREST ##############################

#tuning grid
mg3 <- expand_grid(
  rep = REPEATS,
  fold = FOLDS,
  svd = SVD,
  frail_lab = c('Msk_prob', 'Fall_risk', 'Nutrition', 'Resp_imp'),
  ntree       = 2,
  mtry        = signif(seq(7, 45, length.out = 3), 2),
  sample_frac = signif(seq(0.6, 1, length.out = 3), 1),
  case_weights = c(TRUE, FALSE)
)

#label sample fraction (for naming .csv files)
mg3 <- mutate(mg3, sample_frac_l = ifelse(sample_frac == 0.6, 6,
                                          ifelse(sample_frac == 0.8, 8,
                                                 ifelse(sample_frac == 1.0, 10, NA))))

#randomize grid
mg3 <- mg3[sample(nrow(mg3)),]

for (r in 1:nrow(mg3)){
  do.call(rf, as.list(mg3[r,]))
}
