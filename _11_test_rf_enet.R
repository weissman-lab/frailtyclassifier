library(data.table)
library(dplyr)
library(tidyr)
library(stringr)
library(boot)
library(glmnet)
library(ranger)
library(PRROC)
library(foreach)
library(doParallel)
registerDoParallel(detectCores())

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

# bootstrap scaled Brier score
scaled_Brier_boot <- function(data, idx) {
  df <- data[idx, ]
  predictions <- df[, 1]
  observations <- df[, 2]
  1 - (Brier(predictions, observations, 1) / 
         Brier(mean(observations), observations, 1))
}

# bootstrap multiclass scaled Brier score
multi_scaled_Brier_boot <- function(data, idx) {
  df <- data[idx, ]
  predictions <- df[, 1:3]
  observations <- df[, 4:6]
  event_rate_matrix <- matrix(colMeans(observations), 
                              ncol = length(colMeans(observations)), 
                              nrow = nrow(observations),
                              byrow = TRUE)
  1 - (multi_Brier(predictions, observations) / 
         multi_Brier(event_rate_matrix, observations))
}

# set directories based on location
dirs = c(paste0('/gwshare/frailty/output/saved_models/'),
         '/Users/martijac/Documents/Frailty/frailty_classifier/output/saved_models/',
         '/media/drv2/andrewcd2/frailty/output/saved_models/')
for (d in 1:length(dirs)) {
  if (dir.exists(dirs[d])) {
    rootdir = dirs[d]
  }
}

#constants
frail_lab <- c('Msk_prob', 'Fall_risk', 'Nutrition', 'Resp_imp')
models <- c('enet', 'rf', 'nn_single')
batches <- c('AL01', 'AL02', 'AL03', 'AL04', 'AL05')

#gather enet performance
ep_list <- list()
for (b in 1:length(batches)){
  ep <- fread(paste0(rootdir,
                     batches[b], '/lin_trees_enet/exp',
                     batches[b], '_enet_performance.csv'))
  ep$model <- 'enet'
  ep$batch <- batches[b]
  ep_list[[b]] <- ep
}
enet_performance <- rbindlist(ep_list)

#gather rf performance
rf_list <- list()
for (b in 1:length(batches)){
  rf <- fread(paste0(rootdir,
                     batches[b], '/lin_trees/exp',
                     batches[b], '_rf_performance.csv'))
  rf$model <- 'rf'
  rf$batch <- batches[b]
  rf_list[[b]] <- rf
}
rf_performance <- rbindlist(rf_list)

#gather single-task NN performance
nn_single_performance_w2v <- fread(paste0(rootdir,
                                      tail(batches, 1),
                                      '/learning_curve_stask.csv'))
nn_single_performance_w2v$text <- 'word2vec'
nn_single_performance_bert <- fread(paste0(rootdir,
                                      tail(batches, 1),
                                      '/learning_curve_stask_bert.csv'))
nn_single_performance_bert$text <- 'BioClinicalBERT'
nn_single_performance_roberta <- fread(paste0(rootdir,
                                           tail(batches, 1),
                                           '/learning_curve_stask_roberta.csv'))
nn_single_performance_roberta$text <- 'RoBERTa'
nn_single_performance <- rbind(nn_single_performance_w2v,
                               nn_single_performance_bert,
                               nn_single_performance_roberta)
nn_single_performance[, 'sbrier_neg'] <- 
  apply(nn_single_performance[, .SD, .SDcols = (paste0(frail_lab, '_neg'))],
        1, max, na.rm=TRUE)
nn_single_performance[, 'sbrier_neut'] <- 
  apply(nn_single_performance[, .SD, .SDcols = (paste0(frail_lab, '_neut'))],
        1, max, na.rm=TRUE)
nn_single_performance[, 'sbrier_pos'] <- 
  apply(nn_single_performance[, .SD, .SDcols = (paste0(frail_lab, '_pos'))],
        1, max, na.rm=TRUE)
nn_single_performance[, 'sbrier_multi'] <- nn_single_performance[, 'brier_all']
nn_single_performance[, 'frail_lab'] <- str_sub(nn_single_performance$tags, 3, -3)
nn_single_performance[, 'model'] <- 'nn_single'
nn_single_performance[, 'cv_repeat'] <- nn_single_performance[, 'repeat']


#list text features/embeddings
rf_enet_text <- c('embed', '300', '1000')
nn_text <- c('word2vec', 'BioClinicalBERT', 'RoBERTa')

#list relevant hyperparams
hyperparams_enet <- c('SVD', 'lambda', 'alpha', 'case_weights')
hyperparams_rf <- c('SVD', 'mtry', 'sample_frac', 'case_weights')
hyperparams_nn <- c('text', 'n_dense', 'n_units', 'dropout', 'l1_l2_pen',
                    'use_case_weights')

# step 1: find the best hyperparams for each aspect and calculate performance
# step 2: using the best hyperparams only, calcuLate the mean scaled brier
# across all aspects (sbrier_multi_all) for each fold
# step 3: find the mean and sd for performance across all folds
# note: this should be run separately for each batch & text feature
perf_calc_MEAN <- function(raw_perf){
  #drop repeat 3 fold 9 (performance outlier - likely collinearity issue)
  raw_perf <- filter(raw_perf, !(cv_repeat == 3 & fold == 9))
  raw_perf <- as.data.frame(raw_perf)
  if (raw_perf$model[1] == 'enet') {
    hyperparams = hyperparams_enet
  } else if (raw_perf$model[1] == 'rf') {
    hyperparams = hyperparams_rf
  } else if (raw_perf$model[1] == 'nn_single') {
    hyperparams = hyperparams_nn
    }
  # step 1:
  group1 <- c('frail_lab', hyperparams)
  step_1 <- raw_perf %>%
    group_by_at(vars(all_of(group1))) %>%
    summarise_at(vars(grep('sbrier', colnames(raw_perf), value = TRUE)),
                 list(mean = mean, sd = sd), na.rm = TRUE) %>%
    na.omit() %>%
    ungroup() %>%
    group_by(frail_lab) %>%
    arrange(desc(sbrier_multi_mean), .by_group = TRUE) %>%
    slice(1)
  step_1$model <- raw_perf$model[1]
  step_1$batch <- raw_perf$batch[1]
  #step 2:
  hypcols <- c('frail_lab', hyperparams)
  step_1$hyperp <- do.call(paste0, c(step_1[, hypcols]))
  raw_perf$hyperp <- do.call(paste0, c(raw_perf[, hypcols]))
  step_2 <- raw_perf %>%
    filter(hyperp %in% step_1$hyperp) %>%
    group_by(cv_repeat, fold) %>%
    summarise_at(vars(grep('sbrier', colnames(raw_perf), value = TRUE)),
                 list(all = mean), na.rm = TRUE) %>%
    ungroup()
  #step 3:
  step_3 <- step_2 %>%
    summarise_at(vars(grep('sbrier', colnames(step_2), value = TRUE)),
                 list(mean = mean, sd = sd))
  step_3$model <- raw_perf$model[1]
  step_3$batch <- raw_perf$batch[1]
  return(list(select(step_1, -'hyperp'), step_3))
}

# CROSS VALIDATION HYPERPARAMETERS SUMMARY
#Get best enet hyperparams
txt_l <- list()
for (t in 1:length(rf_enet_text)) {
  perf_l <- list()
  for (b in 1:length(batches)) {
    perf <- perf_calc_MEAN(enet_performance[(batch == batches[b] &
                                              SVD == rf_enet_text[t]),])[[1]]
    perf_l[[b]] <- perf
  }
  txt_l <- c(txt_l, perf_l)
}
enet_hyperparams <- rbindlist(txt_l)

#Get best rf hyperparams
txt_l <- list()
for (t in 1:length(rf_enet_text)) {
  perf_l <- list()
  for (b in 1:length(batches)) {
    perf <- perf_calc_MEAN(rf_performance[(batch == batches[b] &
                                            SVD == rf_enet_text[t]),])[[1]]
    perf_l[[b]] <- perf
  }
  txt_l <- c(txt_l, perf_l)
}
rf_hyperparams <- rbindlist(txt_l)

#Get best nn single task hyperparams
txt_l <- list()
for (t in 1:length(nn_text)) {
  perf_l <- list()
  for (b in 1:length(batches)) {
    perf <- perf_calc_MEAN(nn_single_performance[(batch == batches[b] &
                                                    text == nn_text[t]),])[[1]]
    perf_l[[b]] <- perf
  }  
  txt_l <- c(txt_l, perf_l)
}
nn_single_hyperparams <- rbindlist(txt_l)

#Get best nn multi task hyperparams
# (fewer steps to summarize because all aspects get the same hyperparams)
nn_multi_performance_w2v <- fread(paste0(rootdir,
                                          tail(batches, 1),
                                          '/learning_curve_mtask.csv'))
nn_multi_performance_w2v$text <- 'word2vec'
nn_multi_performance_bert <- fread(paste0(rootdir,
                                           tail(batches, 1),
                                           '/learning_curve_mtask_bert.csv'))
nn_multi_performance_bert$text <- 'BioClinicalBERT'
nn_multi_performance_roberta <- fread(paste0(rootdir,
                                              tail(batches, 1),
                                              '/learning_curve_mtask_roberta.csv'))
nn_multi_performance_roberta$text <- 'RoBERTa'
nn_multi_performance <- rbind(nn_multi_performance_w2v,
                              nn_multi_performance_bert,
                              nn_multi_performance_roberta)
nn_multi_performance[, 'cv_repeat'] <- nn_multi_performance[, 'repeat']
#drop repeat 3 fold 9 (performance outlier - likely collinearity issue)
nn_multi_performance <- filter(nn_multi_performance, !(cv_repeat == 3 & fold == 9))
nn_multi_performance[, 'sbrier_multi_all'] <- nn_multi_performance[, 'brier_mean_aspects']
group <- c('batch', hyperparams_nn)
nn_multi_hyperparams <- nn_multi_performance %>%
  group_by_at(vars(all_of(group))) %>%
  summarise_at(vars(grep('Fall_risk|Msk_prob|Nutrition|Resp_imp|brier', colnames(nn_multi_performance), value = TRUE)),
               list(mean = mean,
                    sd = sd)) %>%
  ungroup() %>%
  group_by(batch, text) %>%
  arrange(desc(sbrier_multi_all_mean)) %>%
  slice(1) %>%
  ungroup()
nn_multi_hyperparams$model <- 'nn_multi'


# CROSS VALIDATION PERFORMANCE SUMMARY
mod_l <- list()
for (m in 1:length(models)) {
  txt_l <- list()
  if (models[m] == 'nn_single') {
    text = nn_text
  } else {
    text = rf_enet_text
  }
  for (t in 1:length(text)) {
    perf_l <- list()
    for (b in 1:length(batches)) {
      if (models[m] == 'nn_single') {
      perf <- perf_calc_MEAN(get(paste0(models[m], '_performance'))[batch == batches[b] &
                                                                      text == text[t],])
      } else {
      perf <- perf_calc_MEAN(get(paste0(models[m], '_performance'))[batch == batches[b] &
                                                                      SVD == text[t],])
      }
       perf <- perf[[2]][c('model', 'batch', 'sbrier_multi_all_mean',
                                 'sbrier_multi_all_sd')]
       perf$text <- text[t]
       perf_l[[b]] <- perf
    }
    txt_l <- c(txt_l, perf_l)
  }
  mod_l <- c(mod_l, txt_l)
}
train_performance_mean <- rbindlist(mod_l)
nn_multi_perf <- select(nn_multi_hyperparams,
                        c('batch', 'model',
                          'text',
                          'sbrier_multi_all_mean',
                          'sbrier_multi_all_sd'))
train_performance_mean <- rbind(train_performance_mean, nn_multi_perf)

fwrite(train_performance_mean,
       paste0('/gwshare/frailty/output/figures_tables/all_train_cv_performance.csv'))



# TEST SET PERFORMANCE

boot_reps <- 1000

#multi-task neural nets
bbs <- list()
for (b in 1:length(batches)){
  tbs <- list()
  for (t in 1:length(nn_text)){
    if (nn_text[t] == 'word2vec') { tl <- ''
    } else if (nn_text[t] == 'BioClinicalBERT') { tl <- '_bioclinicalbert'
    } else if (nn_text[t] == 'RoBERTa') { tl <- '_roberta'}
    test_preds <- fread(paste0(rootdir, batches[b],
                               '/final_model/test_preds/test_preds_',
                               batches[b],
                               tl,
                               '.csv'))
    sbs <- list()
    for (f in 1:length(frail_lab)){
      neg <- grep(paste0(frail_lab[f], '_neg'), colnames(test_preds), value = TRUE)
      neut <- grep(paste0(frail_lab[f], '_neut'), colnames(test_preds), value = TRUE)
      pos <- grep(paste0(frail_lab[f], '_pos'), colnames(test_preds), value = TRUE)
      cols <- c(neg, neut, pos)
      preds <- grep('_pred', cols, value = TRUE)
      preds <- test_preds[, ..preds]
      obs <- grep('_pred', cols, value = TRUE, invert = TRUE)
      obs <- test_preds[, ..obs]
      sb <- data.frame(frail_lab = frail_lab[f])
      sb$batch <- batches[b]
      sb$hyperparams <- mutate(nn_multi_hyperparams[nn_multi_hyperparams$batch == batches[b]&
                                                      nn_multi_hyperparams$text == nn_text[t], ],
                               hyperparams = paste0(text,
                                                    ' dense_', n_dense,
                                                    ' units_', n_units,
                                                    ' dropout_', dropout,
                                                    ' l1l2_', l1_l2_pen,
                                                    ' cw_', use_case_weights))$hyperparams
      #Scaled Brier scores with bootstrapped CIs
      b_multi <- cbind(preds, obs)
      sbrier_multi <- boot(b_multi, multi_scaled_Brier_boot, R = boot_reps, parallel = 'multicore')
      sb$sbrier_multi <- sbrier_multi$t0
      sb$sbrier_multi_sd <- sd(sbrier_multi$t)
      b_neg <- cbind(preds[[1]], obs[[1]])
      sbrier_neg <- boot(b_neg, scaled_Brier_boot, R = boot_reps, parallel = 'multicore')
      sb$sbrier_neg <- sbrier_neg$t0
      sb$sbrier_neg_sd <- sd(sbrier_neg$t)
      b_neut <- cbind(preds[[2]], obs[[2]])
      sbrier_neut <- boot(b_neut, scaled_Brier_boot, R = boot_reps, parallel = 'multicore')
      sb$sbrier_neut <- sbrier_neut$t0
      sb$sbrier_neut_sd <- sd(sbrier_neut$t)
      b_pos <- cbind(preds[[3]], obs[[3]])
      sbrier_pos <- boot(b_pos, scaled_Brier_boot, R = boot_reps, parallel = 'multicore')
      sb$sbrier_pos <- sbrier_pos$t0
      sb$sbrier_pos_sd <- sd(sbrier_pos$t)
      #Precision-recall area under the curve
      sb$PR_AUC_neg = pr.curve(scores.class0 = preds[[1]],
                               weights.class0 = obs[[1]])$auc.integral
      sb$PR_AUC_neut = pr.curve(scores.class0 = preds[[2]],
                                        weights.class0 = obs[[2]])$auc.integral
      sb$PR_AUC_pos = pr.curve(scores.class0 = preds[[3]],
                                       weights.class0 = obs[[3]])$auc.integral
      #Receiver operating characteristic area under the curve
      sb$ROC_AUC_neg = roc.curve(scores.class0 = preds[[1]],
                                 weights.class0 = obs[[1]])$auc
      sb$ROC_AUC_neut = roc.curve(scores.class0 = preds[[2]],
                                  weights.class0 = obs[[2]])$auc
      sb$ROC_AUC_pos = roc.curve(scores.class0 = preds[[3]],
                                 weights.class0 = obs[[3]])$auc
      sbs[[f]] <- sb
    }
    sbs_l <- rbindlist(sbs)
    tbs[[t]] <- sbs_l
  }
  tbs_l <- rbindlist(tbs)
  bbs[[b]] <- tbs_l
}
nn_multi_test_perf <- rbindlist(bbs)
nn_multi_test_perf$model <- 'nn_multi'


#single-task neural nets
bbs <- list()
for (b in 1:length(batches)){
  tbs <- list()
  for (t in 1:length(nn_text)){
    sbs <- list()
    for (f in 1:length(frail_lab)){
      if (nn_text[t] == 'word2vec') { tl <- ''
      } else if (nn_text[t] == 'BioClinicalBERT') { tl <- '_bioclinicalbert'
      } else if (nn_text[t] == 'RoBERTa') { tl <- '_roberta'}
      test_preds <- fread(paste0(rootdir,
                                 batches[b],
                                 '/final_model/test_preds/test_preds_',
                                 batches[b],
                                 '_',
                                 frail_lab[f],
                                 tl,
                                 '.csv'))
      neg <- grep(paste0(frail_lab[f], '_neg'), colnames(test_preds), value = TRUE)
      neut <- grep(paste0(frail_lab[f], '_neut'), colnames(test_preds), value = TRUE)
      pos <- grep(paste0(frail_lab[f], '_pos'), colnames(test_preds), value = TRUE)
      cols <- c(neg, neut, pos)
      preds <- grep('_pred', cols, value = TRUE)
      preds <- test_preds[, ..preds]
      obs <- grep('_pred', cols, value = TRUE, invert = TRUE)
      obs <- test_preds[, ..obs]
      sb <- data.frame(frail_lab = frail_lab[f])
      sb$batch <- batches[b]
      sb$hyperparams <- mutate(nn_single_hyperparams[nn_single_hyperparams$batch == batches[b] &
                                                       nn_single_hyperparams$text == nn_text[t] &
                                                       nn_single_hyperparams$frail_lab == frail_lab[f], ],
                               hyperparams = paste0(text,
                                                    ' dense_', n_dense,
                                                    ' units_', n_units,
                                                    ' dropout_', dropout,
                                                    ' l1l2_', l1_l2_pen,
                                                    ' cw_', use_case_weights))$hyperparams
      #Scaled Brier scores with bootstrapped CIs
      b_multi <- cbind(preds, obs)
      sbrier_multi <- boot(b_multi, multi_scaled_Brier_boot, R = boot_reps, parallel = 'multicore')
      sb$sbrier_multi <- sbrier_multi$t0
      sb$sbrier_multi_sd <- sd(sbrier_multi$t)
      b_neg <- cbind(preds[[1]], obs[[1]])
      sbrier_neg <- boot(b_neg, scaled_Brier_boot, R = boot_reps, parallel = 'multicore')
      sb$sbrier_neg <- sbrier_neg$t0
      sb$sbrier_neg_sd <- sd(sbrier_neg$t)
      b_neut <- cbind(preds[[2]], obs[[2]])
      sbrier_neut <- boot(b_neut, scaled_Brier_boot, R = boot_reps, parallel = 'multicore')
      sb$sbrier_neut <- sbrier_neut$t0
      sb$sbrier_neut_sd <- sd(sbrier_neut$t)
      b_pos <- cbind(preds[[3]], obs[[3]])
      sbrier_pos <- boot(b_pos, scaled_Brier_boot, R = boot_reps, parallel = 'multicore')
      sb$sbrier_pos <- sbrier_pos$t0
      sb$sbrier_pos_sd <- sd(sbrier_pos$t)
      #Precision-recall area under the curve
      sb$PR_AUC_neg = pr.curve(scores.class0 = preds[[1]],
                               weights.class0 = obs[[1]])$auc.integral
      sb$PR_AUC_neut = pr.curve(scores.class0 = preds[[2]],
                                weights.class0 = obs[[2]])$auc.integral
      sb$PR_AUC_pos = pr.curve(scores.class0 = preds[[3]],
                               weights.class0 = obs[[3]])$auc.integral
      #Receiver operating characteristic area under the curve
      sb$ROC_AUC_neg = roc.curve(scores.class0 = preds[[1]],
                                 weights.class0 = obs[[1]])$auc
      sb$ROC_AUC_neut = roc.curve(scores.class0 = preds[[2]],
                                  weights.class0 = obs[[2]])$auc
      sb$ROC_AUC_pos = roc.curve(scores.class0 = preds[[3]],
                                 weights.class0 = obs[[3]])$auc
      sbs[[f]] <- sb
    }
    sbs_l <- rbindlist(sbs)
    tbs[[t]] <- sbs_l
  }
  tbs_l <- rbindlist(tbs)
  bbs[[b]] <- tbs_l
}
nn_single_test_perf <- rbindlist(bbs)
nn_single_test_perf$model <- 'nn_single'


# # delete outdirs
# batches <- c('AL01', 'AL02', 'AL03', 'AL04', 'AL05')
# for (b in 1:length(batches)){
#   batch_root <- paste0(rootdir, batches[b], '/')
#   outdir <- paste0(batch_root, 'lin_trees_final_test/')
#   # enet_modeldir <- paste0(outdir,'enet_models/')
#   # enet_coefsdir <- paste0(outdir, 'enet_coefs/')
#   # enet_predsdir <- paste0(outdir, 'enet_preds/')
#   unlink(outdir, recursive = TRUE)
#   # unlink(enet_modeldir, recursive = TRUE)
#   # unlink(enet_coefsdir, recursive = TRUE)
#   # unlink(enet_predsdir, recursive = TRUE)
# }

# make outdirs
batches <- c('AL01', 'AL02', 'AL03', 'AL04', 'AL05')
for (b in 1:length(batches)){
  batch_root <- paste0(rootdir, batches[b], '/')
  outdir <- paste0(batch_root, 'lin_trees_final_test/')
  dir.create(outdir)
  
  enet_modeldir <- paste0(outdir,'enet_models/')
  enet_coefsdir <- paste0(outdir, 'enet_coefs/')
  enet_predsdir <- paste0(outdir, 'enet_preds/')
  dir.create(enet_modeldir)
  dir.create(enet_coefsdir)
  dir.create(enet_predsdir)
  
  rf_modeldir <- paste0(outdir,'rf_models/')
  rf_importancedir <- paste0(outdir, 'rf_importance/')
  rf_predsdir <- paste0(outdir, 'rf_preds/')
  dir.create(rf_modeldir)
  dir.create(rf_importancedir)
  dir.create(rf_predsdir)
}

#check for models that have already been completed & remove them from the grid
rf_hyperparams_full <- rf_hyperparams
for (b in 1:length(batches)){
  batch_root <- paste0(rootdir, batches[b], '/')
  outdir <- paste0(batch_root, 'lin_trees_final_test/')
  rf_modeldir <- paste0(outdir,'rf_models/')
  rf_hyperparams <- rf_hyperparams %>%
    mutate(filename = 
             paste0(batch, '_', SVD, '_', frail_lab, '_performance.csv')) %>%
    filter(!filename %in% list.files(rf_modeldir))%>%
    select(-'filename')
}

#RFs
if ((nrow(rf_hyperparams) == 0) == FALSE) {
for (r in 1:nrow(rf_hyperparams)){
  #set directories
  batch_root <- paste0(rootdir, rf_hyperparams$batch[r], '/')
  outdir <- paste0(batch_root, 'lin_trees_final_test/')
  rf_modeldir <- paste0(outdir,'rf_models/')
  rf_importancedir <- paste0(outdir, 'rf_importance/')
  rf_predsdir <- paste0(outdir, 'rf_preds/')
  
  # get matching training and test data & labels
  train_df <- fread(paste0(
    batch_root, 'processed_data/full_set/full_df.csv'))
  pca_cols <- grep('pca', colnames(train_df), value = TRUE)
  test_df <- fread(paste0(
    batch_root, 'processed_data/test_set/full_df.csv'))
  pca_cols_test <- grep('pca', colnames(test_df), value = TRUE)
  if (identical(pca_cols, pca_cols_test) == FALSE)
    stop("train PCA does not match test PCA")
  
  # load caseweights (training only)
  if (rf_hyperparams$case_weights[r] == TRUE){
    cw_train_only <- fread(paste0(
      batch_root, 'processed_data/full_set/full_caseweights.csv'))[[paste0(rf_hyperparams$frail_lab[r], '_cw')]]
    # check for matching
    if ((length(cw_train_only) == nrow(train_df)) == FALSE)
      stop("caseweights do not match training data")
  } else {
    cw_train_only <- NULL
  }
  
  # embeddings
  if (rf_hyperparams$SVD[r] == 'embed'){
    # x_train
    train_embed <- fread(paste0(
      batch_root, 'processed_data/full_set/full_embed_min_max_mean_SENT.csv'),
      drop = 1)
    if (identical(train_df$sentence_id, train_embed$sentence_id) == FALSE)
      stop("train embeddings do not match structured data")
    emb_cols <- grep('min_|max_|mean_', colnames(train_embed), value = TRUE)
    x_train <- data.matrix(cbind(train_embed[, ..emb_cols],
                                 train_df[, ..pca_cols]))
    # x_test
    test_embed <- fread(paste0(
      batch_root, 'processed_data/test_set/full_embed_min_max_mean_SENT.csv'),
      drop = 1)
    if (identical(test_df$sentence_id, test_embed$sentence_id) == FALSE)
      stop("test embeddings do not match structured data")
    emb_cols_test <- grep('min_|max_|mean_', colnames(test_embed), value = TRUE)
    if (identical(emb_cols, emb_cols_test) == FALSE) stop("train embed does not 
      match test embed")
    # remove extreme outliers from test PCA
    test_pca <- test_df[, ..pca_cols]
    test_pca[test_pca < -4] <- -4
    test_pca[test_pca > 4] <- 4
    x_test <- data.matrix(cbind(test_embed[, ..emb_cols], test_pca))
    
    # 300-d SVD
  } else if (rf_hyperparams$SVD[r] == '300'){
    # x_train
    train_svd_300 <- fread(paste0(
      batch_root, 'processed_data/full_set/full_svd300.csv'),
      skip = 1, drop = 1)
    colnames(train_svd_300) <- c('sentence_id', paste0('svd', seq(1:300)))
    if (identical(train_df$sentence_id, train_svd_300$sentence_id) == FALSE)
      stop("train 300-d svd does not match structured data")
    svd_cols <- grep('svd', colnames(train_svd_300), value = TRUE)
    x_train <- data.matrix(cbind(train_svd_300[, ..svd_cols],
                                 train_df[, ..pca_cols]))
    # x_test
    test_svd_300 <- fread(paste0(
      batch_root, 'processed_data/test_set/full_svd300.csv'),
      skip = 1, drop = 1)
    colnames(test_svd_300) <- c('sentence_id', paste0('svd', seq(1:300)))
    if (identical(test_df$sentence_id, test_svd_300$sentence_id) == FALSE)
      stop("test 300-d svd does not match structured data")
    # remove extreme outliers from test PCA
    test_pca <- test_df[, ..pca_cols]
    test_pca[test_pca < -4] <- -4
    test_pca[test_pca > 4] <- 4
    x_test <- data.matrix(cbind(test_svd_300[, ..svd_cols], test_pca))
    
    # 1000-d SVD
  } else if (rf_hyperparams$SVD[r] == '1000'){
    # x_train
    train_svd_1000 <- fread(paste0(
      batch_root, 'processed_data/full_set/full_svd1000.csv'),
      skip = 1, drop = 1)
    colnames(train_svd_1000) <- c('sentence_id', paste0('svd', seq(1:1000)))
    if (identical(train_df$sentence_id, train_svd_1000$sentence_id) == FALSE)
      stop("train 1000-d svd does not match structured data")
    svd_cols <- grep('svd', colnames(train_svd_1000), value = TRUE)
    x_train <- data.matrix(cbind(train_svd_1000[, ..svd_cols],
                                 train_df[, ..pca_cols]))
    # x_test
    test_svd_1000 <- fread(paste0(
      batch_root, 'processed_data/test_set/full_svd1000.csv'),
      skip = 1, drop = 1)
    colnames(test_svd_1000) <- c('sentence_id', paste0('svd', seq(1:1000)))
    if (identical(test_df$sentence_id, test_svd_1000$sentence_id) == FALSE)
      stop("test 1000-d svd does not match structured data")
    # remove extreme outliers from test PCA
    test_pca <- test_df[, ..pca_cols]
    test_pca[test_pca < -4] <- -4
    test_pca[test_pca > 4] <- 4
    x_test <- data.matrix(cbind(test_svd_1000[, ..svd_cols], test_pca))
  }
  
  #labels
  y_cols <- c(paste0(rf_hyperparams$frail_lab[r], '_neut'),
              paste0(rf_hyperparams$frail_lab[r], '_pos'),
              paste0(rf_hyperparams$frail_lab[r], '_neg'))
  y_train <- train_df[, ..y_cols] %>%
    as_tibble() %>%
    mutate(factr = ifelse(.[[1]] == 1, 1,
                          ifelse(.[[2]] == 1, 2,
                                 ifelse(.[[3]] == 1, 3, NA))))
  y_train_factor <- as.factor(y_train$factr)
  y_test <- test_df[, ..y_cols]
  
  frail_rf <- ranger(y = y_train_factor,
                     x = x_train,
                     num.threads = detectCores(),
                     probability = TRUE,
                     num.trees = 400,
                     mtry = rf_hyperparams$mtry[r],
                     sample.fraction = rf_hyperparams$sample_frac[r],
                     case.weights = cw_train_only,
                     oob.error = FALSE,
                     importance = 'impurity')

  #save variable importance
  importance <- importance(frail_rf)
  i_names <- names(importance)
  importance <- transpose(as.data.table(importance))
  colnames(importance) <- i_names
  importance$batch <- rf_hyperparams$batch[r]
  importance$frail_lab <- rf_hyperparams$frail_lab[r]
  importance$SVD <- rf_hyperparams$SVD[r]
  importance$mtry <- rf_hyperparams$mtry[r]
  importance$sample_frac <- rf_hyperparams$sample_frac[r]
  importance$case_weights <- rf_hyperparams$case_weights[r]
  fwrite(importance, 
         paste0(rf_importancedir, rf_hyperparams$batch[r], '_',
                rf_hyperparams$SVD[r], '_',
                rf_hyperparams$frail_lab[r], '_importance.csv'))
  
  #make predictions on test fold
  preds <- predict(frail_rf, data=x_test)$predictions
  colnames(preds) <- y_cols
  preds_save <- as.data.table(preds)
  preds_save$sentence_id <- test_df$sentence_id
  
  #save predictions
  fwrite(preds_save, 
         paste0(rf_predsdir, rf_hyperparams$batch[r], '_',
                rf_hyperparams$SVD[r], '_',
                rf_hyperparams$frail_lab[r], '_importance.csv'))
  
  #label each row
  hyper_grid <- data.frame(batch = rf_hyperparams$batch[r])
  hyper_grid$frail_lab <- rf_hyperparams$frail_lab[r]
  hyper_grid$SVD <- rf_hyperparams$SVD[r]
  hyper_grid$mtry <- rf_hyperparams$mtry[r]
  hyper_grid$sample_frac <- rf_hyperparams$sample_frac[r]
  hyper_grid$case_weights <- rf_hyperparams$case_weights[r]
  #Scaled Brier scores with bootstrapped CIs
  b_multi <- cbind(preds, y_test)
  sbrier_multi <- boot(b_multi, multi_scaled_Brier_boot, R = boot_reps, parallel = 'multicore')
  hyper_grid$sbrier_multi <- sbrier_multi$t0
  hyper_grid$sbrier_multi_sd <- sd(sbrier_multi$t)
  b_neut <- cbind(preds[, 1], y_test[[1]])
  sbrier_neut <- boot(b_neut, scaled_Brier_boot, R = boot_reps, parallel = 'multicore')
  hyper_grid$sbrier_neut <- sbrier_neut$t0
  hyper_grid$sbrier_neut_sd <- sd(sbrier_neut$t)
  b_pos <- cbind(preds[, 2], y_test[[2]])
  sbrier_pos <- boot(b_pos, scaled_Brier_boot, R = boot_reps, parallel = 'multicore')
  hyper_grid$sbrier_pos <- sbrier_pos$t0
  hyper_grid$sbrier_pos_sd <- sd(sbrier_pos$t)
  b_neg <- cbind(preds[, 3], y_test[[3]])
  sbrier_neg <- boot(b_neg, scaled_Brier_boot, R = boot_reps, parallel = 'multicore')
  hyper_grid$sbrier_neg <- sbrier_neg$t0
  hyper_grid$sbrier_neg_sd <- sd(sbrier_neg$t)
  #Precision-recall area under the curve
  hyper_grid$PR_AUC_neut <- pr.curve(scores.class0 = preds[, 1],
                                     weights.class0 = y_test[[1]])$auc.integral
  hyper_grid$PR_AUC_pos <- pr.curve(scores.class0 = preds[, 2],
                                    weights.class0 = y_test[[2]])$auc.integral
  hyper_grid$PR_AUC_neg <- pr.curve(scores.class0 = preds[, 3],
                                    weights.class0 = y_test[[3]])$auc.integral
  #Receiver operating characteristic area under the curve
  hyper_grid$ROC_AUC_neut <- roc.curve(scores.class0 = preds[, 1],
                                       weights.class0 = y_test[[1]])$auc
  hyper_grid$ROC_AUC_pos <- roc.curve(scores.class0 = preds[, 2],
                                      weights.class0 = y_test[[2]])$auc
  hyper_grid$ROC_AUC_neg <- roc.curve(scores.class0 = preds[, 3],
                                      weights.class0 = y_test[[3]])$auc
  
  #save hyper_grid for each rf run
  fwrite(hyper_grid, 
         paste0(rf_modeldir, rf_hyperparams$batch[r], '_',
                rf_hyperparams$SVD[r], '_',
                rf_hyperparams$frail_lab[r], '_performance.csv'))
}}


#summarize results
rf_hyperparams_full <- filter(rf_hyperparams_full, batch %in% batches)
md_l <- list()
for (w in 1:nrow(rf_hyperparams_full)){
  batch_root <- paste0(rootdir, rf_hyperparams_full$batch[w], '/')
  outdir <- paste0(batch_root, 'lin_trees_final_test/')
  rf_modeldir <- paste0(outdir,'rf_models/')
  md <- fread(paste0(rf_modeldir, rf_hyperparams_full$batch[w], '_',
                     rf_hyperparams$SVD[r], '_',
                     rf_hyperparams_full$frail_lab[w], '_performance.csv'))
  md$model <- 'rf'
  md_l[[w]] <- md
}
rf_test_perf <- rbindlist(md_l)

cols <- c('model', 'batch', 'frail_lab',
          grep('sbrier', colnames(rf_test_perf), value = TRUE))
print(rf_test_perf[, ..cols])

fwrite(rf_test_perf,
       paste0('/gwshare/frailty/output/figures_tables/RF_test_set_performance.csv'))






#check for models that have already been completed & remove them from the grid
enet_hyperparams_full <- enet_hyperparams
for (b in 1:length(batches)){
  batch_root <- paste0(rootdir, batches[b], '/')
  outdir <- paste0(batch_root, 'lin_trees_final_test/')
  enet_modeldir <- paste0(outdir,'enet_models/')
  enet_hyperparams <- enet_hyperparams %>%
    mutate(filename = 
             paste0(batch, '_', SVD, '_', frail_lab, '_performance.csv')) %>%
    filter(!filename %in% list.files(enet_modeldir))%>%
    select(-'filename')
}

#fit model with best hyperparams on full training set, then predict on test set
if ((nrow(enet_hyperparams) == 0) == FALSE) {
enet_error = foreach (r = 1:nrow(enet_hyperparams), .errorhandling = "pass") %dopar% {
  tc_error <- tryCatch(
    {
      #set directories
      batch_root <- paste0(rootdir, enet_hyperparams$batch[r], '/')
      outdir <- paste0(batch_root, 'lin_trees_final_test/')
      enet_modeldir <- paste0(outdir, 'enet_models/')
      enet_coefsdir <- paste0(outdir, 'enet_coefs/')
      enet_predsdir <- paste0(outdir, 'enet_preds/')
    
      # get matching training and test data & labels
      train_df <- fread(paste0(
        batch_root, 'processed_data/full_set/full_df.csv'))
      pca_cols <- grep('pca', colnames(train_df), value = TRUE)
      test_df <- fread(paste0(
        batch_root, 'processed_data/test_set/full_df.csv'))
      pca_cols_test <- grep('pca', colnames(test_df), value = TRUE)
      if (identical(pca_cols, pca_cols_test) == FALSE)
      stop("train PCA does not match test PCA")
      
      # load caseweights (training only)
      if (enet_hyperparams$case_weights[r] == TRUE){
        cw_train_only <- fread(paste0(
          batch_root, 'processed_data/full_set/full_caseweights.csv'))[[paste0(enet_hyperparams$frail_lab[r], '_cw')]]
      # check for matching
      if ((nrow(cw_train_only) == nrow(train_df)) == FALSE)
        stop("caseweights do not match training data")
      } else {
        cw_train_only <- NULL
      }
      
      # embeddings
      if (enet_hyperparams$SVD[r] == 'embed'){
        # x_train
        train_embed <- fread(paste0(
          batch_root, 'processed_data/full_set/full_embed_min_max_mean_SENT.csv'),
                             drop = 1)
        if (identical(train_df$sentence_id, train_embed$sentence_id) == FALSE)
          stop("train embeddings do not match structured data")
        emb_cols <- grep('min_|max_|mean_', colnames(train_embed), value = TRUE)
        x_train <- data.matrix(cbind(train_embed[, ..emb_cols],
                                         train_df[, ..pca_cols]))
        # x_test
        test_embed <- fread(paste0(
          batch_root, 'processed_data/test_set/full_embed_min_max_mean_SENT.csv'),
                            drop = 1)
        if (identical(test_df$sentence_id, test_embed$sentence_id) == FALSE)
          stop("test embeddings do not match structured data")
        emb_cols_test <- grep('min_|max_|mean_', colnames(test_embed), value = TRUE)
        if (identical(emb_cols, emb_cols_test) == FALSE) stop("train embed does not 
      match test embed")
        # remove extreme outliers from test PCA
        test_pca <- test_df[, ..pca_cols]
        test_pca[test_pca < -4] <- -4
        test_pca[test_pca > 4] <- 4
        x_test <- data.matrix(cbind(test_embed[, ..emb_cols], test_pca))
        
      # 300-d SVD
      } else if (enet_hyperparams$SVD[r] == '300'){
        # x_train
        train_svd_300 <- fread(paste0(
          batch_root, 'processed_data/full_set/full_svd300.csv'),
                             skip = 1, drop = 1)
        colnames(train_svd_300) <- c('sentence_id', paste0('svd', seq(1:300)))
        if (identical(train_df$sentence_id, train_svd_300$sentence_id) == FALSE)
        stop("train 300-d svd does not match structured data")
        svd_cols <- grep('svd', colnames(train_svd_300), value = TRUE)
        x_train <- data.matrix(cbind(train_svd_300[, ..svd_cols],
                                         train_df[, ..pca_cols]))
        # x_test
        test_svd_300 <- fread(paste0(
          batch_root, 'processed_data/test_set/full_svd300.csv'),
                              skip = 1, drop = 1)
        colnames(test_svd_300) <- c('sentence_id', paste0('svd', seq(1:300)))
        if (identical(test_df$sentence_id, test_svd_300$sentence_id) == FALSE)
          stop("test 300-d svd does not match structured data")
        # remove extreme outliers from test PCA
        test_pca <- test_df[, ..pca_cols]
        test_pca[test_pca < -4] <- -4
        test_pca[test_pca > 4] <- 4
        x_test <- data.matrix(cbind(test_svd_300[, ..svd_cols], test_pca))
      
      # 1000-d SVD
      } else if (enet_hyperparams$SVD[r] == '1000'){
        # x_train
        train_svd_1000 <- fread(paste0(
          batch_root, 'processed_data/full_set/full_svd1000.csv'),
                              skip = 1, drop = 1)
        colnames(train_svd_1000) <- c('sentence_id', paste0('svd', seq(1:1000)))
        if (identical(train_df$sentence_id, train_svd_1000$sentence_id) == FALSE)
        stop("train 1000-d svd does not match structured data")
        svd_cols <- grep('svd', colnames(train_svd_1000), value = TRUE)
        x_train <- data.matrix(cbind(train_svd_1000[, ..svd_cols],
                                          train_df[, ..pca_cols]))
        # x_test
        test_svd_1000 <- fread(paste0(
          batch_root, 'processed_data/test_set/full_svd1000.csv'),
                               skip = 1, drop = 1)
        colnames(test_svd_1000) <- c('sentence_id', paste0('svd', seq(1:1000)))
        if (identical(test_df$sentence_id, test_svd_1000$sentence_id) == FALSE)
          stop("test 1000-d svd does not match structured data")
        # remove extreme outliers from test PCA
        test_pca <- test_df[, ..pca_cols]
        test_pca[test_pca < -4] <- -4
        test_pca[test_pca > 4] <- 4
        x_test <- data.matrix(cbind(test_svd_1000[, ..svd_cols], test_pca))
      }
    
      #labels
      y_cols <- c(paste0(enet_hyperparams$frail_lab[r], '_neut'),
                  paste0(enet_hyperparams$frail_lab[r], '_pos'),
                  paste0(enet_hyperparams$frail_lab[r], '_neg'))
      y_train <- data.matrix(train_df[, ..y_cols])
      y_test <- data.matrix(test_df[, ..y_cols])
      
      lambda_seq <- c(10^seq(1, log10(enet_hyperparams$lambda[r]),
                             length.out = 5))
      #lambda_seq <- seq(100, enet_hyperparams$lambda[r], length.out = 10)
      #lambda_seq <- signif(c(10^seq(2, 0, length.out = 5)), 4)
      
      #train model
      frail_logit <- glmnet(x_train, 
                            y_train,
                            family = 'multinomial',
                            alpha = enet_hyperparams$alpha[r],
                            lambda = lambda_seq,
                            weights = cw_train_only,
                            maxit = 1e7)
      
      #save coefficients
      coefs <- predict(frail_logit, x_test, type = 'coefficients')
      for (c in 1:length(coefs)){
        coefs_s <- coefs[[y_cols[c]]]
        coefs_save <- as.data.table(t(as.matrix(coefs_s)))
        colnames(coefs_save)[1] <- 'intercept'
        coefs_save$lambda <- as.character(frail_logit$lambda)
        coefs_save$frail_lab <- y_cols[c]
        coefs_save$SVD <- enet_hyperparams$SVD[r]
        coefs_save$alpha <- enet_hyperparams$alpha[r]
        coefs_save$case_weights <- enet_hyperparams$case_weights[r]
        fwrite(coefs_save,
               paste0(enet_coefsdir, enet_hyperparams$batch[r], '_',
                      enet_hyperparams$SVD[r], '_',
                      y_cols[c], '_coefs.csv'))
      }
      #make predictions on validation fold for each alpha
      alpha_preds <- predict(frail_logit, x_test, type = 'response')
      #get the predictions we care about
      preds <- alpha_preds[, , length(lambda_seq)]
      preds_save <- as.data.table(preds)
      preds_save$lambda <- lambda_seq[length(lambda_seq)]
      preds_save$sentence_id <- test_df$sentence_id
      fwrite(preds_save, 
             paste0(enet_predsdir, enet_hyperparams$batch[r], '_',
                    enet_hyperparams$SVD[r], '_',
                    enet_hyperparams$frail_lab[r], '_preds.csv'))
      #performance metrics
      hyper_grid <- data.frame(batch = enet_hyperparams$batch[r])
      hyper_grid$frail_lab <- enet_hyperparams$frail_lab[r]
      hyper_grid$SVD <- enet_hyperparams$SVD[r]
      hyper_grid$lamba <- lambda_seq[length(lambda_seq)]
      hyper_grid$alpha <- enet_hyperparams$alpha[r]
      hyper_grid$case_weights <- enet_hyperparams$case_weights[r]
      
      #Scaled Brier scores with bootstrapped CIs
      b_multi <- cbind(preds, y_test)
      sbrier_multi <- boot(b_multi, multi_scaled_Brier_boot, R = boot_reps, parallel = 'multicore')
      hyper_grid$sbrier_multi <- sbrier_multi$t0
      hyper_grid$sbrier_multi_sd <- sd(sbrier_multi$t)
      b_neut <- cbind(preds[, 1], y_test[, 1])
      sbrier_neut <- boot(b_neut, scaled_Brier_boot, R = boot_reps, parallel = 'multicore')
      hyper_grid$sbrier_neut <- sbrier_neut$t0
      hyper_grid$sbrier_neut_sd <- sd(sbrier_neut$t)
      b_pos <- cbind(preds[, 2], y_test[, 2])
      sbrier_pos <- boot(b_pos, scaled_Brier_boot, R = boot_reps, parallel = 'multicore')
      hyper_grid$sbrier_pos <- sbrier_pos$t0
      hyper_grid$sbrier_pos_sd <- sd(sbrier_pos$t)
      b_neg <- cbind(preds[, 3], y_test[, 3])
      sbrier_neg <- boot(b_neg, scaled_Brier_boot, R = boot_reps, parallel = 'multicore')
      hyper_grid$sbrier_neg <- sbrier_neg$t0
      hyper_grid$sbrier_neg_sd <- sd(sbrier_neg$t)
      #Precision-recall area under the curve
      hyper_grid$PR_AUC_neut = pr.curve(scores.class0 = preds[, 1],
                                           weights.class0 = y_test[, 1])$auc.integral
      hyper_grid$PR_AUC_pos = pr.curve(scores.class0 = preds[, 2],
                                          weights.class0 = y_test[, 2])$auc.integral
      hyper_grid$PR_AUC_neg = pr.curve(scores.class0 = preds[, 3],
                                          weights.class0 = y_test[, 3])$auc.integral
      #Receiver operating characteristic area under the curve
      hyper_grid$ROC_AUC_neut = roc.curve(scores.class0 = preds[, 1],
                                             weights.class0 = y_test[, 1])$auc
      hyper_grid$ROC_AUC_pos = roc.curve(scores.class0 = preds[, 2],
                                            weights.class0 = y_test[, 2])$auc
      hyper_grid$ROC_AUC_neg = roc.curve(scores.class0 = preds[, 3],
                                            weights.class0 = y_test[, 3])$auc
      
      #save performance
      fwrite(hyper_grid, 
             paste0(enet_modeldir, enet_hyperparams$batch[r], '_',
                    enet_hyperparams$SVD[r], '_',
                    enet_hyperparams$frail_lab[r], '_performance.csv'))
    },
    
    #writing generic error messages to trace back later. Was not able to get 
    #foreach to reliably output the real error message AND where it occurred
    error = function(cond) {
      return(
        paste0('error in: ', enet_hyperparams$batch[r], '_',
                    enet_hyperparams$frail_lab[r]))
    })
  return(tc_error)
}}

fwrite(as.data.table(enet_error), 
       paste0(rootdir, 'final_test_rf_enet_error.txt'))


#summarize results
enet_hyperparams_full <- filter(enet_hyperparams_full, batch %in% batches)
md_l <- list()
for (w in 1:nrow(enet_hyperparams_full)){
  batch_root <- paste0(rootdir, enet_hyperparams_full$batch[w], '/')
  outdir <- paste0(batch_root, 'lin_trees_final_test/')
  enet_modeldir <- paste0(outdir,'enet_models/')
  md <- fread(paste0(enet_modeldir, enet_hyperparams_full$batch[w], '_',
                     enet_hyperparams$SVD[w], '_',
                     enet_hyperparams_full$frail_lab[w], '_performance.csv'))
  md$model <- 'enet'
  md_l[[w]] <- md
}
enet_test_perf <- rbindlist(md_l)

cols <- c('model', 'batch', 'frail_lab',
          grep('sbrier', colnames(enet_test_perf), value = TRUE))
print(enet_test_perf[, ..cols])

fwrite(enet_test_perf,
       paste0('/gwshare/frailty/output/figures_tables/enet_test_set_performance.csv'))



enet_test_perf$hyperparams <- 
  mutate(enet_hyperparams_full,
         hyperparams = paste0(SVD,
                              ' lambda_', lambda,
                              ' alpha_', alpha,
                              ' cw_', case_weights))$hyperparams
rf_test_perf$hyperparams <- 
  mutate(rf_hyperparams_full,
         hyperparams = paste0(SVD,
                              ' mtry_', mtry,
                              ' samplefrac_', sample_frac,
                              ' cw_', case_weights))$hyperparams
nn_single_test_perf
nn_multi_test_perf

cols <- c('model', 'batch', 'frail_lab', 'hyperparams',
          'sbrier_multi', 'sbrier_neg', 'sbrier_neut', 'sbrier_pos',
          'PR_AUC_neg', 'PR_AUC_neut', 'PR_AUC_pos',
          'ROC_AUC_neg', 'ROC_AUC_neut', 'ROC_AUC_pos')

all_test_perf <- rbind(enet_test_perf[, ..cols],
                       rf_test_perf[, ..cols],
                       nn_single_test_perf[, ..cols],
                       nn_multi_test_perf[, ..cols])

fwrite(all_test_perf,
       paste0('/gwshare/frailty/output/figures_tables/all_test_set_performance.csv'))

