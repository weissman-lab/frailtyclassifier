library(data.table)
library(dplyr)
library(tidyr)
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
models <- c('enet', 'rf')
#enet_batches <- c('AL01', 'AL02', 'AL03', 'AL04')
# rf_batches <- c('AL01', 'AL02', 'AL03', 'AL04', 'AL05')
enet_batches <- c('AL01', 'AL03')
rf_batches <- c('AL01', 'AL03')

#gather enet performance
ep_list <- list()
for (b in 1:length(enet_batches)){
  ep <- fread(paste0(rootdir,
                     enet_batches[b], '/lin_trees_enet/exp',
                     enet_batches[b], '_enet_performance.csv'))
  ep$model <- 'enet'
  ep$batch <- enet_batches[b]
  ep_list[[b]] <- ep
}
enet_performance <- rbindlist(ep_list)

#gather rf performance
rf_list <- list()
for (b in 1:length(rf_batches)){
  rf <- fread(paste0(rootdir,
                     rf_batches[b], '/lin_trees/exp',
                     rf_batches[b], '_rf_performance.csv'))
  rf$model <- 'rf'
  rf$batch <- rf_batches[b]
  rf_list[[b]] <- rf
}
rf_performance <- rbindlist(rf_list)

#list relevant hyperparams
hyperparams_enet <- c('SVD', 'lambda', 'alpha', 'case_weights')
hyperparams_rf <- c('SVD', 'mtry', 'sample_frac', 'case_weights')

# step 1: find the best hyperparams for each aspect and calculate performance
# step 2: using the best hyperparams only, calcuLate the mean scaled brier
# across all aspects (sbrier_multi_all) for each fold
# step 3: find the mean and sd for performance across all folds
# note: this should be run separately for each batch
perf_calc_MEAN <- function(raw_perf){
  raw_perf <- as.data.frame(raw_perf)
  if (raw_perf$model[1] == 'enet') {
    hyperparams = hyperparams_enet
  } else if (raw_perf$model[1] == 'rf') {
    hyperparams = hyperparams_rf
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


#Get best enet hyperparams
perf_l <- list()
for (b in 1:length(enet_batches)) {
  perf <- perf_calc_MEAN(enet_performance[batch == enet_batches[b],])[[1]]
  perf_l[[b]] <- perf
}
enet_hyperparams <- rbindlist(perf_l)

#Get best rf hyperparams
perf_l <- list()
for (b in 1:length(rf_batches)) {
  perf <- perf_calc_MEAN(rf_performance[batch == rf_batches[b],])[[1]]
  perf_l[[b]] <- perf
}
rf_hyperparams <- rbindlist(perf_l)


#batch ->
# for each row in enet_hyperparams train then test 
# for each row in f_hyperparams train then test

# # make outdirs
# batches <- c('AL01', 'AL02', 'AL03', 'AL04', 'AL05')
# for (b in 1:length(batches)){
#   batch_root <- paste0(rootdir, batches[b], '/')
#   outdir <- paste0(batch_root, 'lin_trees_final_test_practice/')
#   rf_coefsdir <- paste0(outdir, 'rf_coefs/')
#   unlink(rf_coefsdir, recursive = TRUE)
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
             paste0(batch, '_', frail_lab, '_performance.csv')) %>%
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
    x_test <- data.matrix(cbind(test_embed[, ..emb_cols],
                                test_df[, ..pca_cols]))
    
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
    x_test <- data.matrix(cbind(test_svd_300[, ..svd_cols],
                                test_df[, ..pca_cols]))
    
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
    x_test <- data.matrix(cbind(test_svd_1000[, ..svd_cols],
                                test_df[, ..pca_cols]))
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
                rf_hyperparams$frail_lab[r], '_importance.csv'))
  
  #make predictions on test fold
  preds <- predict(frail_rf, data=x_test)$predictions
  colnames(preds) <- y_cols
  preds_save <- as.data.table(preds)
  preds_save$sentence_id <- test_df$sentence_id
  
  #save predictions
  fwrite(preds_save, 
         paste0(rf_predsdir, rf_hyperparams$batch[r], '_',
                rf_hyperparams$frail_lab[r], '_importance.csv'))
  
  #label each row
  hyper_grid <- data.frame(batch = rf_hyperparams$batch[r])
  hyper_grid$frail_lab <- rf_hyperparams$frail_lab[r]
  hyper_grid$SVD <- rf_hyperparams$SVD[r]
  hyper_grid$mtry <- rf_hyperparams$mtry[r]
  hyper_grid$sample_frac <- rf_hyperparams$sample_frac[r]
  hyper_grid$case_weights <- rf_hyperparams$case_weights[r]
  #single class scaled Brier scores
  hyper_grid$sbrier_neut <- scaled_Brier(preds[, 1], y_test[[1]], 1)
  hyper_grid$sbrier_pos <- scaled_Brier(preds[, 2], y_test[[2]], 1)
  hyper_grid$sbrier_neg <- scaled_Brier(preds[, 3], y_test[[3]], 1)
  #multiclass scaled brier score
  hyper_grid$sbrier_multi <- multi_scaled_Brier(preds, y_test)
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
                rf_hyperparams$frail_lab[r], '_performance.csv'))
}}


#summarize results
rf_hyperparams_full <- filter(rf_hyperparams_full, batch %in% rf_batches)
md_l <- list()
for (w in 1:nrow(rf_hyperparams_full)){
  batch_root <- paste0(rootdir, rf_hyperparams_full$batch[w], '/')
  outdir <- paste0(batch_root, 'lin_trees_final_test/')
  rf_modeldir <- paste0(outdir,'rf_models/')
  md <- fread(paste0(rf_modeldir, rf_hyperparams_full$batch[w], '_',
                     rf_hyperparams_full$frail_lab[w], '_performance.csv'))
  md_l[[w]] <- md
}
rf_test_perf <- rbindlist(md_l)

cols <- c('batch', 'frail_lab',
          grep('sbrier', colnames(rf_test_perf), value = TRUE))
print(rf_test_perf[, ..cols])

fwrite(rf_test_perf, paste0(rootdir, 'figures_tables/RF_test_set_performance.csv'))






#check for models that have already been completed & remove them from the grid
enet_hyperparams_full <- enet_hyperparams
for (b in 1:length(batches)){
  batch_root <- paste0(rootdir, batches[b], '/')
  outdir <- paste0(batch_root, 'lin_trees_final_test/')
  enet_modeldir <- paste0(outdir,'enet_models/')
  enet_hyperparams <- enet_hyperparams %>%
    mutate(filename = 
             paste0(batch, '_', frail_lab, '_performance.csv')) %>%
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
      enet_modeldir <- paste0(outdir,'enet_models/')
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
        x_test <- data.matrix(cbind(test_embed[, ..emb_cols],
                                        test_df[, ..pca_cols]))
      
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
        x_test <- data.matrix(cbind(test_svd_300[, ..svd_cols],
                                          test_df[, ..pca_cols]))
      
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
        x_test <- data.matrix(cbind(test_svd_1000[, ..svd_cols],
                                           test_df[, ..pca_cols]))
      }
    
      #labels
      y_cols <- c(paste0(enet_hyperparams$frail_lab[r], '_neut'),
                  paste0(enet_hyperparams$frail_lab[r], '_pos'),
                  paste0(enet_hyperparams$frail_lab[r], '_neg'))
      y_train <- data.matrix(train_df[, ..y_cols])
      y_test <- data.matrix(test_df[, ..y_cols])
      
      lambda_seq <- c(signif(c(10^seq(2, 0, length.out = 5)), 4),
                      enet_hyperparams$lambda[r])
      
      lambda_seq <- signif(c(10^seq(2, 0, length.out = 5)), 4)
      
      #train model
      frail_logit <- glmnet(x_train, 
                            y_train,
                            family = 'multinomial',
                            alpha = enet_hyperparams$alpha[r],
                            lambda = lambda_seq,
                            weights = cw_train_only)
      
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
                    enet_hyperparams$frail_lab[r], '_preds.csv'))
      #performance metrics
      hyper_grid <- data.frame(batch = enet_hyperparams$batch[r])
      hyper_grid$frail_lab <- enet_hyperparams$frail_lab[r]
      hyper_grid$SVD <- enet_hyperparams$SVD[r]
      hyper_grid$lamba <- lambda_seq[length(lambda_seq)]
      hyper_grid$alpha <- enet_hyperparams$alpha[r]
      hyper_grid$case_weights <- enet_hyperparams$case_weights[r]
      #single class scaled Brier scores
      hyper_grid$sbrier_neut = scaled_Brier(preds[, 1], y_test[, 1], 1)
      hyper_grid$sbrier_pos = scaled_Brier(preds[, 2], y_test[, 2], 1)
      hyper_grid$sbrier_neg = scaled_Brier(preds[, 3], y_test[, 3], 1)
      #multiclass scaled brier score
      hyper_grid$sbrier_multi <- multi_scaled_Brier(preds, y_test)
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
enet_hyperparams_full <- filter(enet_hyperparams_full, batch %in% enet_batches)
md_l <- list()
for (w in 1:nrow(enet_hyperparams_full)){
  batch_root <- paste0(rootdir, enet_hyperparams_full$batch[w], '/')
  outdir <- paste0(batch_root, 'lin_trees_final_test/')
  enet_modeldir <- paste0(outdir,'enet_models/')
  md <- fread(paste0(enet_modeldir, enet_hyperparams_full$batch[w], '_',
                     enet_hyperparams_full$frail_lab[w], '_performance.csv'))
  md_l[[w]] <- md
}
enet_test_perf <- rbindlist(md_l)

cols <- c('batch', 'frail_lab',
          grep('sbrier', colnames(enet_test_perf), value = TRUE))
print(enet_test_perf[, ..cols])
