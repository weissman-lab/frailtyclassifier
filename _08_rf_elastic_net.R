library(data.table)
library(measures)
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

#set directories based on location
dirs = c('/Users/martijac/Documents/Frailty/frailty_classifier/output/lin_trees_SENT/',
         '/media/drv2/andrewcd2/frailty/output/lin_trees_SENT/',
         '/share/gwlab/frailty/output/lin_trees_SENT/')
for (d in 1:length(dirs)) {
  if (dir.exists(dirs[d])) {
    datadir = dirs[d]
  }
}
SVDdir <- paste0(datadir, 'svd/') 
embeddingsdir <- paste0(datadir, 'embeddings/')
trtedatadir <- paste0(datadir, 'trtedata/')

#new output directory for each experiment:
outdir <- paste0(datadir, 'exp', exp, '/')
#directory for performance for each enet model:
enet_modeldir <- paste0(outdir,'enet_models/')
#directory for duration for each enet model:
enet_durationdir <- paste0(outdir,'enet_durations/')
#directory for predictions:
enet_predsdir <- paste0(outdir,'enet_preds/')
#make directories
dir.create(outdir)
dir.create(enet_modeldir)
dir.create(enet_durationdir)
dir.create(enet_predsdir)


#scaled Brier score
scaled_brier_score <- function(pred, obs, event_rate_matrix) {
  1 - (multiclass.Brier(pred, obs) / multiclass.Brier(event_rate_matrix, obs))
}

#set seed
seed = 92120
#Include structured data?
inc_struc = TRUE

#listing features (to prevent accidentally including a label)
x_vars <- c("length", "n_encs", "n_ed_visits", "n_admissions", 
            "days_hospitalized", "mean_sys_bp", "mean_dia_bp", "sd_sys_bp", 
            "sd_dia_bp", "bmi_mean", "bmi_slope", "max_o2", "spo2_worst", "TSH", 
            "sd_TSH", "n_TSH", 'n_unique_meds', 'elixhauser', 'n_comorb', 'AGE',
            'SEXFemale', 'SEXMale', 'MARITAL_STATUSMarried', 'MARITAL_STATUSOther',
            'MARITAL_STATUSSingle', 'MARITAL_STATUSWidowed', 'EMPY_STATFull.Time',
            'EMPY_STATNot.Employed', 'EMPY_STATOther', 'EMPY_STATPart.Time', 
            'EMPY_STATRetired', 'RACEOther', 'RACEWhite', 'LANGUAGEOther', 
            'LANGUAGESpanish', 'MV_n_encs', 'MV_n_ed_visits', 'MV_n_admissions', 
            'MV_days_hospitalized', 'MV_mean_sys_bp', 'MV_mean_dia_bp', 
            'MV_sd_sys_bp', 'MV_sd_dia_bp', 'MV_bmi_mean', 'MV_bmi_slope', 
            'MV_max_o2', 'MV_spo2_worst', 'MV_TSH', 'MV_sd_TSH', 'MV_n_TSH', 
            'MV_n_unique_meds', 'MV_AGE', 'MV_SEX', 'MV_MARITAL_STATUS', 
            'MV_EMPY_STAT', 'MV_RACE', 'MV_LANGUAGE')
  
#repeated k-fold cross validation
repeats <- 1
for (p in 1:length(repeats)) {
  
  #load data in parallel
  folds <- 1
  for (d in 1:length(folds)) {
    assign(paste0('r', repeats[p], '_f', folds[d], '_tr'),
           fread(paste0(trtedatadir, 'r', repeats[p], '_f', folds[d], '_tr_df.csv')))
    assign(paste0('r', repeats[p], '_f', folds[d], '_te'),
           fread(paste0(trtedatadir, 'r', repeats[p], '_f', folds[d], '_te_df.csv')))
  }
  svd <- c('embed', '300', '1000')
  for (s in 1:length(svd)) {
    if (svd[s] == 'embed') {
      train <- foreach (d = 1:length(folds)) %dopar% {
        #load embeddings for each fold (drop index)
        embeddings_tr <- fread(paste0(embeddingsdir, 'r', repeats[p], '_f', folds[d], '_tr_embed_min_max_mean_SENT.csv'), drop = 1)
        #test that embeddings notes match training/test notes before dropping the 'notes' column
        if (identical(distinct(get(paste0('r', repeats[p], '_f', folds[d], '_tr')), note)$note, distinct(embeddings_tr, note)$note) == FALSE) stop("embeddings do not match training data")
        #embeddings with or without structured data
        if (inc_struc == FALSE) {
          #drop 'note' and 'sentence_id' column
          embeddings_tr <- data.matrix(embeddings_tr[, -c('sentence_id', 'note')])
        } else {
          #drop 'note' and 'sentence_id' column & concatenate embeddings with structured data
          embeddings_tr <- data.matrix(cbind(embeddings_tr[, -c('sentence_id', 'note')], get(paste0('r', repeats[p], '_f', folds[d], '_tr'))[, ..x_vars]))
        }
        return(embeddings_tr)
      }
      test <- foreach (d = 1:length(folds)) %dopar% {
        #load embeddings for each fold (drop index)
        embeddings_te <- fread(paste0(embeddingsdir, 'r', repeats[p], '_f', folds[d], '_te_embed_min_max_mean_SENT.csv'), drop = 1)
        #test that embeddings notes match training/test notes before dropping the 'notes' column
        if (identical(distinct(get(paste0('r', repeats[p], '_f', folds[d], '_te')), note)$note, distinct(embeddings_te, note)$note) == FALSE) stop("embeddings do not match test data")
        #embeddings with or without structured data
        if (inc_struc == FALSE) {
          #drop 'note' and 'sentence_id' column
          embeddings_te <- data.matrix(embeddings_te[, -c('sentence_id', 'note')])
        } else {
          #drop 'note' and 'sentence_id' column & concatenate embeddings with structured data
          embeddings_te <- data.matrix(cbind(embeddings_te[, -c('sentence_id', 'note')], get(paste0('r', repeats[p], '_f', folds[d], '_te'))[, ..x_vars]))
        }
        return(embeddings_te)
      }
    } else {
        train <- foreach (d = 1:length(folds)) %dopar% {
          #svd with or without structured data
          if (inc_struc == FALSE) {
            #load only the svd - drop first 2 columns (index and note label)
            x_train <- data.matrix(fread(paste0(SVDdir, 'r', repeats[p], '_f', folds[d], '_tr_svd', svd[s], '.csv'), skip = 1, drop = 1))
          } else {
            #concatenate svd with structured data
            x_train <- data.matrix(cbind(fread(paste0(SVDdir, 'r', repeats[p], '_f', folds[d], '_tr_svd', svd[s], '.csv'), skip = 1, drop = 1), get(paste0('r', repeats[p], '_f', folds[d], '_tr'))[, ..x_vars]))
          }
          #test that svd row length matches training/test row length
          if ((nrow(x_train) == nrow(get(paste0('r', repeats[p], '_f', folds[d], '_tr')))) == FALSE) stop("svd do not match training data")
          return(x_train)
        }
        test <- foreach (d = 1:length(folds)) %dopar% {
          #load labels and structured data
          df_te <- fread(paste0(trtedatadir, 'r', repeats[p], '_f', folds[d], '_te_df.csv'))
          #svd with or without structured data
          if (inc_struc == FALSE) {
            #load only the svd - drop first 2 columns (index and note label)
            x_test <- data.matrix(fread(paste0(SVDdir, 'r', repeats[p], '_f', folds[d], '_te_svd', svd[s], '.csv'), skip = 1, drop = 1))
          } else {
            #concatenate svd with structured data
            x_test <- data.matrix(cbind(fread(paste0(SVDdir, 'r', repeats[p], '_f', folds[d], '_te_svd', svd[s], '.csv'), skip = 1, drop = 1), get(paste0('r', repeats[p], '_f', folds[d], '_te'))[, ..x_vars]))
          }
          #test that svd row length matches training/test row length
          if ((nrow(x_test) == nrow(get(paste0('r', repeats[p], '_f', folds[d], '_te')))) == FALSE) stop("svd do not match training data")
          return(x_test)
        }
      }
    assign(paste0('r', repeats[p], '_s_', svd[s], '_x_train'), train)
    assign(paste0('r', repeats[p], '_s_', svd[s], '_x_test'), test)
  }
  #foreach output is in list format
  #rename each element of the list as a df
  for (s in 1:length(svd)) {
    for (d in 1:length(folds)) {
      assign(paste0('r', repeats[p], '_f', folds[d], '_s_', svd[s], '_x_train'), get(paste0('r', repeats[p], '_s_', svd[s], '_x_train'))[[d]])
      assign(paste0('r', repeats[p], '_f', folds[d], '_s_', svd[s], '_x_test'), get(paste0('r', repeats[p], '_s_', svd[s], '_x_test'))[[d]])
    }
  }
  
  #clear lists of dfs
  for (s in 1:length(svd)) {
    rm(list = paste0('r', repeats[p], '_s_', svd[s], '_x_train'))
    rm(list = paste0('r', repeats[p], '_s_', svd[s], '_x_test'))
  }
  gc()
  
  #run glmnet grid separately for each svd/embedding type to reduce memory use
  svd <- c('embed', '300', '1000')
  for (s in 1:length(svd)){
    
    #hyperparameter grid for experiments
    #set sequence of lambda values to test
    lambda_seq <- c(10^seq(2, -5, length.out = 25))
    #model grid 1
    mg1 <- expand_grid(
      fold = 1,
      svd = svd[s],
      frail_lab = c('Msk_prob', 'Fall_risk', 'Nutrition', 'Resp_imp'),
      alpha = c(0.9, 0.5, 0.1),
    )
    
    # # small test grid
    # #set sequence of lambda values to test
    # lambda_seq <- signif(c(10^seq(-2, -3, length.out = 3)), 4)
    # #model grid 1
    # mg1 <- expand_grid(
    #   fold = seq(1,10),
    #   svd = svd[s],
    #   frail_lab = c('Msk_prob', 'Fall_risk', 'Nutrition', 'Resp_imp'),
    #   alpha = c(0.9, 0.5),
    # )
    
    #label alpha (for naming .csv files)
    mg1 <- mutate(mg1, alpha_l = ifelse(alpha == 0.9, 9,
                                      ifelse(alpha == 0.5, 5,
                                             ifelse(alpha == 0.1, 1, NA))))
    
    #check for models that have already been completed & remove them from the grid
    mg1 <- mg1 %>%
      mutate(filename = paste0('exp', exp, '_hypergrid_r', repeats[p], '_f', fold, '_', frail_lab, '_svd_', svd, '_alpha', alpha_l, '.csv')) %>%
      filter(!filename %in% list.files(enet_modeldir)) %>%
      select(-'filename')
    
    #run glmnet if incomplete
    if ((nrow(mg1) == 0) == FALSE) {
      #run for first model grid
      mg <- mg1
      foreach (r = 1:nrow(mg), .errorhandling = "remove") %dopar% {
        #get matching training and test labels
        x_train <- get(paste0('r', repeats[p], '_f', mg$fold[r], '_s_', mg$svd[r], '_x_train'))
        x_test <- get(paste0('r', repeats[p], '_f', mg$fold[r], '_s_', mg$svd[r], '_x_test'))
        y_cols <- c(paste0(mg$frail_lab[r], '_neut'),
                    paste0(mg$frail_lab[r], '_pos'),
                    paste0(mg$frail_lab[r], '_neg'))
        y_train <- data.matrix(get(paste0('r', repeats[p], '_f', mg$fold[r], '_tr'))[, ..y_cols])
        y_test <- data.matrix(get(paste0('r', repeats[p], '_f', mg$fold[r], '_te'))[, ..y_cols])
        #measure CPU time for glmnet
        benchmark <- benchmark("glmnet" = {
        #train model
        frail_logit <- glmnet(x_train, 
                              y_train,
                              family = 'multinomial',
                              alpha = mg$alpha[r],
                              lambda = lambda_seq)
        }, replications = 1
        )
        #save benchmarking
        fwrite(benchmark, paste0(enet_durationdir, 'exp', exp, '_duration_hyper_r', repeats[p], '_f', mg$fold[r], '_', mg$frail_lab[r], '_svd_', mg$svd[r], '_alpha', mg$alpha_l[r], '.txt'))
        #make predictions on test fold for each alpha
        alpha_preds <- predict(frail_logit, x_test, type = 'response')
        #set lambas as dimnames for 3rd dimension
        dimnames(alpha_preds)[[3]] <- lambda_seq
        #save predictions
        preds_s <- list()
        for (d in 1:dim(alpha_preds)[3]) {
          preds_save <- as.data.table(alpha_preds[, , d])
          preds_save$lambda <- dimnames(alpha_preds)[[3]][d]
          preds_save$sentence_id <- get(paste0('r', repeats[p], '_f', mg$fold[r], '_te'))$sentence_id
          preds_save$note <- get(paste0('r', repeats[p], '_f', mg$fold[r], '_te'))$note
          preds_s[[d]] <- preds_save
        }
        preds_save <- rbindlist(preds_s)
        fwrite(preds_save, paste0(enet_predsdir, 'exp', exp, '_preds_r', repeats[p], '_f', mg$fold[r], '_', mg$frail_lab[r], '_svd_', mg$svd[r], '_alpha', mg$alpha_l[r], '.csv'))
        #build hyperparameter grid
        hyper_grid <- expand.grid(
          frail_lab = NA,
          cv_repeat = NA,
          fold = NA,
          SVD = NA,
          lambda = rep(NA, dim(alpha_preds)[3]), #lambdas are in the 3rd dimension of this array
          alpha = NA,
          df = NA,
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
          hyper_grid$frail_lab[l] <- mg$frail_lab[r]
          hyper_grid$cv_repeat[l] <- repeats[p]
          hyper_grid$fold[l] <- mg$fold[r]
          hyper_grid$SVD[l] <- mg$svd[r]
          hyper_grid$alpha[l] <- mg$alpha[r]
          #lambda
          hyper_grid$lambda[l] <- frail_logit$lambda[l]
          #number of nonzero coefficients (Df)
          hyper_grid$df[l] <- frail_logit$df[l]
          #preds for this lambda
          preds <- alpha_preds[, , l]
          #single class Brier scores
          hyper_grid$bscore_neut[l] = Brier(preds[, 1], y_test[, 1], 0, 1)
          hyper_grid$bscore_pos[l] = Brier(preds[, 2], y_test[, 2], 0, 1)
          hyper_grid$bscore_neg[l] = Brier(preds[, 3], y_test[, 3], 0, 1)
          #single class scaled Brier scores
          hyper_grid$sbrier_neut[l] = BrierScaled(preds[, 1], y_test[, 1], 0, 1)
          hyper_grid$sbrier_pos[l] = BrierScaled(preds[, 2], y_test[, 2], 0, 1)
          hyper_grid$sbrier_neg[l] = BrierScaled(preds[, 3], y_test[, 3], 0, 1)
          #set column names for multiclass.Brier
          colnames(preds) <- c(1, 2, 3)
          #get a single categorical outcome variable for multiclass.Brier
          factr <- y_test %>%
            as_tibble() %>%
            mutate(factr = ifelse(.[[1]] == 1, 1,
                                  ifelse(.[[2]] == 1, 2,
                                         ifelse(.[[3]] == 1, 3, NA))))
          obs <- factr$factr
          #multiclass brier score
          hyper_grid$bscore_multi[l] <- multiclass.Brier(preds, obs)
          hyper_grid$bscore_multi[l] <- multiclass.Brier(preds, obs)
          #make event rate matrix for multiclass scaled brier
          er <- matrix(0, nrow=length(obs), ncol=3)
          er[, 1] <- sum(obs == 1)/length(obs)
          er[, 2] <- sum(obs == 2)/length(obs)
          er[, 3] <- sum(obs == 3)/length(obs)
          colnames(er) = c(1, 2, 3)
          #multiclass scaled brier score
          hyper_grid$sbrier_multi[l] <- scaled_brier_score(preds, obs, er)
          #Precision-recall area under the curve
          hyper_grid$PR_AUC_neut[l] = pr.curve(scores.class0 = preds[, 1], weights.class0 = y_test[, 1])$auc.integral
          hyper_grid$PR_AUC_pos[l] = pr.curve(scores.class0 = preds[, 2], weights.class0 = y_test[, 2])$auc.integral
          hyper_grid$PR_AUC_neg[l] = pr.curve(scores.class0 = preds[, 3], weights.class0 = y_test[, 3])$auc.integral
          #Receiver operating characteristic area under the curve
          hyper_grid$ROC_AUC_neut[l] = roc.curve(scores.class0 = preds[, 1], weights.class0 = y_test[, 1])$auc
          hyper_grid$ROC_AUC_pos[l] = roc.curve(scores.class0 = preds[, 2], weights.class0 = y_test[, 2])$auc
          hyper_grid$ROC_AUC_neg[l] = roc.curve(scores.class0 = preds[, 3], weights.class0 = y_test[, 3])$auc
        }
        
        #save hyper_grid for each glmnet run
        fwrite(hyper_grid, paste0(enet_modeldir, 'exp', exp, '_hypergrid_r', repeats[p], '_f', mg$fold[r], '_', mg$frail_lab[r], '_svd_', mg$svd[r], '_alpha', mg$alpha_l[r], '.csv'))
        #remove objects & garbage collection
        rm(x_train, x_test, y_train, y_test, frail_logit, benchmark, alpha_preds, preds_save, preds_s, preds, obs, er, hyper_grid)
        gc()
      }
    }
    gc()
  }
  
  #RANDOM FOREST
  
  #directory for performance for each rf model:
  rf_modeldir <- paste0(outdir,'rf_models/')
  #directory for duration for each rf model:
  rf_durationdir <- paste0(outdir,'rf_durations/')
  #directory for predictions:
  rf_predsdir <- paste0(outdir,'rf_preds/')
  #make directories
  dir.create(rf_durationdir)
  dir.create(rf_modeldir)
  dir.create(rf_predsdir)
  
  #experiment grid
  mg3 <- expand_grid(
    fold = 1,
    svd = c('embed', '300', '1000'),
    frail_lab = c('Msk_prob', 'Fall_risk', 'Nutrition', 'Resp_imp'),
    ntree       = 400,
    mtry        = signif(seq(7, 45, length.out = 3), 2),
    sample_frac = signif(seq(0.6, 1, length.out = 3), 1)
  )
  
  #small test grid
  # mg3 <- expand_grid(
  #   fold = seq(1,10),
  #   svd = c('embed', '300'),
  #   frail_lab = c('Fall_risk', 'Nutrition'),
  #   ntree       = 400,
  #   mtry        = signif(seq(7, 45, length.out = 2), 2),
  #   sample_frac = signif(seq(0.6, 1, length.out = 2), 1)
  # )
  
  #label sample fraction (for naming .csv files)
  mg3 <- mutate(mg3, sample_frac_l = ifelse(sample_frac == 0.6, 6,
                                          ifelse(sample_frac == 0.8, 8,
                                                 ifelse(sample_frac == 1.0, 10, NA))))
  #check for models that have already been completed & remove them from the grid
  mg3 <- mg3 %>%
    mutate(filename = paste0('exp', exp, '_hypergrid_r', repeats[p], '_f', fold, '_', frail_lab, '_svd_', svd, '_mtry', mtry, '_sfrac', sample_frac_l, '.csv')) %>%
    filter(!filename %in% list.files(rf_modeldir)) %>%
    select(-'filename')
  
  #load caseweights (weight non-neutral tokens by the inverse of their prevalence)
  #e.g. 1.3% of fall_risk tokens are non-neutral. Therefore, non-neutral tokens are weighted * (1/0.013)
  folds <- seq(1, 10)
  for (d in 1:length(folds)) {
    assign(paste0('r', repeats[p], '_f', folds[d], '_tr_cw'), fread(paste0(trtedatadir, 'r', repeats[p], '_f', folds[d], '_tr_cw.csv')))
  }
  #run glmnet if incomplete
  if ((nrow(mg3) == 0) == FALSE) {
    for (r in 1:nrow(mg3)){
      #get matching training and test labels
      x_train <- get(paste0('r', repeats[p], '_f', mg3$fold[r], '_s_', mg3$svd[r], '_x_train'))
      x_test <- get(paste0('r', repeats[p], '_f', mg3$fold[r], '_s_', mg3$svd[r], '_x_test'))
      y_cols <- c(paste0(mg3$frail_lab[r], '_neut'),
                  paste0(mg3$frail_lab[r], '_pos'),
                  paste0(mg3$frail_lab[r], '_neg'))
      y_train <- get(paste0('r', repeats[p], '_f', mg3$fold[r], '_tr'))[, ..y_cols] %>%
        as_tibble() %>%
        mutate(factr = ifelse(.[[1]] == 1, 1,
                              ifelse(.[[2]] == 1, 2,
                                     ifelse(.[[3]] == 1, 3, NA))))
      y_train_factor <- y_train$factr
      y_test <- get(paste0('r', repeats[p], '_f', mg3$fold[r], '_te'))[, ..y_cols]
      #get matching caseweights
      cw <- get(paste0('r', repeats[p], '_f', mg3$fold[r], '_tr_cw'))[[paste0(mg3$frail_lab[r], '_cw')]]
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
                           seed = seed)
      }, replications = 1
      )
      #save benchmarking
      fwrite(benchmark, paste0(rf_durationdir, 'exp', exp, '_duration_hyper_r', repeats[p], '_f', mg3$fold[r], '_', mg3$frail_lab[r], '_svd_', mg3$svd[r], '_mtry_', mg3$mtry[r], '_sfrac_', mg3$sample_frac_l[r], '.csv'))
      #make predictions on test fold
      preds <- predict(frail_rf, data=x_test)$predictions
      colnames(preds) <- y_cols
      preds_save <- as.data.table(preds)
      preds_save$sentence_id <- get(paste0('r', repeats[p], '_f', mg3$fold[r], '_te'))$sentence_id
      preds_save$note <- get(paste0('r', repeats[p], '_f', mg3$fold[r], '_te'))$note
      #save predictions
      fwrite(as.data.table(preds_save), paste0(rf_predsdir, 'exp', exp, '_preds_r', repeats[p], '_f', mg3$fold[r], '_', mg3$frail_lab[r], '_svd_', mg3$svd[r], '_mtry_', mg3$mtry[r], '_sfrac_', mg3$sample_frac_l[r], '.csv'))
  
      #label each row
      hyper_grid <- data.frame(frail_lab = mg3$frail_lab[r])
      hyper_grid$cv_repeat <- repeats[p]
      hyper_grid$fold <- mg3$fold[r]
      hyper_grid$SVD <- mg3$svd[r]
      hyper_grid$mtry <- mg3$mtry[r]
      hyper_grid$sample_frac <- mg3$sample_frac[r]
      hyper_grid$num_indep_vars <- frail_rf$`num.independent.variables`
  
      #single class Brier scores
      hyper_grid$bscore_neut <- Brier(preds[, 1], y_test[[1]], 0, 1)
      hyper_grid$bscore_pos = Brier(preds[, 2], y_test[[2]], 0, 1)
      hyper_grid$bscore_neg = Brier(preds[, 3], y_test[[3]], 0, 1)
      #single class scaled Brier scores
      hyper_grid$sbrier_neut = BrierScaled(preds[, 1], y_test[[1]], 0, 1)
      hyper_grid$sbrier_pos = BrierScaled(preds[, 2], y_test[[2]], 0, 1)
      hyper_grid$sbrier_neg = BrierScaled(preds[, 3], y_test[[3]], 0, 1)
      #set column names for multiclass.Brier
      colnames(preds) <- c(1, 2, 3)
      #get a single categorical outcome variable for multiclass.Brier
      factr <- y_test %>%
        as_tibble() %>%
        mutate(factr = ifelse(.[[1]] == 1, 1,
                              ifelse(.[[2]] == 1, 2,
                                     ifelse(.[[3]] == 1, 3, NA))))
      obs <- factr$factr
      #multiclass brier score
      hyper_grid$bscore_multi <- multiclass.Brier(preds, obs)
      hyper_grid$bscore_multi <- multiclass.Brier(preds, obs)
      #make event rate matrix for multiclass scaled brier
      er <- matrix(0, nrow=length(obs), ncol=3)
      er[, 1] <- sum(obs == 1)/length(obs)
      er[, 2] <- sum(obs == 2)/length(obs)
      er[, 3] <- sum(obs == 3)/length(obs)
      colnames(er) = c(1, 2, 3)
      #multiclass scaled brier score
      hyper_grid$sbrier_multi <- scaled_brier_score(preds, obs, er)
      #Precision-recall area under the curve
      hyper_grid$PR_AUC_neut = pr.curve(scores.class0 = preds[, 1], weights.class0 = y_test[[1]])$auc.integral
      hyper_grid$PR_AUC_pos = pr.curve(scores.class0 = preds[, 2], weights.class0 = y_test[[2]])$auc.integral
      hyper_grid$PR_AUC_neg = pr.curve(scores.class0 = preds[, 3], weights.class0 = y_test[[3]])$auc.integral
      #Receiver operating characteristic area under the curve
      hyper_grid$ROC_AUC_neut = roc.curve(scores.class0 = preds[, 1], weights.class0 = y_test[[1]])$auc
      hyper_grid$ROC_AUC_pos = roc.curve(scores.class0 = preds[, 2], weights.class0 = y_test[[2]])$auc
      hyper_grid$ROC_AUC_neg = roc.curve(scores.class0 = preds[, 3], weights.class0 = y_test[[3]])$auc
      #save hyper_grid for each rf run
      fwrite(hyper_grid, paste0(rf_modeldir, 'exp', exp, '_hypergrid_r', repeats[p], '_f', mg3$fold[r], '_', mg3$frail_lab[r], '_svd_', mg3$svd[r], '_mtry_', mg3$mtry[r], '_sfrac_', mg3$sample_frac_l[r], '.csv'))
      gc()
    }
  }
  #remove all objects related to current CV repeat
  objects_cv_repeat <- grep(paste0('r', repeats[p], '_f'), names(.GlobalEnv), value=TRUE)
  rm(list = objects_cv_repeat)
  gc()
}


#ELASTIC NET - summary
#Summarize performance for all completed enet models
enet_output <- grep('_hypergrid_', list.files(enet_modeldir), value = TRUE)
enet_output <- lapply(paste0(enet_modeldir, enet_output), fread)
enet_output <- rbindlist(enet_output)
#Save
fwrite(enet_output, paste0(outdir, 'exp', exp, '_enet_performance.csv'))
#Summarize benchmarking for all completed enet models
enet_bench <- grep('_duration_hyper_', list.files(enet_durationdir), value = TRUE)
enet_bench <- lapply(paste0(enet_durationdir, enet_bench), fread)
enet_bench <- rbindlist(enet_bench)
#Save
fwrite(enet_bench, paste0(outdir, 'exp', exp, '_enet_cpu_time.csv'))

#RANDOM FOREST - summary
#Summarize performance for all completed RF models
rf_output <- grep('_hypergrid_', list.files(rf_modeldir), value = TRUE)
rf_output <- lapply(paste0(rf_modeldir, rf_output), fread)
rf_output <- rbindlist(rf_output)
#Save
fwrite(rf_output, paste0(outdir, 'exp', exp, '_rf_performance.csv'))
#Summarize benchmarking for all completed RF models
rf_bench <- grep('_duration_hyper_', list.files(rf_durationdir), value = TRUE)
rf_bench <- lapply(paste0(rf_durationdir, rf_bench), fread)
rf_bench <- rbindlist(rf_bench)
#Save
fwrite(rf_bench, paste0(outdir, 'exp', exp, '_rf_cpu_time.csv'))

