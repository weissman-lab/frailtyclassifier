library(data.table)
library(dplyr)
library(tidyr)
library(ggplot2)
library(ggrepel)
library(stringr)


# set directories based on location
dirs = c(paste0('/gwshare/frailty/output/saved_models/'),
         '/Users/martijac/Documents/Frailty/frailty_classifier/output/',
         '/media/drv2/andrewcd2/frailty/output/')
for (d in 1:length(dirs)) {
  if (dir.exists(dirs[d])) {
    rootdir = dirs[d]
  }
}

#constants
frail_lab <- c('Msk_prob', 'Fall_risk', 'Nutrition', 'Resp_imp')
batches <- c('AL01', 'AL02', 'AL03', 'AL04', 'AL05')


# Calculate performance. Skip step #1 &2 for multi-task. Must go through 
# steps 1 & 2 for single task models (NN, RF, enet) in order to properly
# calculate mean and se across folds given that each frailty aspect can have a
# different set of optimal hyperparameters.
# 1.) for each batch, find the best hyperparameters for each model type (RF, 
# enet, singleNN) for each frailty aspect
# 2.) go back to each repeat/fold and find the mean performance across aspects
# (brier_mean_aspects) for those hyperparameters (ie there may be different 
# hyperparams for each frailty aspect)
# 3.) find the mean and se for perfomance across all repeats/folds.

#list relevant hyperparams
hyperparams_enet <- c('SVD', 'lambda', 'alpha', 'case_weights')
hyperparams_rf <- c('SVD', 'mtry', 'sample_frac', 'case_weights')
hyperparams_nn <- c('n_dense', 'n_units', 'dropout', 'l1_l2_pen',
                           'use_case_weights')
#################### MEAN ####################
#calculate performance
#note: this should be run separately for each batch so that it finds the best
#hyperparams specific to each bach
perf_calc_MEAN <- function(raw_perf){
  raw_perf <- as.data.frame(raw_perf)
  if (raw_perf$model[1] == 'enet') {
    hyperparams = hyperparams_enet
    } else if (raw_perf$model[1] == 'rf') {
      hyperparams = hyperparams_rf
      } else if (raw_perf$model[1] == 'nn_single') {
        hyperparams = hyperparams_nn}
  #step 1 (best hyperparams for each frailty aspect)
  group1 <- c('frail_lab', hyperparams)
  step_1 <- raw_perf %>%
    group_by_at(vars(all_of(group1))) %>%
    summarise_at(vars(grep('sbrier', colnames(raw_perf), value = TRUE)),
                 list(mean = mean, sd = sd)) %>%
    na.omit() %>%
    ungroup() %>%
    group_by(frail_lab) %>%
    arrange(desc(sbrier_multi_mean), .by_group = TRUE) %>%
    slice(1)
  #step 2 (mean performance of best hyperparams across all aspects for each fold)
  hypcols <- c('frail_lab', hyperparams)
  step_1$hyperp <- do.call(paste0, c(step_1[, hypcols]))
  raw_perf$hyperp <- do.call(paste0, c(raw_perf[, hypcols]))
  step_2 <- raw_perf %>%
    filter(hyperp %in% step_1$hyperp) %>%
    group_by(cv_repeat, fold) %>%
    summarise_at(vars(grep('sbrier', colnames(raw_perf), value = TRUE)),
                 list(all = mean)) %>%
    ungroup()
  #step 3 (mean & sd for performance of best hyperparams across folds)
  step_3 <- step_2 %>%
    summarise_at(vars(grep('sbrier', colnames(step_2), value = TRUE)),
                 list(mean = mean, sd = sd))
  step_3$model <- raw_perf$model[1]
  step_3$batch <- raw_perf$batch[1]
  return(list(select(step_1, -'hyperp'), step_3))
}

#combine summary performance for all batches & models
mod_l <- list()
for (m in 1:length(models)) {
  if (models[m] == 'nn_single') {
    batches <- batches
  } else if (models[m] == 'enet'){
    batches <- enet_batches
  } else if (models[m] == 'rf'){
    batches <- rf_batches
  }
  perf_l <- list()
  for (b in 1:length(batches)) {
    perf <- perf_calc_MEAN(get(paste0(models[m], '_performance'))[batch == batches[b],])
    perf_l[[b]] <- perf[[2]][c('model', 'batch', 'sbrier_multi_all_mean',
                                    'sbrier_multi_all_sd')]
  }
  mod_l <- c(mod_l, perf_l)
}
all_performance_mean <- rbindlist(mod_l)



#################### MEDIAN ####################
#calculate performance
#note: this should be run separately for each batch so that it finds the best
#hyperparams specific to each bach
perf_calc_MEDIAN <- function(raw_perf){
  raw_perf <- as.data.frame(raw_perf)
  if (raw_perf$model[1] == 'enet') {
    hyperparams = hyperparams_enet
  } else if (raw_perf$model[1] == 'rf') {
    hyperparams = hyperparams_rf
  } else if (raw_perf$model[1] == 'nn_single') {
    hyperparams = hyperparams_nn}
  #step 1 (best hyperparams for each frailty aspect)
  group1 <- c('frail_lab', hyperparams)
  step_1 <- raw_perf %>%
    group_by_at(vars(all_of(group1))) %>%
    summarise_at(vars(grep('sbrier', colnames(raw_perf), value = TRUE)),
                 list(median = median,
                      iqr25 = ~quantile(., probs = 0.25, na.rm = TRUE),
                      iqr75 = ~quantile(., probs = 0.75, na.rm = TRUE))) %>%
    na.omit() %>%
    ungroup() %>%
    group_by(frail_lab) %>%
    arrange(desc(sbrier_multi_median), .by_group = TRUE) %>%
    slice(1)
  #step 2 (median performance of best hyperparams across all aspects for each fold)
  hypcols <- c('frail_lab', hyperparams)
  step_1$hyperp <- do.call(paste0, c(step_1[, hypcols]))
  raw_perf$hyperp <- do.call(paste0, c(raw_perf[, hypcols]))
  step_2 <- raw_perf %>%
    filter(hyperp %in% step_1$hyperp) %>%
    group_by(cv_repeat, fold) %>%
    summarise_at(vars(grep('sbrier', colnames(raw_perf), value = TRUE)),
                 list(all = median)) %>%
    ungroup()
  #step 3 (median & se for performance of best hyperparams across folds)
  step_3 <- step_2 %>%
    summarise_at(vars(grep('sbrier', colnames(step_2), value = TRUE)),
                 list(median = median,
                      iqr25 = ~quantile(., probs = 0.25),
                      iqr75 = ~quantile(., probs = 0.75)))
  step_3$model <- raw_perf$model[1]
  step_3$batch <- raw_perf$batch[1]
  return(list(select(step_1, -'hyperp'), step_3))
}

#combine summary performance for all batches & models
mod_l <- list()
for (m in 1:length(models)) {
  if (models[m] == 'nn_single') {
    batches <- nn_batches
  } else if (models[m] == 'enet'){
    batches <- enet_batches
  } else if (models[m] == 'rf'){
    batches <- rf_batches
  }
  perf_l <- list()
  for (b in 1:length(batches)) {
    perf <- perf_calc_MEDIAN(get(paste0(models[m], '_performance'))[batch == batches[b],])
    perf_l[[b]] <- perf[[2]][c('model', 'batch', 'sbrier_multi_all_median',
                               'sbrier_multi_all_iqr25', 'sbrier_multi_all_iqr75')]
  }
  mod_l <- c(mod_l, perf_l)
}
all_performance_median <- rbindlist(mod_l)



# add multi-task NN performance
# (fewer steps to summarize because all aspects get the same hyperparams)
nn_multi_performance <- fread(paste0(rootdir,
                                     'saved_models/',
                                     tail(nn_batches, 1),
                                     '/learning_curve_mtask.csv'))
nn_multi_performance[, 'cv_repeat'] <- nn_multi_performance[, 'repeat']
nn_multi_performance[, 'sbrier_multi_all'] <- nn_multi_performance[, 'brier_mean_aspects']
group <- c('batch', hyperparams_nn)
nn_multi_performance_summ <- nn_multi_performance %>%
  group_by_at(vars(all_of(group))) %>%
  summarise_at(vars(grep('Fall_risk|Msk_prob|Nutrition|Resp_imp|brier', colnames(nn_multi_performance), value = TRUE)),
               list(mean = mean,
                    sd = sd,
                    median = median,
                    iqr25 = ~quantile(., probs = 0.25),
                    iqr75 = ~quantile(., probs = 0.75)))
nn_multi_performance_summ[, 'model'] <- 'nn_multi'
nn_multi_perf_mean <- nn_multi_performance_summ %>%
  group_by(batch) %>%
  arrange(desc(sbrier_multi_all_mean)) %>%
  select(c('model', 'batch', 'sbrier_multi_all_mean', 'sbrier_multi_all_sd')) %>%
  slice(1) %>%
  ungroup()
nn_multi_perf_median <- nn_multi_performance_summ %>%
  group_by(batch) %>%
  arrange(desc(sbrier_multi_all_median)) %>%
  select(c('model', 'batch', 'sbrier_multi_all_median', 'sbrier_multi_all_iqr25', 'sbrier_multi_all_iqr75')) %>%
  slice(1) %>%
  ungroup()

#combine with other models
all_performance_mean <- rbind(all_performance_mean, nn_multi_perf_mean)
all_performance_median <- rbind(all_performance_median, nn_multi_perf_median)




#plot best performance: MEAN
ggplot(all_performance_mean, aes(x = batch, y = sbrier_multi_all_mean, group=model)) +
  #geom_line(aes(color = model)) +
  geom_pointrange(aes(ymin = (sbrier_multi_all_mean - sbrier_multi_all_sd),
                      ymax = (sbrier_multi_all_mean + sbrier_multi_all_sd),
                      color = model),
                  position = 'jitter') +
  labs(title = 'Learning curve',
       y = 'Scaled Brier Score, mean (SD)',
       x = 'Batch') +
  theme_bw() +
  theme(legend.position = 'bottom')

#table
fwrite(filter(all_performance_mean, batch == 'AL04'),
       paste0(rootdir, 'figures_tables/AL03_model_performance_mean.csv'))

filter(all_performance_mean, batch == 'AL03')

#plot best performance: MEDIAN
ggplot(all_performance_median, aes(x = batch, y = sbrier_multi_all_median, group=model)) +
  #geom_line(aes(color = model)) +
  geom_pointrange(aes(ymin = sbrier_multi_all_iqr25,
                      ymax = sbrier_multi_all_iqr75,
                      color = model),
                  position = 'jitter') +
  labs(title = 'Learning curve',
       y = 'Scaled Brier Score, median (IQR)',
       x = 'Batch') +
  theme_bw() +
  theme(legend.position = 'bottom')

fwrite(filter(all_performance_median, batch == 'AL03'),
       paste0(rootdir, 'figures_tables/AL03_model_performance_median.csv'))



#plot best performance for multi_NN by aspect: MEAN
nn_asp <- list()
for (f in 1:length(frail_lab)) {
  cols <- paste0(frail_lab[f], '_mean')
  nn <- nn_multi_performance_summ %>%
    group_by(batch) %>%
    arrange(desc(.data[[cols]])) %>%
    slice(1) %>%
    select('batch', paste0(frail_lab[f], '_mean'), paste0(frail_lab[f], '_sd'))
  nn_asp[[f]] <- nn
}
NN_best_aspects <- Reduce(function(x,y) merge(x,y,by="batch",all=TRUE), nn_asp)
nba_mean <- NN_best_aspects %>%
  pivot_longer(cols = paste0(frail_lab, '_mean'),
               names_to = c("Aspect", "crd"),
               names_sep = "_mean",
               values_to = 'sbrier_mean')%>%
  select('batch', 'Aspect', 'sbrier_mean')
nba_sd <- NN_best_aspects %>%
  pivot_longer(cols = paste0(frail_lab, '_sd'),
               names_to = c("Aspect", "crd"),
               names_sep = "_sd",
               values_to = 'sbrier_sd') %>%
  select('batch', 'Aspect', 'sbrier_sd')
NN_asp_mean <-left_join(nba_mean, nba_sd, by = c('batch', 'Aspect'))

#plot
ggplot(NN_asp_mean, aes(x = batch, y = sbrier_mean, group=Aspect)) +
  #geom_line(aes(color = Aspect)) +
  geom_pointrange(aes(ymin = (sbrier_mean - sbrier_sd),
                      ymax = (sbrier_mean + sbrier_sd),
                      color = Aspect),
                  position = 'jitter') +
  labs(title = 'Learning curve',
       y = 'Scaled Brier Score, mean (sd)',
       x = 'Batch') +
  theme_bw()+
  theme(legend.position = 'bottom')


#plot best performance for multi_NN by aspect: MEDIAN
nn_asp <- list()
for (f in 1:length(frail_lab)) {
  cols <- paste0(frail_lab[f], '_median')
  nn <- nn_multi_performance %>%
    group_by(batch) %>%
    arrange(desc(.data[[cols]])) %>%
    slice(1) %>%
    select('batch', paste0(frail_lab[f], '_median'),
           paste0(frail_lab[f], '_iqr25'),
           paste0(frail_lab[f], '_iqr75'))
  nn_asp[[f]] <- nn
}
NN_best_aspects <- Reduce(function(x, y) merge(x, y, by="batch", all=TRUE), nn_asp)
nba_median <- NN_best_aspects %>%
  pivot_longer(cols = paste0(frail_lab, '_median'),
               names_to = c("Aspect", "crd"),
               names_sep = "_median",
               values_to = 'sbrier_median')%>%
  select('batch', 'Aspect', 'sbrier_median')
nba_iqr25 <- NN_best_aspects %>%
  pivot_longer(cols = paste0(frail_lab, '_iqr25'),
               names_to = c("Aspect", "crd"),
               names_sep = "_iqr25",
               values_to = 'sbrier_iqr25') %>%
  select('batch', 'Aspect', 'sbrier_iqr25')
nba_iqr75 <- NN_best_aspects %>%
  pivot_longer(cols = paste0(frail_lab, '_iqr75'),
               names_to = c("Aspect", "crd"),
               names_sep = "_iqr75",
               values_to = 'sbrier_iqr75') %>%
  select('batch', 'Aspect', 'sbrier_iqr75')
NN_asp_median <-left_join(nba_median, nba_iqr25, by = c('batch', 'Aspect'))
NN_asp_median <-left_join(NN_asp_median, nba_iqr75, by = c('batch', 'Aspect'))

ggplot(NN_asp_median, aes(x = batch, y = sbrier_median, group=Aspect)) +
  #geom_line(aes(color = Aspect)) +
  geom_pointrange(aes(ymin = sbrier_iqr25,
                      ymax = sbrier_iqr75,
                      color = Aspect),
                  position = 'jitter') +
  labs(title = 'Learning curve',
       y = 'Scaled Brier Score, median (IQR)',
       x = 'Batch') +
  theme_bw() +
  theme(legend.position = 'bottom')



#plot performance by aspect for enet
enet_AL04 <- perf_calc_MEAN(enet_performance[batch == 'AL04', ])[[1]]
enet_AL04['batch'] <- 'AL04'
enet_AL03 <- perf_calc_MEAN(enet_performance[batch == 'AL03', ])[[1]]
enet_AL03['batch'] <- 'AL03'
enet_asp_mean <- rbind(enet_AL03, enet_AL04)


ggplot(enet_asp_mean, aes(x = batch, y = sbrier_multi_mean, group=frail_lab)) +
  geom_pointrange(aes(ymin = (sbrier_multi_mean - sbrier_multi_sd),
                      ymax = (sbrier_multi_mean + sbrier_multi_sd),
                      color = frail_lab),
                  position = 'jitter') +
  labs(title = 'Learning curve',
       y = 'Scaled Brier Score, mean (SD)',
       x = 'Batch') +
  theme_bw() +
  theme(legend.position = 'bottom')



###########
# Historical learning curve
lc_historical <- fread(paste0(rootdir, 'figures_tables/learning_curve_AL_mean.csv'))

ggplot(lc_historical, aes(x = batch, y = brier_aspectwise, group=1)) +
  geom_line() +
  geom_pointrange(aes(ymin = (brier_aspectwise - brier_aspectwise_se),
                      ymax = (brier_aspectwise + brier_aspectwise_se))) +
  labs(title = 'Learning curve',
       y = 'Scaled Brier Score, mean (SE)',
       x = 'Batch') +
  theme(legend.position = 'bottom') +
  theme_bw()

lc_historical <- fread(paste0(rootdir, 'figures_tables/learning_curve_AL_median.csv'))

ggplot(lc_historical, aes(x = batch, y = brier_aspectwise, group=1)) +
  geom_line() +
  geom_pointrange(aes(ymin = brier_aspectwise_iqr25,
                      ymax = brier_aspectwise_iqr75)) +
  labs(title = 'Learning curve',
       y = 'Scaled Brier Score, median (IQR)',
       x = 'Batch') +
  theme(legend.position = 'bottom') +
  theme_bw()





# count sentences for each frailty aspect for each batch
s_l <- list()
for (b in 1:length(nn_batches)) {
  d <- fread(paste0(rootdir,
               'saved_models/',
               nn_batches[b],
               '/processed_data/full_set/full_df.csv'))
  sent <- data.frame(batch = nn_batches[b])
  sent$n_sent <- nrow(d)
  sent$n_patients <- length(unique(d$PAT_ID))
  sent$Msk_pos <- sum(d$Msk_prob_pos)
  sent$Resp_pos <- sum(d$Resp_imp_pos)
  sent$Fall_pos <- sum(d$Fall_risk_pos)
  sent$Nutrition_pos <- sum(d$Nutrition_pos)
  sent$Msk_neg <- sum(d$Msk_prob_neg)
  sent$Resp_neg <- sum(d$Resp_imp_neg)
  sent$Fall_neg <- sum(d$Fall_risk_neg)
  sent$Nutrition_neg <- sum(d$Nutrition_neg)
  s_l[[b]] <- sent
}
sent_count <- rbindlist(s_l)

# check notes count for each batch
note_raw <- fread(paste0(rootdir,
                         '/notes_labeled_embedded_SENTENCES/notes_train_official.csv'))
note_count <- note_raw %>%
  group_by(batch) %>%
  count()
AL00 = sum(note_count[note_count$batch %in% c('batch_01', 'batch_02', 
                                              'batch_03'), ]$n)
AL01 = sum(note_count[note_count$batch %in% c('AL00'), ]$n, AL00)
AL02 = sum(note_count[note_count$batch %in% c('AL01', 'AL01_v2',
                                              'AL01_v2ALTERNATE'), ]$n, AL01)
AL03 = sum(note_count[note_count$batch %in% c('AL02_v2'), ]$n, AL02)
AL04 = sum(note_count[note_count$batch %in% c('AL03'), ]$n, AL03)
check_note_count <- data.frame(n_patients= c(AL00, AL01, AL02, AL03, AL04),
                         batch = c('AL00', 'AL01', 'AL02', 'AL03', 'AL04'))
if (identical(sent_count$n_patients, check_note_count$n_patients[2:5]) == FALSE)
  stop("note counts do not match")

#Total sentence counts for latest batch
AL04 <- filter(sent_count, batch == 'AL04')
Msk_sent <- AL04 %>%
  pivot_longer(cols = grep('Msk', colnames(sent_count), value = TRUE),
               names_to = 'Class',
               values_to = 'Count') %>%
  mutate(Aspect = 'Msk') %>%
  select('Class', 'Count', 'Aspect')
Resp_sent <- AL04 %>%
  pivot_longer(cols = grep('Resp', colnames(sent_count), value = TRUE),
               names_to = 'Class',
               values_to = 'Count') %>%
  mutate(Aspect = 'Respiratory') %>%
  select('Class', 'Count', 'Aspect')
Fall_sent <- AL04 %>%
  pivot_longer(cols = grep('Fall', colnames(sent_count), value = TRUE),
               names_to = 'Class',
               values_to = 'Count') %>%
  mutate(Aspect = 'Fall risk') %>%
  select('Class', 'Count', 'Aspect')
Nutrition_sent <- AL04 %>%
  pivot_longer(cols = grep('Nutrition', colnames(sent_count), value = TRUE),
               names_to = 'Class',
               values_to = 'Count') %>%
  mutate(Aspect = 'Nutrition') %>%
  select('Class', 'Count', 'Aspect')
asp_sent_count <- rbind(Msk_sent, Resp_sent, Fall_sent, Nutrition_sent)
ggplot(asp_sent_count, aes(x = Class, y = Count, group = Aspect)) +
  geom_col(aes(fill = Aspect)) +
  theme_bw() +
  geom_label_repel(aes(label=Count), 
                   direction = 'y', 
                   nudge_y = 0.5, 
                   segment.size=0) +
  labs(title = 'Number of sentences by aspect') +
  theme(legend.position = 'bottom')

asp_sent_count_short <- asp_sent_count %>%
  group_by(Aspect) %>%
  summarise(Count = sum(Count))
ggplot(asp_sent_count_short, aes(x = Aspect, y = Count, group = Aspect)) +
  geom_col(aes(fill = Aspect)) +
  theme_bw() +
  geom_label_repel(aes(label=Count), 
                   direction = 'y', 
                   nudge_y = 0.5, 
                   segment.size=0) +
  labs(title = 'Number of sentences by aspect')

#Percent non-neutral tokens
rowSums(AL04[, 4:ncol(AL04)])/AL04$n_sent

#patients by batch
ggplot(sent_count, aes(x = batch, y = n_patients)) +
  geom_col() +
  theme_bw() +
  geom_label_repel(aes(label=n_patients), direction = 'y', nudge_y = -0.05, segment.size=0)

#sentences by batch
ggplot(sent_count, aes(x = batch, y = n_sent)) +
  geom_col() +
  theme_bw() +
  geom_label_repel(aes(label=n_sent), direction = 'y', nudge_y = 0.5, segment.size=0) +
  labs(title = 'Number of sentences by batch',
       x = 'Batch',
       y = 'Number of sentences') 



#hyperparameters with best performance in the most recent batch
best_enet <- perf_calc(enet_performance[batch == 'AL04', ])[[1]]
best_rf <- perf_calc(rf_performance[batch == 'AL04', ])[[1]]
best_nn_single <- perf_calc(nn_single_performance[batch == 'AL04', ])[[1]]
print(best_enet)
print(best_rf) 
print(best_nn_single)



#CPU time
rf_time <- fread(paste0(
  rootdir, 
  'lin_trees_SENT/exp1292021/exp1292021_rf_cpu_time.csv'))
enet_time <- fread(paste0(
  rootdir,
  'lin_trees_SENT/exp1292021/exp1292021_enet_cpu_time.csv'))
nn_single_performance$test <- nn_single_performance$model
nn_single_performance$elapsed <- nn_single_performance$runtime

time_sum <- function(time) {
  d <- data.frame(model = time[['test']][1],
             mean = round(mean(time[['elapsed']]), 2),
             sd = round(sd(time[['elapsed']]), 2),
             median = round(median(time[['elapsed']]), 2),
             iqr25 = round(quantile(time[['elapsed']], probs = 0.25), 2),
             iqr75 = round(quantile(time[['elapsed']], probs = 0.75), 2),
             row.names = NULL)
  if (d[['model']] == 'rf'){
    d['Processor'] <- '64 CPU'
  }
  if (d[['model']] == 'glmnet'){
    d['Processor'] <- '1 CPU'
  }
  if (d[['model']] == 'nn_single'){
    d['Processor'] <- '1 GPU'
  }
  return(d)
}

rf_time <- time_sum(rf_time)
enet_time <- time_sum(enet_time)
enet_time[2:6] <- round(enet_time[2:6]/25, 1)
nn_time <- time_sum(nn_single_performance)
times <- rbind(rf_time, enet_time, nn_time)



#Calibration
op_func <- function(r, f, frail_lab, svd, alpha){
  #load obs for one of the folds
  obs <- fread(paste0('/Users/martijac/Documents/Frailty/frailty_classifier/output/lin_trees_SENT/exp1292021/r', r, '_f', f, '_va_df.csv'))
  obs <- obs[, c('sentence_id', grep(frail_lab, colnames(obs), value = TRUE), 'sentence'), with = FALSE]
  #get the preds for the best model
  preds <- fread(paste0('/Users/martijac/Documents/Frailty/frailty_classifier/output/lin_trees_SENT/exp1292021/enet_preds/exp1292021_preds_r', r, '_f', f, '_', frail_lab, '_svd_', svd, '_alpha', alpha, '_cw0.csv'))
  preds <- preds[preds$lambda == 0.004217, ]
  #combine
  obs_preds <- cbind(obs=obs, pred=preds)
  return(obs_preds)
}

library(gmish)
#pick a reasonably good fold
# enet_performance %>%
#   filter(frail_lab == 'Resp_imp' & SVD == 'embed' & lambda == 0.004217 & alpha == 0.1 & case_weights == FALSE) %>%
#   arrange(desc(sbrier_pos)) %>%
#   select(c('frail_lab', 'cv_repeat', 'fold', 'SVD', 'lambda', 'alpha', 'case_weights', 'sbrier_multi', 'sbrier_pos', 'sbrier_neg'))
obs_preds <- op_func(1, 2, 'Resp_imp', 'embed', 1)
#calib plot
mc_calib_plot(obs.Resp_imp_pos + obs.Resp_imp_neg ~ pred.Resp_imp_pos +
                pred.Resp_imp_neg, data = obs_preds, cuts = 5) +
  labs(title = 'Respiratory') +
  theme(legend.position = 'bottom') 
  

#pick a reasonably good fold
# enet_performance %>%
#   filter(frail_lab == 'Msk_prob' & SVD == 'embed' & lambda == 0.004217 & alpha == 0.1 & case_weights == FALSE) %>%
#   arrange(desc(sbrier_pos)) %>%
#   select(c('frail_lab', 'cv_repeat', 'fold', 'SVD', 'lambda', 'alpha', 'case_weights', 'sbrier_multi', 'sbrier_pos', 'sbrier_neg'))
obs_preds <- op_func(3, 1, 'Msk_prob', 'embed', 1)
#calib plot
mc_calib_plot(obs.Msk_prob_pos + obs.Msk_prob_neg ~ pred.Msk_prob_pos +
                pred.Msk_prob_neg, data = obs_preds, cuts = 5) +
  labs(title = 'Musculoskeletal') +
  theme(legend.position = 'bottom') 


#pick a reasonably good fold
# enet_performance %>%
#   filter(frail_lab == 'Nutrition' & SVD == '1000' & lambda == 0.004217 & alpha == 0.1 & case_weights == FALSE) %>%
#   arrange(desc(sbrier_pos)) %>%
#   select(c('frail_lab', 'cv_repeat', 'fold', 'SVD', 'lambda', 'alpha', 'case_weights', 'sbrier_multi', 'sbrier_pos', 'sbrier_neg'))
obs_preds <- op_func(1, 8, 'Nutrition', '1000', 1)
#calib plot
mc_calib_plot(obs.Nutrition_pos + obs.Nutrition_neg ~ pred.Nutrition_pos +
                pred.Nutrition_neg, data = obs_preds, cuts = 5) +
  labs(title = 'Nutrition') +
  theme(legend.position = 'bottom') 

#pick a reasonably good fold
# enet_performance %>%
#   filter(frail_lab == 'Fall_risk' & SVD == 'embed' & lambda == 0.004217 & alpha == 0.1 & case_weights == FALSE) %>%
#   arrange(desc(sbrier_pos)) %>%
#   select(c('frail_lab', 'cv_repeat', 'fold', 'SVD', 'lambda', 'alpha', 'case_weights', 'sbrier_multi', 'sbrier_pos', 'sbrier_neg'))
obs_preds <- op_func(3, 3, 'Fall_risk', 'embed', 1)
#calib plot
mc_calib_plot(obs.Fall_risk_pos + obs.Fall_risk_neg ~ pred.Fall_risk_pos + 
                pred.Fall_risk_neg, data = obs_preds, cuts = 5) +
  labs(title = 'Fall Risk') +
  theme(legend.position = 'bottom') 





#TF-IDF vs embeddings overall
summary(lm(sbrier_multi ~ SVD, 
              data = enet_performance[SVD %in% c('1000', 'embed') &
                                        batch == 'AL04', ]))
summary(lm(sbrier_multi ~ SVD, 
           data = rf_performance[SVD %in% c('1000', 'embed') &
                                    batch == 'AL04', ]))

#TF-IDF vs embeddings by frailty aspect by class for the best 5 models
top_50 <- function(raw_perf){
  raw_perf <- as.data.frame(raw_perf)
  if (raw_perf$model[1] == 'enet') {
    hyperparams = hyperparams_enet
  } else if (raw_perf$model[1] == 'rf') {
    hyperparams = hyperparams_rf
  } else if (raw_perf$model[1] == 'nn_single') {
    hyperparams = hyperparams_nn}
  group1 <- c('frail_lab', hyperparams)
  step_1 <- raw_perf %>%
    filter(batch == 'AL04') %>%
    group_by_at(vars(all_of(group1))) %>%
    summarise_at(vars(grep('sbrier', colnames(raw_perf), value = TRUE)),
                 list(mean = mean, se = se)) %>%
    na.omit() %>%
    ungroup() %>%
    mutate(tfidf_embed = ifelse(SVD %in% c('300', '1000'),
                                'TF-IDF',
                                'Embeddings')) %>%
    group_by(frail_lab, tfidf_embed) %>%
    arrange(desc(sbrier_multi_mean), .by_group = TRUE) %>%
    slice(1:50)
  return(step_1)
}

enet_top50 <- top_50(enet_performance)

ggplot(enet_top50, aes(x = frail_lab, y = sbrier_multi_mean, ymin = (sbrier_multi_mean - sbrier_multi_se), ymax = (sbrier_multi_mean + sbrier_multi_se)))+
  geom_pointrange(stat='identity', position = 'jitter', aes(color = tfidf_embed))+
  scale_color_discrete(name = 'Frailty aspect') +
  labs(title = 'Text features (top 50 linear models)', x = 'Text features', y = 'Scaled Brier score') +
  theme_bw() +
  theme(legend.position = 'bottom')

rf_top50 <- top_50(rf_performance)

ggplot(rf_top50, aes(x = frail_lab, y = sbrier_multi_mean, ymin = (sbrier_multi_mean - sbrier_multi_se), ymax = (sbrier_multi_mean + sbrier_multi_se)))+
  geom_pointrange(stat='identity', position = 'jitter', aes(color = tfidf_embed))+
  scale_color_discrete(name = 'Frailty aspect') +
  labs(title = 'Text features (top 50 random forests)', x = 'Text features', y = 'Scaled Brier score') +
  theme_bw() +
  theme(legend.position = 'bottom')

