library(data.table)
library(dplyr)
library(ggplot2)
library(ggrepel)
library(stringr)

se <- function(x) {
  # remove NA (necessary if there is an error (e.g. test fold is completely 
  # missing pos or neg class))
  x <- x[!is.na(x)]
  sd(x)/sqrt(length(x))
}

# set directories based on location
dirs = c(paste0('/gwshare/frailty/output/'),
         '/Users/martijac/Documents/Frailty/frailty_classifier/output/',
         '/media/drv2/andrewcd2/frailty/output/')
for (d in 1:length(dirs)) {
  if (dir.exists(dirs[d])) {
    rootdir = dirs[d]
  }
}

#constants
frail_lab <- c('Msk_prob', 'Fall_risk', 'Nutrition', 'Resp_imp')
models <- c('enet', 'rf', 'nn_single')
rfenet_batches <- c('AL03', 'AL04')
nn_batches <- c('AL01', 'AL02', 'AL03', 'AL04')

#gather performance
ep_list <- list()
for (b in 1:length(rfenet_batches)){
  if (rfenet_batches[b] == 'AL03') {
    ep <- fread(paste0(rootdir,
                    'lin_trees_SENT/exp1292021/exp1292021_enet_performance.csv'))
  } else {
    ep <- fread(paste0(rootdir,
                       'saved_models/', rfenet_batches[b], '/lin_trees/exp',
                       rfenet_batches[b], '_enet_performance.csv'))
  }
  ep$model <- 'enet'
  ep$batch <- rfenet_batches[b]
  ep_list[[b]] <- ep
}
enet_performance <- rbindlist(ep_list)

rf_list <- list()
for (b in 1:length(rfenet_batches)){
  if (rfenet_batches[b] == 'AL03') {
    rf <- fread(paste0(rootdir,
                       'lin_trees_SENT/exp1292021/exp1292021_rf_performance.csv'))
  } else {
    rf <- fread(paste0(rootdir,
                       'saved_models/', rfenet_batches[b], '/lin_trees/exp',
                       rfenet_batches[b], '_rf_performance.csv'))
  }
  rf$model <- 'rf'
  rf$batch <- rfenet_batches[b]
  rf_list[[b]] <- rf
}
rf_performance <- rbindlist(rf_list)

nn_single_performance <- fread(paste0(rootdir,
                               'figures_tables/learning_curve_stask.csv'))
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
hyperparams_nn_single <- c('n_dense', 'n_units', 'dropout', 'l1_l2_pen',
                           'use_case_weights')
#calculate performance
#note: this should be run separately for each batch so that it finds the best
#hyperparams specific to each bach
perf_calc <- function(raw_perf){
  raw_perf <- as.data.frame(raw_perf)
  if (raw_perf$model[1] == 'enet') {
    hyperparams = hyperparams_enet
    } else if (raw_perf$model[1] == 'rf') {
      hyperparams = hyperparams_rf
      } else if (raw_perf$model[1] == 'nn_single') {
        hyperparams = hyperparams_nn_single}
  
  #step 1 (best hyperparams for each frailty aspect)
  group1 <- c('frail_lab', hyperparams)
  step_1 <- raw_perf %>%
    group_by_at(vars(all_of(group1))) %>%
    summarise_at(vars(grep('sbrier', colnames(raw_perf), value = TRUE)),
                 list(mean = mean, se = se)) %>%
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
  
  #step 3 (mean & se for performance of best hyperparams across folds)
  step_3 <- step_2 %>%
    summarise_at(vars(grep('sbrier', colnames(step_2), value = TRUE)),
                 list(mean = mean, se = se))
  
  step_3$model <- raw_perf$model[1]
  step_3$batch <- raw_perf$batch[1]
  
  return(list(select(step_1, -'hyperp'), step_3))
}

#combine summary performance for all batches & models
mod_l <- list()
for (m in 1:length(models)) {
  if (models[m] == 'nn_single') {
    batches <- nn_batches
  } else {
    batches <- rfenet_batches
  }
  perf_l <- list()
  for (b in 1:length(batches)) {
    perf <- perf_calc(get(paste0(models[m], '_performance'))[batch == batches[b],])
    perf_l[[b]] <- perf[[2]][c('model', 'batch', 'sbrier_multi_all_mean',
                                    'sbrier_multi_all_se')]
  }
  mod_l <- c(mod_l, perf_l)
}
all_performance <- rbindlist(mod_l)


#add multi-task NN
nn_multi_performance <- fread(paste0(rootdir,
                              'figures_tables/learning_curve_mtask.csv'))
nn_multi_performance[, 'model'] <- 'nn_multi'
nn_multi_performance[, 'sbrier_multi_all_mean'] <- nn_multi_performance[, 'brier_mean_aspects_mean']
nn_multi_performance[, 'sbrier_multi_all_se'] <- nn_multi_performance[, 'brier_mean_aspects_se']
nn_multi_performance <- nn_multi_performance %>%
  group_by(batch) %>%
  arrange(desc(sbrier_multi_all_mean)) %>%
  slice(1)
all_performance <- rbind(all_performance,
                         nn_multi_performance[, c('model', 'batch', 
                         'sbrier_multi_all_mean', 'sbrier_multi_all_se')])

#plot best performance
ggplot(all_performance, aes(x = batch, y = sbrier_multi_all_mean, group=model)) +
  geom_line(aes(color = model)) +
  geom_pointrange(aes(ymin = (sbrier_multi_all_mean - sbrier_multi_all_se),
                      ymax = (sbrier_multi_all_mean + sbrier_multi_all_se),
                      color = model)) +
  labs(title = 'Learning curve',
       y = 'Scaled Brier Score',
       x = 'Batch') +
  theme(legend.position = 'bottom') +
  theme_bw()

#combine summary performance for all batches & models
asp_l <- list()
for (f in 1:length(frail_lab)){
  mod_l <- list()
  for (m in 1:length(models)) {
    if (models[m] == 'nn_single') {
      batches <- nn_batches
    } else {
      batches <- rfenet_batches
    }
    perf_l <- list()
    for (b in 1:length(batches)) {
      m_p <- get(paste0(models[m], '_performance'))
      bat <- batches[b]
      fra <- frail_lab[f]
      perf <- perf_calc(filter(m_p, (batch ==  bat&
                            frail_lab == fra)))
      perf_l[[b]] <- perf[[2]][c('model', 'batch', 'sbrier_multi_all_mean',
                                 'sbrier_multi_all_se')]
      perf_l[[b]][, 'frail_lab'] <- frail_lab[f]
    }
    mod_l <- c(mod_l, perf_l)
  }
  asp_l <- c(asp_l, mod_l)
}
aspect_performance <- rbindlist(asp_l)


ggplot(aspect_performance[frail_lab == 'Resp_imp', ], aes(x = batch, y = sbrier_multi_all_mean, group=model)) +
  geom_line(aes(color = model)) +
  geom_pointrange(aes(ymin = (sbrier_multi_all_mean - sbrier_multi_all_se),
                      ymax = (sbrier_multi_all_mean + sbrier_multi_all_se),
                      color = model)) +
  labs(title = 'Learning curve',
       y = 'Scaled Brier Score',
       x = 'Batch') +
  theme(legend.position = 'bottom') +
  theme_bw()

ggplot(aspect_performance[frail_lab == 'Msk_prob', ], aes(x = batch, y = sbrier_multi_all_mean, group=model)) +
  geom_line(aes(color = model)) +
  geom_pointrange(aes(ymin = (sbrier_multi_all_mean - sbrier_multi_all_se),
                      ymax = (sbrier_multi_all_mean + sbrier_multi_all_se),
                      color = model)) +
  labs(title = 'Learning curve',
       y = 'Scaled Brier Score',
       x = 'Batch') +
  theme(legend.position = 'bottom') +
  theme_bw()

ggplot(aspect_performance[frail_lab == 'Nutrition', ], aes(x = batch, y = sbrier_multi_all_mean, group=model)) +
  geom_line(aes(color = model)) +
  geom_pointrange(aes(ymin = (sbrier_multi_all_mean - sbrier_multi_all_se),
                      ymax = (sbrier_multi_all_mean + sbrier_multi_all_se),
                      color = model)) +
  labs(title = 'Learning curve',
       y = 'Scaled Brier Score',
       x = 'Batch') +
  theme(legend.position = 'bottom') +
  theme_bw()



###########
# Historical learning curve
lc_historical <- fread(paste0(rootdir, 'figures_tables/learning_curve_AL.csv'))

ggplot(lc_historical, aes(x = batch, y = brier_aspectwise, group=1)) +
  geom_line() +
  geom_pointrange(aes(ymin = (brier_aspectwise - brier_aspectwise_se),
                      ymax = (brier_aspectwise + brier_aspectwise_se))) +
  labs(title = 'Learning curve',
       y = 'Scaled Brier Score',
       x = 'Batch') +
  theme(legend.position = 'bottom') +
  theme_bw()



# count notes for each batch
note_raw <- fread(paste0(rootdir,
                           '/notes_labeled_embedded_SENTENCES/notes_train_official.csv'))
note_count <- note_raw %>%
  group_by(batch) %>%
  count()
AL01 = sum(note_count[note_count$batch %in% c('batch_01', 'batch_02', 
                                              'batch_03', 'AL00'), ]$n)
AL02 = sum(note_count[note_count$batch %in% c('AL01', 'AL01_v2',
                                              'AL01_v2ALTERNATE'), ]$n, AL01)
AL03 = sum(note_count[note_count$batch %in% c('AL02_v2'), ]$n, AL02)
AL04 = sum(note_count[note_count$batch %in% c('AL03'), ]$n, AL03)
note_count <- data.frame(n_notes= c(AL01, AL02, AL03, AL04),
                         batch = c('AL01', 'AL02', 'AL03', 'AL04'))

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
  sent$n_Msk_pos <- sum(d$Msk_prob_pos)
  sent$n_Resp_pos <- sum(d$Resp_imp_pos)
  sent$n_Fall_pos <- sum(d$Fall_risk_pos)
  sent$n_Nutrition_pos <- sum(d$Nutrition_pos)
  sent$n_Msk_neg <- sum(d$Msk_prob_neg)
  sent$n_Resp_neg <- sum(d$Resp_imp_neg)
  sent$n_Fall_neg <- sum(d$Fall_risk_neg)
  sent$n_Nutrition_neg <- sum(d$Nutrition_neg)
  s_l[[b]] <- sent
}
sent_count <- rbindlist(s_l)

#patients from AL03
AL02_new_pts <- note_raw[note_raw$batch %in% c('AL01', 'AL01_v2',
                                               'AL01_v2ALTERNATE'), ]$PAT_ID
AL03_new_pts <- note_raw[note_raw$batch %in% c('AL02_v2'), ]$PAT_ID
AL02 <- fread(paste0(rootdir,
                     'saved_models/AL02/processed_data/full_set/full_df.csv'))
AL03 <- fread(paste0(rootdir,
                'saved_models/AL03/processed_data/full_set/full_df.csv'))
nrow(filter(AL02, PAT_ID %in% AL02_new_pts))
nrow(filter(AL03, PAT_ID %in% AL03_new_pts))

#combine counts
note_sent_count <- left_join(sent_count, note_count)

#histograms
ggplot(note_sent_count, aes(x = batch, y = n_notes)) +
  geom_col() +
  theme_bw() +
  geom_label_repel(aes(label=n_notes), direction = 'y', nudge_y = -0.05, segment.size=0)

ggplot(sent_count, aes(x = batch, y = n_patients)) +
  geom_col() +
  theme_bw() +
  geom_label_repel(aes(label=n_patients), direction = 'y', nudge_y = -0.05, segment.size=0)

ggplot(sent_count, aes(x = batch, y = n_sent)) +
  geom_col() +
  theme_bw() +
  geom_label_repel(aes(label=n_sent), direction = 'y', nudge_y = -0.05, segment.size=0)

ggplot(sent_count, aes(x = batch, y = n_sent)) +
  geom_col() +
  theme_bw() +
  geom_label_repel(aes(label=n_sent), direction = 'y', nudge_y = -0.05, segment.size=0)

# sentence count histogram grouped by frailty aspect
###########################################




#hyperparameters with best performance in the most recent batch
best_enet <- perf_calc(enet_performance[batch == 'AL04', ])[[1]]
best_rf <- perf_calc(rf_performance[batch == 'AL04', ])[[1]]
best_nn_single <- perf_calc(nn_single_performance[batch == 'AL04', ])[[1]]
print(best_enet)
print(best_rf) 
print(best_nn_single)















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
    hyperparams = hyperparams_nn_single}
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

