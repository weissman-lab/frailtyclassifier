library(data.table)
library(dplyr)
library(tidyr)
library(ggplot2)
library(ggrepel)
library(stringr)
library(gmish)
library(PRROC)
library(ROCR)
library(caret)
library(ggsci)

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
batches <- c('AL01', 'AL02', 'AL03', 'AL04', 'AL05')
classes <- c('Positive', 'Negative', 'Neutral')
cls <- c('_pos', '_neg', '_neut')





########################### HISTORICAL LEARNING CURVE #########################

#load cross validated training performance
train_performance_mean <- fread(paste0(rootdir,
                                       'figures_tables/all_train_cv_performance.csv'))
lc_newer <- train_performance_mean[(batch %in% c('AL03', 'AL04', 'AL05') &
                                      model == 'nn_multi' &
                                      text == 'word2vec'), -1]
lc_newer <- lc_newer[, 1:3]

#load performance from AL01 and AL02 (before cross validation & sentences)
lc_older <- fread(paste0(rootdir,
                              'figures_tables/learning_curve_AL_mean.csv'),
                       drop = 1)
lc_older <- lc_older[batch %in% c('AL01', 'AL02'), ]
colnames(lc_older) <- colnames(lc_newer)
#combine
lc_historical <- rbind(lc_older, lc_newer)

ggplot(lc_historical, aes(x = batch, y = sbrier_multi_all_mean, group=1)) +
  geom_line() +
  geom_pointrange(aes(ymin = (sbrier_multi_all_mean - sbrier_multi_all_sd),
                      ymax = (sbrier_multi_all_mean + sbrier_multi_all_sd))) +
  labs(title = 'Learning curve',
       y = 'Scaled Brier Score, mean (SE)',
       x = 'Batch') +
  theme(legend.position = 'bottom') +
  theme_bw()





########################## TEST SET PERFORMANCE BY BATCH ###################### 

# Load test set performance
all_test_perf <- fread(paste0(
  rootdir,
  'figures_tables/all_test_set_performance.csv'))

all_test_perf <- mutate(all_test_perf,
                      text = str_split(hyperparams, ' ', simplify = TRUE)[, 1],
                      text = ifelse(text == 'embed', 'word2vec',
                                    ifelse(text == '1000', 'TF-IDF 1000-d',
                                           ifelse(text == '300', 'TF-IDF 1000-d', text))))

all_test_perf <- all_test_perf %>%
  mutate(Model = ifelse(model == 'enet', 'Elastic net',
                        ifelse(model == 'nn_multi', 'Multi-task NN',
                               ifelse(model == 'nn_single', 'Single-task NN',
                                      ifelse(model == 'rf', 'Random forest', NA)))))

all_test_perf_multiaspect <- all_test_perf %>%
  group_by(Model, batch, text) %>%
  summarise_at(vars(grep('sbrier', colnames(all_test_perf), value = TRUE)),
               list(mean = mean), na.rm = TRUE)

best_test_perf_multiaspect <- all_test_perf_multiaspect %>%
  group_by(Model, batch) %>%
  arrange(desc(sbrier_multi_mean)) %>%
  slice(1) %>%
  ungroup()

#plot
ggplot(best_test_perf_multiaspect, aes(x = batch, y = sbrier_multi_mean, group = Model)) +
  geom_line(aes(color = Model)) +
  geom_errorbar(aes(ymin = (sbrier_multi_mean - sbrier_multi_sd_mean),
                      ymax = (sbrier_multi_mean + sbrier_multi_sd_mean),
                      color = Model,
                    width = 0.1)) +
  labs(title = 'Performance on test set by model type',
       y = 'Scaled Brier Score',
       x = 'Batch') +
  theme_bw() +
  theme(legend.position = 'bottom') +
  scale_color_nejm()




############################# TEST SET BEST MODEL ##############################
test_all <- all_test_perf_multiaspect[all_test_perf_multiaspect$batch == 'AL05', ]

test_all <- test_all %>%
  group_by(Model) %>%
  arrange(desc(sbrier_multi_mean), .by_group = TRUE)

test_all['Text features'] <- test_all$text
test_all['Multiclass SBS (SD)'] <- paste0(
  as.character(round(test_all$sbrier_multi_mean, 2)),
  ' (',
  as.character(round(test_all$sbrier_multi_sd_mean, 2)),
  ')')
test_all['Positive SBS (SD)'] <- paste0(
  as.character(round(test_all$sbrier_pos_mean, 2)),
  ' (',
  as.character(round(test_all$sbrier_pos_sd_mean, 2)),
  ')')
test_all['Negative SBS (SD)'] <- paste0(
  as.character(round(test_all$sbrier_neg_mean, 2)),
  ' (',
  as.character(round(test_all$sbrier_neg_sd_mean, 2)),
  ')')
test_all['Neutral SBS (SD)'] <- paste0(
  as.character(round(test_all$sbrier_neut_mean, 2)),
  ' (',
  as.character(round(test_all$sbrier_neut_sd_mean, 2)),
  ')')

cols <- c('Model', 'Text features',
          grep('(SD)', colnames(test_all), value = TRUE))

test_all <- rbind(test_all[(test_all$Model == 'Elastic net'), cols],
                  test_all[(test_all$Model == 'Random forest'), cols],
                  test_all[(test_all$Model == 'Single-task NN'), cols],
                  test_all[(test_all$Model == 'Multi-task NN'), cols])

fwrite(test_all,
       paste0(rootdir,
              'figures_tables/table3.csv'))




############################### SENTENCE COUNTS ###############################

# count sentences for each frailty aspect for each batch in training set
if (file.exists(paste0(rootdir,
                      'figures_tables/sentence_counts_train.csv'))){
  sent_count_train <- fread(paste0(rootdir,
                             'figures_tables/sentence_counts_train.csv'))
} else{
  s_l <- list()
  for (b in 1:length(batches)) {
    d <- fread(paste0(rootdir,
                      'saved_models/',
                      batches[b],
                      '/processed_data/full_set/full_df.csv'))
    sent <- data.frame(batch = batches[b])
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
  sent_count_train <- rbindlist(s_l)
  
  fwrite(sent_count,
         paste0(rootdir,
                'figures_tables/sentence_counts_train.csv'))
}

# count sentences for each frailty aspect for test notes
if (file.exists(paste0(rootdir,
                       'figures_tables/sentence_counts_test.csv'))){
  sent_count_test <- fread(paste0(rootdir,
                             'figures_tables/sentence_counts_test.csv'))
} else{
    d <- fread(paste0(rootdir,
                      'saved_models/AL05/processed_data/test_set/full_df.csv'))
    sent <- data.frame(batch = 'test')
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

    sent_count_test <- sent
    fwrite(sent,
         paste0(rootdir,
                'figures_tables/sentence_counts_test.csv'))
}


#sentences by batch
ggplot(sent_count_train, aes(x = batch, y = n_sent)) +
  geom_col() +
  theme_bw() +
  geom_label_repel(aes(label=n_sent), direction = 'y', nudge_y = 10, segment.size=0) +
  labs(title = 'Cumulative number of sentences',
       x = 'Batch')  +
  theme(axis.title.y=element_blank(),
        axis.text.y=element_blank(),
        axis.ticks.y=element_blank(),
        legend.position = 'none')+
  ylim(0, 80000)

#Total sentence counts for latest batch
AL04 <- filter(sent_count_train, batch == 'AL04')
Msk_sent <- AL04 %>%
  pivot_longer(cols = grep('Msk', colnames(sent_count_train), value = TRUE),
               names_to = 'Class',
               values_to = 'Count') %>%
  mutate(Aspect = 'Msk') %>%
  select('Class', 'Count', 'Aspect')
Resp_sent <- AL04 %>%
  pivot_longer(cols = grep('Resp', colnames(sent_count_train), value = TRUE),
               names_to = 'Class',
               values_to = 'Count') %>%
  mutate(Aspect = 'Respiratory') %>%
  select('Class', 'Count', 'Aspect')
Fall_sent <- AL04 %>%
  pivot_longer(cols = grep('Fall', colnames(sent_count_train), value = TRUE),
               names_to = 'Class',
               values_to = 'Count') %>%
  mutate(Aspect = 'Fall risk') %>%
  select('Class', 'Count', 'Aspect')
Nutrition_sent <- AL04 %>%
  pivot_longer(cols = grep('Nutrition', colnames(sent_count_train), value = TRUE),
               names_to = 'Class',
               values_to = 'Count') %>%
  mutate(Aspect = 'Nutrition') %>%
  select('Class', 'Count', 'Aspect')
asp_sent_count <- rbind(Msk_sent, Resp_sent, Fall_sent, Nutrition_sent)

#number of sentences by aspect (pos & neg)
ggplot(asp_sent_count, aes(x = Class, y = Count, group = Aspect)) +
  geom_col(aes(fill = Aspect)) +
  theme_bw() +
  geom_label_repel(aes(label=Count), 
                   direction = 'y', 
                   nudge_y = 0.5, 
                   segment.size=0) +
  labs(title = 'Number of sentences by aspect') +
  theme(legend.position = 'bottom')

#Number of sentences by aspect (total)
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

#Percent neutral sentences
1 - rowSums(AL04[, 4:ncol(AL04)])/AL04$n_sent

#patients by batch
ggplot(sent_count_train, aes(x = batch, y = n_patients)) +
  geom_col() +
  theme_bw() +
  geom_label_repel(aes(label=n_patients), direction = 'y', nudge_y = -0.05, segment.size=0)


# Compare train & test sentence counts
fc <- rbind(sent_count_train[sent_count_train$batch == 'AL05'], sent_count_test)
final_counts <- transpose(fc)
final_counts$label <- colnames(fc)
colnames(final_counts) <- c('Train', 'Test', '')
final_counts <- final_counts[-1, ]
final_counts <- final_counts[, c('', 'Train', 'Test')]
train_tot <- as.numeric(final_counts[1,2])
test_tot <- as.numeric(final_counts[1,3])
final_counts$Train <- paste0(final_counts$Train,
                             ' (', 
                             round(as.numeric(final_counts$Train)/train_tot*100, 
                                   1),
                             '%)')
final_counts$Test <- paste0(final_counts$Test,
                             ' (', 
                             round(as.numeric(final_counts$Test)/test_tot*100, 
                                   1),
                             '%)')
fwrite(final_counts,
       paste0(rootdir,
              'figures_tables/sentence_counts_all.csv'))






################################ COMPUTE TIME ################################   
#load data
nn_single_performance <- fread(paste0(rootdir,
                                      'saved_models/',
                                      tail(batches, 1),
                                      '/learning_curve_stask.csv'))
nn_single_performance[, 'model'] <- 'nn_single'
nn_single_performance$test <- nn_single_performance$model
nn_single_performance$elapsed <- nn_single_performance$runtime
nn_single_all_time <- nn_single_performance[batch == 'AL05']
nn_multi_performance <- fread(paste0(rootdir,
                                     'saved_models/',
                                     tail(batches, 1),
                                     '/learning_curve_mtask.csv'))
nn_multi_performance[, 'model'] <- 'nn_multi'
nn_multi_performance$test <- nn_multi_performance$model
nn_multi_performance$elapsed <- nn_multi_performance$runtime
nn_multi_all_time <- nn_multi_performance[batch == 'AL05']
rf_time <- fread(paste0(
  rootdir, 
  'saved_models/AL05/lin_trees/expAL05_rf_cpu_time.csv'))
enet_time <- fread(paste0(
  rootdir,
  'saved_models/AL05/lin_trees_enet/expAL05_enet_cpu_time.csv'))

#function to summarize
time_sum <- function(time) {
  #divide by 25 lambdas per row for glmnet
  if (time[['test']][1] == 'glmnet'){
    time[['elapsed']] <- time[['elapsed']]/25
  }
  d <- data.frame(model = time[['test']][1],
             mean = round(mean(time[['elapsed']]), 1),
             sd = round(sd(time[['elapsed']]), 1),
             median = round(median(time[['elapsed']]), 1),
             iqr25 = round(quantile(time[['elapsed']], probs = 0.25), 1),
             iqr75 = round(quantile(time[['elapsed']], probs = 0.75), 1),
             row.names = NULL)
  d['Seconds per model, mean (SD)'] <- paste0(as.character(d$mean),
                                                      ' (',
                                                      as.character(d$sd),
                                                      ')')
  d['Seconds per model, median (IQR)'] <- paste0(as.character(d$median),
                                                         ' (',
                                                         as.character(d$iqr25),
                                                         ' - ',
                                                         as.character(d$iqr75),
                                                         ')')
  if (d[['model']] == 'rf'){
    d['Processor'] <- '64 CPU'
    d['Total models (n)'] <- nrow(time)
    d['Total compute time (hrs)'] <- round(sum(time[['elapsed']])/60/60, 1)
  }
  if (d[['model']] == 'glmnet'){
    d['Processor'] <- '1 CPU'
    d['Total models (n)'] <- nrow(time)*25
    d['Total compute time (hrs)'] <- round(sum(time[['elapsed']])*25/60/60, 1)
    #note: total compute time for glmnet is for 1 CPU. We ran 96 CPUs in parallel.
  }
  if (d[['model']] == 'nn_single' | d[['model']] == 'nn_multi'){
    d['Processor'] <- '1 GPU'
    d['Total models (n)'] <- nrow(time)
    d['Total compute time (hrs)'] <- round(sum(time[['elapsed']])/60/60, 1)
  }
  return(d[c(1, seq(7, ncol(d)-1, 1))])
}
#summarize
rf_time <- time_sum(rf_time)
enet_time <- time_sum(enet_time)
nn_single_time <- time_sum(nn_single_all_time)
nn_multi_time <- time_sum(nn_multi_all_time)
times_mean_median <- rbind(rf_time, enet_time, nn_single_time, nn_multi_time)
#save
fwrite(times_mean_median,
       paste0(rootdir,
              'figures_tables/compute_time.csv'))








################################# CALIBRATION #################################
#load obs
#gather obs, preds, and sentences
op_func <- function(frail, model) {
  obs <- fread(paste0(
    rootdir,
    'saved_models/AL05/processed_data/test_set/full_df.csv'))
  # Rename cols to classes
  cols <- paste0(frail, cls)
  for (c in seq_along(cols)){
    colnames(obs)[grep(paste0(cols[c]), colnames(obs))] <- classes[c]
  }
  obs <- obs[, c('sentence_id', 'sentence', classes),
             with = FALSE]
  #get the preds for the best model
  preds <- fread(paste0(
    rootdir,
    'saved_models/AL05/lin_trees_final_test/',
    model,
    '_preds/AL05_',
    frail,
    '_preds.csv'))
  for (c in seq_along(cols)){
    colnames(preds)[grep(paste0(cols[c]), colnames(preds))] <- classes[c]
  }
  if (identical(obs$sentence_id, preds$sentence_id) == FALSE)
    stop ("obs do not match preds")
  #combine
  obs_preds <- cbind(obs = obs,
                     pred = preds[, c('Positive', 'Negative', 'Neutral'),
                                  with = FALSE])
  return(obs_preds)
}

# Respiratory
op <- op_func('Resp_imp', 'enet')
mc_calib_plot(obs.Positive + obs.Negative + obs.Neutral ~ pred.Positive +
                pred.Negative + pred.Neutral, data = op, cuts = 5) +
  labs(title = 'Respiratory') +
  theme(legend.position = 'none') +
  scale_color_nejm(labels = c('Negative', 'Neutral', 'Positive'))
# Msk
op <- op_func('Msk_prob', 'enet')
mc_calib_plot(obs.Positive + obs.Negative + obs.Neutral ~ pred.Positive +
                pred.Negative + pred.Neutral, data = op, cuts = 5) +
  labs(title = 'Musculoskeletal') +
  theme(legend.position = 'none') +
  scale_color_nejm()
# Fall risk
op <- op_func('Fall_risk', 'enet')
mc_calib_plot(obs.Positive + obs.Negative + obs.Neutral ~ pred.Positive +
                pred.Negative + pred.Neutral, data = op, cuts = 5) +
  labs(title = 'Fall risk') +
  theme(legend.position = 'none') +
  scale_color_nejm()
# Nutrition
op <- op_func('Nutrition', 'enet')
mc_calib_plot(obs.Positive + obs.Negative + obs.Neutral ~ pred.Positive +
                pred.Negative + pred.Neutral, data = op, cuts = 5) +
  labs(title = 'Nutrition') +
  theme(legend.position = 'none') +
  scale_color_nejm()





################################# PR CURVES #################################

#PR curve function. Uses output from op_func
pr_curv <- function(proc_op){
  PR_pos <- pr.curve(scores.class0 = proc_op$pred.Positive,
                     weights.class0 = proc_op$obs.Positive, curve = TRUE)
  PR_pos <-data.frame(PR_pos$curve)
  PR_pos$Class <- "Positive"
  PR_neg <- pr.curve(scores.class0 = proc_op$pred.Negative,
                     weights.class0 = proc_op$obs.Negative, curve = TRUE)
  PR_neg <-data.frame(PR_neg$curve)
  PR_neg$Class <- "Negative"
  PR_neut <- pr.curve(scores.class0 = proc_op$pred.Neutral,
                      weights.class0 = proc_op$obs.Neutral, curve = TRUE)
  PR_neut <-data.frame(PR_neut$curve)
  PR_neut$Class <- "Neutral"
  PR_resp <- rbind(PR_pos, PR_neg, PR_neut)
  return(PR_resp)
}

# Respiratory
op <- op_func('Resp_imp', 'enet')
pr_plt <- pr_curv(op)
ggplot(pr_plt, aes(x = X1, y = X2, group = Class)) +
  geom_line(aes(color = Class)) +
  labs(title = 'Respiratory Impairment',
       y = 'Precision',
       x = 'Recall') +
  theme_bw() +
  theme(legend.position = 'none') +
  scale_color_nejm()
# Msk
op <- op_func('Msk_prob', 'enet')
pr_plt <- pr_curv(op)
ggplot(pr_plt, aes(x = X1, y = X2, group = Class)) +
  geom_line(aes(color = Class)) +
  labs(title = 'Musculoskeletal',
       y = 'Precision',
       x = 'Recall') +
  theme_bw() +
  theme(legend.position = 'none') +
  scale_color_nejm()
# Fall risk
op <- op_func('Fall_risk', 'enet')
pr_plt <- pr_curv(op)
ggplot(pr_plt, aes(x = X1, y = X2, group = Class)) +
  geom_line(aes(color = Class)) +
  labs(title = 'Fall Risk',
       y = 'Precision',
       x = 'Recall') +
  theme_bw() +
  theme(legend.position = 'none') +
  scale_color_nejm()
# Nutrition
op <- op_func('Nutrition', 'enet')
pr_plt <- pr_curv(op)
ggplot(pr_plt, aes(x = X1, y = X2, group = Class)) +
  geom_line(aes(color = Class)) +
  labs(title = 'Nutrition',
       y = 'Precision',
       x = 'Recall') +
  theme_bw() +
  theme(legend.position = 'none') +
  scale_color_nejm()







############################# ALGORITHMIC EQUITY #############################

#load all patient data
pat_data <- fread(paste0(
  rootdir,
  'figures_tables/patient_struc_data.csv'))
rc <- c('White', 'Non_white')
gd <- c('Male', 'Female')
White <- pat_data[pat_data$Race_white == 1, ]$PAT_ID
Non_white <- pat_data[!pat_data$Race_white == 1, ]$PAT_ID
Male <- pat_data[pat_data$SEX_Male == 1, ]$PAT_ID
Female <- pat_data[pat_data$SEX_Female == 1, ]$PAT_ID

#gather obs and preds by group
eq_perf <- function(frail, model, group) {
  obs <- fread(paste0(
    rootdir,
    'saved_models/AL05/processed_data/test_set/full_df.csv'))
  # Rename cols to classes
  cols <- paste0(frail, cls)
  for (c in seq_along(cols)){
    colnames(obs)[grep(paste0(cols[c]), colnames(obs))] <- classes[c]
  }
  obs <- obs[, c('PAT_ID', 'sentence_id', classes),
             with = FALSE]
  #get obs by group
  if (group == 'Male'){
    obs_race <- obs[obs$PAT_ID %in% Male, ]
  } else if (group == 'Female') {
    obs_race <- obs[obs$PAT_ID %in% Female, ]
  } else if (group == 'White') {
    obs_race <- obs[obs$PAT_ID %in% White, ]
  } else if (group == 'Non_white') {
    obs_race <- obs[obs$PAT_ID %in% Non_white, ]
  }
  #get the preds for the best model
  preds <- fread(paste0(
    rootdir,
    'saved_models/AL05/lin_trees_final_test/',
    model,
    '_preds/AL05_',
    frail,
    '_preds.csv'))
  for (c in seq_along(cols)){
    colnames(preds)[grep(paste0(cols[c]), colnames(preds))] <- classes[c]
  }
  if (identical(obs$sentence_id, preds$sentence_id) == FALSE)
    stop ("obs do not match preds")
  #get preds by group
  preds_race <- preds[preds$sentence_id %in% obs_race$sentence_id, ]
  op <- cbind(pred = preds_race, obs = obs_race)
  return(op)
}

# calculate brier by race
rc_brier <- function(frail, model) {
  rp <- list()
  for (r in seq_along(rc)){
    hyper_grid <- data.frame(frail_lab = frail)
    hyper_grid$model <- model
    hyper_grid$Patient_race <- rc[r]
    op <- eq_perf(frail, model, rc[r])
    obs <- op[, c('obs.Neutral', 'obs.Positive', 'obs.Negative')]
    preds <- op[, c('pred.Neutral', 'pred.Positive', 'pred.Negative')]
    hyper_grid[['sbrier_multi']] <- multi_scaled_Brier(preds, obs)
    hyper_grid[['sbrier_pos']] <-
      scaled_Brier(op[['pred.Positive']], op[['obs.Positive']], 1)
    hyper_grid[['sbrier_neg']] <-
      scaled_Brier(op[['pred.Negative']], op[['obs.Negative']], 1)
    hyper_grid[['sbrier_neut']] <-
      scaled_Brier(op[['pred.Neutral']], op[['obs.Neutral']], 1)
    rp[[r]] <- hyper_grid
  }
  perf_eq <- rbindlist(rp)
  return(perf_eq)
}

# calculate brier by gender
gen_brier <- function(frail, model) {
  rp <- list()
  for (r in seq_along(gd)){
    hyper_grid <- data.frame(frail_lab = frail)
    hyper_grid$model <- model
    hyper_grid$Gender <- gd[r]
    op <- eq_perf(frail, model, gd[r])
    obs <- op[, c('obs.Neutral', 'obs.Positive', 'obs.Negative')]
    preds <- op[, c('pred.Neutral', 'pred.Positive', 'pred.Negative')]
    hyper_grid[['sbrier_multi']] <- multi_scaled_Brier(preds, obs)
    hyper_grid[['sbrier_pos']] <-
      scaled_Brier(op[['pred.Positive']], op[['obs.Positive']], 1)
    hyper_grid[['sbrier_neg']] <-
      scaled_Brier(op[['pred.Negative']], op[['obs.Negative']], 1)
    hyper_grid[['sbrier_neut']] <-
      scaled_Brier(op[['pred.Neutral']], op[['obs.Neutral']], 1)
    rp[[r]] <- hyper_grid
  }
  perf_eq <- rbindlist(rp)
  return(perf_eq)
}

#Race calculate and combine
race_sbrier <- rbind(rc_brier('Resp_imp', 'enet'),
                     rc_brier('Msk_prob', 'enet'),
                     rc_brier('Nutrition', 'enet'),
                     rc_brier('Fall_risk', 'enet'))
cols <- grep('sbrier', colnames(race_sbrier), value = TRUE)
race_sbrier[, (cols) := round(.SD, 2), .SDcols = cols]
colnames(race_sbrier) <- c('Frailty aspect', 'Model', 'Patient race',
                            'Multi-class SBS', 'Positive SBS',
                            'Negative SBS', 'Neutral SBS')
#save
fwrite(race_sbrier[, -'Model'],
       paste0(rootdir,
              'figures_tables/race_sbrier.csv'))

#Gender calculate and combine
gen_sbrier <- rbind(gen_brier('Resp_imp', 'enet'),
                    gen_brier('Msk_prob', 'enet'),
                    gen_brier('Nutrition', 'enet'),
                    gen_brier('Fall_risk', 'enet'))
cols <- grep('sbrier', colnames(gen_sbrier), value = TRUE)
gen_sbrier[, (cols) := round(.SD, 2), .SDcols = cols]
colnames(gen_sbrier) <- c('Frailty aspect', 'Model', 'Gender',
                           'Multi-class SBS', 'Positive SBS',
                           'Negative SBS', 'Neutral SBS')
#save
fwrite(gen_sbrier[, -'Model'],
       paste0(rootdir,
              'figures_tables/gender_sbrier.csv'))

#Race and Gender calculate and combine
race_sbrier$Group <- race_sbrier$`Patient race`
race_sbrier[Group == 'Non_white', 'N (%)'] <- 
  paste0(
    as.character(length(Non_white)),
    ' (',
    as.character(round(length(Non_white)/nrow(pat_data), 2)*100),
    '%)')
race_sbrier[Group == 'White', 'N (%)'] <-
  paste0(
    as.character(length(White)),
    ' (',
    as.character(round(length(White)/nrow(pat_data), 2)*100),
    '%)')
gen_sbrier$Group <- gen_sbrier$Gender
gen_sbrier[Group == 'Female', 'N (%)'] <-
  paste0(
    as.character(length(Female)),
    ' (',
    as.character(round(length(Female)/nrow(pat_data), 2)*100),
    '%)')
gen_sbrier[Group == 'Male', 'N (%)'] <-
  paste0(
    as.character(length(Male)),
    ' (',
    as.character(round(length(Male)/nrow(pat_data), 2)*100),
    '%)')
cols <- c('Frailty aspect', 'Model', 'Group',
                          'Multi-class SBS', 'Positive SBS',
                          'Negative SBS', 'Neutral SBS')
group_sbrier <- arrange(rbind(race_sbrier[, ..cols],
                    gen_sbrier[, ..cols]), `Frailty aspect`)
#save
fwrite(group_sbrier[, -'Model'],
       paste0(rootdir,
              'figures_tables/group_sbrier.csv'))


# Race Positive predictive value by threshold
race_ppv_thresh <- function(frail, model) {
  #calculate threshold and PPV
  rp <- list()
  for (r in seq_along(rc)){
    op <- eq_perf(frail, model, rc[r])
    op <- arrange(op, desc(pred.Positive))
    ppv_posclass <- data.frame(Threshold = op$pred.Positive,
                               test_pos = 1:nrow(op),
                               true_pos = cumsum(op$obs.Positive))
    ppv_posclass$PPV <- signif(ppv_posclass$true_pos/ppv_posclass$test_pos, 3)
    ppv_posclass$Prediction <- 'Positive class'
    
    op <- arrange(op, desc(pred.Negative))
    ppv_negclass <- data.frame(Threshold = op$pred.Negative,
                               test_pos = 1:nrow(op),
                               true_pos = cumsum(op$obs.Negative))
    ppv_negclass$PPV <- signif(ppv_negclass$true_pos/ppv_negclass$test_pos, 3)
    ppv_negclass$Prediction <- 'Positive class'
    ppv_negclass$Prediction <- 'Negative class'
    cols <- c('Threshold', 'PPV', 'Prediction')
    ppv <- rbind(ppv_posclass[, cols], ppv_negclass[, cols])
    ppv$Race <- rc[r]
    rp[[r]] <- ppv
  }
  ppv <- rbindlist(rp)
  return(ppv)
}

# STOPPED HERE - NEED TO REPEAT FOR EACH ASPECT
# THEN DO THE SAME BELOW FOR GENDER.
# THEN PUT IN POWERPOINT
# THEN CHANGE THE COLORS FOR THE OTHER PLOTS
# UPDATE FIGURE LEGENDS AND BLURBS WITH COLORS

#Respiratory
ppv <- race_ppv_thresh('Resp_imp', 'enet')
ggplot(ppv, aes(x = Threshold, y = PPV, color = Race, linetype = Prediction)) +
  geom_line(aes(color = Race, linetype = Prediction)) +
  labs(title = 'Respiratory',
       y = 'PPV',
       x = 'Threshold') +
  theme_bw() +
  theme(legend.position = 'none') +
  scale_color_nejm()

#Musculoskeletal
ppv <- race_ppv_thresh('Msk_prob', 'enet')
ggplot(ppv, aes(x = Threshold, y = PPV, color = Race, linetype = Prediction)) +
  geom_line(aes(color = Race, linetype = Prediction)) +
  labs(title = 'Musculoskeletal',
       y = 'PPV',
       x = 'Threshold') +
  theme_bw() +
  theme(legend.position = 'none') +
  scale_color_nejm()

#Fall risk
ppv <- race_ppv_thresh('Msk_prob', 'enet')
ggplot(ppv, aes(x = Threshold, y = PPV, color = Race, linetype = Prediction)) +
  geom_line(aes(color = Race, linetype = Prediction)) +
  labs(title = 'Fall risk',
       y = 'PPV',
       x = 'Threshold') +
  theme_bw() +
  theme(legend.position = 'none') +
  scale_color_nejm()

#Nutrition
ppv <- race_ppv_thresh('Msk_prob', 'enet')
ggplot(ppv, aes(x = Threshold, y = PPV, color = Race, linetype = Prediction)) +
  geom_line(aes(color = Race, linetype = Prediction)) +
  labs(title = 'Nutrition',
       y = 'PPV',
       x = 'Threshold') +
  theme_bw() +
  theme(legend.position = 'none') +
  scale_color_nejm()

# Gender Positive predictive value by threshold
gender_ppv_thresh <- function(frail, model) {
  #calculate threshold and PPV
  rp <- list()
  for (r in seq_along(gd)){
    op <- eq_perf(frail, model, gd[r])
    op <- arrange(op, desc(pred.Positive))
    ppv_posclass <- data.frame(Threshold = op$pred.Positive,
                               test_pos = 1:nrow(op),
                               true_pos = cumsum(op$obs.Positive))
    ppv_posclass$PPV <- signif(ppv_posclass$true_pos/ppv_posclass$test_pos, 3)
    ppv_posclass$Prediction <- 'Positive class'
    
    op <- arrange(op, desc(pred.Negative))
    ppv_negclass <- data.frame(Threshold = op$pred.Negative,
                               test_pos = 1:nrow(op),
                               true_pos = cumsum(op$obs.Negative))
    ppv_negclass$PPV <- signif(ppv_negclass$true_pos/ppv_negclass$test_pos, 3)
    ppv_negclass$Prediction <- 'Positive class'
    ppv_negclass$Prediction <- 'Negative class'
    cols <- c('Threshold', 'PPV', 'Prediction')
    ppv <- rbind(ppv_posclass[, cols], ppv_negclass[, cols])
    ppv$Gender <- gd[r]
    rp[[r]] <- ppv
  }
  ppv <- rbindlist(rp)
  return(ppv)
}

#Respiratory
ppv <- gender_ppv_thresh('Resp_imp', 'enet')
ggplot(ppv, aes(x = Threshold, y = PPV, color = Gender, linetype = Prediction)) +
  geom_line(aes(color = Gender, linetype = Prediction)) +
  labs(title = 'Respiratory',
       y = 'PPV',
       x = 'Threshold') +
  theme_bw() +
  theme(legend.position = 'none') +
  scale_color_nejm()

#Musculoskeletal
ppv <- gender_ppv_thresh('Msk_prob', 'enet')
ggplot(ppv, aes(x = Threshold, y = PPV, color = Gender, linetype = Prediction)) +
  geom_line(aes(color = Gender, linetype = Prediction)) +
  labs(title = 'Musculoskeletal',
       y = 'PPV',
       x = 'Threshold') +
  theme_bw() +
  theme(legend.position = 'none') +
  scale_color_nejm()

#Fall risk
ppv <- gender_ppv_thresh('Msk_prob', 'enet')
ggplot(ppv, aes(x = Threshold, y = PPV, color = Gender, linetype = Prediction)) +
  geom_line(aes(color = Gender, linetype = Prediction)) +
  labs(title = 'Fall risk',
       y = 'PPV',
       x = 'Threshold') +
  theme_bw() +
  theme(legend.position = 'none') +
  scale_color_nejm()

#Nutrition
ppv <- gender_ppv_thresh('Msk_prob', 'enet')
ggplot(ppv, aes(x = Threshold, y = PPV, color = Gender, linetype = Prediction)) +
  geom_line(aes(color = Gender, linetype = Prediction)) +
  labs(title = 'Nutrition',
       y = 'PPV',
       x = 'Threshold') +
  theme_bw() +
  theme(legend.position = 'none') +
  scale_color_nejm()

# plot ppv by threshold
ppv1 <- race_ppv_thresh('Resp_imp', 'enet')
ppv1$Group <- ppv1$Race
ppv2 <- gender_ppv_thresh('Resp_imp', 'enet')
ppv2$Group <- ppv2$Gender
ppv <- rbind(ppv1[, c('Threshold', 'PPV', 'Prediction', 'Group')],
      ppv2[, c('Threshold', 'PPV', 'Prediction', 'Group')])
ggplot(ppv, aes(x = Threshold, y = PPV, color = Group, linetype = Prediction)) +
  geom_line(aes(color = Group, linetype = Prediction)) +
  labs(title = 'Respiratory',
       y = 'PPV',
       x = 'Threshold') +
  theme_bw() +
  theme(legend.position = 'none')
#scale_color_manual(values = wes_palette(n = 4, name = "Cavalcanti1", type = 'discrete'))

ppv <- race_ppv_thresh('Msk_prob', 'enet')
ppv1$Group <- ppv1$Race
ppv2 <- gender_ppv_thresh('Msk_prob', 'enet')
ppv2$Group <- ppv2$Gender
ppv <- rbind(ppv1[, c('Threshold', 'PPV', 'Prediction', 'Group')],
             ppv2[, c('Threshold', 'PPV', 'Prediction', 'Group')])
ggplot(ppv, aes(x = Threshold, y = PPV, color = Group, linetype = Prediction)) +
  geom_line(aes(color = Group, linetype = Prediction)) +
  labs(title = 'Musculoskeletal',
       y = 'PPV',
       x = 'Threshold') +
  theme_bw() +
  theme(legend.position = 'none')

ppv <- race_ppv_thresh('Fall_risk', 'enet')
ppv1$Group <- ppv1$Race
ppv2 <- gender_ppv_thresh('Fall_risk', 'enet')
ppv2$Group <- ppv2$Gender
ppv <- rbind(ppv1[, c('Threshold', 'PPV', 'Prediction', 'Group')],
             ppv2[, c('Threshold', 'PPV', 'Prediction', 'Group')])
ggplot(ppv, aes(x = Threshold, y = PPV, color = Group, linetype = Prediction)) +
  geom_line(aes(color = Group, linetype = Prediction)) +
  labs(title = 'Fall risk',
       y = 'PPV',
       x = 'Threshold') +
  theme_bw() +
  theme(legend.position = 'none')

ppv <- race_ppv_thresh('Nutrition', 'enet')
ppv1$Group <- ppv1$Race
ppv2 <- gender_ppv_thresh('Nutrition', 'enet')
ppv2$Group <- ppv2$Gender
ppv <- rbind(ppv1[, c('Threshold', 'PPV', 'Prediction', 'Group')],
             ppv2[, c('Threshold', 'PPV', 'Prediction', 'Group')])
ggplot(ppv, aes(x = Threshold, y = PPV, color = Group, linetype = Prediction)) +
  geom_line(aes(color = Group, linetype = Prediction)) +
  labs(title = 'Nutrition',
       y = 'PPV',
       x = 'Threshold') +
  theme_bw() +
  theme(legend.position = 'right')








############################### ERROR ANALYSIS ###############################

#general ppv function (from above but not specific to race/gender)
ppv_thresh <- function(frail, model, thresh){
  op <- op_func(frail, model)
  op <- arrange(op, desc(pred.Positive))
  ppv_posclass <- data.frame(Threshold = op$pred.Positive,
                             test_pos = 1:nrow(op),
                             true_pos = cumsum(op$obs.Positive))
  ppv_posclass$PPV <- signif(ppv_posclass$true_pos/ppv_posclass$test_pos, 3)
  ppv_posclass$Prediction <- 'Positive class'
  
  op <- arrange(op, desc(pred.Negative))
  ppv_negclass <- data.frame(Threshold = op$pred.Negative,
                             test_pos = 1:nrow(op),
                             true_pos = cumsum(op$obs.Negative))
  ppv_negclass$PPV <- signif(ppv_negclass$true_pos/ppv_negclass$test_pos, 3)
  ppv_negclass$Prediction <- 'Positive class'
  ppv_negclass$Prediction <- 'Negative class'
  cols <- c('Threshold', 'PPV', 'Prediction')
  ppv <- rbind(ppv_posclass[, cols], ppv_negclass[, cols])
  ppv_thresh <- ppv %>%
    filter(PPV > thresh) %>%
    arrange(Threshold) %>%
    group_by(Prediction) %>%
    slice(1)
  pos_thresh <- ppv_thresh[ppv_thresh$Prediction == 'Positive class', ]$Threshold
  neg_thresh <- ppv_thresh[ppv_thresh$Prediction == 'Negative class', ]$Threshold
  op$pred_pos_bin <- ifelse(op$pred.Positive > pos_thresh, 1, 0)
  op$pred_neg_bin <- ifelse(op$pred.Negative > neg_thresh, 1, 0)
  return(op)
}

#get binary predictions for a given sensitivity threshold
sensitivity_thresh <- function(frail, model, thresh){
  op <- op_func(frail, model)
  op <- arrange(op, desc(pred.Positive))
  sens_posclass <- data.frame(Threshold = op$pred.Positive,
                             true_pos = cumsum(op$obs.Positive),
                             total_pos = sum(op$obs.Positive))
  sens_posclass$Sensitivity <- signif(sens_posclass$true_pos/sens_posclass$total_pos, 3)
  sens_posclass$Prediction <- 'Positive class'
  
  op <- arrange(op, desc(pred.Negative))
  sens_negclass <- data.frame(Threshold = op$pred.Negative,
                             true_pos = cumsum(op$obs.Negative),
                             total_pos = sum(op$obs.Negative))
  sens_negclass$Sensitivity <- signif(sens_negclass$true_pos/sens_negclass$total_pos, 3)
  sens_negclass$Prediction <- 'Negative class'
  cols <- c('Threshold', 'Sensitivity', 'Prediction')
  sens <- rbind(sens_posclass[, cols], sens_negclass[, cols])
  sens_thresh <- sens %>%
    filter(Sensitivity > thresh) %>%
    arrange(desc(Threshold)) %>%
    group_by(Prediction) %>%
    slice(1)
  pos_thresh <- sens_thresh[sens_thresh$Prediction == 'Positive class', ]$Threshold
  neg_thresh <- sens_thresh[sens_thresh$Prediction == 'Negative class', ]$Threshold
  op$pred_pos_bin <- ifelse(op$pred.Positive > pos_thresh, 1, 0)
  op$pred_neg_bin <- ifelse(op$pred.Negative > neg_thresh, 1, 0)
  return(op)
}


#Respiratory
op <- ppv_thresh('Resp_imp', 'enet', 0.75)
#positive class confusion matrix
confusionMatrix(factor(op$pred_pos_bin), factor(op$obs.Positive), positive = '1')
#negative class confusion matrix
confusionMatrix(factor(op$pred_neg_bin), factor(op$obs.Negative), positive = '1')
#Positive class false positive sentences
op[(op$pred_pos_bin == 1 & op$obs.Positive == 0), ]$obs.sentence
#Positive class false negative sentences
op[(op$pred_pos_bin == 0 & op$obs.Positive == 1), ]$obs.sentence
#Negative class false positive sentences
op[(op$pred_neg_bin == 1 & op$obs.Negative == 0), ]$obs.sentence
#Negative class false negative sentences
op[(op$pred_neg_bin == 0 & op$obs.Negative == 1), ]$obs.sentence
#Sensitivity
sens <- sensitivity_thresh('Resp_imp', 'enet', 0.80)
#Positive class sensitivity
confusionMatrix(factor(sens$pred_pos_bin), factor(sens$obs.Positive), positive = '1')
#Negative class sensitivity
confusionMatrix(factor(sens$pred_neg_bin), factor(sens$obs.Negative), positive = '1')


#Musculoskeletal
op <- ppv_thresh('Msk_prob', 'enet', 0.75)
#confusion matrix
confusionMatrix(factor(op$pred_pos_bin), factor(op$obs.Positive), positive = '1')
#negative class confusion matrix
confusionMatrix(factor(op$pred_neg_bin), factor(op$obs.Negative), positive = '1')
#Positive class false positive sentences
op[(op$pred_pos_bin == 1 & op$obs.Positive == 0), ]$obs.sentence
#Positive class false negative sentences
op[(op$pred_pos_bin == 0 & op$obs.Positive == 1), ]$obs.sentence
#Negative class false positive sentences
op[(op$pred_neg_bin == 1 & op$obs.Negative == 0), ]$obs.sentence
#Negative class false negative sentences
op[(op$pred_neg_bin == 0 & op$obs.Negative == 1), ]$obs.sentence
#Sensitivity
sens <- sensitivity_thresh('Msk_prob', 'enet', 0.80)
#Positive class sensitivity
confusionMatrix(factor(sens$pred_pos_bin), factor(sens$obs.Positive), positive = '1')
#Negative class sensitivity
confusionMatrix(factor(sens$pred_neg_bin), factor(sens$obs.Negative), positive = '1')


#Fall risk
op <- ppv_thresh('Fall_risk', 'enet', 0.75)
#confusion matrix
confusionMatrix(factor(op$pred_pos_bin), factor(op$obs.Positive), positive = '1')
#Positive class false positive sentences
op[(op$pred_pos_bin == 1 & op$obs.Positive == 0), ]$obs.sentence
#Positive class false negative sentences
op[(op$pred_pos_bin == 0 & op$obs.Positive == 1), ]$obs.sentence
#Negative class false positive sentences
op[(op$pred_neg_bin == 1 & op$obs.Negative == 0), ]$obs.sentence
#Negative class false negative sentences
op[(op$pred_neg_bin == 0 & op$obs.Negative == 1), ]$obs.sentence
#Sensitivity
sens <- sensitivity_thresh('Fall_risk', 'enet', 0.80)
#Positive class sensitivity
confusionMatrix(factor(sens$pred_pos_bin), factor(sens$obs.Positive), positive = '1')
#Negative class sensitivity
confusionMatrix(factor(sens$pred_neg_bin), factor(sens$obs.Negative), positive = '1')

#Nutrition
op <- ppv_thresh('Nutrition', 'enet', 0.33)
#confusion matrix
confusionMatrix(factor(op$pred_pos_bin), factor(op$obs.Positive), positive = '1')
#Positive class false positive sentences
op[(op$pred_pos_bin == 1 & op$obs.Positive == 0), ]$obs.sentence
#Positive class false negative sentences
op[(op$pred_pos_bin == 0 & op$obs.Positive == 1), ]$obs.sentence
#Negative class false positive sentences
op[(op$pred_neg_bin == 1 & op$obs.Negative == 0), ]$obs.sentence
#Negative class false negative sentences
op[(op$pred_neg_bin == 0 & op$obs.Negative == 1), ]$obs.sentence
#Sensitivity
sens <- sensitivity_thresh('Nutrition', 'enet', 0.80)
#Positive class sensitivity
confusionMatrix(factor(sens$pred_pos_bin), factor(sens$obs.Positive), positive = '1')
#Negative class sensitivity
confusionMatrix(factor(sens$pred_neg_bin), factor(sens$obs.Negative), positive = '1')




############################# TF-IDF vs EMBEDDINGS #############################
# # the code below analyzes performance of TF-IDF vs embeddings in cross 
# # validation on the training set. We only ran the best models on the test set,
# # so we never ran any of the TF-IDF models becasue they were always worse
# # than embeddings
# enet_test_perf <- fread(paste0(
#   rootdir, 'figures_tables/enet_test_set_performance.csv'))
# summary(lm(sbrier_multi ~ SVD, 
#            data = enet_performance[SVD %in% c('1000', 'embed') &
#                                      batch == 'AL05', ]))
# summary(lm(sbrier_multi ~ SVD, 
#            data = rf_performance[SVD %in% c('1000', 'embed') &
#                                    batch == 'AL05', ]))
# #TF-IDF vs embeddings by frailty aspect by class for the best 5 models
# top_50 <- function(raw_perf){
#   raw_perf <- as.data.frame(raw_perf)
#   if (raw_perf$model[1] == 'enet') {
#     hyperparams = hyperparams_enet
#   } else if (raw_perf$model[1] == 'rf') {
#     hyperparams = hyperparams_rf
#   } else if (raw_perf$model[1] == 'nn_single') {
#     hyperparams = hyperparams_nn}
#   group1 <- c('frail_lab', hyperparams)
#   step_1 <- raw_perf %>%
#     filter(batch == 'AL04') %>%
#     group_by_at(vars(all_of(group1))) %>%
#     summarise_at(vars(grep('sbrier', colnames(raw_perf), value = TRUE)),
#                  list(mean = mean, se = se)) %>%
#     na.omit() %>%
#     ungroup() %>%
#     mutate(tfidf_embed = ifelse(SVD %in% c('300', '1000'),
#                                 'TF-IDF',
#                                 'Embeddings')) %>%
#     group_by(frail_lab, tfidf_embed) %>%
#     arrange(desc(sbrier_multi_mean), .by_group = TRUE) %>%
#     slice(1:50)
#   return(step_1)
# }
# enet_top50 <- top_50(enet_performance)
# ggplot(enet_top50, aes(x = frail_lab, y = sbrier_multi_mean, ymin = (sbrier_multi_mean - sbrier_multi_se), ymax = (sbrier_multi_mean + sbrier_multi_se)))+
#   geom_pointrange(stat='identity', position = 'jitter', aes(color = tfidf_embed))+
#   scale_color_discrete(name = 'Frailty aspect') +
#   labs(title = 'Text features (top 50 linear models)', x = 'Text features', y = 'Scaled Brier score') +
#   theme_bw() +
#   theme(legend.position = 'bottom')
# rf_top50 <- top_50(rf_performance)
# ggplot(rf_top50, aes(x = frail_lab, y = sbrier_multi_mean, ymin = (sbrier_multi_mean - sbrier_multi_se), ymax = (sbrier_multi_mean + sbrier_multi_se)))+
#   geom_pointrange(stat='identity', position = 'jitter', aes(color = tfidf_embed))+
#   scale_color_discrete(name = 'Frailty aspect') +
#   labs(title = 'Text features (top 50 random forests)', x = 'Text features', y = 'Scaled Brier score') +
#   theme_bw() +
#   theme(legend.position = 'bottom')










