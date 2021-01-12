

#
# Outdated script to process output from _08_rf_elastic_net.R -- now the output 
# is much more concise and can be easily imported into a notebook
#

library(data.table)
library(dplyr)
library(tidyr)
library(gmish)
library(ggplot2)
library(openxlsx)

#set experiment numbers
nnet_exp <- 'exp111120_nnet_str'
rf_exp <- 'exp110320_rf_tfidfstr'
enet_exp <- 'exp110320_enetstr'

#set directories based on exp numbers
dirs = c('/Users/martijac/Documents/Frailty/frailty_classifier/output/', '/media/drv2/andrewcd2/frailty/output/', '/share/gwlab/frailty/')
for (d in 1:length(dirs)) {
  if (dir.exists(dirs[d])) {
    outdir = dirs[d]
  }
}
nnet_dir <- paste0(outdir, 'n_nets/', nnet_exp, '/')
rf_dir <- paste0(outdir, 'lin_trees/', rf_exp, '/')
enet_dir <- paste0(outdir, 'lin_trees/', enet_exp, '/')

#standard error
se <- function(x) {
  #remove NA (necessary if there is a major fold error (e.g. test fold is completely missing pos/neg class))
  x <- x[!is.na(x)]
  sd(x)/sqrt(length(x))
  }

#mean & se of loss across fold 
mean_se <- function(loss) {
  loss <- loss %>%
    mutate(mean_sbrier_neut = signif(rowMeans(loss[,grep('sbrier_neut', colnames(loss), value = TRUE)], na.rm = TRUE), 2),
           se_sbrier_neut = signif(apply(loss[,grep('sbrier_neut', colnames(loss), value = TRUE)], 1, se), 1),
           mean_sbrier_pos = signif(rowMeans(loss[,grep('sbrier_pos', colnames(loss), value = TRUE)], na.rm = TRUE), 2),
           se_sbrier_pos = signif(apply(loss[,grep('sbrier_pos', colnames(loss), value = TRUE)], 1, se), 1),
           mean_sbrier_neg = signif(rowMeans(loss[,grep('sbrier_neg', colnames(loss), value = TRUE)], na.rm = TRUE), 2),
           se_sbrier_neg = signif(apply(loss[,grep('sbrier_neg', colnames(loss), value = TRUE)], 1, se), 1),
           mean_sbrier_all = signif(rowMeans(loss[,grep('sbrier_mean', colnames(loss), value = TRUE)], na.rm = TRUE), 2),
           se_sbrier_all = signif(apply(loss[,grep('sbrier_mean', colnames(loss), value = TRUE)], 1, se), 1),
           mean_entropy = signif(rowMeans(loss[,grep('cross_entropy_2', colnames(loss), value = TRUE)], na.rm = TRUE), 2),
           se_entropy = signif(apply(loss[,grep('cross_entropy_2', colnames(loss), value = TRUE)], 1, se), 1),
           mean_bscore_neut = signif(rowMeans(loss[,grep('bscore_neut', colnames(loss), value = TRUE)], na.rm = TRUE), 2),
           se_bscore_neut = signif(apply(loss[,grep('bscore_neut', colnames(loss), value = TRUE)], 1, se), 1),
           mean_bscore_pos = signif(rowMeans(loss[,grep('bscore_pos', colnames(loss), value = TRUE)], na.rm = TRUE), 2),
           se_bscore_pos = signif(apply(loss[,grep('bscore_pos', colnames(loss), value = TRUE)], 1, se), 1),
           mean_bscore_neg = signif(rowMeans(loss[,grep('bscore_neg', colnames(loss), value = TRUE)], na.rm = TRUE), 2),
           se_bscore_neg = signif(apply(loss[,grep('bscore_neg', colnames(loss), value = TRUE)], 1, se), 1),
           mean_bscore_all = signif(rowMeans(loss[,grep('bscore_all', colnames(loss), value = TRUE)], na.rm = TRUE), 2),
           se_bscore_all = signif(apply(loss[,grep('bscore_all', colnames(loss), value = TRUE)], 1, se), 1)) %>%
    select(-grep('fold', colnames(loss)))
  return(loss)}

######## Summarize neural net output #########

#plot loss by batch size
cols <- 'batch_size'
epochs <- 1:10
cols <- append(cols, epochs)
batch_loss <- fread(paste0(nnet_dir, 'batch_loss_grace.csv'), skip = 1)
colnames(batch_loss) <- cols
batch_loss_long <- pivot_longer(batch_loss, cols = 2:11, names_to = 'epoch', values_to = 'cross_entropy')
ggplot(batch_loss_long, aes(x = reorder(as.integer(epoch), sort(as.integer(epoch))), y = cross_entropy, group = batch_size)) +
  geom_line(aes(color = as.factor(batch_size))) +
  labs(title = 'Loss by epoch stratified by batch size', y = 'Categorical cross entropy', x = 'Epoch', color = 'Batch size')

#training loss by model type
cols <- 'model'
epochs <- 1:10
cols <- append(cols, epochs)
train_loss <- fread(paste0(nnet_dir, 'deep_loss.csv'), skip = 1)
colnames(train_loss) <- cols
train_loss_long <- pivot_longer(train_loss, cols = 2:11, names_to = 'epoch', values_to = 'cross_entropy')
#put legend labels in order of loss in 10th epoch
train_loss_long$model <- factor(train_loss_long$model, levels = arrange(filter(train_loss_long, epoch == 10), cross_entropy)$model)
ggplot(train_loss_long, aes(x = reorder(as.integer(epoch), sort(as.integer(epoch))), y = cross_entropy, group = model)) +
  geom_line(aes(color = as.factor(model))) +
  labs(title = 'Training loss by epoch stratified by model', y = 'Categorical cross entropy', x = 'Epoch', color = 'Model', 
       caption = 'Model titles include the type of layer (bl = bidirectional LSTM, d = dense) and size of layer (256, 128, or 64 units)')

#validation loss by model type
val_loss <- fread(paste0(nnet_dir, 'deep_val_loss.csv'), skip = 1)
colnames(val_loss) <- cols
val_loss_long <- pivot_longer(val_loss, cols = 2:11, names_to = 'epoch', values_to = 'cross_entropy')
#put legend labels in order of loss in 10th epoch
val_loss_long$model <- factor(val_loss_long$model, levels = arrange(filter(train_loss_long, epoch == 10), cross_entropy)$model)
ggplot(val_loss_long, aes(x = reorder(as.integer(epoch), sort(as.integer(epoch))), y = cross_entropy, group = model)) +
  geom_line(aes(color = as.factor(model))) +
  labs(title = 'Validation loss by epoch stratified by model', y = 'Categorical cross entropy', x = 'Epoch', color = 'Model',
       caption = 'Model titles include the type of layer (bl = bidirectional LSTM, d = dense) and size of layer (256, 128, or 64 units)')


######## Summarize RF output #########

#list all output to read
rf_loss <- expand.grid(
  model = 'rf',
  exp = rf_exp,
  frail_lab = c('Resp_imp', 'Msk_prob', 'Fall_risk', 'Nutrition'),
  fold = seq(1, 10))
#read all output
for (r in 1:nrow(rf_loss)) {
      hyper <- fread(paste0(rf_dir, rf_loss$exp[r], '_hyper_', rf_loss$frail_lab[r], '_fold_', rf_loss$fold[r], '.csv'))
      hyper$model <- rf_loss$model[r]
      hyper$exp <- rf_loss$exp[r]
      hyper$fold <- rf_loss$fold[r]
      assign(paste0('rf_hyper_', rf_loss$frail_lab[r], '_fold_', rf_loss$fold[r]), hyper)
}
#gather into one table
rf_loss_all <- do.call(rbind, mget(objects(pattern = 'rf_hyper_')))
#drop fold 7 for nutrition (test set had 0 positive observations, therefore unable to calculate scaled brier score)
rf_loss_all <- filter(rf_loss_all, !(frail_lab == 'Nutrition' & fold ==7))
#summarize mean and SE for loss across fold
rf_loss_wide <- pivot_wider(rf_loss_all, 
                             id_cols = c('model', 'exp', 'frail_lab', 'fold' ,'SVD', 'ntree', 'mtry', 'sample_frac'), 
                             names_from = fold, names_prefix = 'fold', 
                             values_from = c(grep("bscore_|sbrier_|cross_entropy_", colnames(rf_loss_all), value=TRUE)))
rf_mean_loss <- mean_se(rf_loss_wide)
#save output
write.csv(rf_mean_loss, paste0(rf_dir, rf_exp, '_mean_loss'))


######## Summarize elastic net output #########
#list all output to read
enet_loss <- expand.grid(
  model = 'enet',
  exp = enet_exp,
  fold = seq(1, 10),
  frail_lab = c('Resp_imp', 'Msk_prob', 'Fall_risk', 'Nutrition'),
  class = c('neut', 'pos', 'neg'),
  SVD = c('300', '1000', '3000', 'embed'),
  alpha_l = c(9, 5, 1))
#read all output
for (r in 1:nrow(enet_loss)) {
  hyper <- fread(paste0(enet_dir, enet_loss$exp[r], '_hyper_f', enet_loss$fold[r], '_', enet_loss$frail_lab[r], '_', enet_loss$class[r], '_svd_', enet_loss$SVD[r], '_alpha', enet_loss$alpha_l[r], '.csv'))
  hyper$model <- enet_loss$model[r]
  hyper$exp <- enet_loss$exp[r]
  hyper$fold <- enet_loss$fold[r]
  assign(paste0('enet_f', enet_loss$fold[r], '_', enet_loss$frail_lab[r], '_', enet_loss$class[r], '_s_', enet_loss$SVD[r], '_alpha', enet_loss$alpha_l[r]), hyper)
}
#gather into one table
enet_loss_all <- do.call(rbind, mget(objects(pattern = 'enet_f')))
#drop fold 7 for nutrition (test set had 0 positive observations, therefore unable to calculate scaled brier score)
enet_loss_all <- filter(enet_loss_all, !(frail_lab == 'Nutrition' & fold ==7))
#get arrange scores by hyperparameter (consolidate all classes)
enet_loss_all <- pivot_wider(enet_loss_all, 
                               id_cols = c('model', 'exp', 'fold' ,'SVD', 'frail_lab', 'alpha', 'lambda', 'class'), 
                               names_from = class, 
                               values_from = c(sbrier, bscore, cross_entropy_2))
#get mean across all classes
enet_loss_all <- mutate(enet_loss_all,
                          sbrier_mean = rowMeans(enet_loss_all[,grep('sbrier_', colnames(enet_loss_all), value = TRUE)]),
                          bscore_all = rowMeans(enet_loss_all[,grep('bscore_', colnames(enet_loss_all), value = TRUE)]),
                          cross_entropy_2 = rowMeans(enet_loss_all[,grep('cross_entropy_2', colnames(enet_loss_all), value = TRUE)]))
#summarize mean and SE for loss across fold
enet_loss_wide <- pivot_wider(enet_loss_all, 
                               id_cols = c('model', 'exp', 'fold' ,'SVD', 'frail_lab', 'alpha', 'lambda'), 
                               names_from = fold, names_prefix = 'fold', 
                              values_from = c(grep("bscore_|sbrier_|cross_entropy_", colnames(enet_loss_all), value=TRUE)))
enet_mean_loss <- mean_se(enet_loss_wide)
#save output
write.csv(enet_mean_loss, paste0(enet_dir, enet_exp, '_mean_loss'))



#Comparing to prior embeddings (mean of 11-token window & center word; NO lead/lag)
rf_embed_exp <- 'exp102420_rf_embedstr'
enet_embed_exp <- 'exp102620_logit_embedstr'

rf_embed <- paste0(outdir, '_08_window_classifier_alt/', rf_embed_exp, '/')
enet_embed <- paste0(outdir, '_08_window_classifier_logit/', enet_embed_exp, '/')

#mean & se of loss across fold - wrote out all of the names explicity to help rename and merge all of the output together
rf_mean_se <- function(loss) {
  loss <- loss %>%
    mutate(mean_sbrier_neut = signif(rowMeans(loss[,grep('scaled_brier_neut', colnames(loss), value = TRUE)]), 2),
           se_sbrier_neut = signif(apply(loss[,grep('scaled_brier_neut', colnames(loss), value = TRUE)], 1, se), 1),
           mean_sbrier_pos = signif(rowMeans(loss[,grep('scaled_brier_pos', colnames(loss), value = TRUE)]), 2),
           se_sbrier_pos = signif(apply(loss[,grep('scaled_brier_pos', colnames(loss), value = TRUE)], 1, se), 1),
           mean_sbrier_neg = signif(rowMeans(loss[,grep('scaled_brier_neg', colnames(loss), value = TRUE)]), 2),
           se_sbrier_neg = signif(apply(loss[,grep('scaled_brier_neg', colnames(loss), value = TRUE)], 1, se), 1),
           mean_sbrier_all = signif(rowMeans(loss[,grep('scaled_brier_all', colnames(loss), value = TRUE)]), 2),
           se_sbrier_all = signif(apply(loss[,grep('scaled_brier_all', colnames(loss), value = TRUE)], 1, se), 1),
           mean_entropy = signif(rowMeans(loss[,grep('cross_entropy_2', colnames(loss), value = TRUE)]), 2),
           se_entropy = signif(apply(loss[,grep('cross_entropy_2', colnames(loss), value = TRUE)], 1, se), 1),
           mean_bscore_neut = signif(rowMeans(loss[,grep('cv_brier_neut', colnames(loss), value = TRUE)]), 2),
           se_bscore_neut = signif(apply(loss[,grep('cv_brier_neut', colnames(loss), value = TRUE)], 1, se), 1),
           mean_bscore_pos = signif(rowMeans(loss[,grep('cv_brier_pos', colnames(loss), value = TRUE)]), 2),
           se_bscore_pos = signif(apply(loss[,grep('cv_brier_pos', colnames(loss), value = TRUE)], 1, se), 1),
           mean_bscore_neg = signif(rowMeans(loss[,grep('cv_brier_neg', colnames(loss), value = TRUE)]), 2),
           se_bscore_neg = signif(apply(loss[,grep('cv_brier_neg', colnames(loss), value = TRUE)], 1, se), 1),
           mean_bscore_all = signif(rowMeans(loss[,grep('cv_brier_all', colnames(loss), value = TRUE)]), 2),
           se_bscore_all = signif(apply(loss[,grep('cv_brier_all', colnames(loss), value = TRUE)], 1, se), 1)) %>%
    select(-grep('fold', colnames(loss)))
  return(loss)}
#same function for glmnet
glmnet_mean_se <- function(loss) {
  loss <- loss %>%
    mutate(mean_sbrier_neut = signif(rowMeans(loss[,grep('sbrier_neut', colnames(loss), value = TRUE)]), 2),
           se_sbrier_neut = signif(apply(loss[,grep('sbrier_neut', colnames(loss), value = TRUE)], 1, se), 1),
           mean_sbrier_pos = signif(rowMeans(loss[,grep('sbrier_pos', colnames(loss), value = TRUE)]), 2),
           se_sbrier_pos = signif(apply(loss[,grep('sbrier_pos', colnames(loss), value = TRUE)], 1, se), 1),
           mean_sbrier_neg = signif(rowMeans(loss[,grep('sbrier_neg', colnames(loss), value = TRUE)]), 2),
           se_sbrier_neg = signif(apply(loss[,grep('sbrier_neg', colnames(loss), value = TRUE)], 1, se), 1),
           mean_sbrier_all = signif(rowMeans(loss[,grep('sbrier_all', colnames(loss), value = TRUE)]), 2),
           se_sbrier_all = signif(apply(loss[,grep('sbrier_all', colnames(loss), value = TRUE)], 1, se), 1),
           mean_entropy = signif(rowMeans(loss[,grep('cross_entropy_2', colnames(loss), value = TRUE)]), 2),
           se_entropy = signif(apply(loss[,grep('cross_entropy_2', colnames(loss), value = TRUE)], 1, se), 1),
           mean_bscore_neut = signif(rowMeans(loss[,grep('bscore_neut', colnames(loss), value = TRUE)]), 2),
           se_bscore_neut = signif(apply(loss[,grep('bscore_neut', colnames(loss), value = TRUE)], 1, se), 1),
           mean_bscore_pos = signif(rowMeans(loss[,grep('bscore_pos', colnames(loss), value = TRUE)]), 2),
           se_bscore_pos = signif(apply(loss[,grep('bscore_pos', colnames(loss), value = TRUE)], 1, se), 1),
           mean_bscore_neg = signif(rowMeans(loss[,grep('bscore_neg', colnames(loss), value = TRUE)]), 2),
           se_bscore_neg = signif(apply(loss[,grep('bscore_neg', colnames(loss), value = TRUE)], 1, se), 1),
           mean_bscore_all = signif(rowMeans(loss[,grep('bscore_all', colnames(loss), value = TRUE)]), 2),
           se_bscore_all = signif(apply(loss[,grep('bscore_all', colnames(loss), value = TRUE)], 1, se), 1)) %>%
    select(-grep('fold', colnames(loss)))
  return(loss)}

#RF with Old Embeddings
#list all output to read
rf_loss <- expand.grid(
  model = 'rf_embed',
  frail_lab = c('Resp_imp', 'Msk_prob', 'Fall_risk', 'Nutrition'),
  fold = seq(1, 10))
#get the current experiment number (set at the top)
rf_loss <- mutate(rf_loss, exp = ifelse(model == 'rf_tfidf', rf_tfidf_exp,
                                        ifelse(model == 'rf_embed', rf_embed_exp, NA)))
#read all output
for (r in 1:nrow(rf_loss)) {
  hyper <- fread(paste0(get(paste0(rf_loss$model[r])), rf_loss$exp[r], '_hyper_', rf_loss$frail_lab[r], '_fold_', rf_loss$fold[r], '.csv'))
  hyper$model <- rf_loss$model[r]
  hyper$exp <- rf_loss$exp[r]
  hyper$fold <- rf_loss$fold[r]
  assign(paste0('rfembed_hyper_', rf_loss$frail_lab[r], '_fold_', rf_loss$fold[r]), hyper)
}
#combine into one df
rf_embed_loss <- do.call(rbind, mget(objects(pattern = 'rfembed_hyper_')))
#fix scaled_brier_all (erroneously copied scaled_brier_neut in original script)
rf_embed_loss <- mutate(rf_embed_loss, scaled_brier_all = rowMeans(rf_embed_loss[,c('scaled_brier_neut', 'scaled_brier_pos', 'scaled_brier_neg')]))
#summarize mean and SE for loss across fold
rf_embed_loss <- pivot_wider(rf_embed_loss, 
                             id_cols = c('model', 'exp', 'frail_lab', 'fold', 'ntree', 'mtry', 'sample_frac'), 
                             names_from = fold, names_prefix = 'fold', 
                             values_from = c(scaled_brier_neut, scaled_brier_pos, scaled_brier_neg, scaled_brier_all, cross_entropy_2, cv_brier_neut, cv_brier_pos, cv_brier_neg, cv_brier_all))
rf_embed_loss <- rf_mean_se(rf_embed_loss)
#save output
write.csv(rf_embed_loss, paste0(rf_embed, rf_embed_exp, '_mean_loss'))


#Elastic net with Old Embeddings
#list all output to read
enet_loss <- expand_grid(
  model = 'enet_embed',
  fold = seq(1,10),
  frail_lab = c('Msk_prob', 'Fall_risk', 'Nutrition', 'Resp_imp'),
  alpha_l = c(9, 5, 1),
  class = c('neut', 'pos', 'neg')
)
enet_loss <- mutate(enet_loss, exp = ifelse(model == 'enet_tfidf', enet_tfidf_exp,
                                            ifelse(model == 'enet_embed', enet_embed_exp, NA)))
#read all output - Note: accidentally labeled loss as 'preds' rather than 'hyper', which was previously my convention
for (r in 1:nrow(enet_loss)) {
  hyper <- fread(paste0(get(paste0(enet_loss$model[r])), enet_loss$exp[r], '_preds_f', enet_loss$fold[r], '_', enet_loss$frail_lab[r], '_', enet_loss$class[r], '_alpha', enet_loss$alpha_l[r], '.csv'))
  hyper$model <- enet_loss$model[r]
  hyper$exp <- enet_loss$exp[r]
  hyper$fold <- enet_loss$fold[r]
  assign(paste0('EnetEmbed_f', enet_loss$fold[r], '_', enet_loss$frail_lab[r], '_', enet_loss$class[r], '_alpha', enet_loss$alpha_l[r]), hyper)
}
#gather into one table
enet_embed_loss <- do.call(rbind, mget(objects(pattern = 'EnetEmbed_f')))
#get arrange scores by hyperparameter (consolidate all classes)
enet_embed_loss <- pivot_wider(enet_embed_loss, 
                               id_cols = c('model', 'exp', 'frail_lab', 'fold' ,'alpha', 'lambda'), 
                               names_from = class, 
                               values_from = c(sbrier, bscore, cross_entropy_2))
#get mean across all classes
enet_embed_loss <- mutate(enet_embed_loss,
                          sbrier_all = rowMeans(enet_embed_loss[,grep('sbrier', colnames(enet_embed_loss), value = TRUE)]),
                          bscore_all = rowMeans(enet_embed_loss[,grep('bscore', colnames(enet_embed_loss), value = TRUE)]),
                          cross_entropy_2 = rowMeans(enet_embed_loss[,grep('cross_entropy_2', colnames(enet_embed_loss), value = TRUE)]))
#summarize mean and SE for loss across fold
enet_embed_loss <- pivot_wider(enet_embed_loss, 
                               id_cols = c('model', 'exp', 'frail_lab', 'fold' ,'alpha', 'lambda'), 
                               names_from = fold, names_prefix = 'fold', 
                               values_from = c(sbrier_neut, sbrier_pos, sbrier_neg, sbrier_all, cross_entropy_2, bscore_neut, bscore_pos, bscore_neg, bscore_all))
enet_embed_loss <- glmnet_mean_se(enet_embed_loss)
#save output
write.csv(enet_embed_loss, paste0(enet_embed, enet_embed_exp, '_mean_loss'))



######## Calibration plot ########

#load predictions for the best model
folds <- seq(1, 10)
#add label for reading .csv predictions
best_model <- mutate(rf_embed_best, sample_frac_l = ifelse(sample_frac == 0.6, 6,
                                                              ifelse(sample_frac == 0.8, 8,
                                                                     ifelse(sample_frac == 1.0, 10, NA))))

for (b in 1:nrow(best_model)) {
  #load predictions from all folds (best_model contains avg over all folds)
  for (d in 1:length(folds)) {
    #load predictions for the best model (note: read first row as header (header = TRUE)
    assign(paste0('BestPred_', best_model$frail_lab[b], '_f_', folds[d]),
           fread(paste0(get(paste0(best_model$model[b])), 'preds/', best_model$exp[b], '_preds_', best_model$frail_lab[b], '_fold_', folds[d], '_mtry_', best_model$mtry[b], '_sfr_', best_model$sample_frac_l[b], '.csv'),
                 header = TRUE))
    #rename cols for plots
    df <- get(paste0('BestPred_', best_model$frail_lab[b], '_f_', folds[d]))
    colnames(df)[colnames(df) == '0'] <- 'Neutral'
    colnames(df)[colnames(df) == '1'] <- 'Positive'
    colnames(df)[colnames(df) == '-1'] <- 'Negative'
    assign(paste0('BestPred_', best_model$frail_lab[b], '_f_', folds[d]), df)
    #load obs
    assign(paste0('f', folds[d], '_te'), fread(paste0(obsdir, 'f_', folds[d], '_te_df.csv')))
    y_test_neut <- get(paste0('f', folds[d], '_te'))[[paste0(best_model$frail_lab[b], '_0')]]
    y_test_pos <- get(paste0('f', folds[d], '_te'))[[paste0(best_model$frail_lab[b], '_1')]]
    y_test_neg <- get(paste0('f', folds[d], '_te'))[[paste0(best_model$frail_lab[b], '_-1')]]
    y_test <- cbind(y_test_neut, y_test_pos, y_test_neg)
    assign(paste0('Obs_', best_model$frail_lab[b], '_fold_', folds[d]), y_test)
  }
  #gather preds for all folds
  assign(paste0(best_model$frail_lab[b], '_pred'), do.call(rbind, mget(objects(pattern = paste0('BestPred_', best_model$frail_lab[b])))))
  #gather obs for all folds
  assign(paste0(best_model$frail_lab[b], '_obs'), do.call(rbind, mget(objects(pattern = paste0(paste0('Obs_', best_model$frail_lab[b]))))))
  #combine obs & preds
  assign(paste0(best_model$frail_lab[b], '_obs_pred'), cbind(get(paste0(best_model$frail_lab[b], '_obs')), get(paste0(best_model$frail_lab[b], '_pred'))))
}
#calibration plot
mc_calib_plot(y_test_neut + y_test_pos + y_test_neg ~ Neutral + Positive + Negative, data = Resp_imp_obs_pred,  cuts = 5)+
  theme(legend.position='bottom', plot.title = element_text(hjust = 0.5), text = element_text(size=14)) +
  labs(title = 'Respiratory impairment')
#save
ggsave(filename = paste0(get(paste0(best_model$model[b])), 'resp_imp_calib.pdf'), device= 'pdf') 
#repeat for each aspect
mc_calib_plot(y_test_neut + y_test_pos + y_test_neg ~ Neutral + Positive + Negative, data = Msk_prob_obs_pred,  cuts = 5)+
  theme(legend.position='bottom', plot.title = element_text(hjust = 0.5), text = element_text(size=14)) +
  labs(title = 'Musculoskeletal problem')
ggsave(filename = paste0(get(paste0(best_model$model[b])), 'msk_prob_calib.pdf'), device= 'pdf') 
mc_calib_plot(y_test_neut + y_test_pos + y_test_neg ~ Neutral + Positive + Negative, data = Fall_risk_obs_pred,  cuts = 5)+
  theme(legend.position='bottom', plot.title = element_text(hjust = 0.5), text = element_text(size=14)) +
  labs(title = 'Fall risk')
ggsave(filename = paste0(get(paste0(best_model$model[b])), 'fall_risk_calib.pdf'), device= 'pdf') 
mc_calib_plot(y_test_neut + y_test_pos + y_test_neg ~ Neutral + Positive + Negative, data = Nutrition_obs_pred,  cuts = 5)+
  theme(legend.position='bottom', plot.title = element_text(hjust = 0.5), text = element_text(size=14)) +
  labs(title = 'Nutrition impairment')
ggsave(filename = paste0(get(paste0(best_model$model[b])), 'nutrition_calib.pdf'), device= 'pdf') 


#combine all aspects for 1 fold (more fair than 10 folds for all aspects -- way overestimates confidence of calibration)
#reformat all obs and preds for calib plots
frail_lab = c('Resp_imp', 'Msk_prob', 'Fall_risk', 'Nutrition')
for (f in 1:length(frail_lab)) {
  #arrange obs for fold 1 into long format
  obs_long <- c(get(paste0('Obs_', frail_lab[f], '_fold_1'))[,1], get(paste0('Obs_', frail_lab[f], '_fold_1'))[,2], get(paste0('Obs_', frail_lab[f], '_fold_1'))[,3])
  #arrange preds for fold 1 into long format
  preds_long <- c(get(paste0('BestPred_', frail_lab[f], '_f_1'))$Neutral, get(paste0('BestPred_', frail_lab[f], '_f_1'))$Positive, get(paste0('BestPred_', frail_lab[f], '_f_1'))$Negative)
  #combine obs & preds
  assign(paste0(frail_lab[f], '_fold1_obs_pred'), cbind(obs_long, preds_long))
  #rename cols for plots
  df <- get(paste0(frail_lab[f], '_fold1_obs_pred'))
  colnames(df)[colnames(df) == 'obs_long'] <- paste0(frail_lab[f], '_obs')
  colnames(df)[colnames(df) == 'preds_long'] <- paste0(frail_lab[f])
  #output
  assign(paste0(frail_lab[f], '_fold1_obs_pred'), df)
}
#join together
assign('all_aspects', do.call(cbind, mget(objects(pattern = '_fold1_obs_pred'))))
#rename for plot
colnames(all_aspects)[colnames(all_aspects) == 'Fall_risk'] <- 'Fall risk'
colnames(all_aspects)[colnames(all_aspects) == 'Msk_prob'] <- "Musculoskeletal\nproblem"
colnames(all_aspects)[colnames(all_aspects) == 'Resp_imp'] <- "Respiratory\nimpairment"
#calibration plot
mc_calib_plot(Fall_risk_obs + Msk_prob_obs + Nutrition_obs + Resp_imp_obs ~ `Fall risk` + `Musculoskeletal\nproblem` + Nutrition + `Respiratory\nimpairment`, data = all_aspects, cuts = 5) +
  theme(legend.position='bottom', plot.title = element_text(hjust = 0.5), text = element_text(size=12)) +
  labs(title = 'Calibration across all classes for each fraily aspect')
#write out
ggsave(filename = paste0(get(paste0(best_model$model[b])), 'all_aspects_calib.pdf'), device= 'pdf') 


