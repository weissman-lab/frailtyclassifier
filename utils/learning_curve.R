library(data.table)
library(dplyr)
library(ggplot2)
library(ggrepel)

#set directories based on location
dirs = c(paste0('/gwshare/frailty/output/'),
         '/Users/martijac/Documents/Frailty/frailty_classifier/output/',
         '/media/drv2/andrewcd2/frailty/output/')
for (d in 1:length(dirs)) {
  if (dir.exists(dirs[d])) {
    rootdir = dirs[d]
  }
}

#learning curve raw data
lc <- fread(paste0(rootdir,
                   'figures_tables/learning_curve.csv'))
lc <- lc %>%
  group_by(batch) %>%
  arrange(desc(aspect_mean_all)) %>%
  slice(1)

#count notes for each batch
note_count <- fread(paste0(rootdir,
                '/notes_labeled_embedded_SENTENCES/notes_train_official.csv'))

note_count <- note_count %>%
  group_by(batch) %>%
  count()

AL03 = sum(note_count[note_count$batch %in% c('batch_01', 'batch_02', 'batch_03', 'AL00',
                                   'AL01', 'AL01_v2', 'AL01_v2ALTERNATE',
                                   'AL02_v2'), ]$n)

AL04 = sum(note_count[note_count$batch %in% c('AL03'), ]$n, AL03)

note_count <- data.frame(n_notes= c(AL03, AL04), batch = c('AL03', 'AL04'))

#combine performance & counts
lc_count <- left_join(lc, note_count)
lc_count[,c('batch', 'n_notes', 'aspect_mean_all', 'aspect_se_all')]
#plot best performance & cumulative note count
ggplot(lc_count, aes(x = batch, y = aspect_mean_all, group=1)) +
  geom_line() +
  geom_pointrange(aes(ymin = (aspect_mean_all - aspect_se_all),
                      ymax = (aspect_mean_all + aspect_se_all))) +
  geom_label_repel(aes(label=n_notes), direction = 'y', nudge_y = -0.05, segment.size=0)+
  labs(title = 'Learning curve',
       y = 'Scaled Brier Score',
       x = 'Batch') +
  theme(legend.position = 'bottom') +
  ylim(0, 0.48)
