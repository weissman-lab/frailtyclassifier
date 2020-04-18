




## This script combines and imputes the structured data

library(mgcv)
library(doParallel)
library(foreach)
library(dplyr)
registerDoParallel(detectCores())
`%ni%` <- Negate(`%in%`)
if (grepl("crandrew", getwd())){
  datadir <- "~/projects/GW_PAIR_frailty_classifier/data/"
  outdir <- "~/projects/GW_PAIR_frailty_classifier/output/"
  figdir <- "~/projects/GW_PAIR_frailty_classifier/figures/"
} else {
  datadir <- "/media/drv2/andrewcd2/frailty/data/"
  outdir <- "/media/drv2/andrewcd2/frailty/output/"
  figdir <- "/media/drv2/andrewcd2/frailty/figures/"
}

# read in the csv
df <- read.csv(paste0(outdir, "hyperparameter_gridsearch_11apr_win.csv"))

dim(df)
head(df)
df <- df[!is.na(df$best_loss),]
oob <- df$oob
df$oob <- NULL
head(df)
df$idx <- df$X <- NULL

plot(df)
dev.new()
df$semipar = ifelse(df$semipar == "True", T, F)

hist(df$best_loss %>% log)

m <- lm(time_to_convergence~.-best_loss, data = df)
summary(m)
min(df$best_loss)

library(ranger)
m <- ranger(best_loss~.-time_to_convergence, data = df, importance = 'permutation')
m$variable.importance %>% sort
summary(m)
m

plot(df$n_units, df$best_loss, pch = 19)
plot(df$n_dense, df$best_loss, pch = 19)

j=0
j=j+1
i = colnames(df)[j]
m <- gam(log(best_loss)~ 
           s(window_size)
         + s(n_dense, k = 5)
         + s(l1_l2, k = 5)
         + s(n_units)
         + s(dropout)
         + semipar
           -time_to_convergence, data = df)
summary(m)
plot(m, pages = 1, scheme = 2, all = T)

head(df)
df[which.min(df$best_loss),]


