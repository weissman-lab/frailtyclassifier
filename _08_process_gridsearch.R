




## This script combines and imputes the structured data

library(mgcv)
library(doParallel)
library(foreach)
library(dplyr)
library(ranger)
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
df <- read.csv(paste0(outdir, "hyperparameter_gridsearch_21apr_win.csv"))

df <- df[!is.na(df$best_loss),]
oob <- df$oob
df$oob <-df$idx <- df$X <- NULL
df$l1_l2 <- df$semipar <- NULL

hist(df$time_to_convergence/60, breaks = 10)


df$semipar = ifelse(df$semipar == "True", T, F)

hist(df$best_loss %>% log)

m <- lm(log(best_loss)~.-time_to_convergence, data = df)
summary(m)
min(df$best_loss)

m <- ranger(best_loss~.-time_to_convergence, data = df, importance = 'permutation')
m$variable.importance %>% sort
m

plot(df)

m <- gam(log(best_loss)~ 
           s(window_size, k=10)
         + s(n_dense, k = 5)
         + s(n_units, k=10)
         + s(dropout,k=10)
           -time_to_convergence, data = df, method = 'REML')
summary(m)
plot(m, pages = 1, scheme = 2, all = T)

m1

head(df)
df[which.min(df$best_loss),]


