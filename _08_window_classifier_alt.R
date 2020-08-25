library(ranger)
library(Matrix)
workdir <- paste0(getwd(), '/')
outdir <- paste0(getwd(), '/')

tr_tfidf <- as(Matrix::readMM(paste0(workdir,'tr_df_tfidf.mtx')), "dgCMatrix")
tr_tfidf <- provideDimnames(tr_tfidf)
tr_df <- data.table::fread(paste0(workdir,'tr_df.csv'))

te_tfidf <- as(Matrix::readMM(paste0(workdir,'te_df_tfidf.mtx')), "dgCMatrix")
te_tfidf <- provideDimnames(te_tfidf)
te_df <- data.table::fread(paste0(workdir,'te_df.csv'))

#fit rf
tr_tfidf_rg <- ranger(y = tr_df$Resp_imp_1, x= tr_tfidf, probability = TRUE)

#make predictions
te_tfidf_pred <- predict(tr_tfidf_rg, data=te_tfidf)

#compute microaveraged brier score