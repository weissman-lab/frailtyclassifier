# Contains helper functions for running _08_rf.R


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


load_labels_caseweights <- function(rep, fold){
  tr <- fread(paste0(trvadatadir, 'r', rep, '_f', fold, '_tr_df.csv'))
  va <- fread(paste0(trvadatadir, 'r', rep, '_f', fold, '_va_df.csv'))
  cw <- fread(paste0(cwdir, 'r', rep, '_f', fold, '_tr_caseweights.csv'))
  if (nrow(tr) != nrow(cw)){
    stop("caseweights do not match training data")    
  }
  return(list(tr = tr, va = va, cw = cw))
}


load_x_bioclinicalbert <- function(rep, fold, lab_cw){
  # load embeddings for each fold (drop index and 1st row junk)
  embeddings_tr <- fread(
    paste0(embeddingsdir, 'r', rep, '_f', fold, '_tr_bioclinicalbert.csv'),
    skip = 1, drop = 1)
  colnames(embeddings_tr) <- paste0('d', seq(1, 768))
  pca_tr <- lab_cw$tr
  pca_cols <- grep('pca', colnames(pca_tr), value = TRUE)
  # test that embeddings match structured data
  if ((nrow(embeddings_tr) == nrow(pca_tr)) == FALSE)
    stop("bioclinicalbert does not match training data")
  #embeddings without vs with structured data
  if (INC_STRUC == FALSE) {
    #get only embeddings columns
    embeddings_tr <- data.matrix(embeddings_tr)
  } else {
    # concatenate embeddings with structured data
    embeddings_tr <- data.matrix(cbind(embeddings_tr, pca_tr[, ..pca_cols]))
  }
  embeddings_va <- fread(
    paste0(embeddingsdir, 'r', rep, '_f', fold, '_va_bioclinicalbert.csv'),
    skip = 1, drop = 1)
  colnames(embeddings_va) <- paste0('d', seq(1, 768))
  pca_va <- lab_cw$va
  pca_cols <- grep('pca', colnames(pca_va), value = TRUE)
  # test that embeddings match structured data
  if ((nrow(embeddings_va) == nrow(pca_va)) == FALSE)
    stop("bioclinicalbert does not match validation data")
  #embeddings without vs with structured data
  if (INC_STRUC == FALSE) {
    #get only embeddings columns
    embeddings_va <- data.matrix(embeddings_va)
  } else {
    # concatenate embeddings with structured data
    embeddings_va <- data.matrix(cbind(embeddings_va, pca_va[, ..pca_cols]))
  }
  outlist <- list(tr = embeddings_tr,
                  va = embeddings_va)
  return(outlist)
}


load_x_roberta <- function(rep, fold, lab_cw){
  # load embeddings for each fold (drop index and 1st row junk)
  embeddings_tr <- fread(
    paste0(embeddingsdir, 'r', rep, '_f', fold, '_tr_roberta.csv'),
    skip = 1, drop = 1)
  colnames(embeddings_tr) <- paste0('d', seq(1, 768))
  pca_tr <- lab_cw$tr
  pca_cols <- grep('pca', colnames(pca_tr), value = TRUE)
  # test that embeddings match structured data
  if ((nrow(embeddings_tr) == nrow(pca_tr)) == FALSE)
    stop("roberta does not match training data")
  #embeddings without vs with structured data
  if (INC_STRUC == FALSE) {
    #get only embeddings columns
    embeddings_tr <- data.matrix(embeddings_tr)
  } else {
    # concatenate embeddings with structured data
    embeddings_tr <- data.matrix(cbind(embeddings_tr, pca_tr[, ..pca_cols]))
  }
  embeddings_va <- fread(
    paste0(embeddingsdir, 'r', rep, '_f', fold, '_va_roberta.csv'),
    skip = 1, drop = 1)
  colnames(embeddings_va) <- paste0('d', seq(1, 768))
  pca_va <- lab_cw$va
  pca_cols <- grep('pca', colnames(pca_va), value = TRUE)
  # test that embeddings match structured data
  if ((nrow(embeddings_va) == nrow(pca_va)) == FALSE)
    stop("roberta does not match validation data")
  #embeddings without vs with structured data
  if (INC_STRUC == FALSE) {
    #get only embeddings columns
    embeddings_va <- data.matrix(embeddings_va)
  } else {
    # concatenate embeddings with structured data
    embeddings_va <- data.matrix(cbind(embeddings_va, pca_va[, ..pca_cols]))
  }
  outlist <- list(tr = embeddings_tr,
                  va = embeddings_va)
  return(outlist)
}


load_x_embeddings <- function(rep, fold, lab_cw){
  embeddings_tr <- fread(paste0(embeddingsdir, 'r', rep, '_f', fold, '_tr_embed_min_max_mean_SENT.csv'))
  pca_tr <- lab_cw$tr
  emb_cols <- grep('min_|max_|mean_', colnames(embeddings_tr), value = TRUE)
  pca_cols <- grep('pca', colnames(pca_tr), value = TRUE)
  # test that embeddings match structured data
  if (identical(pca_tr$sentence_id, embeddings_tr$sentence_id) == FALSE)
    stop("embeddings do not match training data")
  #embeddings without vs with structured data
  if (INC_STRUC == FALSE) {
    #get only embeddings columns
    embeddings_tr <- data.matrix(embeddings_tr[, ..emb_cols])
  } else {
    # concatenate embeddings with structured data
    embeddings_tr <- data.matrix(cbind(embeddings_tr[, ..emb_cols], pca_tr[, ..pca_cols]))
  }
  embeddings_va <- fread(paste0(embeddingsdir, 'r', rep, '_f', fold, '_va_embed_min_max_mean_SENT.csv'), drop = 1)
  pca_va <- lab_cw$va
  emb_cols <- grep('min_|max_|mean_', colnames(embeddings_va), value = TRUE)
  pca_cols <- grep('pca', colnames(pca_va), value = TRUE)
  # test that embeddings match structured data
  if (identical(pca_va$sentence_id, embeddings_va$sentence_id) == FALSE)
    stop("embeddings do not match validation data")
  #embeddings without vs with structured data
  if (INC_STRUC == FALSE) {
    #get only embeddings columns
    embeddings_va <- data.matrix(embeddings_va[, ..emb_cols])
  } else {
    # concatenate embeddings with structured data
    embeddings_va <- data.matrix(cbind(embeddings_va[, ..emb_cols], pca_va[, ..pca_cols]))
  }
  outlist <- list(tr = embeddings_tr,
                  va = embeddings_va)
  return(outlist)
}

load_x_svd <- function(rep, fold, lab_cw, svd_dim){
  svd_tr <- fread(paste0(SVDdir, 'r', rep, '_f', fold, '_tr_svd', svd_dim, '.csv'),
                  skip = 1,
                  drop = 1)
  colnames(svd_tr) <- c('sentence_id', paste0('svd', seq(1:as.integer(svd_dim))))
  pca_tr <- lab_cw$tr
  svd_cols <- grep('svd', colnames(svd_tr), value = TRUE)
  pca_cols <- grep('pca', colnames(pca_tr), value = TRUE)
  #test that svd match structured data
  if (identical(pca_tr$sentence_id, svd_tr$sentence_id) == FALSE)
    stop("embeddings do not match validation data")
  #svd without vs with structured data
  if (INC_STRUC == FALSE) {
    #load only the svd columns
    svd_tr <- data.matrix(svd_tr[, ..svd_cols])
  } else {
    #concatenate svd with structured data
    svd_tr <- data.matrix(cbind(svd_tr[, ..svd_cols], pca_tr[, ..pca_cols]))
  }
  svd_va <- fread(paste0(SVDdir, 'r', rep, '_f', fold, '_va_svd', svd_dim, '.csv'),
                  skip = 1,
                  drop = 1)
  colnames(svd_va) <- c('sentence_id', paste0('svd', seq(1:as.integer(svd_dim))))
  pca_va <- lab_cw$va
  svd_cols <- grep('svd', colnames(svd_va), value = TRUE)
  pca_cols <- grep('pca', colnames(pca_va), value = TRUE)
  #test that svd match structured data
  if (identical(pca_va$sentence_id, svd_va$sentence_id) == FALSE)
    stop("embeddings do not match validation data")
  #svd without vs with structured data
  if (INC_STRUC == FALSE) {
    #load only the svd columns
    svd_va <- data.matrix(svd_va[, ..svd_cols])
  } else {
    #concatenate svd with structured data
    svd_va <- data.matrix(cbind(svd_va[, ..svd_cols], pca_va[, ..pca_cols]))
  }
  outlist <- list(tr = svd_tr,
                  va = svd_va)
  return(outlist)
}

load_x <- function(rep, fold, lab_cw, svd_dim = NULL){
  if (is.null(svd_dim) | svd_dim == "embed"){
    return(load_x_embeddings(rep, fold, lab_cw))
  } else if (svd_dim == "bioclinicalbert") {
    return(load_x_bioclinicalbert(rep, fold, lab_cw))
  } else if (svd_dim == "roberta") {
    return(load_x_roberta(rep, fold, lab_cw))
  } else {
    return(load_x_svd(rep, fold, lab_cw, svd_dim))
  }
}