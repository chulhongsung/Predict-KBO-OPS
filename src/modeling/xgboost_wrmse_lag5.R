##############################################
############ XGBOOST & WRMSE #################
##############################################

rm(list = ls())
gc(reset = T)

if(Sys.getenv('USERNAME') == 'UOS') setwd('C:\\Users\\UOS\\Desktop\\dacon')
if(Sys.getenv('USERNAME') == 'moon') setwd('D:\\Project\\git\\Predict-KBO-OPS\\src')
if(Sys.getenv('USERNAME') == 'kyucheol') setwd('C:\\Users\\kyucheol\\Dropbox\\dacon')

if(!require(tidyverse)) install.packages('tidyverse')
if(!require(xgboost)) install.packages('xgboost')

library(tidyverse)
library(xgboost)

load('lag5_data.Rdata')

#### xgb.DMatrix

lag5_train_x <- lag5_train_data[,1:97]

dtrain <- xgb.DMatrix(data.matrix(lag5_train_x),info = list(label = lag5_train_data$t_OPS, weight = lag5_train_data$t_AB))

#### XGBOOST Custom eval_metric, objective function => WRMSE

wrmse <- function(preds, dtrain){
  OPS <- getinfo(dtrain, 'label')
  AB <- getinfo(dtrain, 'weight')
  err <- (sum((preds - OPS)^2 * AB) / sum(AB)) %>% sqrt()
  return(list(metric = 'WRMSE', value = err))
}

wrmse_obj <- function(preds, dtrain){
  OPS <- getinfo(dtrain, 'label')
  AB <- getinfo(dtrain, 'weight')
  grad <- (preds - OPS) * AB
  hess <- AB
  return(list(grad = grad, hess = hess))
}

#### XGBOOST Parameters Tuning 

best_param <- list()
best_seednumber <- 1234
best_wrmse <- Inf
best_wrmse_index <- 0

set.seed(526)

for (iter in seq_len(100)){
  
  param <- list(booster = 'dart',
                objective = wrmse_obj,
                eval_metric = wrmse,
                sample_type = sample(c('weighted', 'uniform'), 1),
                normalize_type = sample(c('tree', 'forest'), 1),
                max_depth = sample(6:10, 1),
                eta = runif(1, .01, .3), 
                subsample = runif(1, .6, .9),
                colsample_bytree = runif(1, .5, .8), 
                min_child_weight = sample(1:40, 1),
                max_delta_step = sample(1:10, 1)
  )
  
  cv.nround <- 1000
  
  cv.nfold <- 5 # 5-fold cross-validation
  
  seed.number <- sample.int(10000, 1) # set seed for the cv
  
  set.seed(seed.number)
  
  mdcv <- xgb.cv(data = dtrain, params = param,  
                 nfold = cv.nfold, nrounds = cv.nround,
                 verbose = F, early_stopping_rounds = 8, maximize = FALSE)
  
  min_wrmse_index <- mdcv$best_iteration
  
  min_wrmse <-  mdcv$evaluation_log[min_wrmse_index,]$test_WRMSE_mean
  
  if (min_wrmse < best_wrmse) {
    best_wrmse <- min_wrmse
    best_wrmse_index <- min_wrmse_index
    best_seednumber <- seed.number
    best_param <- param
  }
  cat('Iter::', iter, '\n')
}

#### Train XGBOOST Model With Best Parameters 

nrounds <-  best_wrmse_index

set.seed(best_seednumber)

lag5_test_x <- lag5_test_data[,1:97]

dtest <- xgb.DMatrix(data.matrix(lag5_test_x), info = list(label = lag5_test_data$t_OPS, weight = lag5_test_data$t_AB))

xg_mod <- xgboost(data = dtrain, params = best_param, nround = nrounds, verbose = T)

pred_ops <- predict(xg_mod, dtest)

test_wrmse <- (sum((pred_ops - lag5_test_data$t_OPS)^2 * lag5_test_data$t_AB) / sum(lag5_test_data$t_AB)) %>% sqrt() # 0.1257451

save(best_param, file = 'xgb_best_param.Rdata' )
