rm(list = ls())
gc(reset = T)

if(Sys.getenv('USERNAME') == 'UOS') setwd('C:\\Users\\UOS\\Desktop\\dacon')

if(!require(tidyverse)) install.packages('tidyverse'); library(tidyverse)
if(!require(data.table)) install.packages('data.table'); library(data.table)
if(!require(xgboost)) install.packages('xgboost'); library(xgboost)
if(!require(randomForest)) install.packages('randomForest'); library(randomForest)
if(!require(keras)) install.packages('keras'); library(keras)
if(!require(readr)) install.packages('readr'); library(readr)


#### Data load
load('lag5_AB_data.Rdata')
load('submission_data.Rdata')

#### XGB 

#### xgb.DMatrix

dtrain <- xgb.DMatrix(data = data.matrix(lag5_train_data %>% select(-batter_id, -t_AB, -t_OPS)), info = list(label = lag5_train_data$t_OPS, weight = lag5_train_data$t_AB))
dtest <-  xgb.DMatrix(data = data.matrix(lag5_test_data %>% select(-batter_id, -t_AB, -t_OPS)),info = list(label = lag5_test_data$t_OPS, weight = lag5_test_data$t_AB))

#### XGBOOST Custom eval_metric, objective function

wrmse <- function(preds, dtrain){
  OPS <- getinfo(dtrain, 'label')
  AB <- getinfo(dtrain, 'weight')
  err <- (sum((preds - OPS)^2 * AB) / sum(AB)) %>% sqrt()
  return(list(metric = 'WRMSE', value = err))
}

#### XGBOOST Parameters Tuning 

best_param_lag5 <- list()
best_seednumber <- 1234
best_wrmse <- Inf
best_wrmse_index <- 0

set.seed(1)

for (iter in seq_len(200)){
  
  param <- list(booster = 'dart',
                objective = 'reg:linear',
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
                 verbose = F, early_stopping_rounds = 10, maximize = FALSE)
  
  min_wrmse_index <- mdcv$best_iteration
  
  min_wrmse <-  mdcv$evaluation_log[min_wrmse_index,]$test_WRMSE_mean
  
  if (min_wrmse < best_wrmse) {
    best_wrmse <- min_wrmse
    best_wrmse_index <- min_wrmse_index
    best_seednumber <- seed.number
    best_param_lag5 <- param
  }
  cat('Iter::', iter, '\n')
}

#### Train XGBOOST Model With Best Parameters 

nrounds <- best_wrmse_index

set.seed(best_seednumber)

xg_mod <- xgboost(data = dtrain, params = best_param_lag5, nround = nrounds, verbose = T)

pred_ops <- predict(xg_mod, dtest) # * if_else(lag5_test_data$AB <= 5, 0, 1)

wrmse(pred_ops, dtest) # 0.1206037

save(best_param_lag5, nrounds, file = 'xgb_best_param_ratio_lag5.Rdata')

#### Fit XGB for submission_data

load('xgb_best_param_ratio_lag5.Rdata')

#### train_data for submission

train_data <- rbind(lag5_train_data, lag5_test_data)

dtrain <- xgb.DMatrix(data = data.matrix(train_data %>% select(-batter_id, -t_AB, -t_OPS)), info = list(label = train_data$t_OPS, weight = train_data$t_AB))
dtest <- xgb.DMatrix(data = data.matrix(submission_data %>% select(-batter_id)))

wrmse <- function(preds, dtrain){
  OPS <- getinfo(dtrain, 'label')
  AB <- getinfo(dtrain, 'weight')
  err <- (sum((preds - OPS)^2 * AB) / sum(AB)) %>% sqrt()
  return(list(metric = 'WRMSE', value = err))
}

xg_mod <- xgboost(data = dtrain, params = best_param_lag5, nround = nrounds, verbose = T)

pred_XGB <- predict(xg_mod, dtest)

#### DNN

train <- train_data %>% select(-batter_id, -t_AB, -t_OPS) %>% data.matrix()
weights <- train_data$t_AB
label <- train_data$t_OPS

size <- ncol(train)

model <- keras_model_sequential() %>% 
  layer_dense(units = 200, activation = 'relu', input_shape = size) %>% 
  layer_dense(units = 100, activation = 'relu') %>% 
  layer_dense(units = 70, activation = 'relu') %>% 
  layer_dense(units = 40, activation = 'relu') %>% 
  layer_dense(units = 30, activation = 'relu') %>% 
  layer_dense(units = 20, activation = 'relu') %>% 
  layer_dense(units = 10, activation = 'relu') %>% 
  layer_dense(units = 5, activation = 'relu') %>% 
  layer_dense(units = 1)

# Custom metric
weighted_rmse <- function(y_true, y_pred, weigths){
  K <- backend()
  weights <- K$variable(weights)
  loss <- ((K$sum(weights * (K$pow(y_true - y_pred, 2))))/K$sum(weights))%>% sqrt()
  loss
}

wrmse <- custom_metric("wrmse", function(y_true, y_pred) {
  weighted_rmse(y_true, y_pred, weigths)
}) 

model %>% compile(
  loss = 'mse',
  optimizer = optimizer_adam(),
  metrics = wrmse 
)

print_dot_callback <- callback_lambda(
  on_epoch_end = function(epoch, logs) {
    if (epoch %% 80 == 0) cat("\n")
    cat(".")
  }
)    

epochs <- 1000

# Fit the model and store training stats
history <- model %>% fit(
  train,
  label,
  sample_weight = weights,
  epochs = epochs,
  validation_split = 0.2,
  verbose = 0,
  callbacks = list(print_dot_callback)
)

plot(history)

submission <- submission_data %>% select(-batter_id) %>% data.matrix()
pred_DNN <- model %>% predict(submission)

#### RandomForest

train.x <- train_data %>% select(-batter_id, -t_AB, -t_OPS)
train.y <- train_data$t_OPS

rf_model <- randomForest(x = train.x, y = train.y, ntree = 5000)

submission_rf <- apply(submission, 2, function(x) ifelse(is.infinite(x), 1, x)) %>% data.matrix()

pred_RF <- predict(rf_model, submission_rf)


#### Ensemble Model

pred_sheet <- cbind(submission_data$batter_id, pred_XGB, pred_DNN, pred_RF) %>% as_tibble() %>% mutate_all(as.numeric)

colnames(pred_sheet) <- c('batter_id', 'pred_XGB', 'pred_DNN', 'pred_RF')

pred_sheet <- pred_sheet %>%  mutate(pred_OPS = pred_XGB*0.4 + pred_DNN*0.3 + pred_RF*0.3)

pred_sheet$pred_OPS <- pred_sheet$pred_OPS * if_else(as.numeric(submission_data$AB) < 5, 0, 1)

#### Final submission 

submission_OPS <- pred_sheet %>% select(batter_id, pred_OPS)

submission_form <- fread('data/submission.csv', header = T, encoding = 'UTF-8') %>% as_tibble()

submission_form <- submission_form %>% left_join(submission_OPS, by = 'batter_id')

write_excel_csv(submission_form, 'final_submission.csv')

