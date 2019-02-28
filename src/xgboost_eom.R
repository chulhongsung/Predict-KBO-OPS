library(tidyverse)
library(data.table)
library(xgboost)
rm(list = ls()) ; gc(reset = T)

psb <- fread('C:/Users/kyucheol/Dropbox/dacon/Pre_Season_Batter.csv', encoding = 'UTF-8') %>% as.tibble()
rsb <- fread('C:/Users/kyucheol/Dropbox/dacon/Regular_Season_Batter.csv', encoding = 'UTF-8') %>% as.tibble()
rsbd <- fread('C:/Users/kyucheol/Dropbox/dacon/Regular_Season_Batter_Day_by_Day.csv', encoding = 'UTF-8') %>% as.tibble()
submis <- fread('C:/Users/kyucheol/Dropbox/dacon/submission.csv', encoding = 'UTF-8') %>% as.tibble()

#### train data
batter_id_2017 <- rsb %>% filter(year == 2017) %>% group_by(batter_id) %>% distinct(batter_id) %>% ungroup()

tmp_train <- rsb %>% filter(batter_id %in% batter_id_2017$batter_id, year %in% 2005:2016) %>% arrange(batter_id)

tmp_train <- tmp_train %>% select(c(1, 3, 7:20))

tmp_train <- tibble(batter_id = rep(unique(tmp_train$batter_id), each = 12), year = rep(2005:2016, length(unique(tmp_train$batter_id)))) %>% left_join(tmp_train, by = c('batter_id', 'year'))

tmp_train <- apply(tmp_train, 2, function(x) ifelse(is.na(x), 0, x)) %>% as.tibble()

tmp_train <- tmp_train %>% arrange(batter_id)

dim(tmp_train)
# 2364/12 = 197 
#### 197*(1+12*14) 의 행렬을 생성(즉, id, 그리고 나머지 14개변수*12년)
train_x <- data.matrix(tmp_train[,-c(1,2)])

train_x <- matrix(as.vector(t(train_x)), nrow = 197, , byrow = T) %>% as.tibble()

x_name <- NULL
for(i in 12:1){
  x_name_tmp <- paste(names(tmp_train)[-(2:1)], rep(i,14), sep ='_')
  x_name <- c(x_name, x_name_tmp)
}

names(train_x) <- x_name

tmp_train_x <- cbind(batter_id = unique(tmp_train$batter_id), train_x) # train_x에 batter_id있는 data.


#### train_y : 2017년도의 OPS
train_y <- rsb %>% filter(year == 2017 , batter_id %in% tmp_train$batter_id) %>% 
  select(batter_id, AB, OPS) %>% arrange(batter_id)

train_y <- apply(train_y, 2, function(x) ifelse(is.na(x), 0, x)) %>% as.tibble()

#### test data 

batter_id_2018 <- rsb %>% filter(year == 2018) %>% distinct(batter_id) 

#### 2006 ~ 2017

tmp_test <- rsb %>% filter(batter_id %in% batter_id_2018$batter_id) %>% filter(year %in% 2006:2017)

tmp_test <- tmp_test %>% select(c(1, 3, 7:20))

tmp_test <- tibble(batter_id = rep(unique(tmp_test$batter_id), each = 12), year = rep(2006:2017, length(unique(tmp_test$batter_id)))) %>% left_join(tmp_test, by = c('batter_id', 'year'))

tmp_test <- apply(tmp_test, 2, function(x) ifelse(is.na(x), 0, x)) %>% as.tibble()

tmp_test <- tmp_test %>% arrange(batter_id)

test_x <- data.matrix(tmp_test[,-c(1,2)])

test_x <- matrix(as.vector(t(test_x)), nrow = 211, , byrow = T) %>% as.tibble()

names(test_x) <- x_name

tmp_test_x <- cbind(batter_id = unique(tmp_test$batter_id), test_x) # train_x에 batter_id있는 data.

#### test_y : 2018년도의 OPS
test_y <- rsb %>% filter(year == 2018 , batter_id %in% tmp_test$batter_id) %>% 
  select(batter_id, AB, OPS) %>% arrange(batter_id)

test_y <- apply(test_y, 2, function(x) ifelse(is.na(x), 0, x)) %>% as.tibble()


#### XGBoost
params <- list(booster = 'gblinear',
               objective = 'reg:linear',
               subsample = 0.7,
               max_depth = 5,
               colsample_bytree = 0.7,
               eta = 0.037,
               eval_metric = 'rmse',
               min_child_weight = 100)


dtrain <- xgb.DMatrix(data.matrix(train_x), label = train_y$OPS)

set.seed(1)

xgbcv <- xgb.cv(data = dtrain, params = params, nround = 100, prediction = T, maximize = F, nfold = 5,
                early_stopping_rounds = 30)

nrounds <- xgbcv$best_iteration

xgb <- xgb.train(data = dtrain, params = params, nround = nrounds)

dtest <- xgb.DMatrix(data.matrix(test_x))

pred_ops18 <- predict(xgb, dtest)

#### WRMSE 구하기.
sqrt(sum((test_y$OPS - pred_ops18)^2 * test_y$AB )/ sum(test_y$AB))
# WRMSE = 0.1642529