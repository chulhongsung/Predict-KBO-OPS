rm(list =ls())
gc(reset = TRUE)

if(Sys.getenv('USERNAME') == 'UOS') setwd('C:\\Users\\UOS\\Desktop\\dacon')
if(Sys.getenv('USERNAME') == 'moon') setwd('D:\\Project\\git\\Predict-KBO-OPS\\src')
if(Sys.getenv('USERNAME') == 'kyucheol') setwd('C:\\Users\\kyucheol\\Dropbox\\dacon')

load('cnn_data.Rdata')

#### Data dimension
# dim(train_x) # 174 504
# dim(train_y) # 174 3
# dim(test_x) # 201 504
# dim(test_y) # 201 3

#### Ensemble model

#### CNN model

CNN_ensemble <- function(train_x, train_y, test_x, test_y, n_filter, n_epoch, prob, learning_rate){

  if(!requireNamespace("tidyverse")) install.packages("tidyverse"); library(tidyverse)
  if(!requireNamespace("tensorflow")) install.packages("tensorflow"); library(tensorflow)
  
  X <- tf$placeholder(dtype = tf$float32, shape = list(NULL, 504))
  
  X_reshaped <- tf$reshape(X, list(-1L, 36L, 14L, 1L))
  
  AB <- tf$placeholder(dtype = tf$float32, shape = list(NULL, 1L))
  
  OPS <- tf$placeholder(dtype = tf$float32, shape = list(NULL, 1L))
  
  keep_prob <- tf$placeholder(tf$float32)
  
  # 1 model
  
  W1_1 <- tf$Variable(tf$random_normal(list(3L, 14L, 1L, n_filter), stddev = 0.01))
  L1_1 <- tf$nn$conv2d(X_reshaped, W1_1, strides = list(1L, 3L, 1L, 1L), padding = 'VALID')
  L1_2 <- tf$nn$relu(L1_1)
  L1_3 <- tf$nn$max_pool(L1_2, ksize = list(1L, 12L, 1L, 1L), strides = list(1L, 1L, 1L, 1L), padding = "VALID")
  L1_4 <- tf$reshape(L1_3, list(-1L, n_filter))
  L1_5 <- tf$nn$dropout(L1_4, keep_prob = keep_prob)
  
  W1_2 <- tf$Variable(tf$random_normal(list(n_filter, 1L), stddev = 0.01))
  b_1 <- tf$Variable(tf$random_normal(list(1L)))
  
  pred_OPS_1 <- tf$matmul(L1_5, W1_2) + b_1
  
  # 2 model
  
  W2_1 <- tf$Variable(tf$random_normal(list(2L, 14L, 1L, n_filter), stddev = 0.01))
  L2_1 <- tf$nn$conv2d(X_reshaped, W2_1, strides = list(1L, 2L, 1L, 1L), padding = 'VALID')
  L2_2 <- tf$nn$relu(L2_1)
  L2_3 <- tf$nn$max_pool(L2_2, ksize = list(1L, 18L, 1L, 1L), strides = list(1L, 1L, 1L, 1L), padding = "VALID")
  L2_4 <- tf$reshape(L2_3, list(-1L, n_filter))
  L2_5 <- tf$nn$dropout(L2_4, keep_prob = keep_prob)
  
  W2_2 <- tf$Variable(tf$random_normal(list(n_filter, 1L), stddev = 0.01))
  b_2 <- tf$Variable(tf$random_normal(list(1L)))
  
  pred_OPS_2 <- tf$matmul(L2_5, W2_2) + b_2
  
  # 3 model
  
  W3_1 <- tf$Variable(tf$random_normal(list(1L, 14L, 1L, n_filter), stddev = 0.01))
  L3_1 <- tf$nn$conv2d(X_reshaped, W3_1, strides = list(1L, 1L, 1L, 1L), padding = 'VALID')
  L3_2 <- tf$nn$relu(L3_1)
  L3_3 <- tf$nn$max_pool(L3_2, ksize = list(1L, 36L, 1L, 1L), strides = list(1L, 1L, 1L, 1L), padding = "VALID")
  L3_4 <- tf$reshape(L3_3, list(-1L, n_filter))
  L3_5 <- tf$nn$dropout(L3_4, keep_prob = keep_prob)
  
  W3_2 <- tf$Variable(tf$random_normal(list(n_filter, 1L), stddev = 0.01))
  b_3 <- tf$Variable(tf$random_normal(list(1L)))
  
  pred_OPS_3 <- tf$matmul(L3_5, W3_2) + b_3
  
  # w1 <- tf$Variable(tf$random_normal(list(1L)))
  # w2 <- tf$Variable(tf$random_normal(list(1L)))
  # w3 <- tf$Variable(tf$random_normal(list(1L)))
  
  # hypothesis <- w1*pred_OPS_1 + w2*pred_OPS_2 + w3*pred_OPS_3
  
  hypothesis <- 0.5*pred_OPS_1 + 0.3*pred_OPS_2 + 0.2*pred_OPS_3
  
  cost <- tf$sqrt(tf$truediv(tf$reduce_sum(tf$multiply(tf$square(hypothesis - OPS), AB)), tf$reduce_sum(AB)))
  
  optimizer <- tf$train$AdamOptimizer(learning_rate = learning_rate)$minimize(cost)
  
  #### Session 
  
  sess <- tf$Session()
  
  #### Initialize variable
  
  sess$run(tf$global_variables_initializer())
  
  for(epoch in seq_len(n_epoch)){
    
    dict_value <- dict(X = train_x, OPS = data.matrix(train_y[,3]), AB = data.matrix(train_y[,2]), keep_prob = prob)
    accuracy <- sess$run(list(cost, optimizer, hypothesis), feed_dict = dict_value)
    
    total_cost <- accuracy[[1]]
    txt <- sprintf("%s Epoch: %d. Average cost %f ", date(), epoch, total_cost)
    print(txt)
    
  }
  
  PRED_OPS <- sess$run(hypothesis, feed_dict = dict(X = test_x, keep_prob = 9.0))
  
  TEST_WRMSE <- sess$run(cost, feed_dict = dict(X = test_x, OPS = data.matrix(test_y[,3]), AB = data.matrix(test_y[,2]), keep_prob = 1.0))
  cat( "TEST WRMSE:", TEST_WRMSE)
  
  sess$close()
  
  return(list(PRED_OPS = PRED_OPS, TEST_WRMSE = TEST_WRMSE))
}

fit_ensemble <- CNN_ensemble(train_x, train_y, test_x, test_y, n_filter = 6L, n_epoch = 2000, prob =  0.7, learning_rate =  0.001)

fit_ensemble$TEST_WRMSE # 0.1364032

# cnn_ensemble_result <- cbind(fit_ensemble$PRED_OPS, test_y)
# save(cnn_ensemble_result, file = 'cnn_ensemble_result.Rdata')
