rm(list = ls())
gc(reset = TRUE)

if(!require(tidyverse)) install.packages('tidyverse')
if(!require(data.table)) install.packages('data.table')
if(!require(tensorflow)) install.packages('tensorflow')  
require(tidyverse)
require(data.table)
require(tensorflow)

options(warn = -1, tibble.width = Inf)

pre <- fread('data/Pre_Season_Batter.csv', encoding = 'UTF-8') %>% as.tibble
rsb <- fread('data/Regular_Season_Batter.csv', encoding = 'UTF-8') %>% as.tibble
dbd <- fread('data/Regular_Season_Batter_Day_by_Day.csv', encoding = 'UTF-8') %>% as.tibble

#### train data

batter_id_2017 <- rsb %>% filter(year == 2017) %>% group_by(batter_id) %>% distinct(batter_id) %>% ungroup()

#### 2005 ~ 2016

tmp_train <- rsb %>% filter(batter_id %in% batter_id_2017$batter_id) %>% filter(year %in% 2005:2016)

dim(tmp_train) # [1] 1156   29

train_y <- rsb %>% filter((year == 2017) & (batter_id %in% tmp_train$batter_id)) %>% select(batter_id, AB, OPS) %>% arrange(batter_id)
train_y <- train_y[,2:3]

tmp_train <- tmp_train %>% select(c(1, 3, 7:20))

tmp_train <- tibble(batter_id = rep(unique(tmp_train$batter_id), each = 12), year = rep(2005:2016, length(unique(tmp_train$batter_id)))) %>% left_join(tmp_train, by = c('batter_id', 'year'))

tmp_train <- apply(tmp_train, 2, function(x) ifelse(is.na(x), 0, x)) %>% as.tibble()

tmp_train <- tmp_train %>% arrange(batter_id)

#### 12 X 14 행렬을 1 X 168 벡터로 만듬

tmp_train <- tmp_train %>% select(AB:GDP)

# 2364 / 12 = 197
train_mat <- matrix(rep(0, 2364 * 14), nrow = 197)

tmp_train <- data.matrix(tmp_train)


for(i in seq_len(197)){
  for(j in seq_len(12))
  train_mat[i, (14*(j-1)+1):(14*j)] <- tmp_train[12*(i-1)+j,]
}

#### OPS NA!

# train_matrix[194,]
# test_data[194,]

#### test data 

batter_id_2018 <- rsb %>% filter(year == 2018) %>% distinct(batter_id) 

#### 2006 ~ 2017

tmp_test <- rsb %>% filter(batter_id %in% batter_id_2018$batter_id) %>% filter(year %in% 2006:2017)

test_y <- rsb %>% filter((year == 2018) & (batter_id %in% tmp_test$batter_id)) %>% select(batter_id, AB, OPS) %>% arrange(batter_id)

test_y <- test_y[,2:3]

tmp_test <- tmp_test %>% select(c(1, 3, 7:20))

tmp_test <- tibble(batter_id = rep(unique(tmp_test$batter_id), each = 12), year = rep(2006:2017, length(unique(tmp_test$batter_id)))) %>% left_join(tmp_test, by = c('batter_id', 'year'))

tmp_test <- apply(tmp_test, 2, function(x) ifelse(is.na(x), 0, x)) %>% as.tibble()

tmp_test <- tmp_test %>% arrange(batter_id)

tmp_test <- tmp_test %>% select(AB:GDP)

test_mat <- matrix(rep(0, 2532 * 14), nrow = 211)

tmp_test <- data.matrix(tmp_test)

for(i in seq_len(211)){
  for(j in seq_len(12))
    test_mat[i, (14*(j-1)+1):(14*j)] <- tmp_test[12*(i-1)+j,]
}


#### TF code

X <- tf$placeholder(dtype = tf$float32, shape = list(NULL, 168))
X_reshaped <- tf$reshape(X, list(-1L, 12L, 14L, 1L))

AB <- tf$placeholder(dtype = tf$float32, shape = list(NULL, 1L))
Y <- tf$placeholder(dtype = tf$float32, shape = list(NULL, 1L))

#### dropout prob

keep_prob <- 0.7

#### Layer 1 

W1 <- tf$Variable(tf$random_normal(list(1L, 14L, 1L, 10L), stddev = 0.01))
L1 <- tf$nn$conv2d(X_reshaped, W1, strides = list(1L, 1L, 1L, 1L), padding = 'VALID')
L2 <- tf$nn$relu(L1)
L3 <- tf$nn$max_pool(L2, ksize = list(1L, 12L, 1L, 1L), strides = list(1L, 1L, 1L, 1L), padding = "VALID")
L4 <- tf$reshape(L3, list(-1L, 10L))
L5 <- tf$nn$dropout(L4, keep_prob = keep_prob)

#### Layer 2

W2 <- tf$Variable(tf$random_normal(list(10L, 1L), stddev = 0.01))
b <- tf$Variable(tf$random_normal(list(1L)))

#### Predict Y

hypothesis <- tf$matmul(L5, W2) + b

#### Define WRMSE

cost <- tf$sqrt(tf$truediv(tf$reduce_sum(tf$multiply(tf$square(hypothesis - Y), AB)), tf$reduce_sum(AB)))

optimizer <- tf$train$AdamOptimizer(learning_rate = 0.001)$minimize(cost)

#### Session 

sess <- tf$Session()

#### Initialize variable

sess$run(tf$global_variables_initializer())

for(epoch in seq_len(3000)){
  
  dict_value <- dict(X = train_mat[-194,], Y = data.matrix(train_y[-194,2]), AB = data.matrix(train_y[-194,1]))
  accuracy <- sess$run(list(cost, optimizer, hypothesis), feed_dict = dict_value)
  
  total_cost <- accuracy[[1]]
  # txt <- sprintf("%s Epoch: %d. Average cost %f ", date(), epoch, total_cost)
  # print(txt)
  
  if(epoch == seq_len(3000)) pred_Y <- accuracy[[3]]
}

#### 191 observation's OPS is NA!
test_pred_Y <- sess$run(hypothesis, feed_dict = dict(X = test_mat[-191,]))

#### Compare with real test OPS 
View(cbind(test_pred_Y, test_y[-191,2]))

#### MSE
mean((test_pred_Y - test_y[-191,2])^2) %>% sqrt()

#### Test WRMSE 
cat( "TEST WRMSE:", sess$run(cost, feed_dict = dict(X = test_mat[-191,], Y = data.matrix(test_y[-191,2]), AB = data.matrix(test_y[-191,1]))))

#### TEST WRMSE: 0.1132773

#### Session close
sess$close()
