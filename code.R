##############################################
################### Dacon ####################
##############################################

rm(list = ls())
gc(reset = T)

tensorflow::use_python("C:\\ProgramData\\Anaconda3\\python.exe")

if(!require(data.table)) install.packages('data.table')
if(!require(dplyr)) install.packages('dplyr')
if(!require(tensorflow)) install.packages("tensorflow")

require(data.table)
require(dplyr)
require(tensorflow)

setwd('C:\\Users\\moon\\Desktop\\dacon')
list.files('data')

pre_season = fread('data/Pre_Season_Batter.csv', encoding = 'UTF-8')
regular_season = fread('data/Regular_Season_Batter.csv', encoding = 'UTF-8')
regular_day = fread('data/Regular_Season_Batter_Day_by_Day.csv', encoding = 'UTF-8')
submission = fread('data/submission.csv', encoding = 'UTF-8')

regular_season[is.na(regular_season)] = 0 ## Eliminate NA
target_batter_id = submission$batter_id

pre_season = pre_season %>% filter(batter_id %in% target_batter_id)
regular_season = regular_season %>% filter(batter_id  %in% target_batter_id)
regular_day = regular_day %>% filter(batter_id %in% target_batter_id)

### predict
input_dim = 13L
output_dim = 1L

data_scale_function = function(x)
{
  x_return = x %>% select(AB:GDP) 
  x_return_scaled = x_return %>% scale() %>% as.matrix()
  attr(x_return_scaled, 'scaled:center') = NULL
  attr(x_return_scaled, 'scaled:scale') = NULL
  x_return_scaled[is.nan(x_return_scaled)] = x_return[is.nan(x_return_scaled)]
  return(x_return_scaled)
}

opt_prediction_model = function(batter_inx, hidden_dim1, hidden_dim2, max_iter = 1000L, learning_rate = 0.005)
{
  tmp_regular_day = regular_day %>% filter(batter_id == batter_inx) %>% arrange(year, date)
  tmp_regular_season = regular_season %>% filter(batter_id == batter_inx)
  
  tmp_regular_day_list = lapply(tmp_regular_season$year, function(i) 
    tmp_regular_day %>% filter(year == i))
  tmp_regular_day_list_name = lapply(tmp_regular_day_list, function(x) 
    x %>% select(year) %>% unique() %>% as.numeric()) %>% unlist()
  
  names(tmp_regular_day_list) = tmp_regular_day_list_name
  tmp_regular_day_list = tmp_regular_day_list[!is.na(tmp_regular_day_list_name)]
  
  if(all(names(tmp_regular_day_list) == tmp_regular_season$year) & length(tmp_regular_day_list) > 2) 
  {
    tmp_regular_day_list = lapply(tmp_regular_day_list, data_scale_function)
    # print('No Problem!')
  }else{
    # print('Error in Batter Number: ', batter_inx)
    return(list(predicted_test_y = NaN, true_y = NaN))
  }
  
  input_x = tf$placeholder(tf$float32, shape(NULL, input_dim))
  input_y = tf$placeholder(tf$float32, shape())
  output_y = tf$placeholder(tf$float32, shape())
  
  hidden_theta1 = tf$Variable(tf$random_normal(shape(input_dim, hidden_dim1)))
  hidden_bias1 = tf$Variable(tf$random_normal(shape(hidden_dim1)))
  hidden_layer1 = tf$nn$relu(tf$matmul(input_x, hidden_theta1) + hidden_bias1)
  
  hidden_theta2 = tf$Variable(tf$random_normal(shape(hidden_dim1, hidden_dim2)))
  hidden_bias2 = tf$Variable(tf$random_normal(shape(hidden_dim2)))
  hidden_layer2 = tf$reshape(tf$reduce_mean(tf$matmul(hidden_layer1, hidden_theta2) + hidden_bias2, axis = 0L), shape(hidden_dim2, 1L))
  
  hidden_theta3 = tf$Variable(tf$random_normal(shape(1L, hidden_dim2)))
  hidden_bias3 = tf$Variable(tf$random_normal(shape()))
  hidden_theta3_y = tf$Variable(tf$random_normal(shape()))
  
  predicted_y =  tf$matmul(hidden_theta3, hidden_layer2) + hidden_theta3_y * input_y + hidden_bias3
  objective_fun = tf$reduce_sum((output_y - predicted_y)^2)
  
  train_opt = tf$train$AdamOptimizer(learning_rate = learning_rate)$minimize(objective_fun)
  
  sess = tf$Session()
  sess$run(tf$global_variables_initializer())
  
  max_dict_iter = nrow(tmp_regular_season) - 2
  for(iter in 1:max_iter)
  {
    for(dict_iter in 1:max_dict_iter)
    {
      sess$run(train_opt, feed_dict = dict(input_x = tmp_regular_day_list[[dict_iter]],
                                           input_y = tmp_regular_season$OPS[dict_iter],
                                           output_y = tmp_regular_season$OPS[dict_iter + 1]))
    }
  }
  predicted_test_y = sess$run(predicted_y, feed_dict = dict(input_x = tmp_regular_day_list[[max_dict_iter + 1]],
                                                            input_y = tmp_regular_season$OPS[max_dict_iter + 1]))
  true_y = tmp_regular_season$OPS[max_dict_iter + 2]
  sess$close()
  
  return(list(predicted_test_y = predicted_test_y, true_y = true_y))
}

predicted_ops_list = vector('list', length(target_batter_id))
hidden_dim1 = hidden_dim2 = 5L
for(inx in 1:length(target_batter_id))
{
  predicted_ops_list_tmp = opt_prediction_model(batter_inx = target_batter_id[inx], 
                                                hidden_dim1 = hidden_dim1, 
                                                hidden_dim2 = hidden_dim2)
  predicted_ops_list[[inx]] = predicted_ops_list_tmp
  cat(inx, 'complted ! \n')
}

save.image('tmp_result.Rdata')
predicted_ops_vec = lapply(predicted_ops_list, function(x) x$predicted_test_y) %>% unlist()
predicted_ops_vec[predicted_ops_vec <= 0] = 0
predicted_ops_vec[predicted_ops_vec >= 5] = 5

true_ops_vec = lapply(predicted_ops_list, function(x) x$true_y) %>% unlist()

sum((true_ops_vec - predicted_ops_vec)^2, na.rm = T) %>% sqrt()


