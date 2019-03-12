##############################################
############# Random Forest ##################
##############################################

rm(list = ls())
gc(reset = T)

if(Sys.getenv('USERNAME') == 'UOS') setwd('C:\\Users\\UOS\\Desktop\\dacon')
if(Sys.getenv('USERNAME') == 'moon') setwd('D:\\Project\\git\\Predict-KBO-OPS\\src')
if(Sys.getenv('USERNAME') == 'kyucheol') setwd('C:\\Users\\kyucheol\\Dropbox\\dacon')

if(!require(dplyr)) install.packages('dplyr')
if(!require(randomForest)) install.packages('randomForest')
if(!require(e1071)) install.packages('e1071')
require(dplyr)
require(randomForest)
require(e1071)

load('Rdata/data.Rdata')

### Random forest
rf_model = randomForest(x = train_x, y = train_y$OPS, ntree = 5000)
rf_model_predicted = predict(rf_model, test_x)

# train loss
mean((train_y$OPS - rf_model$predicted)^2) # MSE (0.0466)
sqrt(sum((train_y$OPS - rf_model$predicted)^2 * train_y$AB )/sum(train_y$AB)) # WRMSE (0.1528)

# test loss
mean((test_y$OPS - rf_model_predicted)^2) # test MSE (0.0437)
sqrt(sum((test_y$OPS - rf_model_predicted)^2 * test_y$AB )/sum(test_y$AB)) # test WRMSE (0.1477631)

### SVM 
svm_model = svm(x = train_x, y = train_y$OPS)
svm_model_predicted = predict(svm_model, test_x)

# train loss
mean((train_y$OPS - svm_model$fitted)^2) # MSE (0.034)
sqrt(sum((train_y$OPS - svm_model$fitted)^2 * train_y$AB )/sum(train_y$AB)) # WRMSE (0.0805)

# test loss
mean((test_y$OPS - svm_model_predicted)^2) # test MSE (0.044)
sqrt(sum((test_y$OPS - svm_model_predicted)^2 * test_y$AB )/sum(test_y$AB)) # test WRMSE (0.11670373)







