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
require(dplyr)
require(randomForest)

load('Rdata/data.Rdata')

rf_model = randomForest(x = train_x, y = train_y$OPS, ntree = 5000)
rf_model_predicted = predict(rf_model, test_x)

mean((train_y$OPS - rf_model$predicted)^2) # MSE
sqrt(sum((train_y$OPS - rf_model$predicted)^2 * train_y$AB )/sum(train_y$AB)) # WRMSE

mean((test_y$OPS - rf_model_predicted)^2) # test MSE
sqrt(sum((test_y$OPS - rf_model_predicted)^2 * test_y$AB )/sum(test_y$AB)) # test WRMSE (0.1477631)


