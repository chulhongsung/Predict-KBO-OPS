
rm(list = ls())
gc(reset = TRUE)

if(Sys.getenv('USERNAME') == 'UOS') setwd('C:\\Users\\UOS\\Desktop\\dacon')
if(Sys.getenv('USERNAME') == 'moon') setwd('D:\\Project\\git\\Predict-KBO-OPS\\src')
if(Sys.getenv('USERNAME') == 'kyucheol') setwd('C:\\Users\\kyucheol\\Dropbox\\dacon')

if(!require(tidyverse)) install.packages('tidyverse'); require(tidyverse)
if(!require(data.table)) install.packages('data.table'); require(data.table)
if(!require(corrplot)) install.packages('corrplot'); require(corrplot)

options(warn = -1, tibble.width = Inf)

rsb_tmp = fread('data/Regular_Season_Batter.csv', encoding = 'UTF-8') %>% as.tibble

rsb <- rsb_tmp %>% select(batter_id:year,avg:GDP)

colnames(rsb) <- c(colnames(rsb)[1:8], 'TWB', 'THB', colnames(rsb)[11:length(rsb)])

rsb2 <- rsb %>% mutate(submem = if_else(AB < 10, 1, 0))

cm <- cor(rsb2[,4:20])

corrplot(cm, method = 'circle', type = 'upper')

colnames(rsb)

rsb <- rsb %>% mutate(ONB = H - TWB - THB - HR) %>% select(batter_id:H, ONB, TWB:GDP, -TB)

colnames(rsb)

rsb_r <- rsb %>% mutate_at(vars('ONB':'HR'), funs(R = ./H))

new_col <- c('ONB_R', 'TWB_R', 'THB_R', 'HR_R')

density_plot <- function(){
  for(i in new_col){
    print(ggplot(data = rsb_r, aes_string(i)) + geom_density(stat = 'density') + xlab(i))
  }
}

density_plot()

log_density_plot <- function(){
  for(i in new_col){
    print(ggplot(data = rsb_r, aes_string(i)) + geom_density(stat = 'density') + xlab(i) + scale_x_continuous(trans='log10'))
  }
}

log_density_plot()

colnames(rsb_r)

rsb_re <- rsb_r %>% select(batter_id:H, ONB_R, TWB_R, THB_R, HR_R, RBI:GDP)

colnames(rsb_re)
length(rsb_re)

cm <- cor(rsb_re[,4:19])

corrplot(cm, method = 'circle', type = 'upper')

dbd_tmp = fread('data/Regular_Season_Batter_Day_by_Day.csv', encoding = 'UTF-8') %>% as.tibble

#### 전반기, 후반기 나누기.
dbd = dbd_tmp %>%
  mutate(`1B` = H - `2B` - `3B` - `HR`) %>%
  select(batter_id, batter_name, year, date, opposing_team, avg1, AB, R, H, `1B`, everything())

criterion = 7.17 # 전반기, 후반기 기준 날짜.

dbd_fh = dbd %>% filter(date <= criterion) # 전반기 데이터

#### 전반기 OPS 계산하기
rsb_fh_y = dbd_fh %>% group_by(batter_id, batter_name, year) %>%
  summarise(OBP = (sum(H) + sum(BB) + sum(HBP))/(sum(AB) + sum(BB) + sum(HBP)), SLG = (sum(`1B`) + sum(`2B`)*2 + sum(`3B`)*3 + sum(HR)*4)/sum(AB), AB = sum(AB)) %>%
  mutate(OPS = OBP + SLG) %>% ungroup()


#### Lag 3 데이터 만들기.

rsb_re_lag3 <- rsb_re %>%
  group_by(batter_id, batter_name) %>%
  mutate_at(vars('avg':'GDP'), funs(lag1 = lag(., 1), lag2 = lag(., 2), lag3 = lag(., 3))) %>% 
  ungroup()

rsb_re_lag3 <- rsb_re_lag3 %>%
  mutate(t_year = year + 1) # year + 1 로 타겟이 되는 연도 변수 생성

rsb_fh_OPS <- rsb_fh_y %>% select(-OBP, -SLG)

colnames(rsb_fh_OPS) <- c('batter_id', 'batter_name', 'year', 't_AB', 't_OPS')

dataset <- rsb_re_lag3 %>% inner_join(rsb_fh_OPS, by = c('batter_id', 'batter_name', 't_year' = 'year')) # 타겟 변수인 OPS, AB 붙이기

dataset <- dataset %>% select(-batter_name, -year, -t_year)

dataset <- apply(dataset, 2, function(x) ifelse(is.nan(x)|is.na(x), 0, x)) %>% as.tibble() # NA를 0으로 대체.

dataset <- dataset %>% mutate(inx = row_number(), submem = if_else(AB < 10, 0, 1))

#### Stratified Sampling Method 

#### submem에 따라 고정된 비율로 sampling

set.seed(526)

lag3_train_data <- dataset %>% 
  group_by(submem) %>% 
  sample_frac(0.7) %>% 
  ungroup()

train_inx <- lag3_train_data$inx

lag3_test_data <- dataset[-train_inx,]

lag3_train_data <- lag3_train_data %>% select(-inx)

lag3_test_data <- lag3_test_data %>% select(-inx)

save(lag3_train_data, lag3_test_data, file = 'lag3_data.Rdata')

#### Lag 5 데이터 만들기.

rsb_re_lag5 <- rsb_re %>%
  group_by(batter_id, batter_name) %>%
  mutate_at(vars('avg':'GDP'), funs(lag1 = lag(., 1), lag2 = lag(., 2), lag3 = lag(., 3), lag4 = lag(., 4), lag5 = lag(., 5))) %>% 
  ungroup()

rsb_re_lag5 <- rsb_re_lag5 %>%
  mutate(t_year = year + 1) # year + 1 로 타겟이 되는 연도 변수 생성

dataset2 <- rsb_re_lag5 %>% inner_join(rsb_fh_OPS, by = c('batter_id', 'batter_name', 't_year' = 'year')) # 타겟 변수인 OPS, AB 붙이기

dataset2 <- dataset2 %>% select(-batter_name, -year, -t_year)

dataset2 <- apply(dataset2, 2, function(x) ifelse(is.nan(x)|is.na(x), 0, x)) %>% as.tibble() # NA를 0으로 대체.

dataset2 <- dataset2 %>% mutate(inx = row_number(), submem = if_else(AB < 10, 0, 1))

#### Stratified Sampling Method 

#### submem에 따라 고정된 비율로 sampling

set.seed(1)

lag5_train_data <- dataset2 %>% 
  group_by(submem) %>% 
  sample_frac(0.7) %>% 
  ungroup()

train_inx <- lag5_train_data$inx

lag5_test_data <- dataset2[-train_inx,]

lag5_train_data <- lag5_train_data %>% select(-inx)

lag5_test_data <- lag5_test_data %>% select(-inx)

save(lag5_train_data, lag5_test_data, file = 'lag5_data.Rdata')

colnames(rsb)

rsb_div_AB <- rsb %>% mutate_at(vars('H':'GDP'), funs(R = ./AB)) 
rsb_div_AB <- rsb_div_AB %>% select(batter_id:R, H_R:GDP_R)

#### Lag 4 데이터 만들기.

rsb_div_AB_lag4 <- rsb_div_AB %>%
  group_by(batter_id, batter_name) %>%
  mutate_at(vars('H_R':'GDP_R'), funs(lag1 = lag(., 1), lag2 = lag(., 2), lag3 = lag(., 3), lag4 = lag(., 4))) %>% 
  ungroup()

rsb_div_AB_lag4 <- rsb_div_AB_lag4 %>%
  mutate(t_year = year + 1) # year + 1 로 타겟이 되는 연도 변수 생성

dataset3 <- rsb_div_AB_lag4 %>% inner_join(rsb_fh_OPS, by = c('batter_id', 'batter_name', 't_year' = 'year'))

dataset3 <- apply(dataset3, 2, function(x) ifelse(is.nan(x)|is.na(x), 0, x)) %>% as.tibble() # NA를 0으로 대체.

dataset3 <- dataset3 %>% mutate(st_mem = if_else(AB < 10, 0, 1),inx = row_number(), nonz = apply(dataset3 %>% select(avg:GDP_R_lag4), 1 , function(x) sum(x != 0)))

dataset3 <-  # 타겟 변수인 OPS, AB 붙이기

dataset3 <- dataset3 %>% select(-batter_name, -year, -t_year)
                                                                                                 
dataset3 <- dataset3 %>% select(batter_id:GDP_R_lag4, inx, st_mem, nonz, t_AB, t_OPS)                                                                                                 
                                                                                                 
#### Stratified Sampling Method 

#### submem에 따라 고정된 비율로 sampling

set.seed(1)

lag4_train_data <- dataset3 %>% 
  group_by(st_mem) %>% 
  sample_frac(0.7) %>% 
  ungroup()

train_inx <- lag4_train_data$inx

lag4_test_data <- dataset3[-train_inx,]

lag4_train_data <- lag4_train_data %>% select(-inx)

lag4_test_data <- lag4_test_data %>% select(-inx)
                                                                                                 
lag4_train_data <- lag4_train_data %>% mutate_all(as.numeric) %>% filter((batter_id != 45) & (batter_id != 24))
lag4_test_data <- lag4_test_data %>% mutate_all(as.numeric) %>% filter((batter_id != 45) & (batter_id != 24))   

save(lag4_train_data, lag4_test_data, file = 'lag4_AB_data.Rdata')

rsb_div_AB <- rsb %>% mutate_at(vars('H':'GDP'), funs(R = ./AB)) 
rsb_div_AB <- rsb_div_AB %>% select(batter_id:R, H_R:GDP_R)

#### Lag 5 데이터 만들기.

rsb_div_AB_lag5 <- rsb_div_AB %>%
  group_by(batter_id, batter_name) %>%
  mutate_at(vars('H_R':'GDP_R'), funs(lag1 = lag(., 1), lag2 = lag(., 2), lag3 = lag(., 3), lag4 = lag(., 4), lag5 = lag(., 5))) %>% 
  ungroup()

rsb_div_AB_lag5 <- rsb_div_AB_lag5 %>%
  mutate(t_year = year + 1) # year + 1 로 타겟이 되는 연도 변수 생성

dataset3 <- rsb_div_AB_lag5 %>% inner_join(rsb_fh_OPS, by = c('batter_id', 'batter_name', 't_year' = 'year'))

dataset3 <- apply(dataset3, 2, function(x) ifelse(is.nan(x)|is.na(x), 0, x)) %>% as.tibble() # NA를 0으로 대체.

dataset3 <- dataset3 %>% mutate(st_mem = if_else(AB < 10, 0, 1),inx = row_number(), nonz = apply(dataset3 %>% select(avg:GDP_R_lag5), 1 , function(x) sum(x != 0)))

dataset3 <- dataset3 %>% select(-batter_name, -year, -t_year)
                                                                                                 
dataset3 <- dataset3 %>% select(batter_id:GDP_R_lag5, inx, st_mem, nonz, t_AB, t_OPS)                                                                                                 
                                                                                                 
#### Stratified Sampling Method 

#### submem에 따라 고정된 비율로 sampling

set.seed(1)

lag5_train_data <- dataset3 %>% 
  group_by(st_mem) %>% 
  sample_frac(0.7) %>% 
  ungroup()

train_inx <- lag5_train_data$inx

lag5_test_data <- dataset3[-train_inx,]

lag5_train_data <- lag5_train_data %>% select(-inx)

lag5_test_data <- lag5_test_data %>% select(-inx)

lag5_train_data <- lag5_train_data %>% mutate_all(as.numeric) %>% filter((batter_id != 45) & (batter_id != 24))
lag5_test_data <- lag5_test_data %>% mutate_all(as.numeric) %>% filter((batter_id != 45) & (batter_id != 24))
                                                                                                 
save(lag5_train_data, lag5_test_data, file = 'lag5_AB_data.Rdata')

setwd('C:\\Users\\UOS\\Desktop\\dacon\\data')

if(!require(data.table)) install.packages('data.table'); library(data.table)

submission <- fread('submission.csv', header = T, encoding = 'UTF-8') %>% as_tibble()

submission_id <- submission %>% select(batter_id) %>% pull()

submission_data <- rsb_div_AB_lag5 %>%
  filter((year == 2018) & (batter_id %in% submission_id)) 

submission_data <- apply(submission_data, 2, function(x) ifelse(is.nan(x)|is.na(x)|is.infinite(x), 0, x)) %>% as.tibble() 

submission_data <- submission_data %>% mutate(st_mem = if_else(AB < 10, 0, 1), nonz = apply(submission_data %>% select(avg:GDP_R_lag5), 1 , function(x) sum(x != 0)))

submission_data <- submission_data %>% select(-batter_name, -year)
                                                                                                 
submission_data <- submission_data %>% select(batter_id:GDP_R_lag5, st_mem, nonz)
                                                                                                               

save(submission_data, file = 'submission_data.Rdata')
