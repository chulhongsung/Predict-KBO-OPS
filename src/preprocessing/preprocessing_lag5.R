rm(list = ls())
gc(reset = TRUE)

if(Sys.getenv('USERNAME') == 'UOS') setwd('C:\\Users\\UOS\\Desktop\\dacon')
if(Sys.getenv('USERNAME') == 'moon') setwd('D:\\Project\\git\\Predict-KBO-OPS\\src')
if(Sys.getenv('USERNAME') == 'kyucheol') setwd('C:\\Users\\kyucheol\\Dropbox\\dacon')

if(!require(tidyverse)) install.packages('tidyverse'); require(tidyverse)
if(!require(data.table)) install.packages('data.table'); require(data.table)

options(warn = -1, tibble.width = Inf)

# pre_tmp = fread('data/Pre_Season_Batter.csv', encoding = 'UTF-8') %>% as.tibble
rsb_tmp = fread('data/Regular_Season_Batter.csv', encoding = 'UTF-8') %>% as.tibble
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

#### Lag 5 데이터 만들기.
rsb <- rsb_tmp %>% select(1,2,3,5:20)

rsb_lag5 <- rsb %>%
  group_by(batter_id, batter_name) %>%
  mutate_at(vars(colnames(.)[c(4:19)]), funs(lag1 = lag(., 1), lag2 = lag(., 2), lag3 = lag(., 3), lag4 = lag(., 4), lag5 = lag(., 5))) %>% 
  ungroup()

rsb_lag5 <- rsb_lag5 %>%
  mutate(t_year = year + 1) # year + 1 로 타겟이 되는 연도 변수 생성

rsb_fh_OPS <- rsb_fh_y %>% select(1,2,3,6,7)

colnames(rsb_fh_OPS) <- c('batter_id', 'batter_name', 'year', 't_AB', 't_OPS')

dataset <- rsb_lag5 %>% inner_join(rsb_fh_OPS, by = c('batter_id', 'batter_name', 't_year' = 'year')) # 타겟 변수인 OPS, AB 붙이기

dataset <- dataset %>% select(1,4:99,101,102)

dataset <- apply(dataset, 2, function(x) ifelse(is.na(x), 0, x)) %>% as.tibble() # NA를 0으로 대체.

#### Train, Test 7:3 비율로 나누기
set.seed(526)

inx <- sample(nrow(dataset), 7/10*nrow(dataset))

lag5_train_data <- dataset[inx,]
lag5_test_data <- dataset[-inx,]

save(lag5_train_data, lag5_test_data, file = 'lag5_data.Rdata')
