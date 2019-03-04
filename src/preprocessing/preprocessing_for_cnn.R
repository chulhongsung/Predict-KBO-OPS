rm(list = ls())
gc(reset = TRUE)

if(!require(tidyverse)) install.packages('tidyverse'); require(tidyverse)
if(!require(data.table)) install.packages('data.table'); require(data.table)
if(!require(tensorflow)) install.packages('tensorflow'); require(tensorflow)

options(warn = -1, tibble.width = Inf)

setwd('C:\\Users\\UOS\\Desktop\\dacon')

pre_tmp <- fread('data/Pre_Season_Batter.csv', encoding = 'UTF-8') %>% as.tibble
rsb <- fread('data/Regular_Season_Batter.csv', encoding = 'UTF-8') %>% as.tibble
dbd_tmp <- fread('data/Regular_Season_Batter_Day_by_Day.csv', encoding = 'UTF-8') %>% as.tibble

#### 프리시즌 데이터 

pre <- pre_tmp %>% 
  mutate(`1B` = H - `2B` - `3B` - `HR`) %>% 
  select(c(1,2,3,7:9,30,10:20), -TB)

#### 전반기, 후반기 나누기.

dbd <- dbd_tmp %>%
  mutate(`1B` = H - `2B` - `3B` - `HR`) %>% 
  select(batter_id, batter_name, year, date, opposing_team, avg1, AB, R, H, `1B`, everything())

criterion <- 7.17 # 전반기, 후반기 기준 날짜.

dbd_fh <- dbd %>% filter(date <= criterion) # 전반기 데이터

dbd_sh <- dbd %>% filter(date > criterion) # 후반기 데이터

#### 전반기, 후반기 데이터 합치기.

rsb_fh_x <- dbd_fh %>% group_by(batter_id, batter_name, year) %>% summarise_at(colnames(.)[c(7:20)], funs(sum)) %>% ungroup()

rsb_fh_y <- dbd_fh %>% group_by(batter_id, batter_name, year) %>% 
  summarise(OBP = (sum(H) + sum(BB) + sum(HBP))/(sum(AB) + sum(BB) + sum(HBP)), SLG = (sum(`1B`) + sum(`2B`)*2 + sum(`3B`)*3 + sum(HR)*4)/sum(AB)) %>% 
  mutate(OPS = OBP + SLG) %>% ungroup()

rsb_sh_x <- dbd_sh %>% group_by(batter_id, batter_name, year) %>% summarise_at(colnames(.)[7:20], funs(sum)) %>% ungroup()

rsb_sh_y <- dbd_sh %>% group_by(batter_id, batter_name, year) %>% 
  summarise(OBP = (sum(H) + sum(BB) + sum(HBP))/(sum(AB) + sum(BB) + sum(HBP)), SLG = (sum(`1B`) + sum(`2B`)*2 + sum(`3B`)*3 + sum(HR)*4)/sum(AB)) %>% 
  mutate(OPS = OBP + SLG) %>% ungroup()

#### train 데이터 만들기.

batter_id_2017 <- rsb_fh_x %>% filter(year == 2017) %>% distinct(batter_id) %>% pull(batter_id) 

# season은 'pre:프리시즌', 'fh:전반기', 'sh:후반기'을 나타내는 변수.
train_x_fh <- rsb_fh_x %>% filter((batter_id %in% batter_id_2017) & (year %in% 2005:2016)) %>% mutate(season = 'fh')

train_x_sh <- rsb_sh_x %>% filter((batter_id %in% batter_id_2017) & (year %in% 2005:2016)) %>% mutate(season = 'sh')

train_x_pre <- pre %>% filter((batter_id %in% batter_id_2017) & (year %in% 2005:2016)) %>% mutate(season = 'pre')

tmp_train_x <- train_x_fh %>% bind_rows(train_x_sh, train_x_pre)

tmp_train_frame <- tibble(batter_id = rep(batter_id_2017, each = 36),
                year = rep(rep(2005:2016, each = 3),length(batter_id_2017)),
                season = rep(rep(c('pre', 'fh', 'sh'), 12), length(batter_id_2017)))

train_x <- tmp_train_frame %>% 
  left_join(tmp_train_x, by = c('batter_id', 'year', 'season')) %>% 
  select(-c(1:4))

# colnames(train_x)
#  [1] "AB"  "R"   "H"   "1B"  "2B"  "3B"  "HR"  "RBI" "SB"  "CS"  "BB"  "HBP" "SO"  "GDP"

train_x <- apply(train_x, 2, function(x) ifelse(is.na(x), 0, x)) %>% as.tibble() # NA를 0으로 대체.

# 아래의 코드는 이전 코드에서 for loop을 통한 선수들의 년도별로 나누어진 데이터를 하나의 행으로 만드는 코드를 수정.
train_mat <- matrix(c(t(as.matrix(train_x))), nrow = 174, byrow = TRUE) 

train_y <- rsb_fh_y %>% filter((batter_id %in% batter_id_2017) & (year == 2017)) %>% select(batter_id, OPS) %>% pull(OPS)

#### test 데이터 만들기.

batter_id_2018 <- rsb_fh_x %>% filter(year == 2018) %>% distinct(batter_id) %>% pull(batter_id) 

test_x_fh <- rsb_fh_x %>% filter((batter_id %in% batter_id_2018) & (year %in% 2006:2017)) %>% mutate(season = 'fh')

test_x_sh <- rsb_sh_x %>% filter((batter_id %in% batter_id_2018) & (year %in% 2006:2017)) %>% mutate(season = 'sh')

test_x_pre <- pre %>% filter((batter_id %in% batter_id_2018) & (year %in% 2006:2017)) %>% mutate(season = 'pre')

tmp_test_x <- test_x_fh %>% bind_rows(test_x_sh, test_x_pre)

tmp_test_frame <- tibble(batter_id = rep(batter_id_2018, each = 36),
                      year = rep(rep(2006:2017, each = 3),length(batter_id_2018)),
                      season = rep(rep(c('pre', 'fh', 'sh'), 12), length(batter_id_2018)))

test_x <- tmp_test_frame %>% 
  left_join(tmp_test_x, by = c('batter_id', 'year', 'season')) %>% 
  select(-c(1:4))

test_x <- apply(test_x, 2, function(x) ifelse(is.na(x), 0, x)) %>% as.tibble() # NA를 0으로 대체.

# 아래의 코드는 이전 코드에서 for loop을 통한 선수들의 년도별로 나누어진 데이터를 하나의 행으로 만드는 코드를 수정.
test_mat <- matrix(c(t(as.matrix(test_x))), nrow = 202, byrow = TRUE) 

test_y <- rsb_fh_y %>% filter((batter_id %in% batter_id_2018) & (year == 2018)) %>% select(batter_id, OPS) %>% pull(OPS)

# NaN인 선수 빼기

train_x <- train_mat
train_y <- train_y
test_x <- test_mat[-182,]
test_y <- test_y

#### 저장하기

save(train_x, train_y, test_x, test_y, file = 'cnn_data.Rdata')
