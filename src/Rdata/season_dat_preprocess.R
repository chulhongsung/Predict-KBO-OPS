###########################################
##### Preprocessing for season_data #######
###########################################

rm(list = ls())
gc(reset = TRUE)

if(Sys.getenv('USERNAME') == 'UOS') setwd('C:\\Users\\UOS\\Desktop\\dacon')
if(Sys.getenv('USERNAME') == 'moon') setwd('D:\\Project\\git\\Predict-KBO-OPS\\src')
if(Sys.getenv('USERNAME') == 'kyucheol') setwd('C:\\Users\\kyucheol\\Dropbox\\dacon')

if(!require(tidyverse)) install.packages('tidyverse'); require(tidyverse)
if(!require(data.table)) install.packages('data.table'); require(data.table)

rsbd <- fread('data/Regular_Season_Batter_Day_by_Day.csv', encoding = 'UTF-8') %>% as_tibble()

# rsbd를 시즌별 rsb파일로 바꾸는 전처리


rsbd_tmp <- rsbd[complete.cases(rsbd),]

date_0 <- rsbd_tmp %>% arrange(year, date) %>% dplyr::select(year, date, everything()) %>% filter((date>=7) & (date < 9)) %>% 
  transmute(year_date = as.numeric(paste(year, date, sep='')))

date_0 <- c(date_0) %>% unlist %>% unique()

date_1 <- c(min(date_0), date_0)

date_2 <- c(date_0, max(date_0))

# 각 년도별 전반기 종료일!
f_season_end <- date_1[which((((date_2 - date_1)>=0.04) & ((date_2 - date_1)< 0.6)) | ((date_2 - date_1) >= 0.9 & (date_2 - date_1) <= 1 ))] 

f_season_end <- f_season_end[-3] # 이거는 2002년에 날짜가 두개나와서 실제 전반기종료일 찾아보니 3번째는 아닌거여서 걍 뺌.

rm(date_0, date_1, date_2)

f_season_end_tmp <- f_season_end - (2001:2018)*10  # 시즌 종료 date.


# season = 1 : 전반기, season = 2 : 후반기
rsbd_tmp <- rsbd_tmp %>% mutate(season = 2) %>% dplyr::select(batter_id, batter_name, year, date, season, everything())

rsbd_list <- vector('list', length(f_season_end_tmp))

for(i in 1:length(f_season_end_tmp)){
  rsbd_list[[i]] <- rsbd_tmp %>% filter(year == 2000+i) %>% mutate(season = ifelse(date <= f_season_end_tmp[i], 1, 2))
} 
rsbd_tmp <- do.call('rbind', rsbd_list) %>% arrange(batter_id)

# 1루타 개수 구하기.
rsbd_tmp <- rsbd_tmp %>% mutate(`1B` = H - `2B` - `3B` - `HR`) %>% 
  dplyr::select(batter_id, batter_name, year, date, season, opposing_team, avg1, AB, R, H, `1B`, everything())

# 여기서 희생플라이(SF)를 알아야 OBP를구할 수 있는데... 자료가 없음. 크롤링 해야하나..근데 자료가없네.
# 시즌별 누적 출루율 장타율 ops
rsbd_tmp_season <- rsbd_tmp %>% group_by(batter_id, year, season) %>% 
  mutate(cum_OBP = (cumsum(H)+cumsum(BB)+cumsum(HBP))/(cumsum(AB)+cumsum(BB)+cumsum(HBP)),  # +cumsum(SF) 분모에 이거 뺌.
         cum_SLG = (cumsum(`1B`)+ cumsum(`2B`)*2 + cumsum(`3B`)*3 + cumsum(HR)*4)/(cumsum(AB)),
         cum_OPS = cum_OBP + cum_SLG)

aaa <- rsbd_tmp_season %>% group_by(batter_id, year, season) %>% summarise(date = max(date)) %>%
  arrange(batter_id, year, season)

# 시즌이 끝날 때의 누적 OPS만 있는 데이터파일. rsb파일과 비슷하니 필요한 열 가져오기.
rsb_season <- rsbd_tmp_season %>% right_join(aaa)

rsb_season <- rsb_season %>% filter(season == 1)

rsb_season_tmp <- rsb_season %>% mutate(TB = AB * cum_SLG) %>% select(c(1:14, 'TB',everything())) %>% as.tibble()

rsb_season_tmp <- rsb_season_tmp %>% select(c(1,3,8:10, 12:22))

rsb_season_tmp <- rsb_season_tmp[complete.cases(rsb_season_tmp),]

rsb_season_tmp <- rsb_season %>% mutate(TB = AB * cum_SLG) %>% select(c(1:14, 'TB',everything())) %>% as.tibble()
rsb_season_tmp <- rsb_season_tmp %>% select(c(1,3,8:10, 12:22))

#### train data
batter_id_2017 <- rsb_season_tmp %>% filter(year == 2017) %>% group_by(batter_id) %>% distinct(batter_id) %>% ungroup()

tmp_train <- rsb_season_tmp %>% filter(batter_id %in% batter_id_2017$batter_id, year %in% 2016) %>% arrange(batter_id)

sum(is.na(tmp_train))

tmp_train <- apply(tmp_train, 2, function(x) ifelse(is.na(x), 0, x)) %>% as.tibble()

tmp_train <- tmp_train %>% arrange(batter_id)

names(tmp_train)[c(6,7)] <- c('B2','B3')

train_season_x <- data.matrix(tmp_train[,-c(1,2)])

tmp_train_season_x <- cbind(batter_id = unique(tmp_train$batter_id), train_season_x) # train_season_x에 batter_id있는 data.


#### train_season_y : 2017년도의 OPS
train_season_y <- rsb_season %>% filter(year == 2017 , batter_id %in% tmp_train$batter_id) %>% as.data.frame() %>% 
  select(batter_id, AB, cum_OPS) %>% arrange(batter_id) %>% as.tibble()

sum(is.na(train_season_y))

#### test data 

batter_id_2018 <- rsb_season_tmp %>% filter(year == 2018) %>% distinct(batter_id) 

tmp_test <- rsb_season_tmp %>% filter(batter_id %in% batter_id_2018$batter_id) %>% filter(year %in% 2017)

sum(is.na(tmp_test))

tmp_test <- tmp_test %>% arrange(batter_id)

names(tmp_test)[c(6,7)] <- c('B2','B3')

test_season_x <- data.matrix(tmp_test[,-c(1,2)])

tmp_test_season_x <- cbind(batter_id = unique(tmp_test$batter_id), test_season_x) # train_season_x에 batter_id있는 data.

#### test_season_y : 2018년도의 OPS
test_season_y <- rsb_season %>% filter(year == 2018 , batter_id %in% tmp_test$batter_id) %>% as.data.frame() %>% 
  select(batter_id, AB, cum_OPS) %>% arrange(batter_id) %>% as.tibble()

sum(is.na(test_season_y))

test_season_y <- apply(test_season_y, 2, function(x) ifelse(is.na(x), 0, x)) %>% as.tibble()

train_season_x <- train_season_x %>% as.tibble()

test_season_x <- test_season_x %>% as.tibble()

save(train_season_x, train_season_y, test_season_x, test_season_y, file = 'season_data.Rdata')


