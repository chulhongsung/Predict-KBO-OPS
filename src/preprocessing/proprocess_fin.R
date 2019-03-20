rm(list = ls()); gc(reset = T)

if(Sys.getenv('USERNAME') == 'UOS') setwd('C:\\Users\\UOS\\Desktop\\dacon')
if(Sys.getenv('USERNAME') == 'moon') setwd('D:\\Project\\git\\Predict-KBO-OPS\\src')
if(Sys.getenv('USERNAME') == 'kyucheol') setwd('C:\\Users\\kyucheol\\Dropbox\\dacon')

if(!require(tidyverse)) install.packages('tidyverse'); require(tidyverse)
if(!require(data.table)) install.packages('data.table'); require(data.table)

rsbd <- fread('data/Regular_Season_Batter_Day_by_Day.csv', encoding = 'UTF-8') %>% as_tibble()
rsb <- fread('data/Regular_Season_Batter.csv', encoding = 'UTF-8') %>% as_tibble()

# day by day 데이터가 같은날 같은선수가 두번 경기에 임한 경우도 있어서 아래를 꼭 돌려야합니다.
rsbd <- rsbd %>% distinct(batter_id, year, date, .keep_all = T)

rsbd_tmp <- rsbd[complete.cases(rsbd), ]

date_0 <- rsbd_tmp %>% arrange(year, date) %>% dplyr::select(year, date, everything()) %>% filter((date>=7) & (date < 9)) %>% 
  transmute(year_date = as.numeric(paste(year, date, sep='')))

date_0 <- c(date_0) %>% unlist %>% unique()

date_1 <- c(min(date_0), date_0)

date_2 <- c(date_0, max(date_0))

# 각 년도별 전반기 종료일, 주로 날씨에 영향을 받지 않는 한, 최대 2일 쉬기에, 4일이상 경기가 없다면 전반기 끝나는 날이라 생각하자.
f_season_end <- date_1[which((((date_2 - date_1)>=0.04) & ((date_2 - date_1)< 0.6)) | ((date_2 - date_1) >= 0.9 & (date_2 - date_1) <= 1 ))] 
f_season_end <- f_season_end[-3] # 이거는 2002년에 날짜가 두개나와서 실제 전반기종료일 찾아보니 3번째는 아닌여서 뺐다.

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

# 시즌별 누적 출루율 장타율 ops
rsbd_tmp_season <- rsbd_tmp %>% group_by(batter_id, year, season) %>% 
  mutate(AB = cumsum(AB), R = cumsum(R), H = cumsum(H), `1B` = cumsum(`1B`), `2B` = cumsum(`2B`),
         `3B` = cumsum(`3B`), HR = cumsum(HR), RBI = cumsum(RBI), SB = cumsum(SB), CS = cumsum(CS),
         BB = cumsum(BB), HBP = cumsum(HBP), SO = cumsum(SO), GDP = cumsum(GDP),
         OBP = (H+BB+HBP)/(AB+BB+HBP),  # +cumsum(SF) 분모에 이거 뺌.
         SLG = (`1B`+ `2B`*2 + `3B`*3 + HR*4)/(AB),
         OPS = OBP + SLG)

aaa <- rsbd_tmp_season %>% group_by(batter_id, year, season) %>% summarise(date = max(date)) %>%
  arrange(batter_id, year, season)

# 시즌이 끝날 때의 누적 OPS만 있는 데이터파일. rsb파일과 비슷하니 필요한 열 가져오기.
rsb_season <- rsbd_tmp_season %>% right_join(aaa, by = c('batter_id','year','season','date')) %>% ungroup()

# 전반기
rsb_fh <- rsb_season %>% filter(season == 1)

rsb_fh <- rsb_fh %>% mutate(TB = AB * SLG) %>% select(c(1:14, 'TB',everything()))


# 후반기
rsb_sh <- rsb_season %>% filter(season == 2)

rsb_sh <- rsb_sh %>% mutate(TB = AB * SLG) %>% select(c(1:14, 'TB',everything()))

# 전처리 함수
preprocess_fun <- function(dat = rsb_fh, annual = 12){
  ## dat : rsb, rsb_fh, rsb_sh 등 Regular_Season_Batter의 데이터 형식의 파일 (default : rsb_fh(전반기))
  ## annual : 분석하는데 쓰일 연차
  
  dat_tmp <- dat %>% select(c("batter_id","year","AB","R","H","2B","3B","HR","TB","RBI","SB","CS","BB","HBP","SO","GDP"))
  
  #### train data
  batter_id_2017 <- dat_tmp %>% filter(year == 2017) %>% group_by(batter_id) %>% distinct(batter_id) %>% ungroup()
  
  tmp_train <- dat_tmp %>% filter(batter_id %in% batter_id_2017$batter_id, year %in% (2017-annual):2016) %>% arrange(batter_id)
  
  tmp_train <- tibble(batter_id = rep(unique(tmp_train$batter_id), each = annual), year = rep((2017-annual):2016, length(unique(tmp_train$batter_id)))) %>%
    left_join(tmp_train, by = c('batter_id', 'year'))
  
  tmp_train <- apply(tmp_train, 2, function(x) ifelse(is.na(x), 0, x)) %>% as.tibble()
  
  tmp_train <- tmp_train %>% arrange(batter_id)
  
  names(tmp_train)[c(6,7)] <- c('B2','B3')
  
  #### length(batter_id) * (1+annual*14) 의 행렬을 생성(즉, id, 그리고 나머지 14개변수 * annual년)
  train_x <- data.matrix(tmp_train[,-c(1,2)])
  
  train_x <- matrix(as.vector(t(train_x)), nrow = nrow(tmp_train)/annual, byrow = T) %>% as.tibble()
  
  x_name <- NULL
  for(i in annual:1){
    x_name_tmp <- paste(names(tmp_train)[-(2:1)], rep(i,14), sep ='_')
    x_name <- c(x_name, x_name_tmp)
  }
  
  names(train_x) <- x_name
  ##
  
  tmp_train_x <- cbind(batter_id = unique(tmp_train$batter_id), train_x) # train_x에 batter_id있는 data.
  
  
  #### train_y : 2017년도의 AB, OPS
  train_y <- dat %>% filter(year == 2017 , batter_id %in% tmp_train$batter_id) %>% select(batter_id, AB, OPS) %>% arrange(batter_id) 
  
  train_y <- apply(train_y, 2, function(x) ifelse(is.na(x), 0, x)) %>% as.tibble()
  
  #### test data 
  
  batter_id_2018 <- dat_tmp %>% filter(year == 2018) %>% distinct(batter_id) 
  
  tmp_test <- dat_tmp %>% filter(batter_id %in% batter_id_2018$batter_id, year %in% (2018-annual):2017) %>% arrange(batter_id)
  
  tmp_test <- tibble(batter_id = rep(unique(tmp_test$batter_id), each = annual), year = rep((2018-annual):2017, length(unique(tmp_test$batter_id)))) %>% 
    left_join(tmp_test, by = c('batter_id', 'year'))
  
  tmp_test <- apply(tmp_test, 2, function(x) ifelse(is.na(x), 0, x)) %>% as.tibble()
  
  tmp_test <- tmp_test %>% arrange(batter_id)
  
  names(tmp_test)[c(6,7)] <- c('B2','B3')
  
  #### length(batter_id) * (1+annual*14) 의 행렬을 생성(즉, id, 그리고 나머지 14개변수 * annual년)
  test_x <- data.matrix(tmp_test[,-c(1,2)])
  
  test_x <- matrix(as.vector(t(test_x)), nrow = nrow(tmp_test)/annual, byrow = T) %>% as.tibble()
  
  
  names(test_x) <- x_name
  ##
  
  tmp_test_x <- cbind(batter_id = unique(tmp_test$batter_id), test_x) # test_x에 batter_id있는 data.
  
  
  #### test_y : 2018년도의 AB, OPS
  test_y <- dat %>% filter(year == 2018 , batter_id %in% tmp_test$batter_id) %>% select(batter_id, AB, OPS) %>% arrange(batter_id) 
  
  test_y <- apply(test_y, 2, function(x) ifelse(is.na(x), 0, x)) %>% as.tibble()
  return(list(train_x = train_x, train_y = train_y, 
              test_x = test_x, test_y = test_y))
}

save(rsb, rsb_fh, rsb_sh, preprocess_fun, file = 'C:/Users/kyucheol/Dropbox/dacon/dacon/season_and_fun.Rdata')

rm(list = ls())
load(file = 'C:/Users/kyucheol/Dropbox/dacon/dacon/season_and_fun.Rdata')
preprocess_fun()
## preprocess_fun 에는 dat 에 data 즉, rsb, rsb_fh(전반기), rsb_sh(후반기) 의 파일을 넣을 수 있고, 
##                     annual 은 lag개념으로 직전 몇년치를 train데이터로 쓸지 결정함.