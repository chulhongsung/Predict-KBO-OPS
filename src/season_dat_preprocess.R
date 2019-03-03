# rm(list = ls()); gc(reset = T)
library(tidyverse)
library(data.table)

rsbd <- fread('file:///C:/Users/kyucheol/Dropbox/dacon/Regular_Season_Batter_Day_by_Day.csv', encoding = 'UTF-8') %>% as_tibble()

# rsbd를 시즌별 rsb파일로 바꾸는 전처리
rsbd_tmp[!complete.cases(rsbd_tmp), ]

rsbd_tmp <- rsbd_tmp[complete.cases(rsbd_tmp),]

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
