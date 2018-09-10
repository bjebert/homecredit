library(tidyverse)
library(xgboost)
library(magrittr)
set.seed(0)

get_predictions <- function(train, test) {
    tr <- copy(train)
    te <- copy(test)
    
    #---------------------------
    cat("Loading data...\n")
    
    bbalance <- read_csv("data/bureau_balance.csv") 
    bureau <- read_csv("data/bureau.csv")
    cc_balance <- read_csv("data/credit_card_balance.csv")
    payments <- read_csv("data/installments_payments.csv") 
    pc_balance <- read_csv("data/POS_CASH_balance.csv")
    prev <- read_csv("data/previous_application.csv")
    
    
    #I commented out ones that kxx used for creating new variables (e.g. AMT_CREDIT / AMT_ANNUITY)
    #I am forced to do this because the score is poor if I don't.
    
    #tr$AMT_INCOME_TOTAL <- log1p(tr$AMT_INCOME_TOTAL)
    #tr$AMT_CREDIT <- log1p(tr$AMT_CREDIT)
    #tr$AMT_ANNUITY <- log1p(tr$AMT_ANNUITY)
    #tr$AMT_GOODS_PRICE <- log1p(tr$AMT_GOODS_PRICE)
    tr$REGION_POPULATION_RELATIVE <- sqrt(tr$REGION_POPULATION_RELATIVE)
    #tr$DAYS_BIRTH <- sqrt(abs(tr$DAYS_BIRTH))
    #tr$DAYS_EMPLOYED <- sqrt(abs(tr$DAYS_EMPLOYED))
    tr$DAYS_REGISTRATION <- sqrt(abs(tr$DAYS_REGISTRATION))
    #tr$OWN_CAR_AGE <- sqrt(abs(tr$OWN_CAR_AGE))
    tr$APARTMENTS_AVG <- log1p(50*tr$APARTMENTS_AVG)
    tr$YEARS_BEGINEXPLUATATION_AVG <- (tr$YEARS_BEGINEXPLUATATION_AVG)^30
    tr$YEARS_BUILD_AVG <- (tr$YEARS_BUILD_AVG)^3
    tr$COMMONAREA_AVG <- (tr$COMMONAREA_AVG)^(-1/200)
    tr$ELEVATORS_AVG <- (tr$ELEVATORS_AVG)^(1/40)
    tr$ENTRANCES_AVG <- (tr$ENTRANCES_AVG)^(1/3)
    tr$FLOORSMAX_AVG <- (tr$FLOORSMAX_AVG)^(1/2.5)
    tr$FLOORSMIN_AVG <- (tr$FLOORSMIN_AVG)^(1/2.2)
    tr$LANDAREA_VG <- (tr$LANDAREA_AVG)^(1/5)
    tr$LIVINGAPRTMENTS_AVG <- (tr$LIVINGAPARTMENTS_AVG)^(1/3)
    tr$LIVINGAREA_AVG <- (tr$LIVINGAREA_AVG)^(1/3.5)
    tr$NONLIVINGAPARTMENTS_AVG <- (tr$NONLIVINGAPARTMENTS_AVG)^(1/7)
    tr$NONLIVINGAREA_AVG <- (tr$NONLIVINGAREA_AVG)^(1/5)
    tr$TOTALAREA_MODE <- (tr$TOTALAREA_MODE)^(1/3)
    tr$OBS_30_CNT_SOCIAL_CIRCLE <- (tr$OBS_30_CNT_SOCIAL_CIRCLE)^(1/7)
    tr$DEF_30_CNT_SOCIAL_CIRCLE <- (tr$DEF_30_CNT_SOCIAL_CIRCLE)^(1/7)
    tr$OBS_60_CNT_SOCIAL_CIRCLE <- (tr$OBS_60_CNT_SOCIAL_CIRCLE)^(1/7)
    tr$DEF_60_CNT_SOCIAL_CIRCLE <- (tr$DEF_60_CNT_SOCIAL_CIRCLE)^(1/7)
    #tr$DAYS_LAST_PHONE_CHANGE <- (abs(tr$DAYS_LAST_PHONE_CHANGE))^(1/2)
    
    #te$AMT_INCOME_TOTAL <- log1p(te$AMT_INCOME_TOTAL)
    #te$AMT_CREDIT <- log1p(te$AMT_CREDIT)
    #te$AMT_ANNUITY <- log1p(te$AMT_ANNUITY)
    #te$AMT_GOODS_PRICE <- log1p(te$AMT_GOODS_PRICE)
    te$REGION_POPULATION_RELATIVE <- sqrt(te$REGION_POPULATION_RELATIVE)
    #te$DAYS_BIRTH <- sqrt(abs(te$DAYS_BIRTH))
    #te$DAYS_EMPLOYED <- sqrt(abs(te$DAYS_EMPLOYED))
    te$DAYS_REGISTRATION <- sqrt(abs(te$DAYS_REGISTRATION))
    #te$OWN_CAR_AGE <- sqrt(abs(te$OWN_CAR_AGE))
    te$APARTMENTS_AVG <- log1p(50*te$APARTMENTS_AVG)
    te$YEARS_BEGINEXPLUATATION_AVG <- (te$YEARS_BEGINEXPLUATATION_AVG)^30
    te$YEARS_BUILD_AVG <- (te$YEARS_BUILD_AVG)^3
    te$COMMONAREA_AVG <- (te$COMMONAREA_AVG)^(-1/200)
    te$ELEVATORS_AVG <- (te$ELEVATORS_AVG)^(1/40)
    te$ENTRANCES_AVG <- (te$ENTRANCES_AVG)^(1/3)
    te$FLOORSMAX_AVG <- (te$FLOORSMAX_AVG)^(1/2.5)
    te$FLOORSMIN_AVG <- (te$FLOORSMIN_AVG)^(1/2.2)
    te$LANDAREA_VG <- (te$LANDAREA_AVG)^(1/5)
    te$LIVINGAPRTMENTS_AVG <- (te$LIVINGAPARTMENTS_AVG)^(1/3)
    te$LIVINGAREA_AVG <- (te$LIVINGAREA_AVG)^(1/3.5)
    te$NONLIVINGAPARTMENTS_AVG <- (te$NONLIVINGAPARTMENTS_AVG)^(1/7)
    te$NONLIVINGAREA_AVG <- (te$NONLIVINGAREA_AVG)^(1/5)
    te$TOTALAREA_MODE <- (te$TOTALAREA_MODE)^(1/3)
    te$OBS_30_CNT_SOCIAL_CIRCLE <- (te$OBS_30_CNT_SOCIAL_CIRCLE)^(1/7)
    te$DEF_30_CNT_SOCIAL_CIRCLE <- (te$DEF_30_CNT_SOCIAL_CIRCLE)^(1/7)
    te$OBS_60_CNT_SOCIAL_CIRCLE <- (te$OBS_60_CNT_SOCIAL_CIRCLE)^(1/7)
    te$DEF_60_CNT_SOCIAL_CIRCLE <- (te$DEF_60_CNT_SOCIAL_CIRCLE)^(1/7)
    #te$DAYS_LAST_PHONE_CHANGE <- (abs(te$DAYS_LAST_PHONE_CHANGE))^(1/2)
    
    #---------------------------
    cat("Preprocessing...\n")
    
    fn <- funs(mean, sd, min, max, sum, n_distinct, .args = list(na.rm = TRUE))
    
    sum_bbalance <- bbalance %>%
        mutate_if(is.character, funs(factor(.) %>% as.integer)) %>% 
        group_by(SK_ID_BUREAU) %>% 
        summarise_all(fn) 
    rm(bbalance); gc()
    
    sum_bureau <- bureau %>% 
        left_join(sum_bbalance, by = "SK_ID_BUREAU") %>% 
        select(-SK_ID_BUREAU) %>% 
        mutate_if(is.character, funs(factor(.) %>% as.integer)) %>% 
        group_by(SK_ID_CURR) %>% 
        summarise_all(fn)
    rm(bureau, sum_bbalance); gc()
    
    sum_cc_balance <- cc_balance %>% 
        select(-SK_ID_PREV) %>% 
        mutate_if(is.character, funs(factor(.) %>% as.integer)) %>% 
        group_by(SK_ID_CURR) %>% 
        summarise_all(fn)
    rm(cc_balance); gc()
    
    sum_payments <- payments %>% 
        select(-SK_ID_PREV) %>% 
        mutate(PAYMENT_PERC = AMT_PAYMENT / AMT_INSTALMENT,
               PAYMENT_DIFF = AMT_INSTALMENT - AMT_PAYMENT,
               DPD = DAYS_ENTRY_PAYMENT - DAYS_INSTALMENT,
               DBD = DAYS_INSTALMENT - DAYS_ENTRY_PAYMENT,
               DPD = ifelse(DPD > 0, DPD, 0),
               DBD = ifelse(DBD > 0, DBD, 0)) %>% 
        group_by(SK_ID_CURR) %>% 
        summarise_all(fn) 
    rm(payments); gc()
    
    sum_pc_balance <- pc_balance %>% 
        select(-SK_ID_PREV) %>% 
        mutate_if(is.character, funs(factor(.) %>% as.integer)) %>% 
        group_by(SK_ID_CURR) %>% 
        summarise_all(fn)
    rm(pc_balance); gc()
    
    sum_prev <- prev %>%
        select(-SK_ID_PREV) %>% 
        mutate_if(is.character, funs(factor(.) %>% as.integer)) %>% 
        mutate(DAYS_FIRST_DRAWING = ifelse(DAYS_FIRST_DRAWING == 365243, NA, DAYS_FIRST_DRAWING),
               DAYS_FIRST_DUE = ifelse(DAYS_FIRST_DUE == 365243, NA, DAYS_FIRST_DUE),
               DAYS_LAST_DUE_1ST_VERSION = ifelse(DAYS_LAST_DUE_1ST_VERSION == 365243, NA, DAYS_LAST_DUE_1ST_VERSION),
               DAYS_LAST_DUE = ifelse(DAYS_LAST_DUE == 365243, NA, DAYS_LAST_DUE),
               DAYS_TERMINATION = ifelse(DAYS_TERMINATION == 365243, NA, DAYS_TERMINATION),
               APP_CREDIT_PERC = AMT_APPLICATION / AMT_CREDIT) %>% 
        group_by(SK_ID_CURR) %>% 
        summarise_all(fn) 
    rm(prev); gc()
    
    tri <- 1:nrow(tr)
    y <- tr$TARGET
    
    tr_te <- tr %>% 
        select(-TARGET) %>% 
        bind_rows(te) %>%
        left_join(sum_bureau, by = "SK_ID_CURR") %>% 
        left_join(sum_cc_balance, by = "SK_ID_CURR") %>% 
        left_join(sum_payments, by = "SK_ID_CURR") %>% 
        left_join(sum_pc_balance, by = "SK_ID_CURR") %>% 
        left_join(sum_prev, by = "SK_ID_CURR") %>% 
        select(-SK_ID_CURR) %>% 
        mutate_if(is.character, funs(factor(.) %>% as.integer)) %>% 
        mutate(na = apply(., 1, function(x) sum(is.na(x))),
               DAYS_EMPLOYED = ifelse(DAYS_EMPLOYED == 365243, NA, DAYS_EMPLOYED),
               DAYS_EMPLOYED_PERC = sqrt(DAYS_EMPLOYED / DAYS_BIRTH),
               INCOME_CREDIT_PERC = AMT_INCOME_TOTAL / AMT_CREDIT,
               INCOME_PER_PERSON = log1p(AMT_INCOME_TOTAL / CNT_FAM_MEMBERS),
               ANNUITY_INCOME_PERC = sqrt(AMT_ANNUITY / (1 + AMT_INCOME_TOTAL)),
               LOAN_INCOME_RATIO = AMT_CREDIT / AMT_INCOME_TOTAL,
               ANNUITY_LENGTH = AMT_CREDIT / AMT_ANNUITY,
               CHILDREN_RATIO = CNT_CHILDREN / CNT_FAM_MEMBERS, 
               CREDIT_TO_GOODS_RATIO = AMT_CREDIT / AMT_GOODS_PRICE,
               INC_PER_CHLD = AMT_INCOME_TOTAL / (1 + CNT_CHILDREN),
               SOURCES_PROD = EXT_SOURCE_1 * EXT_SOURCE_2 * EXT_SOURCE_3,
               CAR_TO_BIRTH_RATIO = OWN_CAR_AGE / DAYS_BIRTH,
               CAR_TO_EMPLOY_RATIO = OWN_CAR_AGE / DAYS_EMPLOYED,
               PHONE_TO_BIRTH_RATIO = DAYS_LAST_PHONE_CHANGE / DAYS_BIRTH,
               PHONE_TO_EMPLOY_RATIO = DAYS_LAST_PHONE_CHANGE / DAYS_EMPLOYED) 
    
    docs <- str_subset(names(tr), "FLAG_DOC")
    live <- str_subset(names(tr), "(?!NFLAG_)(?!FLAG_DOC)(?!_FLAG_)FLAG_")
    inc_by_org <- tr_te %>% 
        group_by(ORGANIZATION_TYPE) %>% 
        summarise(m = median(AMT_INCOME_TOTAL)) %$% 
        setNames(as.list(m), ORGANIZATION_TYPE)
    
    rm(tr, te, fn, sum_bureau, sum_cc_balance, 
       sum_payments, sum_pc_balance, sum_prev); gc()
    
    tr_te %<>% 
        mutate(DOC_IND_KURT = apply(tr_te[, docs], 1, moments::kurtosis),
               LIVE_IND_SUM = apply(tr_te[, live], 1, sum),
               NEW_INC_BY_ORG = recode(tr_te$ORGANIZATION_TYPE, !!!inc_by_org),
               NEW_EXT_SOURCES_MEAN = apply(tr_te[, c("EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3")], 1, mean),
               NEW_SCORES_STD = apply(tr_te[, c("EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3")], 1, sd))%>%
        mutate_all(funs(ifelse(is.nan(.), NA, .))) %>% 
        mutate_all(funs(ifelse(is.infinite(.), NA, .))) %>% 
        data.matrix()
    
    #---------------------------
    cat("Preparing data...\n")
    dtest <- xgb.DMatrix(data = tr_te[-tri, ])
    tr_te <- tr_te[tri, ]
    tri <- caret::createDataPartition(y, p = 0.9, list = F) %>% c()
    dtrain <- xgb.DMatrix(data = tr_te[tri, ], label = y[tri])
    dval <- xgb.DMatrix(data = tr_te[-tri, ], label = y[-tri])
    cols <- colnames(tr_te)
    
    rm(tr_te, y, tri); gc()
    
    #---------------------------
    cat("Training model...\n")
    p <- list(objective = "binary:logistic",
              booster = "gbtree",
              eval_metric = "auc",
              eta = 0.05,
              max_depth = 6,
              min_child_weight = 30,
              gamma = 0,
              subsample = 0.85,
              colsample_bytree = 0.7,
              colsample_bylevel = 0.632,
              alpha = 0,
              lambda = 0,
              nrounds = 2000)
    
    set.seed(0)
    m_xgb <- xgb.train(p, dtrain, p$nrounds, list(val = dval), print_every_n = 50, early_stopping_rounds = 300)
    
    predictions <- predict(m_xgb, dtest)
    
    return(predictions)
}