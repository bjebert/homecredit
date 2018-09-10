# Forked from https://www.kaggle.com/kailex/tidy-xgb-all-tables-0-796
# Changed to LGBM model
# Changed to one-hot encoding

library(tidyverse)
library(xgboost)
library(magrittr)
library(lightgbm)
library(onehot)
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
    
    #---------------------------
    cat("Preprocessing...\n")
    
    # Set up a function to automatically get aggregated features 
    # Note that funs() is from the dplyr package. 
    # .args is a named list of additional arguments to be added to the functions
    fn <- funs(mean, sd, min, max, sum, n_distinct, .args = list(na.rm = TRUE))
    
    
    # To get all aggregated features for the bureau balance file
    encoder<- onehot(bbalance, stringsAsFactors=TRUE, addNA=FALSE)
    bbalance<- predict(encoder, bbalance)
    
    sum_bbalance <- bbalance %>%
        as_tibble() %>%
        mutate_if(is.character, funs(factor(.) %>% as.integer)) %>%  #change all categorical variables to numerical variables
        group_by(SK_ID_BUREAU) %>% # Aggregate by SK_ID_BUREAU
        summarise_all(fn) 
    rm(bbalance); gc()
    
    #---- bureau file
    encoder<- onehot(bureau, stringsAsFactors=TRUE, addNA=FALSE, max_levels=5)
    bureau<- predict(encoder, bureau)
    
    sum_bureau <- bureau %>% 
        as.tibble() %>%
        left_join(sum_bbalance, by = "SK_ID_BUREAU") %>% 
        select(-SK_ID_BUREAU) %>% 
        mutate_if(is.character, funs(factor(.) %>% as.integer)) %>% 
        group_by(SK_ID_CURR) %>% # Aggregate by SK_ID_CURR
        summarise_all(fn)
    rm(bureau, sum_bbalance); gc()
    
    #----credit card balance file
    encoder<- onehot(cc_balance, stringsAsFactors=TRUE, addNA=FALSE, max_levels=5)
    cc_balance<- predict(encoder, cc_balance)
    
    sum_cc_balance <- cc_balance %>% 
        as.tibble() %>%
        select(-SK_ID_PREV) %>% 
        mutate_if(is.character, funs(factor(.) %>% as.integer)) %>% 
        group_by(SK_ID_CURR) %>% 
        summarise_all(fn)
    rm(cc_balance); gc()
    
    #-----payments file
    encoder<- onehot(payments, stringsAsFactors=TRUE, addNA=FALSE, max_levels=5)
    payments<- predict(encoder, payments)
    
    sum_payments <- payments %>% 
        as.tibble() %>%
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
    
    #-----pc_balance file
    encoder<- onehot(pc_balance, stringsAsFactors=TRUE, addNA=FALSE, max_levels=5)
    pc_balance<- predict(encoder, pc_balance)
    
    sum_pc_balance <- pc_balance %>% 
        as.tibble() %>%
        select(-SK_ID_PREV) %>% 
        mutate_if(is.character, funs(factor(.) %>% as.integer)) %>% 
        group_by(SK_ID_CURR) %>% 
        summarise_all(fn)
    rm(pc_balance); gc()
    
    #----- prev file
    encoder<- onehot(prev, stringsAsFactors=TRUE, addNA=FALSE, max_levels=5)
    prev<- predict(encoder, prev)
    
    sum_prev <- prev %>%
        as.tibble() %>%
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
    
    #---- Merge files
    
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
    
    dtest<- data.matrix(tr_te[-tri,])
    tr_val<- tr_te[tri,]
    tri<- caret::createDataPartition(y, p = 0.9, list = F) %>% c()
    lgb.train = lgb.Dataset(tr_val[tri, ], label = y[tri])
    lgb.valid = lgb.Dataset(tr_val[-tri, ], label = y[-tri])
    cols<- colnames(tr_te)
    rm(tr_te, y, tri); gc()
    
    #------------------------------
    cat("Training LGBM....\n")
    
    lgb.params<- list(objective = "binary",
                      metric = "auc",
                      num_leaves = 32,
                      max_depth=8,
                      min_data_in_leaf = 10,
                      min_sum_hessian_in_leaf = 40,
                      feature_fraction = 0.95,
                      bagging_fraction = 0.87,
                      bagging_freq = 0,
                      lambda_l1 = 0.04, 
                      lambda_l2 = 0.073,
                      min_gain_to_split=0.02
    )
    
    lgb.model <- lgb.train(params = lgb.params,
                           data = lgb.train,
                           valids = list(val = lgb.valid),
                           learning_rate = 0.02,
                           nrounds = 5000,
                           early_stopping_rounds = 200,
                           eval_freq = 50
    )
    
    # Importance Plot
    # lgb.importance(lgb.model, percentage = TRUE) %>% head(20) %>% kable()
    tree_imp <- lgb.importance(lgb.model, percentage = TRUE) %>% head(30)
    lgb.plot.importance(tree_imp, measure = "Gain")
    
    # Make prediction and submission
    lgb_pred <- predict(lgb.model, data = dtest, n = lgb.model$best_iter)
    
    return(lgb_pred)
}