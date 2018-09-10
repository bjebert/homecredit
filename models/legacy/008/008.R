# CV: 0.779
# LB: ?

set.seed(0)
library(data.table)
library(xgboost)

source("analysis/auto_merge.R")
source("~/utilities/col_convert.R")
source("~/utilities/one_hot_encode.R")
source("~/utilities/target_encode.R")
source("~/utilities/remove_categorical.R")

# Initial read ------------------------------------------------------------

bureau <- fread("data/bureau.csv")
bureau_balance <- fread("data/bureau_balance.csv")
credit_card_balance <- fread("data/credit_card_balance.csv")
installments <- fread("data/installments_payments.csv")
pos_cash_balance <- fread("data/POS_CASH_balance.csv")
previous_application <- fread("data/previous_application.csv")

# Preprocess data ---------------------------------------------------------


merge_data <- function(data) {
    
    data <- auto_merge(data, bureau, "SK_ID_CURR", "SK_ID_BUREAU")
    data <- auto_merge(data, credit_card_balance, "SK_ID_CURR", "SK_ID_PREV")
    data <- auto_merge(data, pos_cash_balance, "SK_ID_CURR", "SK_ID_BUREAU")
    data <- auto_merge(data, previous_application, "SK_ID_CURR", "SK_ID_PREV")
    data <- auto_merge(data, installments, "SK_ID_CURR", "SK_ID_PREV")
    
    return(data)
}


preprocess_data <- function(data) {
    data <- col_convert(data, "integer", "numeric")  # Convert integer columns to numeric
    
    # Convert 2-valued character columns to boolean
    data[, IS_CONTRACT_CASH := NAME_CONTRACT_TYPE == "Cash loans"]
    data[, IS_MALE := CODE_GENDER == "M"]
    data[, IS_OWN_CAR := FLAG_OWN_CAR == "Y"]
    data[, IS_OWN_REALTY := FLAG_OWN_REALTY == "Y"]
    data[, IS_EMERGENCY_MODE := ifelse(data[["EMERGENCYSTATE_MODE"]] == "Yes", 
                                       TRUE, 
                                       ifelse(data[["EMERGENCYSTATE_MODE"]] == "No", 
                                              FALSE, 
                                              NA))]
    
    # One-hot encode cols
    data[NAME_TYPE_SUITE == "", NAME_TYPE_SUITE := "NA"]
    data[OCCUPATION_TYPE == "", OCCUPATION_TYPE := "NA"]
    data[FONDKAPREMONT_MODE == "", FONDKAPREMONT_MODE := "NA"]
    data[HOUSETYPE_MODE == "", HOUSETYPE_MODE := "NA"]
    data[WALLSMATERIAL_MODE == "", WALLSMATERIAL_MODE := "NA"]
    
    data <- merge_data(data)
    
    encode_cols <- c("NAME_TYPE_SUITE", "NAME_INCOME_TYPE", "NAME_EDUCATION_TYPE", "NAME_FAMILY_STATUS",
                     "NAME_HOUSING_TYPE", "OCCUPATION_TYPE", "WEEKDAY_APPR_PROCESS_START",
                     "ORGANIZATION_TYPE", "FONDKAPREMONT_MODE", "HOUSETYPE_MODE", "WALLSMATERIAL_MODE")
    data <- one_hot_encode(data, encode_cols)
    
    
    
    x <- colnames(data)
    
    # Remove columns from data
    x <- x[!(x %in% c("origin", "fold", "SK_ID_CURR", "CODE_GENDER"))]
    data <- data[, x, with = FALSE]
    
    colnames(data) <- gsub(" ", "_", colnames(data))
    colnames(data) <- gsub(",", "", colnames(data))
    colnames(data) <- gsub(":", "", colnames(data))
    colnames(data) <- gsub("/", "", colnames(data))
    colnames(data) <- toupper(colnames(data))
    
    return(data)
}

get_predictions <- function(train, test) {
    label_target <- train[["TARGET"]]
    # train_processed <- fread("data/train_processed.csv", header = T)
    # test_processed <- fread("data/test_processed.csv", header = T)

    train_processed <- preprocess_data(train)
    test_processed <- preprocess_data(test)

    data <- target_encode(train_processed, test_processed, "TARGET", c("mean", "sd", "count"))
    train_processed <- data[["train"]]
    test_processed <- data[["test"]]

    train_processed <- col_convert(train_processed, "character", "")
    test_processed <- col_convert(test_processed, "character", "")

    rm(data)  # memory

    train_processed[, TARGET := NULL]
    test_processed[, TARGET := NULL]  # Shouldn't be here, but just in case

    fwrite(train_processed, "data/TEST_train_processed.csv", row.names = F)
    fwrite(test_processed, "data/TEST_test_processed.csv", row.names = F)
    
    Dtrain <- xgb.DMatrix(data = as.matrix(train_processed), label = label_target)
    Dtest <- xgb.DMatrix(data = as.matrix(test_processed))
    
    params <- structure(list(objective = "binary:logistic", eta = 0.107155480326619, 
                             gamma = 2.18815732048824, max_depth = 4L, colsample_bytree = 0.778631998924538, 
                             colsample_bylevel = 0.691564935608767, lambda = 8.11068782955408, 
                             alpha = 7.27082939911634, subsample = 0.979085586033762, 
                             nrounds = 246L), .Names = c("objective", "eta", "gamma", 
                                                         "max_depth", "colsample_bytree", "colsample_bylevel", "lambda", 
                                                         "alpha", "subsample", "nrounds"))
    
    mdl <- xgboost(params = params,
                   data = Dtrain,
                   nrounds = params[["nrounds"]],
                   verbose = 0)
    
    predictions <- predict(mdl, Dtest)
    
    return(predictions)
}