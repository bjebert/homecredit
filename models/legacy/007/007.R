# CV: 0.774
# LB: 0.762

set.seed(0)
library(xgboost)

bureau <- fread("data/bureau.csv")
bureau_balance <- fread("data/bureau_balance.csv")
credit_card_balance <- fread("data/credit_card_balance.csv")
installments <- fread("data/installments_payments.csv")
pos_cash_balance <- fread("data/POS_CASH_balance.csv")
previous_application <- fread("data/previous_application.csv")


merge_data <- function(data) {
    
    # Merge installments
    ins_summary <- installments[, .(UNDERPAY_RATIO = mean(AMT_PAYMENT < AMT_INSTALMENT),
                                    UNDERPAY_COUNT = sum(AMT_PAYMENT < AMT_INSTALMENT),
                                    PAY_RATIO = mean(AMT_PAYMENT / AMT_INSTALMENT),
                                    PAY_COUNT = .N,
                                    LATE_PAYMENTS = sum(DAYS_ENTRY_PAYMENT > DAYS_INSTALMENT),
                                    MEAN_INSTALMENT = mean(AMT_INSTALMENT),
                                    SUM_INSTALMENTS = sum(AMT_INSTALMENT)), by = SK_ID_CURR]
    
    ins_late <- installments[DAYS_ENTRY_PAYMENT > DAYS_INSTALMENT, 
                             .(MEAN_DAYS_LATE = mean(DAYS_ENTRY_PAYMENT - DAYS_INSTALMENT),
                               MEDIAN_DAYS_LATE = median(DAYS_ENTRY_PAYMENT - DAYS_INSTALMENT)),
                             by = SK_ID_CURR]
    
    ins_summary <- merge(ins_summary, ins_late, all.x = TRUE)
    ins_summary[, LATE_PAY_RATIO := LATE_PAYMENTS / PAY_COUNT]
    
    data <- merge(data, ins_summary, by = "SK_ID_CURR", sort = FALSE, all.x = TRUE)
    
    # Merge others
    data <- auto_merge(data, bureau, "SK_ID_CURR", "SK_ID_BUREAU")
    data <- auto_merge(data, credit_card_balance, "SK_ID_CURR", "SK_ID_PREV")
    data <- auto_merge(data, pos_cash_balance, "SK_ID_CURR", "SK_ID_BUREAU")
    data <- auto_merge(data, previous_application, "SK_ID_CURR", "SK_ID_PREV")
    
    return(data)
}


preprocess_data <- function(data) {
    data <- int_to_num_cols(data)  # Convert integer columns to numeric
    
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
    
    # Remove the remaining categorical features
    data <- remove_categorical(data)
    
    x <- colnames(data)
    
    # Remove columns from data
    x <- x[!(x %in% c("origin", "fold", "SK_ID_CURR", "TARGET", "CODE_GENDER"))]
    data <- data[, x, with = FALSE]
    
    return(data)
}


Mode <- function(x) {
    ux <- unique(x)
    ux[which.max(tabulate(match(x, ux)))]
}


remove_categorical <- function(data) {
    data <- as.data.frame(data)
    
    logical_cols <- which(sapply(1:ncol(data), function(x) is.logical(data[, x])))
    data[, logical_cols] <- sapply(data[, logical_cols], as.numeric)
    
    integer_cols <- which(sapply(1:ncol(data), function(x) is.integer(data[, x])))
    data[, integer_cols] <- sapply(data[, integer_cols], as.numeric)
    
    numeric_cols <- which(sapply(1:ncol(data), function(x) is.numeric(data[, x])))
    data <- data[, numeric_cols] 
    
    return(as.data.table(data))
}


one_hot_encode <- function(data, encode_cols) {
    for(ec in encode_cols) {
        onehot_tmp <- as.data.table(model.matrix(~data[[ec]]))
        colnames(onehot_tmp) <- gsub("data\\[\\[ec\\]\\]", sprintf("%s_", ec), colnames(onehot_tmp))
        onehot_tmp[, 1 := NULL]  # Delete intercept
        
        data <- cbind(data, onehot_tmp)
    }
    
    return(data)
}


get_predictions <- function(train, test) {
    train_processed <- preprocess_data(train)
    test_processed <- preprocess_data(test)
    
    params <- list(objective = "binary:logistic",
                   eta = 0.0509253,
                   gamma = 5.126576,
                   max_depth = 10,
                   colsample_bytree = 0.6597428,
                   colsample_bylevel = 0.8344905,
                   lambda = 2.257446,
                   alpha = 5.97798,
                   subsample = 0.999185,
                   nrounds = 134)
    
    Dtrain <- xgb.DMatrix(data = as.matrix(train_processed), label = train[["TARGET"]])
    Dtest <- xgb.DMatrix(data = as.matrix(test_processed))
    
    mdl <- xgboost(params = params,
                   data = Dtrain,
                   nrounds = params[["nrounds"]],
                   verbose = 0)
    
    predictions <- predict(mdl, Dtest)
    
    return(predictions)
}
