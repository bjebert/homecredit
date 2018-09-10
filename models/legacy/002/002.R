# Use xgboost instead of h2o.gbm, and try optimising some hyperparameters

# CV: 0.750
# LB: 0.739

set.seed(0)
library(xgboost)


preprocess_data <- function(data) {
    x <- colnames(data)
    
    # Remove columns from data
    x <- x[!(x %in% c("origin", "fold", "SK_ID_CURR", "TARGET", "CODE_GENDER"))]
    
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
    
    data <- data[, x, with = FALSE]
    data <- remove_categorical(data)
    
    return(data)
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


get_predictions <- function(train, test) {
    train_processed <- preprocess_data(train)
    test_processed <- preprocess_data(test)
    
    # Params found using prototype/xgboost_random_search.R
    params <- structure(list(objective = "binary:logistic", eta = 0.145784390016925, 
                             gamma = 5.75170090002939, max_depth = 3L, colsample_bytree = 0.98115893593058, 
                             colsample_bylevel = 0.592773119919002, lambda = 6.65121562220156, 
                             alpha = 3.7460559764877, subsample = 0.774868534761481, nrounds = 243L), 
                        .Names = c("objective", "eta", "gamma", "max_depth", "colsample_bytree", "colsample_bylevel", 
                                   "lambda", "alpha", "subsample", "nrounds"))
    
    Dtrain <- xgb.DMatrix(data = as.matrix(train_processed), label = train[["TARGET"]])
    Dtest <- xgb.DMatrix(data = as.matrix(test_processed))
    
    mdl <- xgboost(params = params,
                   data = Dtrain,
                   nrounds = params[["nrounds"]],
                   verbose = 0)
    
    predictions <- predict(mdl, Dtest)
    
    return(predictions)
}
