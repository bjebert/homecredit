# Use xgboost instead of h2o.gbm, and try optimising some hyperparameters

# CV: ? (best 0.743)
# LB: ? (best 0.729)

set.seed(0)
library(lightgbm)
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
    
    char_cols <- names(which(sapply(sapply(data[, x, with = FALSE], class), function(x) x == "character")))
    x <- x[!(x %in% char_cols)]
    
    data <- data[, x, with = FALSE]
    return(data)
}




get_predictions <- function(train, test) {
    train_processed <- preprocess_data(train)
    test_processed <- preprocess_data(test)
    
    Dtrain <- xgb.DMatrix(data = as.matrix(train_processed), label = train[["TARGET"]])
    Dtest <- xgb.DMatrix(data = as.matrix(test_processed))
    
    mdl <- xgboost(params = params,
                   data = Dtrain,
                   nrounds = params[["nrounds"]],
                   verbose = 0)
    
    predictions <- predict(mdl, Dtest)
    
    return(predictions)
}
