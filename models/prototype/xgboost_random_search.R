set.seed(0)
library(xgboost)


# Initial read ------------------------------------------------------------


applications <- fread("data/applications.csv", header = TRUE)
train <- applications[origin == "train"]
test <- applications[origin == "val"]

# Preprocess data ---------------------------------------------------------


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


train_processed <- preprocess_data(train)
test_processed <- preprocess_data(test)

# XGB prediction function ----------------------------------------------------------


predict_xgb <- function(train_processed, test_processed, params, x) {
    Dtrain <- xgb.DMatrix(data = as.matrix(train_processed[, x, with = FALSE]), label = train[["TARGET"]])
    Dtest <- xgb.DMatrix(data = as.matrix(test_processed[, x, with = FALSE]))
    
    mdl <- xgboost(params = params,
                   data = Dtrain,
                   nrounds = params[["nrounds"]],
                   verbose = 0)
    
    predictions <- predict(mdl, Dtest)
    
    return(predictions)
}


# Begin random search -----------------------------------------------------

res <- data.table()
cols <- colnames(train_processed)
best_auc <- 0
iter <- 0

while(TRUE) {
    iter <- iter + 1
    
    eta <- runif(1, min = 0.01, max = 0.2)
    gamma <- runif(1, min = 0, max = 6)
    max_depth <- sample(3:20, 1)
    colsample_bytree <- runif(1, min = 0.5, max = 1)
    colsample_bylevel <- runif(1, min = 0.5, max = 1)
    lambda <- runif(1, min = 0, max = 12)
    alpha <- runif(1, min = 0, max = 12)
    subsample <- runif(1, min = 0.5, max = 1)
    nrounds <- sample(1:300, 1)
    
    params <- list(objective = "binary:logistic",
                   eta = eta,
                   gamma = gamma,
                   max_depth = max_depth,
                   colsample_bytree = colsample_bytree,
                   colsample_bylevel = colsample_bylevel,
                   lambda = lambda,
                   alpha = alpha,
                   subsample = subsample,
                   nrounds = nrounds)
    
    x <- c(sample(cols, sample(0:length(cols), 1)))
    
    predictions <- predict_xgb(train_processed, test_processed, params, x)
    auc <- Metrics::auc(test[["TARGET"]], predictions)
    
    res <- rbind(res, data.table(params = list(params), x = list(x), auc = auc))
    
    if(auc > best_auc) {
        print(sprintf("New best AUC: %f (Iteration %d)", auc, iter))
        print(sprintf("Num features: %d", length(x)))
        best_auc = auc
    }
}
