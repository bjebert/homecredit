# CV: 0.788
# LB: 0.783

library(xgboost)
library(data.table)

get_predictions <- function(notused, notused2) {
    train_processed <- fread("data/val_train.csv", header = T)
    test_processed <- fread("data/val_test.csv", header = T)
    
    y_tr <- train_processed[, TARGET]
    y_te <- test_processed[, TARGET]
    
    train_processed[, TARGET := NULL]
    test_processed[, TARGET := NULL]
    
    full_test <- fread("data/full_test.csv", header = T)
    
    
    # XGB prediction function ----------------------------------------------------------
    
    
    predict_xgb <- function(train_processed, test_processed, params, x) {
        Dtrain <- xgb.DMatrix(data = as.matrix(train_processed[, x, with = FALSE]), label = y_tr)
        Dtest <- xgb.DMatrix(data = as.matrix(test_processed[, x, with = FALSE]))
        
        mdl <- xgboost(params = params,
                       data = Dtrain,
                       nrounds = params[["nrounds"]],
                       verbose = 1)
        
        predictions <- predict(mdl, Dtest)
        
        return(list(predictions, mdl))
    }
    
    
    params <- list(objective = "binary:logistic",
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
    
    x <- colnames(train_processed)
    result <- predict_xgb(train_processed, test_processed, params, x)
    predictions <- result[[1]]
    model <- result[[2]]
    
    auc <- Metrics::auc(y_te, predictions)  # 0.78753
    Dtest <- xgb.DMatrix(data = as.matrix(full_test))
    
    predictions <- predict(model, Dtest)
    
    return(predictions)
}

