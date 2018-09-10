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
    
    
    params <- structure(list(objective = "binary:logistic", eval_metric = "auc", 
                   eta = 0.119453180017881, gamma = 7.81538938404992, max_depth = 5L, 
                   colsample_bytree = 0.542258794768713, colsample_bylevel = 0.922964152530767, 
                   lambda = 2.50033100601286, alpha = 11.3304517809302, subsample = 0.982457390171476, 
                   nrounds = 283L), .Names = c("objective", "eval_metric", "eta", 
                                               "gamma", "max_depth", "colsample_bytree", "colsample_bylevel", 
                                               "lambda", "alpha", "subsample", "nrounds"))
    
    x <- colnames(train_processed)
    result <- predict_xgb(train_processed, test_processed, params, x)
    predictions <- result[[1]]
    model <- result[[2]]
    
    # auc <- Metrics::auc(y_te, predictions)  
    # Dtest <- xgb.DMatrix(data = as.matrix(test_processed))
    
    # predictions <- predict(model, Dtest)
    
    return(predictions)
}

predictions <- get_predictions(1, 2)
auc <- Metrics::auc(y_te, predictions)  
