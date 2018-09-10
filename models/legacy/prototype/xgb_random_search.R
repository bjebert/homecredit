library(xgboost)
library(data.table)

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
                   verbose = 0)

    predictions <- predict(mdl, Dtest)

    return(list(predictions, mdl))
}


# params <- list(objective = "binary:logistic",
#           booster = "gbtree",
#           eval_metric = "auc",
#           eta = 0.05,
#           max_depth = 6,
#           min_child_weight = 30,
#           gamma = 0,
#           subsample = 0.85,
#           colsample_bytree = 0.7,
#           colsample_bylevel = 0.632,
#           alpha = 0,
#           lambda = 0,
#           nrounds = 2000)
# 
# x <- colnames(train_processed)
# result <- predict_xgb(train_processed, test_processed, params, x)
# predictions <- result[[1]]
# model <- result[[2]]
# 
# auc <- Metrics::auc(y_te, predictions)  # 0.78753
# Dtest <- xgb.DMatrix(data = as.matrix(full_test))
# 
# predictions <- predict(model, Dtest)

# # Begin random search -----------------------------------------------------

res <- data.table()
cols <- colnames(train_processed)
best_auc <- 0
iter <- 0

while(TRUE) {
    iter <- iter + 1

    # eta <- runif(1, min = 0.01, max = 0.2)
    # gamma <- runif(1, min = 0, max = 10)
    # max_depth <- sample(3:20, 1)
    # colsample_bytree <- runif(1, min = 0.5, max = 1)
    # colsample_bylevel <- runif(1, min = 0.5, max = 1)
    # lambda <- runif(1, min = 0, max = 12)
    # alpha <- runif(1, min = 0, max = 12)
    # subsample <- runif(1, min = 0.5, max = 1)
    # nrounds <- sample(1:300, 1)
    # 
    # params <- list(objective = "binary:logistic",
    #                eval_metric = "auc",
    #                eta = eta,
    #                gamma = gamma,
    #                max_depth = max_depth,
    #                colsample_bytree = colsample_bytree,
    #                colsample_bylevel = colsample_bylevel,
    #                lambda = lambda,
    #                alpha = alpha,
    #                subsample = subsample,
    #                nrounds = nrounds)

    x <- sample(features, sample(1:length(features)), prob = feature_probs)
    params <- sample(params_list, 1, prob = params_probs_norm)[[1]]
    
    result <- predict_xgb(train_processed, test_processed, params, x)
    predictions <- result[[1]]
    model <- result[[2]]

    auc <- Metrics::auc(y_te, predictions)

    gain <- xgb.importance(model, feature_names = cols)

    res <- rbind(res, data.table(params = list(params), x = list(x), auc = auc, gain = list(gain)))

    if(auc > best_auc) {
        print(sprintf("New best AUC: %f (Iteration %d)", auc, iter))
        print(sprintf("Num features: %d", length(x)))
        best_auc = auc
    }
}

# saveRDS(res, "data/big_res.rds")

#
# # Gains
#
gains <- rbindlist(res[, gain])[, .(MeanGain = mean(Gain), N = .N), by = Feature][order(-MeanGain)]
features <- gains[order(-MeanGain), Feature]
feature_probs <- gains[order(-MeanGain), MeanGain]

params_list <- res[order(-auc), params]
params_probs <- res[order(-auc), auc]
params_probs_norm <- (params_probs - min(params_probs)) / (max(params_probs) - min(params_probs))
